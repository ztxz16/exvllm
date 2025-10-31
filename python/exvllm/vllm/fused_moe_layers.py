# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Some code (cuda graph mainly), copy from: https://github.com/guqiong96/Lvllm

from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Callable, Literal, Optional, Union, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG, FusedMoEConfig, FusedMoEParallelConfig,
    FusedMoEQuantConfig, biased_moe_quant_config)
# yapf: enable
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat, FusedMoEModularKernel,
    FusedMoEPermuteExpertsUnpermute, FusedMoEPrepareAndFinalize)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    is_rocm_aiter_moe_enabled)
from vllm.model_executor.layers.fused_moe.routing_simulator import (
    RoutingSimulator)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import (cdiv, direct_register_custom_op, has_deep_ep, has_pplx,
                        round_up)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts, fused_experts
    if has_pplx():
        from .pplx_prepare_finalize import (PplxPrepareAndFinalize,
                                            pplx_hidden_dim_scale_bytes)
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import (DEEPEP_QUANT_BLOCK_SHAPE,
                                                 DeepEPLLPrepareAndFinalize)
else:
    fused_experts = None  # type: ignore
    FusedMoEPermuteExpertsUnpermute = None  # type: ignore
    FusedMoEPrepareAndFinalize = None  # type: ignore
if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
        rocm_aiter_grouped_topk as grouped_topk)
elif current_platform.is_cpu():
    pass
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore


logger = init_logger(__name__)
import threading
import ctypes
import numpy as np
import vllm

from exvllm import ft_kernel
ft_moe_enabled = True

class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):

    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.fused_experts: Optional[FusedMoEModularKernel] = None
        self.topk_indices_dtype = None

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    @staticmethod
    def _maybe_make_prepare_finalize(
        moe: FusedMoEConfig,
        quant_config: Optional[FusedMoEQuantConfig],
    ) -> Optional[FusedMoEPrepareAndFinalize]:
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None

        prepare_finalize: Optional[FusedMoEPrepareAndFinalize] = None

        # TODO: could allow this now
        assert not moe.use_flashinfer_cutlass_kernels, \
            "Must be created in modelopt.py"

        if moe.use_pplx_kernels:
            assert quant_config is not None

            hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
                moe.max_num_tokens,
                moe.hidden_dim,
                moe.in_dtype,
                quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )

            all_to_all_args = dict(
                max_num_tokens=moe.max_num_tokens,
                num_experts=moe.num_experts,
                experts_per_token=moe.experts_per_token,  # topk
                rank=all2all_manager.rank,
                world_size=all2all_manager.world_size,
                # dp_size actually means tp_size, bug in pplx kernels
                dp_size=all2all_manager.tp_group.world_size,
                hidden_dim=moe.hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=hidden_scale_bytes,
            )

            num_dispatchers = (all2all_manager.world_size //
                               all2all_manager.tp_group.world_size)

            # Intranode pplx a2a takes a group name while internode does not.
            if not all2all_manager.internode:
                all_to_all_args[
                    "group_name"] = all2all_manager.cpu_group.group_name

            handle = all2all_manager.get_handle(all_to_all_args)

            prepare_finalize = PplxPrepareAndFinalize(
                handle,
                max_num_tokens=moe.max_num_tokens,
                num_local_experts=moe.num_local_experts,
                num_dispatchers=num_dispatchers,
            )
        elif moe.use_deepep_ht_kernels:
            assert moe.dp_size == all2all_manager.dp_world_size

            all_to_all_args = dict()
            handle = all2all_manager.get_handle(all_to_all_args)
            prepare_finalize = DeepEPHTPrepareAndFinalize(
                handle,
                num_dispatchers=all2all_manager.world_size,
                dp_size=all2all_manager.dp_world_size,
                rank_expert_offset=all2all_manager.rank *
                moe.num_local_experts,
            )

        elif moe.use_deepep_ll_kernels:
            assert quant_config is not None
            all_to_all_args = dict(
                max_num_tokens_per_dp_rank=moe.max_num_tokens,
                token_hidden_size=moe.hidden_dim,
                num_ep_ranks=all2all_manager.world_size,
                num_global_experts=moe.num_experts,
                num_local_experts=moe.num_experts //
                all2all_manager.world_size)
            handle = all2all_manager.get_handle(all_to_all_args)

            # Note: We may want to use FP8 dispatch just to reduce
            # data movement.
            use_fp8_dispatch = (
                quant_config.quant_dtype == current_platform.fp8_dtype()
                and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE)

            prepare_finalize = DeepEPLLPrepareAndFinalize(
                handle,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
            )

        return prepare_finalize

    def maybe_make_prepare_finalize(
            self) -> Optional[FusedMoEPrepareAndFinalize]:
        if self.moe.moe_parallel_config.use_all2all_kernels:
            return FusedMoEMethodBase._maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config)
        else:
            return None

    # Note: init_prepare_finalize should only be called by
    # prepare_communication_buffer_for_model.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        assert self.moe is not None

        # We must get the quant config here so that the layer is
        # completely initialized, i.e. all weights loaded and post
        # processed.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = self.maybe_make_prepare_finalize()

        if prepare_finalize is not None:
            logger.debug("%s for %s(%s)", prepare_finalize.__class__.__name__,
                         self, id(self))
            assert self.topk_indices_dtype is None
            assert self.fused_experts is None, \
                f"Attempt to override experts for {id(self)}!"
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            experts = self.select_gemm_impl(prepare_finalize, layer)
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                experts,
                layer.shared_experts,
            )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize")

    @abstractmethod
    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

from vllm.scalar_type import scalar_types

@CustomOp.register("exvllm_awq_fused_moe")
class ExvllmAwqFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""
    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.weight_bits != 4:
            raise ValueError("AWQMoEMethod only supports 4bit now.")
        self.quant_type = scalar_types.uint4

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        device = torch.cuda.current_device() if current_platform.is_cuda_alike() else "cpu"
        if isinstance(layer, FusedMoE) and ft_moe_enabled:
            device = "cpu"   

        extra_weight_attrs.update({
            "is_transposed":
            True,
            "quant_method":
            FusedMoeWeightScaleSupported.GROUP.value,
        })

        w13_qweight = torch.nn.Parameter(
            torch.empty(num_experts,
                        hidden_size,
                        2 * intermediate_size_per_partition //
                        self.quant_config.pack_factor,
                        dtype=torch.int32, device = device),
            requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(torch.empty(num_experts,
                                           intermediate_size_per_partition,
                                           hidden_size //
                                           self.quant_config.pack_factor,
                                           dtype=torch.int32, device = device),
                               requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = (intermediate_size_per_partition //
                         self.quant_config.group_size)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = torch.nn.Parameter(torch.empty(num_experts,
                                           num_groups_w13,
                                           intermediate_size_per_partition * 2,
                                           dtype=params_dtype, device = device),
                               requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(torch.empty(num_experts,
                                          num_groups_w2,
                                          hidden_size,
                                          dtype=params_dtype, device = device),
                              requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = torch.nn.Parameter(
            torch.empty(num_experts,
                        num_groups_w13,
                        2 * intermediate_size_per_partition //
                        self.quant_config.pack_factor,
                        dtype=torch.int32, device = device),
            requires_grad=False)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = torch.nn.Parameter(torch.empty(num_experts,
                                          num_groups_w2,
                                          hidden_size //
                                          self.quant_config.pack_factor,
                                          dtype=torch.int32, device = device),
                              requires_grad=False)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

        # device = layer.w13_qweight.device
        # layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # skip
        ...

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ...


@CustomOp.register("exvllm_fp8_fused_moe")
class ExvllmFp8FusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None

        self.flashinfer_moe_backend = None
        self.fused_experts: Optional[
            mk.FusedMoEModularKernel] = None  # type: ignore
        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = (not current_platform.has_device_capability(89)
                           or envs.VLLM_TEST_FORCE_FP8_MARLIN)
        # Disable marlin for rocm
        if current_platform.is_rocm():
            self.use_marlin = False

        # Check for DeepGemm support.
        self.allow_deep_gemm = False

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        device = torch.cuda.current_device() if current_platform.is_cuda_alike() else "cpu"
        if isinstance(layer, FusedMoE) and ft_moe_enabled:
            device = "cpu"   
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn
        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}.")
            if (tp_size > 1
                    and intermediate_size_per_partition % block_k != 0):
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}.")

        # WEIGHTS
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype, device = device),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype, device = device),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, 2, dtype=torch.float32, device = device),
                                                  requires_grad=False)
            w2_weight_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32, device = device),
                                                 requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) //
                         block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32, device = device,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32, device = device,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
             value} if self.block_quant else
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8.")

            w13_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32, device = device),
                                                 requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(torch.ones(
                num_experts, dtype=torch.float32, device = device),
                                                requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        ...

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        return None

@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts
            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None  # type: ignore

    def maybe_make_prepare_finalize(
            self) -> Optional[FusedMoEPrepareAndFinalize]:
        if self.rocm_aiter_moe_enabled:
            return None
        else:
            return super().maybe_make_prepare_finalize()

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (prepare_finalize.activation_format ==
                FusedMoEActivationFormat.BatchedExperts):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(self.moe_quant_config)

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        device = torch.cuda.current_device() if current_platform.is_cuda_alike() else "cpu"
        if isinstance(layer, FusedMoE) and ft_moe_enabled:
            device = "cpu"   
        # Fused gate_up_proj (column parallel) 
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype, device=device),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype, device=device),
                                          requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype, device=device),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts,
                                                     hidden_size,
                                                     dtype=params_dtype, device="cpu"),
                                         requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm()
                and weight.stride(-1) == 1
                and (weight.stride(-2) * weight.element_size()) % 512 == 0):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # print("into process")
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        # Lazy import to avoid importing triton.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            shuffle_weights)

        if self.rocm_aiter_moe_enabled:
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data)

            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2

        if current_platform.is_xpu():
            import intel_extension_for_pytorch as ipex
            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
            )
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe
            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import (
                    check_cpu_sgl_kernel)
                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (envs.VLLM_CPU_SGL_KERNEL
                        and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                        and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight)
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight)
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    layer.cpu_fused_moe = cpu_fused_moe.IPEXFusedMOE(layer)
            else:
                layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module) -> Optional[FusedMoEQuantConfig]:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count)

        if self.rocm_aiter_moe_enabled:
            assert self.fused_experts is None
            return self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input)
        elif self.fused_experts is not None:
            if self.moe.has_bias:
                raise ValueError(
                    "FusedMoEModularKernel does not support bias.")
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        else:
            assert fused_experts is not None
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for CPU.")
        return layer.cpu_fused_moe(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for XPU.")
        assert custom_routing_function is None
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU.")
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU.")
        assert activation == "silu", f"{activation} is not supported for TPU."
        assert routed_scaling_factor == 1.0, \
            f"routed_scaling_factor {routed_scaling_factor} is not supported " \
            f"for TPU."
        if enable_eplb is not False or expert_load_view is not None or \
                logical_to_physical_map is not None or \
                logical_replica_count is not None:
            raise NotImplementedError("Expert load balancing is not supported "
                                      "for TPU.")
        return fused_moe_pallas(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                global_num_experts=global_num_experts,
                                expert_map=expert_map,
                                renormalize=renormalize)

    if current_platform.is_tpu():
        forward_native = forward_tpu
    elif current_platform.is_cpu():
        forward_native = forward_cpu
    elif current_platform.is_xpu():
        forward_native = forward_xpu
    else:
        forward_native = forward_cuda


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
) -> tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size: The size of the expert parallel group
            ep_rank: The rank of the current process in the expert parallel
                group
            global_num_experts: The total number of experts in the model.
            expert_placement_strategy: The expert placement strategy.

        Returns:
            tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    if ep_rank < remainder:
        local_num_experts = base_experts + 1
    else:
        local_num_experts = base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx:start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32)
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(ep_rank,
                                         global_num_experts,
                                         ep_size,
                                         dtype=torch.int32)

        expert_map[local_log_experts] = torch.arange(0,
                                                     local_num_experts,
                                                     dtype=torch.int32)
    else:
        raise ValueError("Unsupported expert placement strategy "
                         f"'{expert_placement_strategy}', expected one of "
                         f"{get_args(ExpertPlacementStrategy)}")
    return (local_num_experts, expert_map)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
        Compresses the expert map by removing any -1 entries.

        Args:
            expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
                mapping from global to local index. Contains -1 for experts not
                assigned to the current rank.

        Returns:
            str: A string mapping from local to global index.
                Using str to support hashing for logging once only.
        """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices))


@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        use_ft_moe: bool = True,
    ):
        super().__init__()
        self.use_ft_moe = use_ft_moe and ft_moe_enabled
        self.ft_moe = None
        
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        tp_size_ = (tp_size if tp_size is not None else
                    get_tensor_model_parallel_world_size())
        dp_size_ = (dp_size
                    if dp_size is not None else get_dp_group().world_size)

        self.is_sequence_parallel = is_sequence_parallel
        if self.is_sequence_parallel:
            self.sp_size = tp_size_

        vllm_config = get_current_vllm_config()
        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=tp_size_,
                dp_size_=dp_size_,
                vllm_parallel_config=vllm_config.parallel_config))

        self.global_num_experts = num_experts + num_redundant_experts

        # we are padding globally so EP buffer allocation works
        if quant_config and quant_config.get_name() == "mxfp4":
            from vllm.model_executor.layers.quantization.mxfp4 import (
                Mxfp4Backend, get_mxfp4_backend)
            current_mxfp4_backend = get_mxfp4_backend()
            if (current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
                    or current_mxfp4_backend
                    == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS):
                hidden_size = round_up(hidden_size, 128)
            elif (current_platform.is_rocm() or current_mxfp4_backend
                  == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM or
                  current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16):
                hidden_size = round_up(hidden_size, 256)

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: Optional[torch.Tensor] = None
        self.logical_to_physical_map: Optional[torch.Tensor] = None
        self.logical_replica_count: Optional[torch.Tensor] = None

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, \
                    "EPLB currently only supports even distribution of " \
                    "experts across ranks."
            else:
                assert num_redundant_experts == 0, \
                    "Redundant experts are only supported with EPLB."

            expert_placement_strategy = (
                vllm_config.parallel_config.expert_placement_strategy)
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = ((num_expert_group is not None
                                          and num_expert_group > 1)
                                         and num_redundant_experts == 0
                                         and not self.enable_eplb)

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement.")
                    expert_placement_strategy = "linear"

            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
            )
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, expert_placement_strategy,
                self.local_num_experts, self.global_num_experts,
                get_compressed_expert_map(self.expert_map))
        else:
            self.local_num_experts, self.expert_map = (self.global_num_experts,
                                                       None)

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if vllm_config.model_config is not None:
            model_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            model_dtype = params_dtype

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=model_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
        )
        self.moe_config = moe
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None
        
        from vllm.model_executor.layers.quantization.awq import AWQConfig
        from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        if (isinstance(quant_config, Fp8Config)):
            quant_method = ExvllmFp8FusedMoEMethod(quant_config, moe)
        elif (isinstance(quant_config, AWQMarlinConfig)):
            quant_method = ExvllmAwqFusedMoEMethod(quant_config, moe)
        elif isinstance(quant_config, AWQConfig):
            raise("AWQConfig not support")
        else:
            quant_method = UnquantizedFusedMoEMethod(moe);  

        # quant_method = quant_config.get_quant_method(self, prefix)

        # print("quant_config", quant_config)
        # print("quant_method", quant_method)        

        #quant_method = (UnquantizedFusedMoEMethod(moe) if quant_config is None
        #                else quant_config.get_quant_method(self, prefix))

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if self.enable_eplb:
            from vllm.model_executor.layers.quantization.fp8 import (
                Fp8MoEMethod)
            if not isinstance(quant_method,
                              (Fp8MoEMethod, UnquantizedFusedMoEMethod)):
                # TODO: Add support for additional quantization methods.
                # The implementation for other quantization methods does not
                # contain essential differences, but the current quant API
                # design causes duplicated work when extending to new
                # quantization methods, so I'm leaving it for now.
                # If you plan to add support for more quantization methods,
                # please refer to the implementation in `Fp8MoEMethod`.
                raise NotImplementedError("EPLB is only supported for FP8 "
                                          "quantization for now.")

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod",
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: Optional[torch.Tensor] = None
        self.batched_router_logits: Optional[torch.Tensor] = None

        # TODO(bnell): flashinfer uses non-batched format.
        # Does it really need a batched buffer?
        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or self.moe_config.use_flashinfer_cutlass_kernels):
            if vllm_config.parallel_config.enable_dbo:
                self.batched_hidden_states = torch.zeros(
                    (2, moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (2, moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())
            else:
                self.batched_hidden_states = torch.zeros(
                    (moe.max_num_tokens, self.hidden_size),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

                # Note here we use `num_experts` which is logical expert count
                self.batched_router_logits = torch.zeros(
                    (moe.max_num_tokens, num_experts),
                    dtype=moe.in_dtype,
                    device=torch.cuda.current_device())

    @property
    def shared_experts(self) -> Optional[torch.nn.Module]:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (self.moe_quant_config is not None
                and self.moe_quant_config.quant_dtype == "nvfp4"
                and self.moe_config.use_flashinfer_cutlass_kernels)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self.expert_map is not None
        with self.expert_map.device:
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)

    def _load_per_tensor_weight_scale(self, shard_id: str,
                                      param: torch.nn.Parameter,
                                      loaded_weight: torch.Tensor,
                                      expert_id: int):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(self, shard_dim: int,
                                        loaded_weight: torch.Tensor,
                                        param: torch.Tensor, tp_rank: int):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
                                             shard_size)
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.Tensor,
                  tp_rank: int,
                  load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(self, param: torch.nn.Parameter,
                           loaded_weight: torch.Tensor, expert_id: int):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(self, shard_id: str, expert_data: torch.Tensor,
                    shard_dim: int, loaded_weight: torch.Tensor, tp_rank: int):

        if shard_id == "w2":
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank)
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[False]) -> None:
        ...

    @overload
    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: Literal[True]) -> bool:
        ...

    def weight_loader(self,
                      param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor,
                      weight_name: str,
                      shard_id: str,
                      expert_id: int,
                      return_success: bool = False) -> Optional[bool]:
        # print("weight loader")
        # print("loaded_weight", loaded_weight.shape, loaded_weight)
        # print("weight_name", weight_name)
        # print("shard_id", shard_id)
        # print("expert_id", expert_id)

        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        quant_method_name = self.quant_method.__class__.__name__
        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
                "CompressedTensorsWNA16MarlinMoEMethod",
                "CompressedTensorsWNA16MoEMethod"):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, device=loaded_weight.device, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if ("compressed" in quant_method_name.lower()
                    and param.data[expert_id] != 1
                    and (param.data[expert_id] - loaded_weight).abs() > 1e-5):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=self.tp_rank)
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern(
            )

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = ("weight_scale_2" in weight_name
                             if uses_weight_scale_2 else "weight_scale"
                             in weight_name) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=param,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if ("scale" in weight_name or "zero" in weight_name
                or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank)
            elif quant_method in [
                    FusedMoeWeightScaleSupported.GROUP.value,
                    FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False))
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                WEIGHT_SCALE_SUPPORTED = [
                    e.value for e in FusedMoeWeightScaleSupported
                ]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return True if return_success else None

        return False if return_success else None

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        assert all(weight.is_contiguous() for _, weight in weights)

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1) for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS and weight.shape != torch.Size(
                []) and not name.startswith("_shared_experts.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config(self):
        if self.quant_method.moe_quant_config is None:
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self))

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        indices_type: Optional[torch.dtype] = None,
        enable_eplb: bool = False,
        expert_map: Optional[torch.Tensor] = None,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
            (topk_weights, topk_ids) (tuple[torch.Tensor, torch.Tensor]):
            The weights and *global physical* expert ids of the top-k experts.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            return RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=top_k,
                indices_type=indices_type)

        # DeepSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

            # 1. Convert the logical expert ids to physical expert ids
            # Directly select a random replica for each logical expert

            # TODO: maybe optimize this by using specified kernels,
            # or compute pseudo-random indices by modulo

            # In case `indices_type` is not `torch.long` or `torch.int`,
            # e.g. `torch.uint32` as required by dispatch/combine kernels
            topk_ids_long = topk_ids.long()
            replica_indices = (
                torch.rand_like(topk_ids, dtype=torch.float) *
                logical_replica_count[topk_ids_long]).long().unsqueeze(-1)
            physical_ids = logical_to_physical_map[topk_ids_long].gather(
                -1, replica_indices).squeeze(-1)

            topk_ids = physical_ids

            # 2. Record expert load metrics.

            # TODO(bowen): When using `FusedMoEModularKernel`, this
            # can be done in a more unified way, since
            # `FusedMoEPrepareAndFinalize` will return the expert
            # token count, in some cases directly from the kernel.
            # However, now there are many code paths not using
            # the modular kernel, e.g. calling `fused_experts`,
            # so we decide to keep the logic here.
            #
            # If later refactor moved all the MoE kernel calls
            # to the modular kernel, we can move this logic there
            # to achieve better efficiency.

            # `expert_load_view`: (num_physical_experts,)

            topk_ids_flatten = topk_ids.flatten()

            # Performance optimization:
            # `masked_fill` is significantly faster than `masked_select`
            invalid_mask = topk_ids_flatten < 0
            # Replace invalid expert ids with 0 (just a dummy position)
            # to avoid out-of-bounds errors in scatter_add_
            index = topk_ids_flatten.masked_fill_(invalid_mask, 0)
            # `src` is the valid mask, which is 1 for valid and 0 for invalid
            src = ~invalid_mask

            expert_load_view.scatter_add_(dim=0,
                                          index=index.long(),
                                          src=src.to(expert_load_view))

            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        return (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels)

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """
        The pplx combine kernel reduces across GPU ranks by default.
        """
        if (self.use_pplx_kernels or self.use_deepep_ht_kernels
                or self.use_deepep_ll_kernels):
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(hidden_states,
                                  (0, self.hidden_size - og_hidden_states),
                                  mode='constant',
                                  value=0.0)

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name)
            return fused_output[..., :og_hidden_states]
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits)
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name)
            return (shared_output[..., :og_hidden_states],
                    fused_output[..., :og_hidden_states])

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert (
            self.batched_hidden_states.size(-1) == full_hidden_states.size(-1))
        assert (
            self.batched_router_logits.size(-1) == full_router_logits.size(-1))

        self.ensure_moe_quant_config()

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(
                full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[
                    batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[
                    batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (batched_hidden_states.size(0)  # type: ignore
                    >= chunk_size)
            assert (batched_router_logits.size(0)  # type: ignore 
                    >= chunk_size)
            staged_hidden_states = batched_hidden_states[:
                                                         chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:
                                                         chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)
            
            if self.use_ft_moe and self.ft_moe is not None:
                try:  
                    final_hidden_states = self.forward_ft(staged_hidden_states, staged_router_logits)
                except Exception as e:
                    logger.error(f"Error in forward_ft with chunk size {chunk_size}: {e}") 
                    self.use_ft_moe = False
            else:

                # Matrix multiply.
                final_hidden_states = self.quant_method.apply(
                    layer=self,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                    top_k=self.top_k,
                    renormalize=self.renormalize,
                    use_grouped_topk=self.use_grouped_topk,
                    global_num_experts=self.global_num_experts,
                    expert_map=self.expert_map,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    activation=self.activation,
                    enable_eplb=self.enable_eplb,
                    expert_load_view=self.expert_load_view,
                    logical_to_physical_map=self.logical_to_physical_map,
                    logical_replica_count=self.logical_replica_count,
                )

            assert self.shared_experts is None or isinstance(
                final_hidden_states, tuple)

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states,
                                                        non_blocking=True)
                else:
                    full_shared_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states[0],
                                                        non_blocking=True)
                    full_fused_final_hidden_states[
                        chunk_start:chunk_end, :].copy_(final_hidden_states[1],
                                                        non_blocking=True)

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(max_tokens_across_dispatchers,
                                                 self.sp_size)

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
                range(0, max_tokens_across_dispatchers,
                      moe_dp_chunk_size_per_rank)):
            chunk_start = chunk_start_
            chunk_end = min(chunk_start + moe_dp_chunk_size_per_rank,
                            max_tokens_across_dispatchers)
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(moe_dp_chunk_size_per_rank,
                                               chunk_idx):
                process_chunk(chunk_start,
                              chunk_end,
                              skip_result_store=chunk_start_ >= num_tokens)

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states,
                    full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.quant_method is not None
        self.ensure_moe_quant_config()

        # Route to the chunked forward path using the FlashInfer Cutlass kernel
        # only when data parallelism (DP) is enabled.
        _use_flashinfer_cutlass_kernels = (self.dp_size > 1 and
                                           self.use_flashinfer_cutlass_kernels)  

        if (self.moe_parallel_config.use_pplx_kernels
                or self.moe_parallel_config.use_deepep_ll_kernels
                or _use_flashinfer_cutlass_kernels):
            return self.forward_impl_chunked(hidden_states, router_logits)

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_deepep_ht_kernels
            and not self.moe_config.use_flashinfer_cutlass_kernels)

        # If there are shared experts but we are not using a modular kernel, the
        # shared experts must be called here
        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        if do_naive_dispatch_combine:
            hidden_states, router_logits = get_ep_group().dispatch(
                hidden_states, router_logits)

        if self.use_ft_moe and self.ft_moe is not None:
            try:  
                final_hidden_states = self.forward_ft(hidden_states, router_logits)
            except Exception as e:
                logger.error(f"Error in forward_ft forward: {e}") 
                self.use_ft_moe = False
        else:
            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

        if shared_output is not None:
            assert not isinstance(final_hidden_states, tuple)
            assert self.shared_experts is not None
            final_hidden_states = (
                shared_output,
                final_hidden_states,
            )

        def reduce_output(states: torch.Tensor,
                          do_combine: bool = True) -> torch.Tensor:
            if do_naive_dispatch_combine and do_combine:
                states = get_ep_group().combine(states)

            if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
                states = self.maybe_all_reduce_tensor_model_parallel(states)

            return states

        if self.shared_experts is None:
            assert not isinstance(final_hidden_states, tuple)
            return reduce_output(final_hidden_states)
        else:
            return (
                reduce_output(final_hidden_states[0], do_combine=False),
                reduce_output(final_hidden_states[1]),
            )
            
    def process_weights_after_loading(self):
        # print("into process_weights_after_loading")
        # print("quant_method", self.quant_method)

        if not self.use_ft_moe:
            return
        try:  
            find_weight = False 
            
            if (isinstance(self.quant_method, ExvllmFp8FusedMoEMethod)):
                self._process_fp8_weights()
                find_weight = True
            elif (isinstance(self.quant_method, ExvllmAwqFusedMoEMethod)):
                self._process_awq_weights()
                find_weight = True
            elif hasattr(self, 'w13_qweight') and hasattr(self, 'w2_qweight'): 
                #self._process_gguf_weights()
                raise("unsupport model")
                find_weight = True
            elif hasattr(self, 'w13_weight') and hasattr(self, 'w2_weight'): 
                self._process_regular_weights()
                find_weight = True
            
            if not find_weight: 
                logger.error("weight not found in layer")
                return
            
            self._initialize_cuda_graph_buffers()
            logger.info(f"Initialized {self.layer_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.layer_name}: {e}")
            self.use_ft_moe = False
            self.ft_moe = None
            
    def get_fastllm_type_from_dtype(self, dtype):
            if dtype == torch.float32:
                return 0  # fastllm::DataType::FLOAT32
            elif dtype == torch.float16:
                return 7  # fastllm::DataType::FLOAT16
            elif dtype == torch.bfloat16:
                return 1  # fastllm::DataType::BFLOAT16
            elif dtype == torch.float8_e4m3fn:
                return 10 # fastllm::DataType::FFP8_E4M3
            else:
                raise ValueError(f"Unsupported dtype {dtype}")
        
    def _process_regular_weights(self): 
        # print("_process_regular_weights")
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
         
        if not w13_weight.device.type == 'cpu' or not w2_weight.device.type == 'cpu':
            logger.warning("Moe weights on GPU...")
            return
              
        gate_up_dtype = w13_weight.dtype
        down_dtype = w2_weight.dtype 
        gate_ft_type = self.get_fastllm_type_from_dtype(gate_up_dtype)
        up_ft_type = self.get_fastllm_type_from_dtype(gate_up_dtype)
        down_ft_type = self.get_fastllm_type_from_dtype(down_dtype)
        hidden_ft_type = self.get_fastllm_type_from_dtype(self.moe_config.in_dtype)
          
        
        num_experts, total_intermediate_size, hidden_size = self.w13_weight.shape
        intermediate_size = total_intermediate_size // 2
          
        gate_proj_view = self.w13_weight.narrow(1, 0, intermediate_size).contiguous()
        up_proj_view = self.w13_weight.narrow(1, intermediate_size, intermediate_size).contiguous()
        
        gate_proj_ptr = gate_proj_view.data_ptr()
        up_proj_ptr = up_proj_view.data_ptr()
        down_proj_ptr = self.w2_weight.contiguous().data_ptr()
            
        gate = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, gate_proj_ptr, gate_ft_type)
        up = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, up_proj_ptr, up_ft_type)
        down = ft_kernel.FastllmLinearWeight(self.local_num_experts, hidden_size, intermediate_size, down_proj_ptr, down_ft_type)
        self.ft_moe = ft_kernel.FastllmMoe(self.local_num_experts, self.top_k, self.hidden_size, intermediate_size, hidden_ft_type, gate, up, down)
         
        del gate_proj_view, up_proj_view,  
        del self.w13_weight, self.w2_weight
        
        import gc
        gc.collect()

    def _process_fp8_weights(self): 
        # print("_process_fp8_weights")
        
        w13_weight = self.w13_weight
        w13_weight_scale_inv = self.w13_weight_scale_inv
        w2_weight = self.w2_weight
        w2_weight_scale_inv = self.w2_weight_scale_inv

        # print("w13_weight", w13_weight.shape, w13_weight.device)
        # print("w13_weight_scale_inv", w13_weight_scale_inv.shape, w13_weight_scale_inv.device, w13_weight_scale_inv.dtype)
        # print("w2_weight", w2_weight.shape, w2_weight.device)
        # print("w2_weight_scale_inv", w2_weight_scale_inv.shape, w2_weight_scale_inv.device, w13_weight_scale_inv.dtype)
         
        if not w13_weight.device.type == 'cpu' or not w2_weight.device.type == 'cpu' or not w13_weight_scale_inv.device.type == 'cpu' or not w2_weight_scale_inv.device.type == 'cpu':
            logger.warning("Moe weights on GPU...")
            return
              
        gate_up_dtype = w13_weight.dtype
        down_dtype = w2_weight.dtype 

        gate_ft_type = self.get_fastllm_type_from_dtype(gate_up_dtype)
        up_ft_type = self.get_fastllm_type_from_dtype(gate_up_dtype)
        down_ft_type = self.get_fastllm_type_from_dtype(down_dtype)
        hidden_ft_type = self.get_fastllm_type_from_dtype(self.moe_config.in_dtype)
          
        num_experts, total_intermediate_size, hidden_size = self.w13_weight.shape
        intermediate_size = total_intermediate_size // 2
        
        gate_proj_view = self.w13_weight.narrow(1, 0, intermediate_size).contiguous()
        up_proj_view = self.w13_weight.narrow(1, intermediate_size, intermediate_size).contiguous()
        
        if (len(self.quant_config.weight_block_size) == 2):
            block_gate_0 = block_down_0 = block_up_0 = self.quant_config.weight_block_size[0]
            block_gate_1 = block_down_1 = block_up_1 = self.quant_config.weight_block_size[1]
        gate_proj_scale_view = self.w13_weight_scale_inv.narrow(1, 0, intermediate_size // block_gate_0).contiguous()
        up_proj_scale_view = self.w13_weight_scale_inv.narrow(1, intermediate_size // block_gate_0, intermediate_size // block_gate_0).contiguous()
        
        gate_proj_ptr = gate_proj_view.data_ptr()
        up_proj_ptr = up_proj_view.data_ptr()
        down_proj_ptr = self.w2_weight.contiguous().data_ptr()

        gate_proj_scale_ptr = gate_proj_scale_view.data_ptr()
        up_proj_scale_ptr = up_proj_scale_view.data_ptr()
        down_proj_scale_ptr = self.w2_weight_scale_inv.contiguous().data_ptr()
        
        # print("into new fp8 create")
        gate = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, gate_proj_ptr, gate_ft_type, gate_proj_scale_ptr, 0, block_gate_0, block_gate_1)
        up = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, up_proj_ptr, up_ft_type, up_proj_scale_ptr, 0, block_up_0, block_up_1)
        down = ft_kernel.FastllmLinearWeight(self.local_num_experts, hidden_size, intermediate_size, down_proj_ptr, down_ft_type, down_proj_scale_ptr, 0, block_down_0, block_down_1)
        self.ft_moe = ft_kernel.FastllmMoe(self.local_num_experts, self.top_k, self.hidden_size, intermediate_size, hidden_ft_type, gate, up, down)
         
        del gate_proj_view, up_proj_view,  
        del gate_proj_scale_view, up_proj_scale_view
        del self.w13_weight, self.w2_weight
        
        import gc
        gc.collect()
    
    def _process_awq_weights(self): 
        # print("_process_awq_weights")
        
        w13_qweight = self.w13_qweight
        w13_scales = self.w13_scales
        w13_qzeros = self.w13_qzeros

        w2_qweight = self.w2_qweight
        w2_scales = self.w2_scales
        w2_qzeros = self.w2_qzeros

        # print("w13_qweight", w13_qweight.shape, w13_qweight.dtype)
        # print("w13_scales", w13_scales.shape, w13_scales.dtype)
        # print("w13_qzeros", w13_qzeros.shape, w13_qzeros.dtype)

        # print("w2_qweight", w2_qweight.shape, w2_qweight.dtype)
        # print("w2_scales", w2_scales.shape, w2_scales.dtype)
        # print("w2_qzeros", w2_qzeros.shape, w2_qzeros.dtype)
         
        if not w13_qweight.device.type == 'cpu' or not w13_scales.device.type == 'cpu' or not w13_qzeros.device.type == 'cpu' or not w2_qweight.device.type == 'cpu' or not w2_scales.device.type == 'cpu' or not w2_qzeros.device.type == 'cpu':
            logger.warning("Moe weights on GPU...")
            return

        gate_ft_type = 1001
        up_ft_type = 1001
        down_ft_type = 1001
        hidden_ft_type = self.get_fastllm_type_from_dtype(self.moe_config.in_dtype)
          
        intermediate_size = w2_qweight.shape[1]
        
        gate_proj_view = self.w13_qweight.narrow(2, 0, w13_qweight.shape[2] // 2).contiguous()
        up_proj_view = self.w13_qweight.narrow(2, w13_qweight.shape[2] // 2, w13_qweight.shape[2] // 2).contiguous()
        
        gate_proj_scale_view = self.w13_scales.narrow(2, 0, w13_scales.shape[2] // 2).contiguous()
        up_proj_scale_view = self.w13_scales.narrow(2, w13_scales.shape[2] // 2, w13_scales.shape[2] // 2).contiguous()

        gate_proj_qzeros_view = self.w13_qzeros.narrow(2, 0, w13_qzeros.shape[2] // 2).contiguous()
        up_proj_qzeros_view = self.w13_qzeros.narrow(2, w13_qzeros.shape[2] // 2, w13_qzeros.shape[2] // 2).contiguous()
        
        gate_proj_scale_fp32 = gate_proj_scale_view.float().contiguous()
        up_proj_scale_fp32 = up_proj_scale_view.float().contiguous()
        down_proj_scale_fp32 = w2_scales.float().contiguous()

        gate_proj_ptr = gate_proj_view.data_ptr()
        up_proj_ptr = up_proj_view.data_ptr()
        down_proj_ptr = self.w2_qweight.contiguous().data_ptr()

        gate_proj_scale_ptr = gate_proj_scale_fp32.data_ptr()
        up_proj_scale_ptr = up_proj_scale_fp32.data_ptr()
        down_proj_scale_ptr = down_proj_scale_fp32.contiguous().data_ptr()

        gate_proj_qzeros_ptr = gate_proj_qzeros_view.data_ptr()
        up_proj_qzeros_ptr = up_proj_qzeros_view.data_ptr()
        down_proj_qzeros_ptr = self.w2_qzeros.contiguous().data_ptr()
        
        hidden_size = self.hidden_size
        
        # print("into new awq create")
        gate = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, gate_proj_ptr, gate_ft_type, gate_proj_scale_ptr, gate_proj_qzeros_ptr, 128, 128)
        up = ft_kernel.FastllmLinearWeight(self.local_num_experts, intermediate_size, hidden_size, up_proj_ptr, up_ft_type, up_proj_scale_ptr, up_proj_qzeros_ptr, 128, 128)
        down = ft_kernel.FastllmLinearWeight(self.local_num_experts, hidden_size, intermediate_size, down_proj_ptr, down_ft_type, down_proj_scale_ptr, down_proj_qzeros_ptr, 128, 128)
        self.ft_moe = ft_kernel.FastllmMoe(self.local_num_experts, self.top_k, self.hidden_size, intermediate_size, hidden_ft_type, gate, up, down)
         
        del gate_proj_view, up_proj_view,  
        del gate_proj_scale_view, up_proj_scale_view
        del gate_proj_qzeros_view, up_proj_qzeros_view
        del self.w13_qweight, self.w2_qweight

        del gate_proj_scale_fp32
        del up_proj_scale_fp32
        del down_proj_scale_fp32
        
        import gc
        gc.collect()
                
    def forward_ft(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        return self._process_valid_inputs(hidden_states, router_logits)
 
    
    def _initialize_cuda_graph_buffers(self): 
        if not hasattr(FusedMoE, 'cuda_graphs'): 
            max_num_batched_tokens = 2048 
            FusedMoE.cuda_graphs = [1, 2, 4] + list(range(8, max_num_batched_tokens+1, 8))
             
            FusedMoE.input_tensor_cpu = []
            FusedMoE.expert_ids_cpu = []
            FusedMoE.weights_cpu = []
            FusedMoE.output_cpu = []
            FusedMoE.bsz_tensor_cpu = []
            FusedMoE.output_gpu = []
            
            num_experts_per_tok = self.top_k
            hidden_size = self.hidden_size
            buff_dtype = self.moe_config.in_dtype
             
            FusedMoE.output_gpu = [
                torch.zeros((batch_size, hidden_size), device="cuda", dtype=buff_dtype)
                for batch_size in FusedMoE.cuda_graphs
            ]
            
            FusedMoE.input_tensor_cpu = [
                torch.zeros((batch_size, self.hidden_size), device="cpu", dtype=buff_dtype, pin_memory=True)
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.expert_ids_cpu = [
                torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.long, pin_memory=True)
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.weights_cpu = [
                torch.zeros((batch_size, num_experts_per_tok), device="cpu", dtype=torch.float32, pin_memory=True)
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.output_cpu = [
                torch.zeros((batch_size, hidden_size), device="cpu", pin_memory=True, dtype=buff_dtype)
                for batch_size in FusedMoE.cuda_graphs
            ]
            FusedMoE.bsz_tensor_cpu = [
                torch.zeros((1), device="cpu", dtype=torch.int32, pin_memory=True)
                for _ in range(len(FusedMoE.cuda_graphs))
            ]
         
    def _find_best_graph_index(self, total_tokens: int) -> int:
        if not hasattr(FusedMoE, 'cuda_graphs') or not FusedMoE.cuda_graphs:
            raise ValueError("No CUDA graphs initialized.")
        
        cuda_graphs = FusedMoE.cuda_graphs
        
        low, high = 0, len(cuda_graphs) - 1
        best_index = len(cuda_graphs) - 1  
        
        while low <= high:
            mid = (low + high) // 2
            if cuda_graphs[mid] >= total_tokens:
                best_index = mid
                high = mid - 1
            else:
                low = mid + 1
         
        if best_index >= len(cuda_graphs):
            best_index = len(cuda_graphs) - 1
             
        if cuda_graphs[best_index] < total_tokens:
            raise ValueError(f"No suitable CUDA graph found for {total_tokens} tokens. "
                            f"Maximum available buffer size: {cuda_graphs[-1]}")
        
        return best_index

    def _process_valid_inputs(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """Process inputs that are guaranteed to be valid (non-NaN)"""
        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            expert_map=self.expert_map,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )
         
        from vllm.forward_context import get_forward_context
        from vllm.config import CUDAGraphMode
        from vllm.compilation.monitor import cudagraph_capturing_enabled  
        from vllm.utils import current_stream, weak_ref_tensors 
  
            
        def get_cuda_stream_ptr(stream: torch.cuda.Stream) -> int:
                """
                Get the underlying CUDA stream pointer from a torch.cuda.Stream object.
                """
                if hasattr(stream, 'cuda_stream'):
                    return stream.cuda_stream
                elif hasattr(stream, 'stream'):
                    return stream.stream
                else:
                    # Fallback to using the default stream
                    return 0
                
        forward_context = get_forward_context()
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode
        batch_descriptor = forward_context.batch_descriptor 
         
        is_decode_mode = getattr(batch_descriptor, 'uniform_decode', False) 
        
        if(hasattr(batch_descriptor, 'num_tokens')):
            batch_size = batch_descriptor.num_tokens
        else:
            batch_size = hidden_states.size(0)
        
        stream = current_stream()
        stream_ptr = get_cuda_stream_ptr(stream) 
        non_blocking = True
        
        try:      
            graph_index = self._find_best_graph_index(batch_size) 
            
            input_tensor_cpu = FusedMoE.input_tensor_cpu
            expert_ids_cpu = FusedMoE.expert_ids_cpu
            weights_cpu = FusedMoE.weights_cpu
            output_cpu = FusedMoE.output_cpu
            bsz_tensor_cpu = FusedMoE.bsz_tensor_cpu
            output_gpu = FusedMoE.output_gpu 
                 
            bsz_tensor_cpu[graph_index][0] = batch_size 

            input_tensor_cpu[graph_index][:batch_size].copy_(hidden_states, non_blocking=non_blocking)
            expert_ids_cpu[graph_index][:batch_size].copy_(topk_ids, non_blocking=non_blocking)
            weights_cpu[graph_index][:batch_size].copy_(topk_weights, non_blocking=non_blocking) 
            input_ptr = input_tensor_cpu[graph_index].data_ptr()
            expert_ids_ptr = expert_ids_cpu[graph_index].data_ptr()
            weights_ptr = weights_cpu[graph_index].data_ptr()
            output_ptr = output_cpu[graph_index].data_ptr() 

            # print("submit_with_cuda_stream") 
            self.ft_moe.submit_with_cuda_stream(
                stream_ptr, 
                hidden_states.size(0),                                   # qlen
                expert_ids_cpu[graph_index].size(1),                     # k
                expert_ids_ptr,                  # expert_ids
                weights_ptr,                     # weights
                input_ptr,                       # input
                output_ptr,                      # output 
                bsz_tensor_cpu[graph_index].data_ptr()                   # bsz_tensor
            )  
            self.ft_moe.sync_with_cuda_stream(stream_ptr) 
            output_gpu[graph_index][:batch_size].copy_(output_cpu[graph_index][:batch_size], non_blocking=non_blocking) 
            return weak_ref_tensors(output_gpu[graph_index][:batch_size])  
       
        except Exception as e:
            logger.error(f"ft_moe forward failed with error: {e}, falling back to default path")
            self.use_ft_moe = False
            raise RuntimeError("ft_moe forward failed, fallback to default MoE implementation")
  
    @classmethod
    def make_expert_params_mapping(
            cls,
            ckpt_gate_proj_name: str,
            ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int,
            num_redundant_experts: int = 0) -> list[tuple[str, str, int, str]]:

        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = \
            EplbState.build_initial_global_physical_to_logical_map(
            num_experts, num_redundant_experts)

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
             expert_id, shard_id) for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s 
 
    

def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.needs_fixed_stride_order, ),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
