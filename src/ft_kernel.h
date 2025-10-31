#ifndef MOE_C_API_H
#define MOE_C_API_H

#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
    #ifdef MOE_EXPORTS
        #define MOE_API __declspec(dllexport)
    #else
        #define MOE_API __declspec(dllimport)
    #endif
#else
    #define MOE_API
#endif

typedef void* FastllmLinearWeightHandle;
typedef void* FastllmMoeHandle;

// FastllmLinearWeight 相关函数
MOE_API FastllmLinearWeightHandle fastllm_linear_weight_create(
    int batch, int k, int m, void* data, int dataType);

MOE_API FastllmLinearWeightHandle fastllm_linear_weight_create_quantized(
    int batch, int k, int m, void* data, int dataType,
    void* scales, void* zeros, int blockK, int blockM);

MOE_API void fastllm_linear_weight_destroy(FastllmLinearWeightHandle handle);

MOE_API int fastllm_linear_weight_get_batch(FastllmLinearWeightHandle handle);
MOE_API int fastllm_linear_weight_get_k(FastllmLinearWeightHandle handle);
MOE_API int fastllm_linear_weight_get_m(FastllmLinearWeightHandle handle);
MOE_API int fastllm_linear_weight_get_block(FastllmLinearWeightHandle handle);
MOE_API int fastllm_linear_weight_get_dataType(FastllmLinearWeightHandle handle);

MOE_API void fastllm_linear_weight_set_batch(FastllmLinearWeightHandle handle, int batch);
MOE_API void fastllm_linear_weight_set_k(FastllmLinearWeightHandle handle, int k);
MOE_API void fastllm_linear_weight_set_m(FastllmLinearWeightHandle handle, int m);
MOE_API void fastllm_linear_weight_set_block(FastllmLinearWeightHandle handle, int block);

// FastllmMoe 相关函数
MOE_API FastllmMoeHandle fastllm_moe_create(
    int expertNum, int routedExpertNum, int hiddenSize, 
    int intermediateSize, int hiddenType,
    FastllmLinearWeightHandle gate, 
    FastllmLinearWeightHandle up, 
    FastllmLinearWeightHandle down);

MOE_API void fastllm_moe_destroy(FastllmMoeHandle handle);

MOE_API int fastllm_moe_get_expertNum(FastllmMoeHandle handle);
MOE_API int fastllm_moe_get_routedExpertNum(FastllmMoeHandle handle);
MOE_API int fastllm_moe_get_hiddenSize(FastllmMoeHandle handle);
MOE_API int fastllm_moe_get_intermediateSize(FastllmMoeHandle handle);
MOE_API int fastllm_moe_get_hiddenType(FastllmMoeHandle handle);

MOE_API void fastllm_moe_warm_up(FastllmMoeHandle handle);

MOE_API void fastllm_moe_sync_with_cuda_stream(
    FastllmMoeHandle handle, intptr_t user_cuda_stream);

MOE_API void fastllm_moe_submit_with_cuda_stream(
    FastllmMoeHandle handle, intptr_t user_cuda_stream, 
    int qlen, int k, const uint64_t* expert_ids, 
    const float* weights, const void* input, void* output, 
    int* batch_size_tensor);

MOE_API void fastllm_moe_forward(
    FastllmMoeHandle handle, int qlen, int k,
    const uint64_t* expert_ids, const float* weights,
    const void* input, void* output);

#ifdef __cplusplus
}
#endif

#endif // MOE_C_API_H
