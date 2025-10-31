#include "ft_kernel.h"
#include "moe.h"

extern "C" {

// FastllmLinearWeight 相关函数实现
FastllmLinearWeightHandle fastllm_linear_weight_create(
    int batch, int k, int m, void* data, int dataType) {
    return new fastllm::FastllmLinearWeight(
        batch, k, m, data, (fastllm::DataType)dataType);
}

FastllmLinearWeightHandle fastllm_linear_weight_create_quantized(
    int batch, int k, int m, void* data, int dataType,
    void* scales, void* zeros, int blockK, int blockM) {
    return new fastllm::FastllmLinearWeight(
        batch, k, m, data, (fastllm::DataType)dataType,
        scales, zeros, blockK, blockM);
}

void fastllm_linear_weight_destroy(FastllmLinearWeightHandle handle) {
    delete static_cast<fastllm::FastllmLinearWeight*>(handle);
}

int fastllm_linear_weight_get_batch(FastllmLinearWeightHandle handle) {
    return static_cast<fastllm::FastllmLinearWeight*>(handle)->batch;
}

int fastllm_linear_weight_get_k(FastllmLinearWeightHandle handle) {
    return static_cast<fastllm::FastllmLinearWeight*>(handle)->k;
}

int fastllm_linear_weight_get_m(FastllmLinearWeightHandle handle) {
    return static_cast<fastllm::FastllmLinearWeight*>(handle)->m;
}

int fastllm_linear_weight_get_block(FastllmLinearWeightHandle handle) {
    return static_cast<fastllm::FastllmLinearWeight*>(handle)->block;
}

int fastllm_linear_weight_get_dataType(FastllmLinearWeightHandle handle) {
    return static_cast<int>(static_cast<fastllm::FastllmLinearWeight*>(handle)->dataType);
}

void fastllm_linear_weight_set_batch(FastllmLinearWeightHandle handle, int batch) {
    static_cast<fastllm::FastllmLinearWeight*>(handle)->batch = batch;
}

void fastllm_linear_weight_set_k(FastllmLinearWeightHandle handle, int k) {
    static_cast<fastllm::FastllmLinearWeight*>(handle)->k = k;
}

void fastllm_linear_weight_set_m(FastllmLinearWeightHandle handle, int m) {
    static_cast<fastllm::FastllmLinearWeight*>(handle)->m = m;
}

void fastllm_linear_weight_set_block(FastllmLinearWeightHandle handle, int block) {
    static_cast<fastllm::FastllmLinearWeight*>(handle)->block = block;
}

// FastllmMoe 相关函数实现
FastllmMoeHandle fastllm_moe_create(
    int expertNum, int routedExpertNum, int hiddenSize, 
    int intermediateSize, int hiddenType,
    FastllmLinearWeightHandle gate, 
    FastllmLinearWeightHandle up, 
    FastllmLinearWeightHandle down) {
    return new fastllm::FastllmMoe(
        expertNum, routedExpertNum, hiddenSize, intermediateSize,
        (fastllm::DataType)hiddenType,
        static_cast<fastllm::FastllmLinearWeight*>(gate),
        static_cast<fastllm::FastllmLinearWeight*>(up),
        static_cast<fastllm::FastllmLinearWeight*>(down));
}

void fastllm_moe_destroy(FastllmMoeHandle handle) {
    delete static_cast<fastllm::FastllmMoe*>(handle);
}

int fastllm_moe_get_expertNum(FastllmMoeHandle handle) {
    return static_cast<fastllm::FastllmMoe*>(handle)->expertNum;
}

int fastllm_moe_get_routedExpertNum(FastllmMoeHandle handle) {
    return static_cast<fastllm::FastllmMoe*>(handle)->routedExpertNum;
}

int fastllm_moe_get_hiddenSize(FastllmMoeHandle handle) {
    return static_cast<fastllm::FastllmMoe*>(handle)->hiddenSize;
}

int fastllm_moe_get_intermediateSize(FastllmMoeHandle handle) {
    return static_cast<fastllm::FastllmMoe*>(handle)->intermediateSize;
}

int fastllm_moe_get_hiddenType(FastllmMoeHandle handle) {
    return static_cast<int>(static_cast<fastllm::FastllmMoe*>(handle)->hiddenType);
}

void fastllm_moe_warm_up(FastllmMoeHandle handle) {
    static_cast<fastllm::FastllmMoe*>(handle)->warm_up();
}

void fastllm_moe_sync_with_cuda_stream(
    FastllmMoeHandle handle, intptr_t user_cuda_stream) {
    static_cast<fastllm::FastllmMoe*>(handle)->sync_with_cuda_stream(user_cuda_stream);
}

void fastllm_moe_submit_with_cuda_stream(
    FastllmMoeHandle handle, intptr_t user_cuda_stream, 
    int qlen, int k, const uint64_t* expert_ids, 
    const float* weights, const void* input, void* output, 
    int* batch_size_tensor) {
    static_cast<fastllm::FastllmMoe*>(handle)->submit_with_cuda_stream(
        user_cuda_stream, qlen, k, expert_ids, weights, input, output, batch_size_tensor);
}

void fastllm_moe_forward(
    FastllmMoeHandle handle, int qlen, int k,
    const uint64_t* expert_ids, const float* weights,
    const void* input, void* output) {
    static_cast<fastllm::FastllmMoe*>(handle)->forward(
        qlen, k, expert_ids, weights, input, output);
}

} // extern "C"
