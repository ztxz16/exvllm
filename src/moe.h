#include "fastllm.h"
#include <atomic>
#include <cuda_runtime.h> 

namespace fastllm {
    struct FastllmMoe {
        int expertNum, routedExpertNum, hiddenSize, intermediateSize;
        DataType hiddenType;
        FastllmLinearWeight *gate, *up, *down;

        FastllmMoe (int expertNum, int routedExpertNum, int hiddenSize, int intermediateSize,
                DataType hiddenType, FastllmLinearWeight *gate, FastllmLinearWeight *up, FastllmLinearWeight *down) :
                expertNum(expertNum), routedExpertNum(routedExpertNum), hiddenSize(hiddenSize), intermediateSize(intermediateSize), 
                hiddenType(hiddenType), gate(gate), up(up), down(down) {}
        
        void warm_up() {}

        void sync();
        
        void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);

        void forwardOne(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);

        void forwardMulti(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);

        void submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids, 
                             const float* weights, const void* input, void* output, int* bsz_tensor);
                             
        void sync_with_cuda_stream(intptr_t user_cuda_stream);

        std::atomic <bool> sync_flag{false};

        struct FastllmMoeDataManager {
            std::vector <float, aligned_allocator<float, 64> > fp32OutputVector, newOutput, gate_out, up_out;
            std::vector <uint8_t, aligned_allocator<uint8_t, 64> > oriInput, newInput, temp;
        } fastllmMoeDataManager;
    };
    
    struct FastllmModeForwardParams {
        FastllmMoe* moe_ptr;
        int qlen;
        int k;
        const uint64_t* expert_ids;
        const float* weights;
        const void* input;
        void* output;
        int* bsz_tensor;
        cudaEvent_t* forward_done_event;
    };
}