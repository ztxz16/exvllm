#include <cstring>
#include "moe.h"
#include "fastllm.h"

namespace fastllm {
    extern FP8E4M3ToFP32Manager fp8e4m3tofp32;

    static void silu(float* gate, float* up, int n) {
        int i = 0;
#if defined(__AVX2__)
    const __m256 v_log2e = _mm256_set1_ps(1.44269504089f);  
    const __m256 v_ln2 = _mm256_set1_ps(0.69314718056f);  
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_neg_inf = _mm256_set1_ps(-128.0f);       
    const __m256 v_pos_inf = _mm256_set1_ps(127.0f);       
    for (; i + 7 < n; i += 8) {
        __m256 v_gate = _mm256_loadu_ps(gate + i);  
        __m256 v_up = _mm256_loadu_ps(up + i);      

        __m256 v_x = _mm256_mul_ps(v_gate, v_log2e);

        v_x = _mm256_max_ps(_mm256_min_ps(v_x, v_pos_inf), v_neg_inf);

        __m256i v_k = _mm256_cvtps_epi32(v_x);         
        __m256 v_k_f = _mm256_cvtepi32_ps(v_k);        
        __m256 v_r = _mm256_sub_ps(v_x, v_k_f);         

        __m256i v_k_bias = _mm256_add_epi32(v_k, _mm256_set1_epi32(127));  
        __m256i v_k_bits = _mm256_slli_epi32(v_k_bias, 23);              
        __m256 v_two_k = _mm256_castsi256_ps(v_k_bits);                 

        __m256 v_t = _mm256_mul_ps(v_r, v_ln2);                       
        __m256 v_t2 = _mm256_mul_ps(v_t, v_t);                        
        __m256 v_t3 = _mm256_mul_ps(v_t2, v_t);                     
        __m256 v_t4 = _mm256_mul_ps(v_t3, v_t);                     
        __m256 v_two_r = _mm256_add_ps(v_one, 
            _mm256_fmadd_ps(v_t, v_one, 
                _mm256_fmadd_ps(v_t2, _mm256_set1_ps(1.0f/2.0f), 
                    _mm256_fmadd_ps(v_t3, _mm256_set1_ps(1.0f/6.0f), 
                        _mm256_mul_ps(v_t4, _mm256_set1_ps(1.0f/24.0f))))));

        __m256 v_two_x = _mm256_mul_ps(v_two_k, v_two_r);

        __m256 v_denom = _mm256_add_ps(v_one, v_two_x);
        __m256 v_sigmoid = _mm256_div_ps(v_two_x, v_denom);

        __m256 v_swish = _mm256_mul_ps(v_gate, v_sigmoid);
        __m256 v_out = _mm256_mul_ps(v_up, v_swish);

        _mm256_storeu_ps(gate + i, v_out);
    }
#endif
        for (; i < n; ++i) {
            gate[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
        }
    }

    extern BF16ToFP32Manager bf16tofp32;

    extern bool LinearBFloat16BFloat16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16_FP8E4M3BLOCK128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16_AWQ4BIT128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool AWQ4BIT128_TO_BFloat16_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb);
                        
    void FastllmGemm (int n, int m, int k, 
        const void *A, long lda, // A [n * m], lda = bytes for 1 row in A
        const void *B, long ldb, // B [k * m], ldb = bytes for 1 row in B
        void *C, long ldc, // C[n * k], ldc = bytes for 1 row in C
        int st, int end, // calc C[0 : n, st : end]
        DataType AType, DataType BType, DataType CType
    ) {
        if (AType == DataType::FLOAT32) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::FLOAT32) {
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            float *floatB = (float*)((uint8_t*)B + j * ldb);
                            float sum = 0.0f;
                            for (int l = 0; l < m; l++) {
                                sum += floatA[l] * floatB[l];
                            }
                            floatC[j] = sum;
                        }
                    }
                } else if (BType == DataType::BFLOAT16) {
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            uint16_t *floatB = (uint16_t*)((uint8_t*)B + j * ldb);
                            float sum = 0.0f;
                            for (int l = 0; l < m; l++) {
                                sum += floatA[l] * bf16tofp32.dict[floatB[l]];
                            }
                            floatC[j] = sum;
                        }
                    }
                }
            }
        } else if (AType == DataType::BFLOAT16) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::BFLOAT16) {                    
                    LinearBFloat16BFloat16_Kernel((uint16_t*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                } else if (BType == FP8_E4M3_BLOCK_128) {
                    // A是BFLOAT16, B是FP8_E4M3_BLOCK_128格式（fp8数据+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        // 转换fp8到bf16，仅转换需要计算的行[st:end]
                        int block_size = 128;
                        int num_blocks = (m + block_size - 1) / block_size;
                        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;  // ldb应该是每行的总字节数
                            uint16_t *bf16B_row = bf16B_temp.data() + (j - st) * m;
                            
                            // 按block进行处理
                            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                                // 计算当前block的大小（最后一个block可能不完整）
                                int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                                
                                // 计算当前block的起始位置
                                // 每个block占用 128字节(fp8) + 4字节(float scale)
                                uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                                uint8_t *fp8_ptr = block_start;
                                float *scale_ptr = (float*)(block_start + block_size);
                                
                                // 转换当前block中的每个fp8到bf16
                                int base_idx = block_idx * block_size;
                                for (int l = 0; l < current_block_size; l++) {
                                    // fp8转fp32并乘以scale
                                    float fp32_val = fp8e4m3tofp32.dict[fp8_ptr[l]] * (*scale_ptr);
                                    
                                    // fp32转bf16
                                    uint32_t val;
                                    memcpy(&val, &fp32_val, sizeof(val));
                                    bf16B_row[base_idx + l] = (uint16_t)(val >> 16);
                                }
                            }
                        }
                        LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_FP8E4M3BLOCK128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    }
                } else if (BType == AWQ_4BIT_128) {
                    // A是BFLOAT16, B是AWQ_4BIT_128格式（uint4权重+zero+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        bool success = AWQ4BIT128_TO_BFloat16_Kernel((uint8_t*)B, bf16B_temp.data(), m, st, end, ldb);
                        LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_AWQ4BIT128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    }
                }
            }
        }
    }

    struct MultiThreadLinearOp : MultiThreadBaseOp {
        uint8_t *inputData;
        uint8_t *weightData;
        uint8_t *outputData;
        DataType inputDataType, weightDataType, outputDataType;
        float *biasData;
        int n, m, k, st, end;

        MultiThreadLinearOp(uint8_t *inputData, DataType inputDataType,
                            uint8_t *weightData, DataType weightDataType,
                            uint8_t *outputData, DataType outputDataType,
                            float *biasData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), inputDataType(inputDataType),
            weightData(weightData), weightDataType(weightDataType),
            outputData(outputData), outputDataType(outputDataType),
            biasData(biasData), n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            FastllmGemm (
                n, m, k, 
                inputData, GetDataBytes(inputDataType, 1, m),
                weightData, GetDataBytes(weightDataType, 1, m),
                outputData, GetDataBytes(outputDataType, 1, k), 
                st, end,
                inputDataType, weightDataType, outputDataType
            );

            // TODO: Add Bias
            if (biasData != nullptr) {
                ErrorInFastLLM("biasData != nullptr");
            }
        }
    };

    void MultiThreadLinear(uint8_t *inputData, DataType inputDataType, 
                            uint8_t *weightData, DataType weightDataType,
                            uint8_t *outputData, DataType outputDataType, 
                            float *biasData, int n, int m, int k) {
        auto *pool = GetAlivePool();
        int threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = std::min(cur + per, k) + (i < k % threadNum);
            ops.push_back(new MultiThreadLinearOp(
                inputData, inputDataType,
                weightData, weightDataType, 
                outputData, outputDataType,
                biasData, 
                n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadDownOp : MultiThreadBaseOp {
        uint8_t *inputData;
        std::vector<void*> *downWeights;  // Pointer to vector of expert weights for this NUMA node
        uint64_t *expert_ids;             // Expert IDs array
        uint8_t *downOutData;
        DataType inputDataType, downDataType, downOutDataType;
        float *weights;  // Array of weights for each expert
        float *lastOutput;
        int n, m, k, hidden_size, num_experts, st, end;
        int output_offset;

        MultiThreadDownOp(uint8_t *inputData, DataType inputDataType,
            std::vector<void*> *downWeights, uint64_t *expert_ids, DataType downDataType,
            uint8_t *downOutData, DataType downOutDataType,
            float *weights, float *lastOutput,
            int n, int m, int k, int hidden_size, int num_experts, int st, int end, 
            int output_offset) : 
            inputData(inputData), inputDataType(inputDataType),
            downWeights(downWeights), expert_ids(expert_ids), downDataType(downDataType),
            downOutData(downOutData), downOutDataType(downOutDataType),
            weights(weights), lastOutput(lastOutput), 
            n(n), m(m), k(k), hidden_size(hidden_size), num_experts(num_experts), st(st), end(end),  
            output_offset(output_offset) {}
            
        void Run() {
            // Process each expert
            for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
                uint64_t expert_id = expert_ids[expert_idx];
                uint8_t *expertInput = inputData + expert_idx * GetDataBytes(inputDataType, 1, m);
                uint8_t *expertOutput = downOutData + expert_idx * GetDataBytes(downOutDataType, 1, output_offset);
                uint8_t *expertDownWeight = (uint8_t*)(*downWeights)[expert_id];
                
                FastllmGemm(
                    n, m, hidden_size,
                    expertInput, GetDataBytes(inputDataType, 1, m),
                    expertDownWeight, GetDataBytes(downDataType, 1, m),
                    expertOutput, GetDataBytes(downOutDataType, 1, output_offset),
                    st, end,
                    inputDataType, downDataType, downOutDataType
                );
                // Accumulate weighted output
                float weight = weights[expert_idx];
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        lastOutput[i * output_offset + j] += weight * ((float*)expertOutput)[i * output_offset + j];
                    }
                }
            }
        }
    };

    void MultiThreadDown(uint8_t *inputData, DataType inputDataType,
                    std::vector<std::vector<void*>> &downWeights, uint64_t *expert_ids, DataType downDataType,
                    uint8_t *downOutData, DataType downOutDataType,
                    float *weights, float *lastOutput,
                    int n, int m, int hidden_size, int num_experts) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();

        std::vector<fastllm::MultiThreadDownOp*> ops;
        ops.resize(numaConfig->threads);

        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            int cur_hidden_size = hidden_size / numaConfig->numaCnt;
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            int per = cur_hidden_size / threadNum;
            int cur = 0;
            size_t downOutOffset = GetDataBytes(downOutDataType, 1, nid * cur_hidden_size);

            for (int i = 0; i < threadNum; i++) {
                int end = std::min(cur + per, cur_hidden_size) + (i < cur_hidden_size % threadNum);
                ops[numaConfig->numaToCpuDict[nid][i].first] = new MultiThreadDownOp(
                    inputData, inputDataType,
                    &downWeights[nid], expert_ids, downDataType,
                    downOutData + downOutOffset, downOutDataType,
                    weights, 
                    (float*)((uint8_t*)lastOutput + downOutOffset),
                    n, m, cur_hidden_size, cur_hidden_size, num_experts, cur, end,
                    hidden_size
                );
                cur = end;
            }
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadGateUpOp : MultiThreadBaseOp {
        uint8_t *inputData;
        std::vector<void*> *gateWeights, *upWeights;  // Pointers to vectors of expert weights for this NUMA node
        uint64_t *expert_ids;                         // Expert IDs array
        uint8_t *gateOutData, *upOutData;
        DataType inputDataType, gateDataType, upDataType, gateOutDataType, upOutDataType;
        int n, m, k, intermediate_size, num_experts, st, end;
        int output_offset;  // Total output size for proper indexing
        
        MultiThreadGateUpOp(uint8_t *inputData, DataType inputDataType,
                        std::vector<void*> *gateWeights, uint64_t *expert_ids, DataType gateDataType,
                        std::vector<void*> *upWeights, DataType upDataType,
                        uint8_t *gateOutData, DataType gateOutDataType,
                        uint8_t *upOutData, DataType upOutDataType,
                        int n, int m, int k, int intermediate_size, int num_experts, int st, int end,
                        int output_offset) : 
            inputData(inputData), inputDataType(inputDataType),
            gateWeights(gateWeights), expert_ids(expert_ids), gateDataType(gateDataType),
            upWeights(upWeights), upDataType(upDataType),
            gateOutData(gateOutData), gateOutDataType(gateOutDataType),
            upOutData(upOutData), upOutDataType(upOutDataType),
            n(n), m(m), k(k), intermediate_size(intermediate_size), num_experts(num_experts), st(st), end(end),
            output_offset(output_offset) {}
            
        void Run() {
            // Process each expert
            for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
                uint64_t expert_id = expert_ids[expert_idx];
                uint8_t *expertGateOut = gateOutData + expert_idx * GetDataBytes(gateOutDataType, 1, output_offset);
                uint8_t *expertUpOut = upOutData + expert_idx * GetDataBytes(upOutDataType, 1, output_offset);
                uint8_t *expertGateWeight = (uint8_t*)(*gateWeights)[expert_id];
                uint8_t *expertUpWeight = (uint8_t*)(*upWeights)[expert_id];
                
                FastllmGemm(
                    n, m, k,
                    inputData, GetDataBytes(inputDataType, 1, m),
                    expertGateWeight, GetDataBytes(gateDataType, 1, m),
                    expertGateOut, GetDataBytes(gateOutDataType, 1, output_offset),
                    st, end,
                    inputDataType, gateDataType, gateOutDataType
                );
                
                FastllmGemm(
                    n, m, k,
                    inputData, GetDataBytes(inputDataType, 1, m),
                    expertUpWeight, GetDataBytes(upDataType, 1, m),
                    expertUpOut, GetDataBytes(upOutDataType, 1, output_offset),
                    st, end,
                    inputDataType, upDataType, upOutDataType
                );
                
                for (int i = 0; i < n; i++) {
                    silu(((float*)expertGateOut) + i * output_offset + st, 
                        ((float*)expertUpOut) + i * output_offset + st, 
                        end - st);
                }
            }
        }
    };

    void MultiThreadGateUp(uint8_t *inputData, DataType inputDataType,
                    std::vector<std::vector<void*>> &gateWeights, uint64_t *expert_ids, DataType gateDataType,
                    std::vector<std::vector<void*>> &upWeights, DataType upDataType,
                    uint8_t *gateOutData, DataType gateOutDataType,
                    uint8_t *upOutData, DataType upOutDataType,
                    int n, int m, int intermediate_size, int num_experts) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        
        std::vector<fastllm::MultiThreadGateUpOp*> ops;
        ops.resize(numaConfig->threads);
        
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            int cur_intermediate_size = intermediate_size / numaConfig->numaCnt;
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            int per = cur_intermediate_size / threadNum;
            int cur = 0;
            size_t gateOutOffset = GetDataBytes(gateOutDataType, 1, nid * cur_intermediate_size);
            size_t upOutOffset = GetDataBytes(upOutDataType, 1, nid * cur_intermediate_size);
            
            for (int i = 0; i < threadNum; i++) {
                int end = std::min(cur + per, cur_intermediate_size) + (i < cur_intermediate_size % threadNum);
                ops[numaConfig->numaToCpuDict[nid][i].first] = new MultiThreadGateUpOp(
                    inputData, inputDataType,
                    &gateWeights[nid], expert_ids, gateDataType,
                    &upWeights[nid], upDataType,
                    gateOutData + gateOutOffset, gateOutDataType,
                    upOutData + upOutOffset, upOutDataType,
                    n, m, cur_intermediate_size, cur_intermediate_size, num_experts, cur, end,
                    intermediate_size
                );
                cur = end;
            }
        }
        
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    struct MultiThreadGateUpBatchOp : MultiThreadBaseOp {
        uint8_t *inputData;
        std::vector<void*> *gateWeights, *upWeights;  // Pointers to vectors of expert weights
        uint8_t *gateOutData, *upOutData;
        DataType inputDataType, gateDataType, upDataType, gateOutDataType, upOutDataType;
        int hidden_size, intermediate_size;
        int *expertOffsets;
        int expert_id, st, end;
        int output_offset;

        MultiThreadGateUpBatchOp(uint8_t *inputData, DataType inputDataType,
                        std::vector<void*> *gateWeights, DataType gateDataType,
                        std::vector<void*> *upWeights, DataType upDataType,
                        uint8_t *gateOutData, DataType gateOutDataType,
                        uint8_t *upOutData, DataType upOutDataType,
                        int hidden_size, int intermediate_size,
                        int *expertOffsets, int expert_id, int st, int end,
                        int output_offset) : 
            inputData(inputData), inputDataType(inputDataType),
            gateWeights(gateWeights), gateDataType(gateDataType),
            upWeights(upWeights), upDataType(upDataType),
            gateOutData(gateOutData), gateOutDataType(gateOutDataType),
            upOutData(upOutData), upOutDataType(upOutDataType),
            hidden_size(hidden_size), intermediate_size(intermediate_size), 
            expertOffsets(expertOffsets), expert_id(expert_id), st(st), end(end),
            output_offset(output_offset) {}

        void Run() {
            size_t inputPerRow = GetDataBytes(inputDataType, 1, hidden_size);
            
            int start = expertOffsets[expert_id];
            int batch_end = expertOffsets[expert_id + 1];
            int batch_size = batch_end - start;
            if (batch_size == 0) return;
            
            const uint8_t* gate_weight = (uint8_t*)(*gateWeights)[expert_id];
            const uint8_t* up_weight = (uint8_t*)(*upWeights)[expert_id];
            
            FastllmGemm(
                batch_size, hidden_size, intermediate_size,
                inputData + start * inputPerRow, GetDataBytes(inputDataType, 1, hidden_size),
                (uint8_t*)gate_weight, GetDataBytes(gateDataType, 1, hidden_size),
                gateOutData + GetDataBytes(gateOutDataType, start, output_offset), 
                GetDataBytes(gateOutDataType, 1, output_offset),
                st, end,
                inputDataType, gateDataType, gateOutDataType
            );
            
            FastllmGemm(
                batch_size, hidden_size, intermediate_size,
                inputData + start * inputPerRow, GetDataBytes(inputDataType, 1, hidden_size),
                (uint8_t*)up_weight, GetDataBytes(upDataType, 1, hidden_size),
                upOutData + GetDataBytes(upOutDataType, start, output_offset), 
                GetDataBytes(upOutDataType, 1, output_offset),
                st, end,
                inputDataType, upDataType, upOutDataType
            );
            
            for (int i = start; i < batch_end; i++) {
                silu(((float*)gateOutData) + i * output_offset + st, 
                    ((float*)upOutData) + i * output_offset + st, 
                    end - st);
            }
        }
    };

    void MultiThreadGateUpBatch(uint8_t *inputData, DataType inputDataType,
                    std::vector<std::vector<void*>> &gateWeights, DataType gateDataType,
                    std::vector<std::vector<void*>> &upWeights, DataType upDataType,
                    uint8_t *gateOutData, DataType gateOutDataType,
                    uint8_t *upOutData, DataType upOutDataType,
                    int hidden_size, int intermediate_size, int expert_num,
                    int *expertOffsets) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        int threadNum = pool->threads.size();
        int stride = 64;  // 固定分成64的小块
        
        // 创建所有任务
        std::vector<std::vector <fastllm::MultiThreadGateUpBatchOp*> > ops;
        ops.resize(numaConfig->numaCnt);

        for (int expert_id = 0; expert_id < expert_num; expert_id++) {
            // 跳过空的专家
            if (expertOffsets[expert_id + 1] - expertOffsets[expert_id] == 0) {
                continue;
            }
            
            int intermediateSizePer = intermediate_size / numaConfig->numaCnt;
            for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                int base = intermediateSizePer * nid;
                size_t gateOutOffset = GetDataBytes(gateOutDataType, 1, base);
                size_t upOutOffset = GetDataBytes(upOutDataType, 1, base);
                
                for (int j = 0; j < intermediateSizePer; j += stride) {
                    int end = std::min(j + stride, intermediateSizePer);
                    ops[nid].push_back(new MultiThreadGateUpBatchOp(
                        inputData, inputDataType,
                        &gateWeights[nid], gateDataType,
                        &upWeights[nid], upDataType,
                        gateOutData + gateOutOffset, gateOutDataType,
                        upOutData + upOutOffset, upOutDataType,
                        hidden_size, intermediateSizePer,
                        expertOffsets, expert_id, j, end,
                        intermediate_size
                    ));
                }
            }
        }

        DynamicScheduleTasks(ops);
    }

    struct MultiThreadDownBatchOp : MultiThreadBaseOp {
        uint8_t *inputData;
        std::vector<void*> *downWeights;  // Pointer to vector of expert weights
        uint8_t *downOutData;
        DataType inputDataType, downDataType, downOutDataType;
        int hidden_size, intermediate_size;
        int *expertOffsets;
        int expert_id, st, end;
        int output_offset;

        MultiThreadDownBatchOp(uint8_t *inputData, DataType inputDataType,
            std::vector<void*> *downWeights, DataType downDataType,
            uint8_t *downOutData, DataType downOutDataType,
            int hidden_size, int intermediate_size,
            int *expertOffsets, int expert_id, int st, int end,
            int output_offset) : 
            inputData(inputData), inputDataType(inputDataType),
            downWeights(downWeights), downDataType(downDataType),
            downOutData(downOutData), downOutDataType(downOutDataType),
            hidden_size(hidden_size), intermediate_size(intermediate_size),
            expertOffsets(expertOffsets), expert_id(expert_id), st(st), end(end),
            output_offset(output_offset) {}

        void Run() {
            size_t tempPerRow = GetDataBytes(inputDataType, 1, intermediate_size);
            
            int start = expertOffsets[expert_id];
            int batch_end = expertOffsets[expert_id + 1];
            int batch_size = batch_end - start;
            if (batch_size == 0) return;
            
            const uint8_t* down_weight = (uint8_t*)(*downWeights)[expert_id];
            
            FastllmGemm(
                batch_size, intermediate_size, hidden_size,
                inputData + start * tempPerRow, GetDataBytes(inputDataType, 1, intermediate_size),
                (uint8_t*)down_weight, GetDataBytes(downDataType, 1, intermediate_size),
                downOutData + GetDataBytes(downOutDataType, start, output_offset), 
                GetDataBytes(downOutDataType, 1, output_offset),
                st, end,
                inputDataType, downDataType, downOutDataType
            );
        }
    };

    void MultiThreadDownBatch(uint8_t *inputData, DataType inputDataType,
                    std::vector<std::vector<void*>> &downWeights, DataType downDataType,
                    uint8_t *downOutData, DataType downOutDataType,
                    int hidden_size, int intermediate_size, int expert_num,
                    int *expertOffsets) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        int threadNum = pool->threads.size();
        int stride = 64;  // 固定分成64的小块
        
        // 创建所有任务
        std::vector<std::vector <fastllm::MultiThreadDownBatchOp*> > ops;
        ops.resize(numaConfig->numaCnt);

        for (int expert_id = 0; expert_id < expert_num; expert_id++) {
            // 跳过空的专家
            if (expertOffsets[expert_id + 1] - expertOffsets[expert_id] == 0) {
                continue;
            }

            int hiddenSizePer = hidden_size / numaConfig->numaCnt;
            for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                int base = hiddenSizePer * nid;
                size_t downOutOffset = GetDataBytes(downOutDataType, 1, base);
                
                for (int j = 0; j < hiddenSizePer; j += stride) {
                    int end = std::min(j + stride, hiddenSizePer);
                    ops[nid].push_back(new MultiThreadDownBatchOp(
                        inputData, inputDataType,
                        &downWeights[nid], downDataType,
                        downOutData + downOutOffset, downOutDataType,
                        hiddenSizePer, intermediate_size,
                        expertOffsets, expert_id, j, end, 
                        hidden_size
                    ));
                }
            }
        }

        DynamicScheduleTasks(ops);
    }


    struct MultiThreadReduceBatchOp : MultiThreadBaseOp {
        uint8_t *downOutData;
        DataType downOutDataType;
        float *weights;
        float *lastOutput;
        int *pos;
        int bsz, k;
        int hidden_size;
        int batch_st, batch_end;  // batch维度的范围
        int hidden_st, hidden_end; // hidden维度的范围
        
        MultiThreadReduceBatchOp(uint8_t *downOutData, DataType downOutDataType,
            float *weights, float *lastOutput,
            int *pos, int bsz, int k, 
            int hidden_size, 
            int batch_st, int batch_end,
            int hidden_st, int hidden_end) : 
            downOutData(downOutData), downOutDataType(downOutDataType),
            weights(weights), lastOutput(lastOutput),
            pos(pos), bsz(bsz), k(k),
            hidden_size(hidden_size), 
            batch_st(batch_st), batch_end(batch_end),
            hidden_st(hidden_st), hidden_end(hidden_end) {}
            
        void Run() {
            for (int i = batch_st; i < batch_end; i++) {
                // 处理第一个专家（初始化输出）
                int curPos = pos[i * k];
                float weight = weights[curPos];
                for (int h = hidden_st; h < hidden_end; h++) {
                    lastOutput[i * hidden_size + h] = weight * ((float*)downOutData)[curPos * hidden_size + h];
                }
                
                // 累加其余专家的贡献
                for (int expert_idx = 1; expert_idx < k; expert_idx++) {
                    curPos = pos[i * k + expert_idx];
                    weight = weights[curPos];
                    for (int h = hidden_st; h < hidden_end; h++) {
                        lastOutput[i * hidden_size + h] += weight * ((float*)downOutData)[curPos * hidden_size + h];
                    }
                }
            }
        }
    };

    void MultiThreadReduceBatch(uint8_t *downOutData, DataType downOutDataType,
                    float *weights, float *lastOutput,
                    int *pos, int bsz, int k,
                    int hidden_size) {
        auto *pool = GetAlivePool();
        int threadNum = pool->threads.size();
        
        // 决定如何划分：尝试创建一个接近正方形的网格
        int batch_blocks = 1, hidden_blocks = threadNum;
        
        // 简单的启发式：如果bsz足够大，尝试在两个维度上划分
        if (bsz >= 4 && threadNum >= 4) {
            // 找到最佳的2D网格划分
            for (int b = 2; b <= std::min(bsz, threadNum); b++) {
                if (threadNum % b == 0) {
                    int h = threadNum / b;
                    if (h <= hidden_size) {
                        batch_blocks = b;
                        hidden_blocks = h;
                    }
                }
            }
        }
        
        std::vector<fastllm::MultiThreadReduceBatchOp*> ops;
        ops.reserve(threadNum);
        
        int batch_per = bsz / batch_blocks;
        int hidden_per = hidden_size / hidden_blocks;
        
        int op_idx = 0;
        for (int b = 0; b < batch_blocks; b++) {
            int batch_st = b * batch_per;
            int batch_end = (b == batch_blocks - 1) ? bsz : (b + 1) * batch_per;
            
            for (int h = 0; h < hidden_blocks; h++) {
                int hidden_st = h * hidden_per;
                int hidden_end = (h == hidden_blocks - 1) ? hidden_size : (h + 1) * hidden_per;
                
                ops.push_back(new MultiThreadReduceBatchOp(
                    downOutData, downOutDataType,
                    weights, lastOutput,
                    pos, bsz, k,
                    hidden_size,
                    batch_st, batch_end,
                    hidden_st, hidden_end));
                
                pool->PushOp(op_idx++, ops.back());
            }
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void FastllmMoe::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, 
                    const void* input, void* output) {
        const float* input_fp32 = static_cast<const float*>(input);
        float* output_fp32 = static_cast<float*>(output);
        size_t ioPerRow = GetDataBytes(this->hiddenType, 1, this->hiddenSize);

        if (qlen < 32) {
            for (int i = 0; i < qlen; i++) {
                this->forwardOne(k, expert_ids + i * k, weights + i * k, 
                    (uint8_t*)input + i * ioPerRow, (uint8_t*)output + i * ioPerRow);
            }
        } else {
            int len = 4096;
            for (int i = 0; i < qlen; i += len) {
                int curLen = std::min(len, qlen - i);
                this->forwardMulti(curLen, k, expert_ids + i * k, weights + i * k, 
                    (uint8_t*)input + i * ioPerRow, (uint8_t*)output + i * ioPerRow);
            }
        }

        sync_flag.store(true, std::memory_order_seq_cst);
        return;
    }

    void FastllmMoe::forwardOne(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
auto st = std::chrono::system_clock::now();
        auto &gate_out = fastllmMoeDataManager.gate_out;
        if (gate_out.size() < k * intermediateSize) {
            gate_out.resize(k * intermediateSize);
        }
        auto &up_out = fastllmMoeDataManager.up_out;
        if (up_out.size() < k * intermediateSize) {
            up_out.resize(k * intermediateSize);
        }
        
        float* output_fp32 = static_cast<float*>(output);
        auto &fp32OutputVector = fastllmMoeDataManager.fp32OutputVector;
        if (hiddenType != DataType::FLOAT32) {
            if (fp32OutputVector.size() < this->hiddenSize) {
                fp32OutputVector.resize(this->hiddenSize);
            }
            output_fp32 = fp32OutputVector.data();
        }
        memset(output_fp32, 0, 1 * hiddenSize * sizeof(float));
// printf("input spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        // Calculate bytes per expert for each weight type
        size_t gateBytesPerExpert = GetDataBytes(gate->dataType, intermediateSize, hiddenSize);
        size_t upBytesPerExpert = GetDataBytes(up->dataType, intermediateSize, hiddenSize);
        size_t downBytesPerExpert = GetDataBytes(down->dataType, hiddenSize, intermediateSize);
        
        // Execute Gate and Up projections for all experts in one launch
        MultiThreadGateUp(
            (uint8_t*)input, hiddenType,
            gate->datas, const_cast<uint64_t*>(expert_ids), gate->dataType,
            up->datas, up->dataType,
            (uint8_t*)gate_out.data(), DataType::FLOAT32,
            (uint8_t*)up_out.data(), DataType::FLOAT32,
            1, hiddenSize, intermediateSize, k
        );
// printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        DataType tempDataType = hiddenType;
        size_t tempPerRow = GetDataBytes(tempDataType, 1, intermediateSize);
        auto &temp = fastllmMoeDataManager.temp;
        if (temp.size() < 1 * k * tempPerRow) {
            temp.resize(1 * k * tempPerRow);
        }
        ConvertFromFloat32 (
            temp.data(), tempDataType, gate_out.data(), 1 * k, intermediateSize
        );
        auto &newOutput = fastllmMoeDataManager.newOutput;
        if (newOutput.size() < k * hiddenSize) {
            newOutput.resize(k * hiddenSize);
        }
// printf("down input spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        // Execute Down projections for all experts in one launch
        MultiThreadDown(
            (uint8_t*)temp.data(), tempDataType,
            down->datas, const_cast<uint64_t*>(expert_ids), down->dataType,
            (uint8_t*)newOutput.data(), DataType::FLOAT32,
            const_cast<float*>(weights), output_fp32,
            1, intermediateSize, hiddenSize, k
        );
// printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        if (hiddenType != DataType::FLOAT32) {
            fastllm::ConvertFromFloat32(output, this->hiddenType, output_fp32, 1, this->hiddenSize);
        }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
    }
    void FastllmMoe::forwardMulti(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
        auto st = std::chrono::system_clock::now();
        float* output_fp32 = static_cast<float*>(output);
        auto &fp32OutputVector = fastllmMoeDataManager.fp32OutputVector;
        if (hiddenType != DataType::FLOAT32) {
            if (fp32OutputVector.size() < qlen * this->hiddenSize) {
                fp32OutputVector.resize(qlen * this->hiddenSize);
            }
            output_fp32 = fp32OutputVector.data();
        }
        // std::fill(output_fp32, output_fp32 + qlen * hiddenSize, 0.0f);
// printf("create output spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        size_t inputPerRow = GetDataBytes(hiddenType, 1, hiddenSize);
        auto &oriInput = fastllmMoeDataManager.oriInput;
        if (oriInput.size() < qlen * inputPerRow) {
            oriInput.resize(qlen * inputPerRow);
        }
        auto &newInput = fastllmMoeDataManager.newInput;
        if (newInput.size() < qlen * k * inputPerRow) {
            newInput.resize(qlen * k * inputPerRow);
        }
        auto &newOutput = fastllmMoeDataManager.newOutput;
        if (newOutput.size() < qlen * k * hiddenSize) {
            newOutput.resize(qlen * k * hiddenSize);
        }
        auto &gate_out = fastllmMoeDataManager.gate_out;
        if (gate_out.size() < qlen * k * intermediateSize) {
            gate_out.resize(qlen * k * intermediateSize);
        }
        auto &up_out = fastllmMoeDataManager.up_out;
        if (up_out.size() < qlen * k * intermediateSize) {
            up_out.resize(qlen * k * intermediateSize);
        }
        DataType tempDataType = hiddenType;
        size_t tempPerRow = GetDataBytes(tempDataType, 1, intermediateSize);
        auto &temp = fastllmMoeDataManager.temp;
        if (temp.size() < qlen * k * tempPerRow) {
            temp.resize(qlen * k * tempPerRow);
        }
// printf("create temp spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        // 处理输入
        std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
        memcpyTasks.resize(qlen);
        for (int i = 0; i < qlen; i++) {
            memcpyTasks[i] = MultiThreadMemcpyMultiLinesTask(
                oriInput.data() + i * inputPerRow, 
                (uint8_t*)input + i * inputPerRow, 
                inputPerRow
            );
        }
        RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
// printf("copy input 0 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        // 重新排列输入和权重，按专家分组
        // 计算每个专家需要处理的样本数量
        std::vector <int> expertOffsets;
        expertOffsets.resize(expertNum + 1);
        for (int i = 0; i < qlen * k; i++) {
            expertOffsets[expert_ids[i] + 1]++;
        }
        for (int i = 1; i < expertNum + 1; i++) {
            expertOffsets[i] += expertOffsets[i - 1];
        }
        std::vector <int> curPos = expertOffsets;
        std::vector <int> pos;
        std::vector <float> oldWeights;
        pos.resize(qlen * k);
        oldWeights.resize(qlen * k);
        
        // 重新排列输入和权重，按专家分组
        // std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
        memcpyTasks.resize(qlen * k);
        for (int i = 0; i < qlen * k; i++) {
            uint64_t eid = expert_ids[i];
            int bid = i / k;
            pos[i] = curPos[eid]++;
            oldWeights[pos[i]] = weights[i];
            memcpyTasks[i] = MultiThreadMemcpyMultiLinesTask(
                newInput.data() + pos[i] * inputPerRow, 
                oriInput.data() + bid * inputPerRow, 
                inputPerRow
            );
        }
// printf("ready copy spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
// printf("copy input spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        // 一次性处理所有专家的 gate 和 up 投影
        MultiThreadGateUpBatch(
            newInput.data(), hiddenType,
            gate->datas, gate->dataType,
            up->datas, up->dataType,
            (uint8_t*)gate_out.data(), DataType::FLOAT32,
            (uint8_t*)up_out.data(), DataType::FLOAT32,
            hiddenSize, intermediateSize, expertNum,
            expertOffsets.data()
        );
        
// printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        RunMultiThreadConvertFromFloat32(
            temp.data(), tempDataType, gate_out.data(), qlen * k, intermediateSize, GetAlivePool()
        );
        
// printf("ConvertFromFloat32 temp spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        
        // 一次性处理所有专家的 down 投影
        MultiThreadDownBatch (
            temp.data(), tempDataType,
            down->datas, down->dataType,
            (uint8_t*)newOutput.data(), DataType::FLOAT32,
            hiddenSize, intermediateSize, expertNum,
            expertOffsets.data()
        );
// printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        MultiThreadReduceBatch (
            (uint8_t*)newOutput.data(), DataType::FLOAT32,
            oldWeights.data(), output_fp32,
            pos.data(), qlen, k,
            hiddenSize
        );
// printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        // 转换回原始数据类型
        if (hiddenType != DataType::FLOAT32) {
            RunMultiThreadConvertFromFloat32(output, this->hiddenType, output_fp32, qlen, this->hiddenSize, GetAlivePool());
        }
        
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
    }

    static void forward_wrapper_fastllmmoe(void* args) {
        FastllmModeForwardParams* params = (FastllmModeForwardParams*)args;
            params->moe_ptr->forward(
                params->qlen, 
                params->k, 
                params->expert_ids, 
                params->weights, 
                params->input, 
                params->output
            );
            // delete params;  !don't delete params here
    }


    void FastllmMoe::submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids, 
                                 const float* weights, const void* input, void* output, int* bsz_tensor) {
        sync_flag.store(false, std::memory_order_seq_cst); 
        FastllmModeForwardParams* params = new FastllmModeForwardParams();
        params->moe_ptr = this;
        params->qlen = qlen;
        params->k = k;
        params->expert_ids = expert_ids;
        params->weights = weights;
        params->input = input;
        params->output = output; 
        params->bsz_tensor = bsz_tensor;
        
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)forward_wrapper_fastllmmoe, params);
    }

    void FastllmMoe::sync() {
        while (!sync_flag.load(std::memory_order_seq_cst));
    }

    static void sync_fastllmmoe_(void * moe_ptr) { 
        FastllmMoe* moe = (FastllmMoe*)moe_ptr;
        moe->sync();
    }

    void FastllmMoe::sync_with_cuda_stream(intptr_t user_cuda_stream) { 
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_fastllmmoe_, (void*)this);
    }
}