//
// Created by huangyuyang on 10/30/25.
//

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#include "fastllm.h"

namespace fastllm {
    void AddBias(float *outputData, float *biasData, int n, int k, int st, int end) {
        if (biasData) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    outputData[i * k + j] += biasData[j];
                }
            }
        }
    }

    extern BF16ToFP32Manager bf16tofp32;
    extern FP8E4M3ToFP32Manager fp8e4m3tofp32;

    bool LinearBFloat16BFloat16_Base_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float sum = 0.0f;
                for (int l = 0; l < m; l++) {
                    sum += bf16tofp32.dict[inputData[i * m + l]] * bf16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = sum;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    extern bool LinearBFloat16BFloat16_AVX512BF16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16BFloat16_AVX2_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);  
    bool LinearBFloat16BFloat16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512BF16) {
            return LinearBFloat16BFloat16_AVX512BF16_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else if (GetCPUInstructInfo()->hasAVX2) {
            return LinearBFloat16BFloat16_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else {
            return LinearBFloat16BFloat16_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
        return false;
    }

    bool LinearBFloat16_FP8E4M3BLOCK128_Base_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        static int block_size = 128;
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        
        // 计算总共有多少个完整的block和最后一个不完整block的大小
        int num_blocks = (m + block_size - 1) / block_size;
        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
                            
            for (int j = st; j < end; j++) {
                uint8_t *rowStart = (uint8_t*)weightData + j * perRow;
                float sum = 0.0f;
                
                // 按block进行处理
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    // 计算当前block的大小（最后一个block可能不完整）
                    int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                    
                    // 计算当前block的起始位置
                    // 每个block占用 128字节(fp8) + 4字节(float scale)
                    uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                    uint8_t *fp8_ptr = block_start;
                    float *scale_ptr = (float*)(block_start + block_size);
                    
                    // 先计算block内的点积，最后再乘以scale
                    float block_sum = 0.0f;
                    int base_idx = block_idx * block_size;
                    
                    for (int l = 0; l < current_block_size; l++) {
                        // 将bf16的A转换为fp32
                        float valA = bf16tofp32.dict[bf16A[base_idx + l]];
                        
                        // 将fp8的B转换为fp32
                        float valB = fp8e4m3tofp32.dict[fp8_ptr[l]];
                        
                        block_sum += valA * valB;
                    }
                    
                    // 整个block的结果乘以scale
                    sum += block_sum * (*scale_ptr);
                }
                
                floatC[j] = sum;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    extern bool LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearBFloat16_FP8E4M3BLOCK128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512BF16) {
            return LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else if (GetCPUInstructInfo()->hasAVX2) {
            return LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else {
            return LinearBFloat16_FP8E4M3BLOCK128_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
        return false;
    }

    bool LinearBFloat16_AWQ4BIT128_Base_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        static constexpr int block_size = 128;
        size_t perRow = GetDataBytes(DataType::AWQ_4BIT_128, 1, m);
        int num_blocks = m / block_size;
        
        // 预分配缓存数组，避免重复分配
        std::vector<float> input_block_sums(num_blocks);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
            
            // 预计算当前输入行的每个block的和
            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                float block_input_sum = 0.0f;
                int block_start_idx = block_idx * block_size;
                int block_end = std::min(block_start_idx + block_size, m);
                
                // 优化: 展开循环，一次处理4个元素
                int k = block_start_idx;
                for (; k + 3 < block_end; k += 4) {
                    block_input_sum += bf16tofp32.dict[bf16A[k]] + 
                                    bf16tofp32.dict[bf16A[k + 1]] +
                                    bf16tofp32.dict[bf16A[k + 2]] +
                                    bf16tofp32.dict[bf16A[k + 3]];
                }
                // 处理剩余元素
                for (; k < block_end; k++) {
                    block_input_sum += bf16tofp32.dict[bf16A[k]];
                }
                input_block_sums[block_idx] = block_input_sum;
            }
            
            // 现在对每个输出列j，使用预计算的input_block_sums
            for (int j = st; j < end; j++) {
                uint8_t *awqB = (uint8_t*)weightData + j * perRow;
                float sum = 0.0f;
                
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    uint8_t *block_start = awqB + block_idx * (block_size/2 + 1 + 4);
                    
                    uint8_t *packedWeights = block_start;
                    uint8_t zero = block_start[block_size/2];
                    float scale = *(float*)(block_start + block_size/2 + 1);
                    
                    // 使用预计算的input_sum
                    float zero_correction = zero * scale * input_block_sums[block_idx];
                    
                    float block_sum = 0.0f;
                    
                    // 优化: 展开循环，一次处理4个权重（2个uint8）
                    int k = 0;
                    int base_idx = block_idx * block_size;
                    
                    for (; k + 3 < block_size && base_idx + k + 3 < m; k += 4) {
                        // 读取2个uint8 = 4个uint4权重
                        uint8_t packed0 = packedWeights[k / 2];
                        uint8_t packed1 = packedWeights[k / 2 + 1];
                        
                        // 解包4个权重
                        int w0 = packed0 & 0xF;
                        int w1 = (packed0 >> 4) & 0xF;
                        int w2 = packed1 & 0xF;
                        int w3 = (packed1 >> 4) & 0xF;
                        
                        // 读取4个输入值
                        float valA0 = bf16tofp32.dict[bf16A[base_idx + k]];
                        float valA1 = bf16tofp32.dict[bf16A[base_idx + k + 1]];
                        float valA2 = bf16tofp32.dict[bf16A[base_idx + k + 2]];
                        float valA3 = bf16tofp32.dict[bf16A[base_idx + k + 3]];
                        
                        // 累加
                        block_sum += valA0 * w0 + valA1 * w1 + valA2 * w2 + valA3 * w3;
                    }
                    
                    // 处理剩余元素
                    for (; k < block_size && base_idx + k < m; k += 2) {
                        float valA0 = bf16tofp32.dict[bf16A[base_idx + k]];
                        float valA1 = (base_idx + k + 1 < m) ? bf16tofp32.dict[bf16A[base_idx + k + 1]] : 0.0f;
                        
                        uint8_t packed = packedWeights[k / 2];
                        int w0 = packed & 0xF;
                        int w1 = (packed >> 4) & 0xF;
                        
                        block_sum += valA0 * w0 + valA1 * w1;
                    }
                    
                    // 应用scale和zero correction
                    sum += block_sum * scale - zero_correction;
                }
                
                floatC[j] = sum;
            }
        }
        return true;
    }
    bool LinearBFloat16_AWQ4BIT128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearBFloat16_AWQ4BIT128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX2) {
            return LinearBFloat16_AWQ4BIT128_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else {
            return LinearBFloat16_AWQ4BIT128_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
    }

    bool AWQ4BIT128_TO_BFloat16_Base_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb) {
        // 参数检查
        if (awqData == nullptr || bf16Data == nullptr) {
            return false;
        }
        if (st < 0 || end <= st || m <= 0) {
            return false;
        }
        if (m % 128 != 0) {
            // AWQ要求m必须是128的倍数
            return false;
        }
        
        const int block_size = 128;
        const int blocks_per_row = m / block_size;
        const int block_bytes = 64 + 1 + 4;
        
        // 转换awq4bit到bf16，处理行[st:end]
        for (int j = st; j < end; j++) {
            // 获取当前行的AWQ数据指针
            uint8_t *awqB = awqData + j * ldb;
            // 获取当前行的BF16输出指针
            uint16_t *bf16B_row = bf16Data + (j - st) * m;
            
            // 遍历每个block
            for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
                // 计算当前block的起始位置
                uint8_t *block_start = awqB + block_idx * block_bytes;
                
                // 解析当前block的组成部分
                uint8_t *packedWeights = block_start;                    // 64字节，存储128个uint4
                uint8_t zero = *(block_start + 64);                      // 1字节zero
                float scale = *(float*)(block_start + 64 + 1);          // 4字节scale
                
                // 处理当前block中的128个元素
                for (int elem_in_block = 0; elem_in_block < block_size; elem_in_block++) {
                    int l = block_idx * block_size + elem_in_block;
                    
                    // 从packed格式中提取uint4权重
                    int weight_idx = elem_in_block / 2;
                    int weight_shift = (elem_in_block & 1) * 4;
                    int w = (packedWeights[weight_idx] >> weight_shift) & 0xF;
                    
                    // 反量化：(w - zero) * scale
                    float fp32_val = (w - zero) * scale;
                    
                    // 转换为bf16
                    uint32_t val;
                    memcpy(&val, &fp32_val, sizeof(val));
                    bf16B_row[l] = (uint16_t)(val >> 16);
                }
            }
        }
        
        return true;
    }
    extern bool AWQ4BIT128_TO_BFloat16_AVX2_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb);
    extern BF16ToFP32Manager bf16tofp32;
    
    bool AWQ4BIT128_TO_BFloat16_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb) {
        if (GetCPUInstructInfo()->hasAVX2) {
            return AWQ4BIT128_TO_BFloat16_AVX2_Kernel(awqData, bf16Data, m, st, end, ldb);
        } else {
            return AWQ4BIT128_TO_BFloat16_Base_Kernel(awqData, bf16Data, m, st, end, ldb);
        }
    }
}