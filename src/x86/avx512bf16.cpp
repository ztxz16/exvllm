//
// Created by huangyuyang on 5/8/25.
//

#include <cstdint>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

#include "fastllm.h"

namespace fastllm {
    extern void AddBiasAVX512(float *outputData, float *biasData, int n, int k, int st, int end);
    
    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx512(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512BF16__
        constexpr int SIMD_WIDTH = 32;  // AVX512 一次处理 32 个 bf16
        int nb = n / SIMD_WIDTH;
        int remainder = n % SIMD_WIDTH;
        if (remainder != 0) {
            printf("In mul_mat_bf16_bf16_direct_avx512, n %% 32 should be 0.");
            exit(0);
        }

        // 累加器 - 注意这里的顺序
        __m512 acc[AROW * BROW];
        
        // 初始化
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm512_setzero_ps();
        }
        
        // 主循环
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                __m512bh a_vec = (__m512bh)_mm512_loadu_si512((__m512i const*)(a_row + i * SIMD_WIDTH));

                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    __m512bh b_vec = (__m512bh)_mm512_loadu_si512((__m512i const*)(b_row + i * SIMD_WIDTH));
                    
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm512_dpbf16_ps(acc[acc_idx], a_vec, b_vec);
                }
            }
        }
        
        // 水平求和并存储
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                float result = _mm512_reduce_add_ps(acc[acc_idx]);
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result;
            }
        }
#endif
    }

    template <int BRow>
    void LinearBFloat16BFloat16_AVX512BF16_Row_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int i, int m, int k, int st, int end) {
        int j = st;
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_bf16_bf16_direct_avx512 <BRow, 5> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float));
        }
        switch (end - j) {
            case 0: break;
            case 1: mul_mat_bf16_bf16_direct_avx512 <BRow, 1> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 2: mul_mat_bf16_bf16_direct_avx512 <BRow, 2> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 3: mul_mat_bf16_bf16_direct_avx512 <BRow, 3> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 4: mul_mat_bf16_bf16_direct_avx512 <BRow, 4> (m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
        }
    }

    bool LinearBFloat16BFloat16_AVX512BF16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        int i = 0;
        for (; i + 4 < n; i += 5) {
            LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <5> (inputData, weightData, biasData, outputData, i, m, k, st, end);
        }
        switch (n - i) {
            case 0: break;
            case 1: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <1> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 2: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <2> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 3: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <3> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 4: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <4> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
        }
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
    }

/*
// 待优化
    template <int BROW, int AROW>
    void mul_mat_bf16_fp8e4m3block128_direct_avx512(
        int n,
        const uint8_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX512BF16__
        static constexpr int block_size = 128;
        static constexpr int SIMD_WIDTH = 32;  // AVX512 processes 32 elements at once
        
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, n);
        float magicScale = pow(2, 120);
        
        // 计算总共有多少个完整的block
        int num_blocks = (n + block_size - 1) / block_size;
        
        // 设置FP8转换掩码
        __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
        __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F);
        
        // 累加器数组 - 用于存储每个输出元素的累加结果
        __m512 acc[BROW * AROW];
        
        for (int i = 0; i < BROW; i++) {
            const uint16_t *bf16B = (const uint16_t*)((const uint8_t*)B + stride_b * i);
            float *floatC = (float*)((uint8_t*)C + i * stride_c);
            
            // 初始化当前行的所有累加器
            for (int j = 0; j < AROW; j++) {
                acc[i * AROW + j] = _mm512_setzero_ps();
            }
            
            // 遍历所有block
            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                int block_start = block_idx * block_size;
                int block_end = std::min(block_start + block_size, n);
                int current_block_size = block_end - block_start;
                
                // 处理每个A矩阵的行
                for (int j = 0; j < AROW; j++) {
                    const uint8_t *rowStart = ((const uint8_t*)A + j * perRow);
                    
                    // 计算当前block的起始位置
                    // 每个block占用 128字节(fp8) + 4字节(float scale)
                    const uint8_t *block_ptr = rowStart + block_idx * (block_size + sizeof(float));
                    const uint8_t *fp8_ptr = block_ptr;
                    float scale = *(const float*)(block_ptr + block_size);
                    __m512 vScale = _mm512_set1_ps(scale);
                    
                    __m512 block_acc = _mm512_setzero_ps();
                    
                    // 处理当前block内的数据，每次32个元素
                    int l = 0;
                    for (; l + SIMD_WIDTH <= current_block_size; l += SIMD_WIDTH) {
                        // 1. Load 32 BF16 inputs from B
                        __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512(
                            (__m512i const*)(bf16B + block_start + l));
                        
                        // 2. Load 32 FP8 weights from current block and convert to BF16
                        __m256i va_bytes = _mm256_loadu_si256((__m256i*)(fp8_ptr + l));
                        
                        // 处理高4位
                        __m256i va_masked_bytes = _mm256_and_si256(va_bytes, v_a_mask_byte);
                        __m512i va_promoted_words = _mm512_cvtepu8_epi16(va_masked_bytes);
                        __m512i v_a_term_shifted = _mm512_slli_epi16(va_promoted_words, 8);
                        
                        // 处理低4位
                        __m256i vb_masked_bytes = _mm256_and_si256(va_bytes, v_b_mask_byte);
                        __m512i vb_promoted_words = _mm512_cvtepu8_epi16(vb_masked_bytes);
                        __m512i v_b_term_shifted = _mm512_slli_epi16(vb_promoted_words, 4);
                        
                        // 组合成BF16格式
                        __m512i v_result = _mm512_or_si512(v_a_term_shifted, v_b_term_shifted);
                        __m512bh v_weights_bf16 = (__m512bh)v_result;
                        
                        // 3. Compute dot product using BF16 operations
                        block_acc = _mm512_dpbf16_ps(block_acc, v_input_bf16, v_weights_bf16);
                    }
                    
                    // 将当前block的结果乘以scale并累加到总结果中
                    acc[i * AROW + j] = _mm512_fmadd_ps(block_acc, vScale, acc[i * AROW + j]);
                }
            }
            
            // 水平求和并存储结果
            for (int j = 0; j < AROW; j++) {
                float result = _mm512_reduce_add_ps(acc[i * AROW + j]) * magicScale;
                floatC[j] = result;
            }
        }
#endif
    }


    template <int BRow>
    void LinearBFloat16BFloat16_AVX512BF16_Row_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int i, int m, int k, int st, int end) {
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        int j = st;
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_bf16_fp8e4m3block128_direct_avx512 <BRow, 5> (m, weightData + j * perRow, perRow, inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float));
        }
        switch (end - j) {
            case 0: break;
            case 1: mul_mat_bf16_fp8e4m3block128_direct_avx512 <BRow, 1> (m, weightData + j * perRow, perRow, inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 2: mul_mat_bf16_fp8e4m3block128_direct_avx512 <BRow, 2> (m, weightData + j * perRow, perRow, inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 3: mul_mat_bf16_fp8e4m3block128_direct_avx512 <BRow, 3> (m, weightData + j * perRow, perRow, inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 4: mul_mat_bf16_fp8e4m3block128_direct_avx512 <BRow, 4> (m, weightData + j * perRow, perRow, inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
        }
    }

    bool LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        int i = 0;
        for (; i + 4 < n; i += 5) {
            LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <5> (inputData, weightData, biasData, outputData, i, m, k, st, end);
        }
        switch (n - i) {
            case 0: break;
            case 1: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <1> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 2: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <2> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 3: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <3> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 4: LinearBFloat16BFloat16_AVX512BF16_Row_Kernel <4> (inputData, weightData, biasData, outputData, i, m, k, st, end); break;
        }
        AddBiasAVX512(outputData, biasData, n, k, st, end);
        return true;
    }
*/
    bool LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX512BF16__
        static int block_size = 128;
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        float magicScale = pow(2, 120);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;

            int j = st;
            __m256i v_a_mask_byte = _mm256_set1_epi8(0x80); 
            __m256i v_b_mask_byte = _mm256_set1_epi8(0x7F); 
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m512 last_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums

                // 获取当前行的起始位置
                uint8_t *rowData = (uint8_t*)weightData + j * perRow;
                
                // 计算需要多少个block（每个block有128个FP8 + 1个float scale）
                const int blockM = 128;
                int numBlocks = (m + blockM - 1) / blockM;
                
                for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                    // 计算当前block在rowData中的偏移
                    // 每个block占用 128 bytes (FP8) + 4 bytes (float scale)
                    size_t blockOffset = blockIdx * (blockM + sizeof(float));
                    
                    // 获取当前block的FP8数据和scale
                    uint8_t *fp8B = rowData + blockOffset;
                    
                    // 计算当前block处理的元素范围
                    int blockStart = blockIdx * blockM;
                    int blockEnd = std::min(blockStart + blockM, m);
                    
                    __m512 v_sum = _mm512_setzero_ps(); // Accumulator for 16 parallel sums
                    
                    // 处理当前block内的数据
                    int l = blockStart;
                    for (; l + 31 < blockEnd; l += 32) {
                        // 1. Load 32 BF16 inputs
                        __m512bh v_input_bf16 = (__m512bh)_mm512_loadu_si512((__m512i const*)(bf16A + l));
                        
                        // 2. Load 32 FP8 weights from current block
                        // 注意：fp8B指向当前block的开始，所以需要用 (l - blockStart) 作为偏移
                        __m256i va_bytes = _mm256_loadu_si256((__m256i*)(fp8B + (l - blockStart)));

                        __m256i va_masked_bytes = _mm256_and_si256(va_bytes, v_a_mask_byte);
                        __m512i va_promoted_words = _mm512_cvtepu8_epi16(va_masked_bytes);
                        __m512i v_a_term_shifted = _mm512_slli_epi16(va_promoted_words, 8);

                        __m256i vb_masked_bytes = _mm256_and_si256(va_bytes, v_b_mask_byte);
                        __m512i vb_promoted_words = _mm512_cvtepu8_epi16(vb_masked_bytes);
                        __m512i v_b_term_shifted = _mm512_slli_epi16(vb_promoted_words, 4);

                        __m512i v_result = _mm512_or_si512(v_a_term_shifted, v_b_term_shifted);
                        __m512bh v_weights_bf16 = (__m512bh)v_result;
                        
                        // 3. Compute dot product: v_sum += v_input_bf16 * v_weights_bf16
                        v_sum = _mm512_dpbf16_ps(v_sum, v_input_bf16, v_weights_bf16);
                    }
                    
                    // 处理剩余的元素（如果有）
                    // TODO: 这里可能需要处理不足32个元素的情况
                    
                    float curScale = *(float*)(fp8B + blockM);  // scale在128个FP8之后
                    __m512 vScale = _mm512_set1_ps(curScale);
                    last_sum = _mm512_fmadd_ps(v_sum, vScale, last_sum);
                }
                
                now += _mm512_reduce_add_ps(last_sum) * magicScale;
                floatC[j] = now;
            }
        }
        return true;
#endif
        return false;
    }
}