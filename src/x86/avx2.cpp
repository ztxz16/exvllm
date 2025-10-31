//
// Created by huangyuyang on 10/30/25.
//

#include <cstdint>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include "fastllm.h"

namespace fastllm {
#ifdef __AVX2__
    // BF16到FP32的转换辅助函数
    inline __m256 bf16_to_fp32_avx2(__m128i bf16_data) {
        // BF16只是FP32的高16位，所以左移16位即可
        __m256i fp32_data = _mm256_cvtepu16_epi32(bf16_data);
        fp32_data = _mm256_slli_epi32(fp32_data, 16);
        return _mm256_castsi256_ps(fp32_data);
    }
#endif

    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx2(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int SIMD_WIDTH_FP32 = 8;  // AVX2 一次处理 8 个 float
        constexpr int SIMD_WIDTH_BF16 = 8;  // 一次加载 8 个 bf16
        
        int nb = n / SIMD_WIDTH_BF16;
        int remainder = n % SIMD_WIDTH_BF16;
        
        // 累加器
        __m256 acc[AROW * BROW];
        
        // 初始化累加器
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }
        
        // 主循环 - 每次处理8个元素
        for (int i = 0; i < nb; ++i) {
            // 对每个A的行
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                // 加载8个BF16值并转换为FP32
                __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a_row + i * SIMD_WIDTH_BF16));
                __m256 a_fp32 = bf16_to_fp32_avx2(a_bf16);
                
                // 对每个B的行
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    // 加载8个BF16值并转换为FP32
                    __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b_row + i * SIMD_WIDTH_BF16));
                    __m256 b_fp32 = bf16_to_fp32_avx2(b_bf16);
                    
                    // 执行FMA操作
                    int acc_idx = ix * BROW + iy;
                    acc[acc_idx] = _mm256_fmadd_ps(a_fp32, b_fp32, acc[acc_idx]);
                }
            }
        }

        // 水平求和并存储结果
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                
                // 水平求和 AVX2版本
                __m256 sum = acc[acc_idx];
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                
                // 继续水平求和
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float result = _mm_cvtss_f32(sum128);
                
                // 存储结果
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = result;
            }
        }
#endif
    }

    // 更高效的版本 - 一次处理多个8元素块
    template <int BROW, int AROW>
    void mul_mat_bf16_bf16_direct_avx2_optimized(
        int n,
        const uint16_t* A,
        size_t stride_a,
        const uint16_t* B,
        size_t stride_b,
        float* C,
        size_t stride_c
    ) {
#ifdef __AVX2__
        constexpr int UNROLL = 4;  // 展开因子
        constexpr int SIMD_WIDTH = 8;
        
        int nb = n / (SIMD_WIDTH * UNROLL);
        int remainder = n % (SIMD_WIDTH * UNROLL);
        
        // 累加器
        __m256 acc[AROW * BROW];
        for (int i = 0; i < AROW * BROW; ++i) {
            acc[i] = _mm256_setzero_ps();
        }
        
        // 主循环 - 展开4次
        for (int i = 0; i < nb; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                
                // 预加载4组A数据
                __m256 a_fp32[UNROLL];
                for (int u = 0; u < UNROLL; ++u) {
                    __m128i a_bf16 = _mm_loadu_si128((__m128i const*)(a_row + (i * UNROLL + u) * SIMD_WIDTH));
                    a_fp32[u] = bf16_to_fp32_avx2(a_bf16);
                }
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    int acc_idx = ix * BROW + iy;
                    
                    // 展开的内部循环
                    for (int u = 0; u < UNROLL; ++u) {
                        __m128i b_bf16 = _mm_loadu_si128((__m128i const*)(b_row + (i * UNROLL + u) * SIMD_WIDTH));
                        __m256 b_fp32 = bf16_to_fp32_avx2(b_bf16);
                        acc[acc_idx] = _mm256_fmadd_ps(a_fp32[u], b_fp32, acc[acc_idx]);
                    }
                }
            }
        }
        
        // 处理剩余部分
        int start = nb * SIMD_WIDTH * UNROLL;
        for (int i = start; i < n; ++i) {
            for (int ix = 0; ix < AROW; ++ix) {
                const uint16_t* a_row = (const uint16_t*)((const char*)A + ix * stride_a);
                uint32_t a_val_int = ((uint32_t)a_row[i]) << 16;
                float a_val = *((float*)&a_val_int);
                
                for (int iy = 0; iy < BROW; ++iy) {
                    const uint16_t* b_row = (const uint16_t*)((const char*)B + iy * stride_b);
                    uint32_t b_val_int = ((uint32_t)b_row[i]) << 16;
                    float b_val = *((float*)&b_val_int);
                    
                    int acc_idx = ix * BROW + iy;
                    float scalar_result = a_val * b_val;
                    
                    // 使用标量加法
                    __m256 temp = acc[acc_idx];
                    float* temp_arr = (float*)&temp;
                    temp_arr[0] += scalar_result;
                    acc[acc_idx] = temp;
                }
            }
        }
        
        // 水平求和并存储
        for (int ix = 0; ix < AROW; ++ix) {
            for (int iy = 0; iy < BROW; ++iy) {
                int acc_idx = ix * BROW + iy;
                
                __m256 sum = acc[acc_idx];
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                
                float* c_row = (float*)((char*)C + iy * stride_c);
                c_row[ix] = _mm_cvtss_f32(sum128);
            }
        }
#endif
    }

    template <int BRow>
    void LinearBFloat16BFloat16_AVX2_Row_Kernel(
        uint16_t *inputData, 
        uint16_t *weightData, 
        float *biasData, 
        float *outputData,
        int i, int m, int k, int st, int end) 
    {
        int j = st;
        for (j = st; j + 4 < end; j += 5) {
            mul_mat_bf16_bf16_direct_avx2_optimized<BRow, 5>(
                m, weightData + j * m, m * sizeof(uint16_t), 
                inputData + i * m, m * sizeof(uint16_t), 
                outputData + i * k + j, k * sizeof(float));
        }
        
        switch (end - j) {
            case 0: break;
            case 1: mul_mat_bf16_bf16_direct_avx2_optimized<BRow, 1>(m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 2: mul_mat_bf16_bf16_direct_avx2_optimized<BRow, 2>(m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 3: mul_mat_bf16_bf16_direct_avx2_optimized<BRow, 3>(m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
            case 4: mul_mat_bf16_bf16_direct_avx2_optimized<BRow, 4>(m, weightData + j * m, m * sizeof(uint16_t), inputData + i * m, m * sizeof(uint16_t), outputData + i * k + j, k * sizeof(float)); break;
        }
    }

    bool LinearBFloat16BFloat16_AVX2_Kernel(
        uint16_t *inputData, 
        uint16_t *weightData, 
        float *biasData, 
        float *outputData,
        int n, int m, int k, int st, int end) 
    {
        int i = 0;
        for (; i + 4 < n; i += 5) {
            LinearBFloat16BFloat16_AVX2_Row_Kernel<5>(inputData, weightData, biasData, outputData, i, m, k, st, end);
        }
        
        switch (n - i) {
            case 0: break;
            case 1: LinearBFloat16BFloat16_AVX2_Row_Kernel<1>(inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 2: LinearBFloat16BFloat16_AVX2_Row_Kernel<2>(inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 3: LinearBFloat16BFloat16_AVX2_Row_Kernel<3>(inputData, weightData, biasData, outputData, i, m, k, st, end); break;
            case 4: LinearBFloat16BFloat16_AVX2_Row_Kernel<4>(inputData, weightData, biasData, outputData, i, m, k, st, end); break;
        }
        
        // 假设你有AVX2版本的AddBias
        // AddBiasAVX2(outputData, biasData, n, k, st, end);
        return true;
    }

    bool LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        static int block_size = 128;
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        float magicScale = pow(2, 120);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
            int j = st;
            __m128i v_a_mask_byte = _mm_set1_epi8(0x80); 
            __m128i v_b_mask_byte = _mm_set1_epi8(0x7F); 
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m256 last_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
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
                    
                    __m256 v_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
                    
                    // 处理当前block内的数据
                    int l = blockStart;
                    for (; l + 15 < blockEnd; l += 16) {
                        // 1. Load 16 BF16 inputs and convert to float
                        __m256i v_input_bf16 = _mm256_loadu_si256((__m256i const*)(bf16A + l));
                        
                        // Convert BF16 to float32 (shift left by 16 bits)
                        __m128i v_input_low = _mm256_extracti128_si256(v_input_bf16, 0);
                        __m128i v_input_high = _mm256_extracti128_si256(v_input_bf16, 1);
                        
                        // Process low 8 BF16 values
                        __m256i v_input_low_32 = _mm256_cvtepu16_epi32(v_input_low);
                        __m256i v_input_low_shifted = _mm256_slli_epi32(v_input_low_32, 16);
                        __m256 v_input_float_low = _mm256_castsi256_ps(v_input_low_shifted);
                        
                        // Process high 8 BF16 values
                        __m256i v_input_high_32 = _mm256_cvtepu16_epi32(v_input_high);
                        __m256i v_input_high_shifted = _mm256_slli_epi32(v_input_high_32, 16);
                        __m256 v_input_float_high = _mm256_castsi256_ps(v_input_high_shifted);
                        
                        // 2. Load 16 FP8 weights from current block
                        __m128i va_bytes = _mm_loadu_si128((__m128i*)(fp8B + (l - blockStart)));
                        
                        // Extract sign and mantissa for FP8 conversion
                        __m128i va_masked_bytes = _mm_and_si128(va_bytes, v_a_mask_byte);
                        __m128i vb_masked_bytes = _mm_and_si128(va_bytes, v_b_mask_byte);
                        
                        // Convert to 16-bit for BF16 format
                        // Low 8 bytes
                        __m128i va_low_bytes = _mm_unpacklo_epi8(va_masked_bytes, _mm_setzero_si128());
                        __m128i vb_low_bytes = _mm_unpacklo_epi8(vb_masked_bytes, _mm_setzero_si128());
                        __m128i v_a_term_low = _mm_slli_epi16(va_low_bytes, 8);
                        __m128i v_b_term_low = _mm_slli_epi16(vb_low_bytes, 4);
                        __m128i v_result_low = _mm_or_si128(v_a_term_low, v_b_term_low);
                        
                        // High 8 bytes
                        __m128i va_high_bytes = _mm_unpackhi_epi8(va_masked_bytes, _mm_setzero_si128());
                        __m128i vb_high_bytes = _mm_unpackhi_epi8(vb_masked_bytes, _mm_setzero_si128());
                        __m128i v_a_term_high = _mm_slli_epi16(va_high_bytes, 8);
                        __m128i v_b_term_high = _mm_slli_epi16(vb_high_bytes, 4);
                        __m128i v_result_high = _mm_or_si128(v_a_term_high, v_b_term_high);
                        
                        // Convert BF16 weights to float32
                        __m256i v_weight_low_32 = _mm256_cvtepu16_epi32(v_result_low);
                        __m256i v_weight_low_shifted = _mm256_slli_epi32(v_weight_low_32, 16);
                        __m256 v_weight_float_low = _mm256_castsi256_ps(v_weight_low_shifted);
                        
                        __m256i v_weight_high_32 = _mm256_cvtepu16_epi32(v_result_high);
                        __m256i v_weight_high_shifted = _mm256_slli_epi32(v_weight_high_32, 16);
                        __m256 v_weight_float_high = _mm256_castsi256_ps(v_weight_high_shifted);
                        
                        // 3. Compute dot product: multiply and accumulate
                        __m256 v_mul_low = _mm256_mul_ps(v_input_float_low, v_weight_float_low);
                        __m256 v_mul_high = _mm256_mul_ps(v_input_float_high, v_weight_float_high);
                        
                        v_sum = _mm256_add_ps(v_sum, v_mul_low);
                        v_sum = _mm256_add_ps(v_sum, v_mul_high);
                    }
                    
                    // 处理剩余的元素（标量处理）
                    for (; l < blockEnd; l++) {
                        // Convert BF16 input to float
                        uint32_t input_val = ((uint32_t)bf16A[l]) << 16;
                        float input_float = *((float*)&input_val);
                        
                        // Convert FP8 weight to BF16 then to float
                        uint8_t fp8_val = fp8B[l - blockStart];
                        uint16_t sign_and_exp = (fp8_val & 0x80) << 8;
                        uint16_t mantissa = (fp8_val & 0x7F) << 4;
                        uint16_t bf16_val = sign_and_exp | mantissa;
                        uint32_t weight_val = ((uint32_t)bf16_val) << 16;
                        float weight_float = *((float*)&weight_val);
                        
                        // Accumulate
                        now += input_float * weight_float;
                    }
                    
                    float curScale = *(float*)(fp8B + blockM);  // scale在128个FP8之后
                    __m256 vScale = _mm256_set1_ps(curScale);
                    last_sum = _mm256_fmadd_ps(v_sum, vScale, last_sum);
                }
                
                // Horizontal sum of last_sum
                __m128 sum_low = _mm256_extractf128_ps(last_sum, 0);
                __m128 sum_high = _mm256_extractf128_ps(last_sum, 1);
                __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                
                now += _mm_cvtss_f32(sum_128) * magicScale;
                floatC[j] = now;
            }
        }
        return true;
#else
        return false;
#endif
    }

#ifdef __AVX2__
    void print_m256i_epi16_v2(const char* name, __m256i vec) {
        int16_t values[16] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)values, vec);
        
        printf("%s: ", name);
        for (int i = 0; i < 16; i++) {
            printf("%d ", values[i]);
        }
        printf("\n");
    }
    void print_m256(const char* name, __m256 vec) {
        alignas(32) float result[8];
        _mm256_store_ps(result, vec);
        
        printf("%s: [", name);
        for (int i = 0; i < 8; i++) {
            printf("%.6f", result[i]);
            if (i < 7) printf(", ");
        }
        printf("]\n");
    }
    void print_m128i(const char* name, __m128i vec) {
        alignas(32) uint16_t result[8];
        _mm_storeu_si128((__m128i*)result, vec);
        
        printf("%s: [", name);
        for (int i = 0; i < 8; i++) {
            printf("%d", (int)result[i]);
            if (i < 7) printf(", ");
        }
        printf("]\n");
    }

    __attribute__((always_inline)) inline __m128i fp32_to_bf16_vec(__m256 float_vals) {
        __m256i shifted = _mm256_srli_epi32(_mm256_castps_si256(float_vals), 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        return _mm_packus_epi32(lo, hi);
    }
#endif

    bool AWQ4BIT128_TO_BFloat16_AVX2_Kernel(uint8_t *awqData, uint16_t *bf16Data, int m, int st, int end, int ldb) {
#ifdef __AVX2__
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
        
        // 用于提取低4位和高4位的掩码
        const __m256i low_mask = _mm256_set1_epi8(0x0F);
        
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
                
                // 准备zero和scale的向量
                __m256 scale_vec = _mm256_set1_ps(scale);
                __m256 zero_vec = _mm256_set1_ps((float)zero);
                
                // 处理当前block中的128个元素，每次处理32个（16字节包含32个int4）
                for (int i = 0; i < 64; i += 16) {  // 64字节，每次处理16字节
                    // 加载16字节（包含32个int4值）
                    __m128i packed_128 = _mm_loadu_si128((__m128i*)(packedWeights + i));
                    
                    // 扩展到256位以便处理
                    __m256i packed = _mm256_cvtepu8_epi16(packed_128);
                    
                    // 提取低4位（偶数索引的int4值）
                    __m256i low_nibbles = _mm256_and_si256(packed, _mm256_set1_epi16(0x0F));
                    // 提取高4位（奇数索引的int4值）
                    __m256i high_nibbles = _mm256_srli_epi16(packed, 4);
                    
                    // 将16位整数转换为32位整数，准备转换为float
                    // 处理低4位的前8个
                    __m256i low_lo_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_nibbles));
                    // 处理低4位的后8个
                    __m256i low_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_nibbles, 1));
                    // 处理高4位的前8个
                    __m256i high_lo_32 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_nibbles));
                    // 处理高4位的后8个
                    __m256i high_hi_32 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_nibbles, 1));
                    
                    // 转换为float
                    __m256 low_lo_f = _mm256_cvtepi32_ps(low_lo_32);
                    __m256 low_hi_f = _mm256_cvtepi32_ps(low_hi_32);
                    __m256 high_lo_f = _mm256_cvtepi32_ps(high_lo_32);
                    __m256 high_hi_f = _mm256_cvtepi32_ps(high_hi_32);
                    
                    // 反量化：(w - zero) * scale
                    low_lo_f = _mm256_mul_ps(_mm256_sub_ps(low_lo_f, zero_vec), scale_vec);
                    low_hi_f = _mm256_mul_ps(_mm256_sub_ps(low_hi_f, zero_vec), scale_vec);
                    high_lo_f = _mm256_mul_ps(_mm256_sub_ps(high_lo_f, zero_vec), scale_vec);
                    high_hi_f = _mm256_mul_ps(_mm256_sub_ps(high_hi_f, zero_vec), scale_vec);
                    
                    // 转换为bf16
                    __m128i bf16_low_lo = fp32_to_bf16_vec(low_lo_f);
                    __m128i bf16_low_hi = fp32_to_bf16_vec(low_hi_f);
                    __m128i bf16_high_lo = fp32_to_bf16_vec(high_lo_f);
                    __m128i bf16_high_hi = fp32_to_bf16_vec(high_hi_f);
                    
                    // 计算输出位置
                    int out_idx = block_idx * block_size + (i * 2);
                    
                    // 交错存储，恢复原始顺序
                    // 需要将低4位和高4位的结果交错存储
                    __m128i tmp0, tmp1, tmp2, tmp3;
                    
                    // 交错低4位和高4位的前16个元素
                    tmp0 = _mm_unpacklo_epi16(bf16_low_lo, bf16_high_lo);
                    tmp1 = _mm_unpackhi_epi16(bf16_low_lo, bf16_high_lo);
                    
                    // 交错低4位和高4位的后16个元素
                    tmp2 = _mm_unpacklo_epi16(bf16_low_hi, bf16_high_hi);
                    tmp3 = _mm_unpackhi_epi16(bf16_low_hi, bf16_high_hi);
                    
                    // 存储32个bf16值
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx), tmp0);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 8), tmp1);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 16), tmp2);
                    _mm_storeu_si128((__m128i*)(bf16B_row + out_idx + 24), tmp3);
                }
            }
        }
        
        return true;
#else
        return false;
#endif
    }
    
    bool LinearBFloat16_AWQ4BIT128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#ifdef __AVX2__
        size_t perRow = GetDataBytes(DataType::AWQ_4BIT_128, 1, m);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
            int j = st;
            // 用于提取低4位和高4位的掩码
            const __m256i low_mask = _mm256_set1_epi8(0x0F);
            
            for (; j < end; j++) {
                float now = 0.0f;
                __m256 last_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
                // 获取当前行的起始位置
                uint8_t *rowData = (uint8_t*)weightData + j * perRow;
                
                // 计算需要多少个block（每个block有128个FP8 + 1个float scale）
                const int block_size = 128;
                const int blocks_per_row = m / block_size;
                const int block_bytes = 64 + 1 + 4;
                
                for (int blockIdx = 0; blockIdx < blocks_per_row; blockIdx++) {
                    // 计算当前block的起始位置
                    uint8_t *block_start = rowData + blockIdx * block_bytes;
                    
                    // 解析当前block的组成部分
                    uint8_t *packedWeights = block_start;                    // 64字节，存储128个uint4
                    uint8_t zero = *(block_start + 64);                      // 1字节zero
                    float scale = *(float*)(block_start + 64 + 1);          // 4字节scale
                    
                    // 准备zero和scale的向量
                    __m256 scale_vec = _mm256_set1_ps(scale);
                    __m256 zero_vec = _mm256_set1_ps((float)zero);
                    
                    // 计算当前block处理的元素范围
                    int blockStart = blockIdx * block_size;
                    int blockEnd = std::min(blockStart + block_size, m);
                    
                    __m256 v_sum = _mm256_setzero_ps(); // Accumulator for 8 parallel sums
                    
                    // 处理当前block内的数据
                    int l = blockStart;
                    for (; l + 15 < blockEnd; l += 16) {
                        // 1. Load 16 BF16 inputs and convert to float
                        __m256i v_input_bf16 = _mm256_loadu_si256((__m256i const*)(bf16A + l));
                        
                        // Convert BF16 to float32 (shift left by 16 bits)
                        __m128i v_input_low = _mm256_extracti128_si256(v_input_bf16, 0);
                        __m128i v_input_high = _mm256_extracti128_si256(v_input_bf16, 1);
                        
                        // Process low 8 BF16 values
                        __m256i v_input_low_32 = _mm256_cvtepu16_epi32(v_input_low);
                        __m256i v_input_low_shifted = _mm256_slli_epi32(v_input_low_32, 16);
                        __m256 v_input_float_low = _mm256_castsi256_ps(v_input_low_shifted);
                        
                        // Process high 8 BF16 values
                        __m256i v_input_high_32 = _mm256_cvtepu16_epi32(v_input_high);
                        __m256i v_input_high_shifted = _mm256_slli_epi32(v_input_high_32, 16);
                        __m256 v_input_float_high = _mm256_castsi256_ps(v_input_high_shifted);
                        
                        // 2. Load and dequantize AWQ4BIT weights
                        // 加载8字节（包含16个int4值）
                        __m128i packed_8bytes = _mm_loadl_epi64((__m128i*)(packedWeights + (l - blockStart) / 2));
                        // 扩展到128位
                        __m128i packed_128 = _mm_cvtepu8_epi16(packed_8bytes);
                        // 提取低4位（偶数索引的int4值）
                        __m128i low_nibbles = _mm_and_si128(packed_128, _mm_set1_epi16(0x0F));
                        // 提取高4位（奇数索引的int4值）  
                        __m128i high_nibbles = _mm_srli_epi16(packed_128, 4);
                        // 转换为32位整数
                        __m256i weight_32_low = _mm256_cvtepu16_epi32(low_nibbles);
                        __m256i weight_32_high = _mm256_cvtepu16_epi32(high_nibbles);
                        // 转换为float
                        __m256 weight_float_tmp_low = _mm256_cvtepi32_ps(weight_32_low);
                        __m256 weight_float_tmp_high = _mm256_cvtepi32_ps(weight_32_high);
                        // 反量化：(w - zero) * scale, scale放到后面乘
                        weight_float_tmp_low = _mm256_sub_ps(weight_float_tmp_low, zero_vec);
                        weight_float_tmp_high = _mm256_sub_ps(weight_float_tmp_high, zero_vec);
                        // 交错低4位和高4位，恢复原始顺序
                        // 需要将weight_float_tmp_low和weight_float_tmp_high交错组合
                        __m256 v_weight_float_low, v_weight_float_high;
                        // 使用unpack操作交错组合
                        __m256 tmp0 = _mm256_unpacklo_ps(weight_float_tmp_low, weight_float_tmp_high);
                        __m256 tmp1 = _mm256_unpackhi_ps(weight_float_tmp_low, weight_float_tmp_high);
                        // 进一步排列以得到正确的顺序
                        v_weight_float_low = _mm256_permute2f128_ps(tmp0, tmp1, 0x20);
                        v_weight_float_high = _mm256_permute2f128_ps(tmp0, tmp1, 0x31);
                        
                        // 3. Compute dot product: multiply and accumulate
                        __m256 v_mul_low = _mm256_mul_ps(v_input_float_low, v_weight_float_low);
                        __m256 v_mul_high = _mm256_mul_ps(v_input_float_high, v_weight_float_high);
                        
                        v_sum = _mm256_add_ps(v_sum, v_mul_low);
                        v_sum = _mm256_add_ps(v_sum, v_mul_high);
                    }
                    
                    last_sum = _mm256_fmadd_ps(v_sum, scale_vec, last_sum);
                }
                
                // Horizontal sum of last_sum
                __m128 sum_low = _mm256_extractf128_ps(last_sum, 0);
                __m128 sum_high = _mm256_extractf128_ps(last_sum, 1);
                __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                
                now += _mm_cvtss_f32(sum_128);
                floatC[j] = now;
            }
        }
        return true;
#else
        return false;
#endif
    }
}