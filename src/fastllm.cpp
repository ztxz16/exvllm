#include "fastllm.h"

namespace fastllm {
    static int threads = 60;
    static AliveThreadPool *fastllmAliveThreadPool = nullptr;
    static NumaConfig *fastllmNumaConfig = nullptr;
    static CPUInstructInfo fastllmCPUInfostructInfo;
    static MachineNumaInfo machineNumaInfo;
    
    MachineNumaInfo::MachineNumaInfo() {
        // 检查 NUMA 是否可用
        if (numa_available() < 0) {
            // NUMA 不可用，使用默认值
            numaCnt = 1;
            threads = 1;
            cpuIds.resize(1);
            cpuIds[0].push_back(0);
            return;
        }
        // 获取系统的 NUMA 节点数量
        int systemNumaNodes = numa_num_configured_nodes();
        if (systemNumaNodes <= 0) {
            systemNumaNodes = 1;
        }
        // 获取每个 NUMA 节点的物理核心数
        // 这里我们取第一个 NUMA 节点的核心数作为参考
        int coresPerNuma = 0;
        for (int node = 0; node < systemNumaNodes; ++node) {
            struct bitmask* cpumask = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, cpumask) == 0) {
                // 统计该节点上的 CPU 数量
                int nodeCoreCnt = 0;
                for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
                    if (numa_bitmask_isbitset(cpumask, cpu)) {
                        nodeCoreCnt++;
                    }
                }
                if (nodeCoreCnt > coresPerNuma) {
                    coresPerNuma = nodeCoreCnt;
                }
            }
            numa_free_cpumask(cpumask);
        }
        // 如果获取失败，使用默认值
        if (coresPerNuma <= 0) {
            coresPerNuma = 1;
        }
        // 读取环境变量 FT_NUMAS
        const char* ftNumas = std::getenv("FT_NUMAS");
        int numaStart = 0;  // 默认从第0个NUMA节点开始
        const char* ftNumasStart = std::getenv("FT_NUMAS_START");
        if (ftNumasStart != nullptr) {
            numaStart = std::atoi(ftNumasStart);
            if (numaStart < 0) {
                numaStart = 0;
            }
        }
        
        if (ftNumas != nullptr) {
            int envNumaCnt = std::atoi(ftNumas);
            if (envNumaCnt > 0 && (envNumaCnt + numaStart) <= systemNumaNodes) {
                numaCnt = envNumaCnt;
            } else {
                numaCnt = systemNumaNodes - numaStart;
                if (numaCnt <= 0) {
                    numaCnt = 1;
                    numaStart = 0;
                }
            }
        } else {
            numaCnt = systemNumaNodes - numaStart;
            if (numaCnt <= 0) {
                numaCnt = 1;
                numaStart = 0;
            }
        }
        
        // 读取环境变量 FT_THREADS
        const char* ftThreads = std::getenv("FT_THREADS");
        if (ftThreads != nullptr) {
            int envThreads = std::atoi(ftThreads);
            if (envThreads > 0) {
                threads = envThreads;
            } else {
                threads = numaCnt * std::max(1, coresPerNuma / 2 - 2);
            }
        } else {
            threads = numaCnt * std::max(1, coresPerNuma / 2 - 2);
        }
        
        // 读取环境变量 FT_THREADS_START
        int threadStart = 0;  // 默认从第0个核心开始
        const char* ftThreadsStart = std::getenv("FT_THREADS_START");
        if (ftThreadsStart != nullptr) {
            threadStart = std::atoi(ftThreadsStart);
            if (threadStart < 0) {
                threadStart = 0;
            }
        }
        
        // 初始化 cpuIds
        cpuIds.resize(numaCnt);
        
        // 收集所有 NUMA 节点的 CPU ID
        std::vector<std::vector<int>> allNumaCpus(systemNumaNodes);
        for (int node = 0; node < systemNumaNodes; ++node) {
            struct bitmask* cpumask = numa_allocate_cpumask();
            if (numa_node_to_cpus(node, cpumask) == 0) {
                for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
                    if (numa_bitmask_isbitset(cpumask, cpu)) {
                        allNumaCpus[node].push_back(cpu);
                    }
                }
            }
            numa_free_cpumask(cpumask);
        }
        
        // 应用线程起始偏移
        if (threadStart > 0) {
            // 计算需要跳过的CPU总数
            int totalCpusToSkip = threadStart;
            
            // 遍历所有NUMA节点，跳过指定数量的CPU
            for (int node = 0; node < systemNumaNodes && totalCpusToSkip > 0; ++node) {
                if (totalCpusToSkip >= allNumaCpus[node].size()) {
                    // 跳过整个NUMA节点
                    totalCpusToSkip -= allNumaCpus[node].size();
                    allNumaCpus[node].clear();
                } else {
                    // 跳过部分CPU
                    allNumaCpus[node].erase(
                        allNumaCpus[node].begin(), 
                        allNumaCpus[node].begin() + totalCpusToSkip
                    );
                    totalCpusToSkip = 0;
                }
            }
        }
        
        // 将 CPU 分配到指定数量的 NUMA 节点，考虑NUMA起始偏移
        if (numaCnt <= systemNumaNodes) {
            // 使用从 numaStart 开始的 numaCnt 个节点
            for (int i = 0; i < numaCnt; ++i) {
                int nodeIdx = numaStart + i;
                if (nodeIdx < systemNumaNodes) {
                    cpuIds[i] = allNumaCpus[nodeIdx];
                }
            }
        } else {
            // numaCnt > systemNumaNodes 的情况（理论上不应该发生）
            // 循环分配，考虑NUMA起始偏移
            for (int i = 0; i < numaCnt; ++i) {
                int nodeIdx = (numaStart + i) % systemNumaNodes;
                cpuIds[i] = allNumaCpus[nodeIdx];
            }
        }
    }

    void SetAliveThreads(int t) {
        threads = t;
        if (fastllmAliveThreadPool != nullptr) {
            fastllmAliveThreadPool->Shutdown();
            delete fastllmAliveThreadPool;
        }
        fastllmAliveThreadPool = new AliveThreadPool(t);
    }

    CPUInstructInfo *GetCPUInstructInfo() {
        return &fastllmCPUInfostructInfo;
    }

    AliveThreadPool *GetAlivePool() {
        if (fastllmAliveThreadPool == nullptr) {
            SetAliveThreads(machineNumaInfo.threads);
        }
        return fastllmAliveThreadPool;
    }

    NumaConfig *GetNumaConfig() {
        auto *pool = GetAlivePool();
        if (fastllmNumaConfig == nullptr) {
            fastllmNumaConfig = new NumaConfig(pool->threads.size(), pool, &machineNumaInfo);
        }
        
        return fastllmNumaConfig;
    }

    std::string GetDataTypeName(DataType type) {
        if (dataTypeNames.find(type) != dataTypeNames.end()) {
            return dataTypeNames[type][0];
        } else {
            return "Type " + std::to_string((int)type);
        }
    }

    uint8_t fp32tofp8e4m3(float val) {
        union {
            float f;
            uint32_t u;
        } fp32;
        
        fp32.f = val;
        uint32_t fp32_bits = fp32.u;
        
        // 提取 float32 的符号、指数和尾数
        uint32_t sign = (fp32_bits >> 31) & 0x1;
        int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127;  // 无偏指数
        uint32_t mantissa = fp32_bits & 0x7FFFFF;
        
        // 处理特殊情况
        if ((fp32_bits & 0x7FFFFFFF) == 0) {
            // ±0 -> 0
            return sign << 7;
        }
        
        // 处理 NaN 和 Inf
        if ((fp32_bits & 0x7F800000) == 0x7F800000) {
            if (mantissa != 0) {
                // NaN -> 0x7F (E4M3 的 NaN 表示)
                return 0x7F;
            } else {
                // Inf -> 最大值或最小值
                return (sign << 7) | 0x7E;  // ±240
            }
        }
        
        // 调整指数到 E4M3 范围
        int32_t fp8_exp = exp + 7;  // 加上 E4M3 的偏移量
        
        // 处理下溢出 (指数太小)
        if (fp8_exp <= 0) {
            // 非规格化数或下溢到0
            if (fp8_exp < -3) {
                // 太小了，直接返回0
                return sign << 7;
            }
            // 尝试表示为非规格化数
            uint32_t shift = 1 - fp8_exp;
            mantissa = (0x800000 | mantissa) >> shift;
            fp8_exp = 0;
            mantissa = mantissa >> 20;  // 只取高3位
        } else if (fp8_exp >= 15) {
            // 上溢出，钳位到最大值
            return (sign << 7) | 0x7E;  // 最大正常值
        } else {
            // 正常范围
            mantissa = mantissa >> 20;  // 取尾数的高3位
        }
        
        // 组装 FP8 E4M3
        uint8_t result = (sign << 7) | (fp8_exp << 3) | (mantissa & 0x7);
        
        // 舍入处理（可选，这里使用最近舍入）
        if (fp8_exp > 0 && fp8_exp < 15) {
            uint32_t round_bit = (fp32_bits >> 19) & 0x1;
            uint32_t sticky_bits = fp32_bits & 0x7FFFF;
            
            if (round_bit && (sticky_bits != 0 || (mantissa & 0x1))) {
                result++;
                // 检查是否产生进位溢出
                if ((result & 0x7F) == 0x7F) {
                    result = (sign << 7) | 0x7E;  // 溢出到最大值
                }
            }
        }
        
        return result;
    }

    BF16ToFP32Manager bf16tofp32;
    FP8E4M3ToFP32Manager fp8e4m3tofp32;

    size_t GetDataBytes(DataType type, size_t rows, size_t columns) {
        if (type == DataType::FLOAT32) {
            return rows * columns * sizeof(float);
        } else if (type == DataType::BFLOAT16 || type == DataType::FLOAT16) {
            return rows * columns * sizeof(uint16_t);
        } else if (type == DataType::FP8_E4M3_BLOCK_128) {
            // columns * [fp8] + ((columns - 1) / 128 + 1) * [float]
            return rows * (columns + ((columns - 1) / 128 + 1) * sizeof(float));
        } else if (type == DataType::FP8_E4M3) {
            return rows * columns * sizeof(uint8_t);
        } else if (type == DataType::AWQ_4BIT_128) {
            int groups = (columns - 1) / 128 + 1;
            size_t colBytes = columns / 2 + groups + groups * sizeof(float);
            return rows * colBytes;
        } else {
            ErrorInFastLLM("GetDataBytes failed. " + std::to_string((int)type) + "\n");
            return 0;
        }
    }

    void ConvertToFloat32(const void *srcData, DataType srcDataType, float *floatData, size_t rows, size_t columns) {
        if (srcDataType == DataType::FLOAT32) {
            memcpy(floatData, srcData, rows * columns * sizeof(float));
        } else if (srcDataType == DataType::BFLOAT16) {
            uint16_t *bf16Src = (uint16_t*)srcData;
            for (int i = 0; i < rows * columns; i++) {
                floatData[i] = bf16tofp32.dict[bf16Src[i]];
            }
        } else if (srcDataType == DataType::FP8_E4M3_BLOCK_128) {
            int block_size = 128;
            int num_blocks_per_row = (columns + block_size - 1) / block_size;
            
            for (int i = 0; i < rows; i++) {
                const uint8_t *srcRow = (const uint8_t*)srcData;
                float *floatRow = floatData + i * columns;
                
                // 计算当前行在srcData中的偏移
                size_t row_offset = 0;
                for (int r = 0; r < i; r++) {
                    int blocks = (columns + block_size - 1) / block_size;
                    row_offset += blocks * (block_size * sizeof(uint8_t) + sizeof(float));
                }
                srcRow += row_offset;
                
                // 处理每个block
                for (int b = 0; b < num_blocks_per_row; b++) {
                    const uint8_t *fp8Block = srcRow + b * (block_size * sizeof(uint8_t) + sizeof(float));
                    const float *scale = (const float*)(fp8Block + block_size);
                    
                    int start = b * block_size;
                    int end = std::min(start + block_size, (int)columns);
                    
                    for (int j = start; j < end; j++) {
                        floatRow[j] = fp8e4m3tofp32.dict[fp8Block[j - start]] * (*scale);
                    }
                }
            }
        } else if (srcDataType == DataType::AWQ_4BIT_128) {
            int block_size = 128;
            int blocks_per_row = columns / block_size;
            
            for (size_t i = 0; i < rows; i++) {
                uint8_t *awqRow = (uint8_t*)srcData + i * (blocks_per_row * (64 + 1 + 4));
                float *floatRow = floatData + i * columns;
                
                // Process each block in the row
                for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
                    // Calculate current block's starting position
                    uint8_t *block_start = awqRow + block_idx * (64 + 1 + 4);
                    
                    // Block components
                    uint8_t *packedWeights = block_start;                    // 64 bytes for 128 uint4 values
                    uint8_t zero = *(block_start + 64);                      // 1 byte zero
                    float scale = *(float*)(block_start + 64 + 1);          // 4 bytes scale
                    
                    // Process 128 elements in current block
                    for (int elem_in_block = 0; elem_in_block < block_size; elem_in_block++) {
                        int col_idx = block_idx * block_size + elem_in_block;
                        
                        // Extract uint4 weight from packed format
                        int weight_idx = elem_in_block / 2;
                        int weight_shift = (elem_in_block & 1) * 4;
                        int w = (packedWeights[weight_idx] >> weight_shift) & 0xF;
                        
                        // Dequantize: (w - zero) * scale
                        floatRow[col_idx] = (w - zero) * scale;
                    }
                }
            }
        } else {
            ErrorInFastLLM("ConvertToFloat32 failed.\n");
        }
    }
    void ConvertFromFloat32(void *dstData, DataType dstDataType, const float *floatData, size_t rows, size_t columns) {
        if (dstDataType == DataType::FLOAT32) {
            memcpy(dstData, floatData, rows * columns * sizeof(float));
        } else if (dstDataType == DataType::BFLOAT16) {
            uint16_t *bf16Dst = (uint16_t*)dstData;
            for (int i = 0; i < rows * columns; i++) {
                uint32_t val;
                memcpy(&val, &floatData[i], sizeof(val));
                bf16Dst[i] = (uint16_t)(val >> 16);
            }
        } else if (dstDataType == DataType::FP8_E4M3_BLOCK_128) {
            int block_size = 128;
            int num_blocks_per_row = (columns + block_size - 1) / block_size;
            
            for (int i = 0; i < rows; i++) {
                uint8_t *dstRow = (uint8_t*)dstData;
                const float *floatRow = floatData + i * columns;
                
                // 计算当前行在dstData中的偏移
                size_t row_offset = 0;
                for (int r = 0; r < i; r++) {
                    int blocks = (columns + block_size - 1) / block_size;
                    row_offset += blocks * (block_size * sizeof(uint8_t) + sizeof(float));
                }
                dstRow += row_offset;
                
                // 处理每个block
                for (int b = 0; b < num_blocks_per_row; b++) {
                    uint8_t *fp8Block = dstRow + b * (block_size * sizeof(uint8_t) + sizeof(float));
                    float *scale = (float*)(fp8Block + block_size);
                    
                    int start = b * block_size;
                    int end = std::min(start + block_size, (int)columns);
                    
                    // 首先计算该block的scale（取该block内的最大绝对值）
                    float max_abs = 0.0f;
                    for (int j = start; j < end; j++) {
                        max_abs = std::max(max_abs, std::abs(floatRow[j]));
                    }
                    
                    // FP8 E4M3的最大值约为448，设置scale
                    *scale = max_abs / 448.0f;
                    if (*scale < 1e-6f) *scale = 1e-6f; // 避免除零
                    
                    // 量化为fp8
                    for (int j = start; j < end; j++) {
                        float scaled_val = floatRow[j] / (*scale);
                        
                        // 裁剪到FP8 E4M3范围并转换
                        scaled_val = std::max(-448.0f, std::min(448.0f, scaled_val));
                        fp8Block[j - start] = fp32tofp8e4m3(scaled_val);
                    }
                }
            }
        } else if (dstDataType == DataType::AWQ_4BIT_128) {
            int block_size = 128;
            int blocks_per_row = columns / block_size;
            
            for (size_t i = 0; i < rows; i++) {
                uint8_t *awqRow = (uint8_t*)dstData + i * (blocks_per_row * (64 + 1 + 4));
                const float *floatRow = floatData + i * columns;
                
                // Process each block in the row
                for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
                    // Calculate current block's starting position
                    uint8_t *block_start = awqRow + block_idx * (64 + 1 + 4);
                    
                    // Find min and max values in this block for quantization
                    float min_val = floatRow[block_idx * block_size];
                    float max_val = floatRow[block_idx * block_size];
                    for (int elem = 1; elem < block_size; elem++) {
                        float val = floatRow[block_idx * block_size + elem];
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                    }
                    
                    // Calculate scale and zero point for 4-bit quantization
                    float scale = (max_val - min_val) / 15.0f;  // 4-bit has range 0-15
                    uint8_t zero = 0;
                    if (scale > 0) {
                        zero = std::round(-min_val / scale);
                        zero = std::min((uint8_t)15, std::max((uint8_t)0, zero));
                    }
                    
                    // Store quantized weights
                    uint8_t *packedWeights = block_start;
                    memset(packedWeights, 0, 64);  // Clear packed weights
                    
                    for (int elem_in_block = 0; elem_in_block < block_size; elem_in_block++) {
                        float val = floatRow[block_idx * block_size + elem_in_block];
                        
                        // Quantize to 4-bit
                        int quantized;
                        if (scale > 0) {
                            quantized = std::round(val / scale + zero);
                            quantized = std::min(15, std::max(0, quantized));
                        } else {
                            quantized = zero;
                        }
                        
                        // Pack into uint8 array (2 uint4 per byte)
                        int weight_idx = elem_in_block / 2;
                        int weight_shift = (elem_in_block & 1) * 4;
                        packedWeights[weight_idx] |= (quantized << weight_shift);
                    }
                    
                    // Store zero and scale
                    *(block_start + 64) = zero;
                    *(float*)(block_start + 64 + 1) = scale;
                }
            }
        } else {
            ErrorInFastLLM("ConvertFromFloat32 failed.\n");
        }
    }

    // 新增一个专门的Op来处理数据类型转换
    struct MultiThreadConvertFromFloat32Op : MultiThreadBaseOp {
        void *dstData;
        DataType dstDataType;
        const float *floatData;
        size_t columns;
        size_t startRow, endRow;  // 处理的行范围 [startRow, endRow)
        MultiThreadConvertFromFloat32Op(void *dstData, DataType dstDataType, 
                                    const float *floatData, size_t columns,
                                    size_t startRow, size_t endRow) :
            dstData(dstData), dstDataType(dstDataType), 
            floatData(floatData), columns(columns),
            startRow(startRow), endRow(endRow) {}
        void Run() {
            // 计算每行的字节大小
            size_t elementSize = 0;
            switch(dstDataType) {
                case DataType::FLOAT16: elementSize = 2; break;
                case DataType::BFLOAT16: elementSize = 2; break;
                case DataType::INT8: elementSize = 1; break;
                case DataType::FLOAT32: elementSize = 4; break;
                // 根据实际的DataType枚举添加更多类型
                default: elementSize = 4;
            }
            
            // 调用原始函数处理指定行范围
            void *dstStart = (char*)dstData + startRow * columns * elementSize;
            const float *srcStart = floatData + startRow * columns;
            size_t rowsToProcess = endRow - startRow;
            
            ConvertFromFloat32(dstStart, dstDataType, srcStart, rowsToProcess, columns);
        }
    };

    // 对应的多线程运行函数
    void RunMultiThreadConvertFromFloat32(void *dstData, DataType dstDataType, 
                                                const float *floatData, size_t rows, 
                                                size_t columns, AliveThreadPool *pool) {
        // 如果数据量较小，直接单线程处理
        if (rows * columns < 10000) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        int threadNum = pool->threads.size();
        threadNum = std::min(threadNum, (int)rows);  // 线程数不超过行数
        
        // 如果行数太少，减少线程数
        if (rows < threadNum) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        size_t rowsPerThread = rows / threadNum;
        size_t curRow = 0;
        
        std::vector<MultiThreadConvertFromFloat32Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            size_t endRow = (i == threadNum - 1) ? rows : curRow + rowsPerThread;
            ops.push_back(new MultiThreadConvertFromFloat32Op(
                dstData, dstDataType, floatData, columns, curRow, endRow));
            curRow = endRow;
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void AwqToFastllmAwq4Bit128(int experts, int n, int m, unsigned int *qweight, unsigned int *qzeros, float *scales, std::vector <uint8_t> &awq4bit128Value) {
        static const int awq_shift[8] = {0,16,4,20,8,24,12,28};
        
        // 每个block的大小：64字节的权重(128个uint4) + 1字节的zero + 4字节的scale
        size_t blockSize = 64 + 1 + 4;  // 69 bytes per block
        size_t blocksPerRow = n / 128;
        size_t rowSize = blocksPerRow * blockSize;
        
        awq4bit128Value.resize((size_t)experts * m * 8 * rowSize);
        
        for (size_t e = 0; e < experts; e++) {
            for (size_t gy = 0; gy < m; gy++) {
                for (size_t y_sub = 0; y_sub < 8; y_sub++) {
                    size_t y = gy * 8 + y_sub;
                    uint8_t *row_ptr = awq4bit128Value.data() + e * (m * 8) * rowSize + y * rowSize;
                    
                    // 按块处理
                    for (int gx = 0; gx < n/128; gx++) {
                        // 当前块的起始位置
                        uint8_t *block_ptr = row_ptr + gx * blockSize;
                        
                        // 1. 写入128个uint4权重（64字节）
                        for (int x = 0; x < 128; x += 2) {
                            size_t global_x = gx * 128 + x;
                            uint8_t packed = 0;
                            int w1 = (qweight[global_x * m + gy] >> awq_shift[y_sub]) & 15;
                            int w2 = (global_x + 1 < n) ? ((qweight[(global_x + 1) * m + gy] >> awq_shift[y_sub]) & 15) : 0;
                            packed = (w1 & 0xF) | ((w2 & 0xF) << 4);
                            block_ptr[x / 2] = packed;
                        }
                        
                        // 2. 写入1个uint8 zero（1字节）
                        int z = (qzeros[gx * m + gy] >> awq_shift[y_sub]) & 15;
                        block_ptr[64] = (uint8_t)z;
                        
                        // 3. 写入1个float scale（4字节）
                        float s = scales[gx * m * 8 + y];
                        *((float*)(block_ptr + 65)) = s;
                    }
                }
            }
            
            // 移动到下一个专家的数据
            qweight += n * m;
            qzeros += (n / 128) * m;
            scales += (n / 128) * (m * 8);
        }
    }


    void Fp8ToFastllmFP8_E4M3_BLOCK128(int experts, int k, int m, uint8_t *fp8, float *scales, int blockK, int blockM, std::vector <uint8_t> &fp8Packed) {
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;
        
        // 计算每行需要的总字节数
        // 每128个fp8需要1个float的scale，所以需要 (m + 127) / 128 个scale
        int numScalesPerRow = (m + 127) / 128;
        int rowSize = m + numScalesPerRow * sizeof(float);
        fp8Packed.resize((size_t)experts * k * rowSize);
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < k; j++) {
                size_t rowIdx = i * k + j;
                size_t packedOffset = rowIdx * rowSize;
                
                // 按照每128个fp8后接一个scale的格式打包
                size_t currentPos = packedOffset;
                
                for (int blockIdx = 0; blockIdx < numScalesPerRow; blockIdx++) {
                    size_t blockStart = blockIdx * 128;
                    size_t blockEnd = std::min(blockStart + 128, (size_t)m);
                    size_t blockSize = blockEnd - blockStart;
                    
                    // 复制当前block的fp8数据
                    for (size_t l = blockStart; l < blockEnd; l++) {
                        size_t srcIdx = i * k * m + j * m + l;
                        fp8Packed[currentPos++] = fp8[srcIdx];
                    }
                    
                    // 在这个block后面添加对应的scale
                    // scale的索引计算：需要根据当前block在整个矩阵中的位置
                    size_t scaleRow = j / blockK;  // 当前行属于哪个scale行块
                    size_t scaleCol = blockStart / blockM;  // 当前block属于哪个scale列块
                    size_t scaleIdx = i * ks * ms + scaleRow * ms + scaleCol;
                    
                    float* scalePtr = (float*)(&fp8Packed[currentPos]);
                    *scalePtr = scales[scaleIdx];
                    currentPos += sizeof(float);
                }
            }
        }
    }

    struct MultiThreadFp8ToFastllmFP8_E4M3_BLOCK128Op : MultiThreadBaseOp {
        int experts, k, m;
        uint8_t *fp8;
        float *scales;
        int blockK, blockM;
        std::vector <uint8_t> *fp8Packed;

        MultiThreadFp8ToFastllmFP8_E4M3_BLOCK128Op(
            int experts,
            int k,
            int m,
            uint8_t *fp8,
            float *scales,
            int blockK,
            int blockM,
            std::vector<uint8_t> *fp8Packed
        ) : experts(experts),
            k(k),
            m(m),
            fp8(fp8),
            scales(scales),
            blockK(blockK),
            blockM(blockM),
            fp8Packed(fp8Packed) {}

        void Run() {
            Fp8ToFastllmFP8_E4M3_BLOCK128(experts, k, m, fp8, scales, blockK, blockM, *fp8Packed);
        }
    };

    void FastllmLinearWeight::Init (int batch, int k, int m, void *data, DataType dataType) {
        this->batch = batch;
        this->k = k;
        this->m = m;
        this->dataType = dataType;

        // printf("into create linear [%d %d %d] (%d) \n", batch, k, m, dataType);
        auto *numaConfig = GetNumaConfig();
        size_t sizePerRow = GetDataBytes(dataType, 1, m);
        if (k % numaConfig->numaCnt != 0) {
            ErrorInFastLLM("Linear weight's size %% numaCnt != 0.");
        }

        int kPerNuma = k / numaConfig->numaCnt;
        datas.resize(numaConfig->numaCnt);
        for (int i = 0; i < numaConfig->numaCnt; i++) {
            // datas[i] = (uint8_t*)allocate_aligned_numa(this->batch * kPerNuma * sizePerRow, i);
            datas[i].resize(this->batch);
            for (int j = 0; j < batch; j++) {
                datas[i][j] = (uint8_t*)allocate_aligned_numa(kPerNuma * sizePerRow, i);
            }
        }
        
        // 创建所有任务
        std::vector<std::vector <fastllm::MultiThreadMemcpyOp*> > ops;
        ops.resize(numaConfig->numaCnt);

        for (int i = 0; i < numaConfig->numaCnt; i++) {
            for (int e = 0; e < this->batch; e++) {
                ops[i].push_back(new MultiThreadMemcpyOp(
                    (uint8_t*)datas[i][e], 
                    (uint8_t*)data + ((size_t)e * k + (size_t)i * kPerNuma) * sizePerRow, 
                    kPerNuma * sizePerRow
                ));
            }
        }

        DynamicScheduleTasks(ops);
    }

    void FastllmLinearWeight::Init(int batch, int k, int m, std::vector <std::vector <uint8_t> > &oriDatas, DataType dataType) {
        this->batch = batch;
        this->k = k;
        this->m = m;
        this->dataType = dataType;

        // printf("into create linear [%d %d %d] (%d) \n", batch, k, m, dataType);
        auto *numaConfig = GetNumaConfig();
        size_t sizePerRow = GetDataBytes(dataType, 1, m);
        if (k % numaConfig->numaCnt != 0) {
            ErrorInFastLLM("Linear weight's size %% numaCnt != 0.");
        }

        int kPerNuma = k / numaConfig->numaCnt;
        datas.resize(numaConfig->numaCnt);
        for (int i = 0; i < numaConfig->numaCnt; i++) {
            // datas[i] = (uint8_t*)allocate_aligned_numa(this->batch * kPerNuma * sizePerRow, i);
            datas[i].resize(this->batch);
            for (int j = 0; j < batch; j++) {
                datas[i][j] = (uint8_t*)allocate_aligned_numa(kPerNuma * sizePerRow, i);
            }
        }
        
        // 创建所有任务
        std::vector<std::vector <fastllm::MultiThreadMemcpyOp*> > ops;
        ops.resize(numaConfig->numaCnt);

        for (int i = 0; i < numaConfig->numaCnt; i++) {
            for (int e = 0; e < this->batch; e++) {
                ops[i].push_back(new MultiThreadMemcpyOp(
                    (uint8_t*)datas[i][e], 
                    (uint8_t*)oriDatas[e].data() + i * kPerNuma * sizePerRow, 
                    kPerNuma * sizePerRow
                ));
            }
        }

        DynamicScheduleTasks(ops);
    }

    FastllmLinearWeight::FastllmLinearWeight (int batch, int k, int m, void *data, DataType dataType) {
        Init(batch, k, m, data, dataType);
    }

    FastllmLinearWeight::FastllmLinearWeight (int batch, int k, int m, void *data, DataType dataType, 
            void *scales, void *zeros, int blockK, int blockM) {
        if (dataType == DataType::FP8_E4M3) {
            // printf("into fp8 moe %d %d %d (%d %d)\n", batch, k, m, blockK, blockM);
            if (blockK == 128 && blockM == 128) {             
            } else {
                ErrorInFastLLM("Unsupport fp8 quant type.");
            }

            std::vector<std::vector <uint8_t> > fp8Packeds;
            fp8Packeds.resize(batch);
            int ks = (k - 1) / blockK + 1;
            int ms = (m - 1) / blockM + 1;
            
            /*for (int i = 0; i < batch; i++) {
                Fp8ToFastllmFP8_E4M3_BLOCK128(1, k, m, 
                    (uint8_t*)data + (size_t)i * k * m, 
                    (float*)scales + (size_t)i * ks * ms, 
                    blockK, blockM, fp8Packeds[i]
                );
            }*/

            // 创建所有任务
            auto *numaConfig = GetNumaConfig();
            std::vector<std::vector <fastllm::MultiThreadFp8ToFastllmFP8_E4M3_BLOCK128Op*> > ops;
            ops.resize(numaConfig->numaCnt);

            /*for (int i = 0; i < batch; i++) {
                Fp8ToFastllmFP8_E4M3_BLOCK128(1, k, m, 
                    (uint8_t*)data + (size_t)i * k * m, 
                    (float*)scales + (size_t)i * ks * ms, 
                    blockK, blockM, fp8Packeds[i]
                );
            }*/

            for (int e = 0; e < batch; e++) {
                ops[e % ops.size()].push_back(new MultiThreadFp8ToFastllmFP8_E4M3_BLOCK128Op(
                    1, k, m, 
                    (uint8_t*)data + (size_t)e * k * m, 
                    (float*)scales + (size_t)e * ks * ms, 
                    blockK, blockM, &fp8Packeds[e]
                ));

            }
            DynamicScheduleTasks(ops);

            this->dataType = DataType::FP8_E4M3_BLOCK_128;
            this->block = blockM;
            this->Init(batch, k, m, fp8Packeds, this->dataType);
        } else if (dataType == DataType::AWQ_4BIT_128) {
            // printf("into AWQ_4BIT_128 moe\n");    
            if (blockM == 128) { 
            } else {
                ErrorInFastLLM("Unsupport awq quant type.");
            }

            std::vector <uint8_t> awqPacked;
            AwqToFastllmAwq4Bit128(batch, m, k / 8, (unsigned int *)data, (unsigned int *)zeros, (float*)scales, awqPacked);
            this->Init(batch, k,m, awqPacked.data(), dataType);
        } else {
            ErrorInFastLLM("FastllmLinearWeight Unsupport data type." + GetDataTypeName(dataType));
        }
    }
}
