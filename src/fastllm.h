#ifndef FASTLLM_H
#define FASTLLM_H

#include <cstdint>
#include <cmath>
#include <deque>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <atomic>
#include <array>

#include "alivethreadpool.h"

#ifndef __aarch64__
// Intrinsics for CPUID
#if defined(_MSC_VER)
    #include <intrin.h> // For __cpuid, __cpuidex, _xgetbv
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h> // For __get_cpuid, __get_cpuid_count
    #include <x86intrin.h> // For _xgetbv (usually included by cpuid.h or available)
    // GCC/Clang might not have _xgetbv as an intrinsic like MSVC,
    // or it might be in a different header.
    // If _xgetbv is not found, you might need to implement it with inline assembly.
    #ifndef _XCR_XFEATURE_ENABLED_MASK // Often defined with _xgetbv
    #define _XCR_XFEATURE_ENABLED_MASK 0
    #if __GNUC__ < 8 and !defined(USE_ROCM)
    static uint64_t _xgetbv(uint32_t xcr_index) {
        uint32_t eax, edx;
        __asm__ __volatile__ (
            "xgetbv"
            : "=a" (eax), "=d" (edx)  // Output operands: eax, edx
            : "c" (xcr_index)         // Input operand: ecx (xcr_index)
            :                         // Clobbered registers (none explicitly clobbered by xgetbv beyond outputs)
        );
        return ((uint64_t)edx << 32) | eax;
    }
    #endif
    #endif
#else
    #warning "CPUID detection not implemented for this compiler."
#endif
#endif // ifndef __aarch64__

namespace fastllm {
    struct CPUInstructInfo {
        bool hasAVX2 = false;
        bool hasAVX512F = false;
        bool hasAVX512BF16 = false;
        bool hasAVX512VNNI = false;
        
        CPUInstructInfo() {
#ifndef __aarch64__
            #if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
            std::array<int, 4> regs; // For EAX, EBX, ECX, EDX
            
            // Step 1: Check OSXSAVE bit (CPUID EAX=1, ECX bit 27)
            // This indicates if the OS supports XGETBV to query enabled AVX features
            bool os_supports_xsave = false;
            #if defined(_MSC_VER)
            __cpuid(regs.data(), 1);
            #else // GCC/Clang
            __get_cpuid(1, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
            #endif
            if (regs[2] & (1 << 27)) { // Check ECX bit 27 (OSXSAVE)
                os_supports_xsave = true;
            }
            
            bool os_avx_enabled = false;
            bool os_avx512_enabled = false;
            if (os_supports_xsave) {
                // Step 2: Check if AVX states (and by extension AVX512 states) are enabled by OS
                // XCR0 register:
                // Bit 1 (SSE state) must be 1
                // Bit 2 (AVX state - YMM registers) must be 1
                // Bits 5,6,7 (AVX512 OPMASK, ZMM_Hi256, Hi16_ZMM states) must be 1 for AVX512
                uint64_t xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK); // _XCR_XFEATURE_ENABLED_MASK is typically 0
                
                // Check for AVX support (bits 1 and 2)
                if ((xcr0 & 0x6) == 0x6) {
                    os_avx_enabled = true;
                    
                    // Check for AVX512 support (bits 1,2,5,6,7)
                    if ((xcr0 & 0xE6) == 0xE6) {
                        os_avx512_enabled = true;
                    }
                }
            }
            
            if (os_avx_enabled) {
                // CPUID with EAX=7, ECX=0 for extended features
                #if defined(_MSC_VER)
                __cpuidex(regs.data(), 7, 0);
                #else // GCC/Clang
                __get_cpuid_count(7, 0, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
                #endif
                
                // AVX2: EAX=7, ECX=0, EBX bit 5
                hasAVX2 = (regs[1] & (1 << 5)) != 0;
                
                // Only check AVX512 features if OS supports AVX512 states
                if (os_avx512_enabled) {
                    // AVX512F: EAX=7, ECX=0, EBX bit 16
                    hasAVX512F = (regs[1] & (1 << 16)) != 0;
                    
                    // AVX512VNNI: EAX=7, ECX=0, ECX bit 11
                    hasAVX512VNNI = (regs[2] & (1 << 11)) != 0;
                    
                    // AVX512_BF16: EAX=7, ECX=1, EAX bit 5
                    // Need to make another CPUID call with ECX=1
                    #if defined(_MSC_VER)
                    __cpuidex(regs.data(), 7, 1);
                    #else // GCC/Clang
                    __get_cpuid_count(7, 1, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
                    #endif
                    hasAVX512BF16 = (regs[0] & (1 << 5)) != 0;
                    
                    // Ensure AVX512_BF16 and AVX512VNNI depend on AVX512F
                    hasAVX512BF16 = hasAVX512BF16 && hasAVX512F;
                    hasAVX512VNNI = hasAVX512VNNI && hasAVX512F;
                }
            }
            // If os_avx_enabled is false, all 'has...' flags will remain false.
            #endif // Compiler check
            // Print the results
            std::string x[2] = {"OFF", "ON"};
            printf("CPU Instruction Info: ");
            printf("[AVX2: %s] ", x[hasAVX2].c_str());
            printf("[AVX512F: %s] ", x[hasAVX512F].c_str());
            printf("[AVX512_VNNI: %s] ", x[hasAVX512VNNI].c_str());
            printf("[AVX512_BF16: %s] ", x[hasAVX512BF16].c_str());
            printf("\n");
#endif // ifndef __aarch64__
        }
    };


    struct FP8E4M3ToFP32Manager {
        float dict[256] = {
            0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480, -0.0, -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625, -0.01171875, -0.013671875, -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375, -0.025390625, -0.02734375, -0.029296875, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0, -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0, -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0, -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, -480
        };
    };

    template<typename T, std::size_t Alignment>
    class aligned_allocator {
    public:
        using value_type = T;
        
        T* allocate(std::size_t n) {
            void* ptr = std::aligned_alloc(Alignment, n * sizeof(T));
            if (!ptr) throw std::bad_alloc();
            return static_cast<T*>(ptr);
        }
        
        void deallocate(T* p, std::size_t) noexcept {
            std::free(p);
        }
        
        template<typename U>
        struct rebind {
            using other = aligned_allocator<U, Alignment>;
        };
    };

    static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
    };

    CPUInstructInfo *GetCPUInstructInfo();

    AliveThreadPool *GetAlivePool();

    NumaConfig *GetNumaConfig();

    // 抽象的动态任务调度函数
    /* template<typename OpType>
    void DynamicScheduleTasks(std::vector<std::vector <OpType*> >& ops) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();

        std::vector <std::deque<OpType*> > tasks;
        tasks.resize(numaConfig->threads);
        int totalOps = 0;
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            totalOps += ops[nid].size();
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            // 分配任务给线程
            int per = ops[nid].size() / threadNum, remain = ops[nid].size() % threadNum;
            int cur = 0;
            for (int i = 0; i < threadNum; i++) {
                int now = per + (i < remain);
                for (int j = cur; j < cur + now; j++) {
                    tasks[numaConfig->numaToCpuDict[nid][i].first].push_back(ops[nid][j]);
                }
                cur += now;
            }
        }
        
        // 动态调度任务
        int pushTasks = 0;
        while (pushTasks < totalOps) {
            for (int i = 0; i < tasks.size(); i++) {
                if (pool->TryWait(i)) {
                    if (tasks[i].size() > 0) {
                        auto *task = tasks[i].front();
                        tasks[i].pop_front();
                        pool->PushOp(i, task);
                        pushTasks++;
                    } else {
                        // 从最忙的线程偷取任务
                        int sel = -1, maxS = 0;
                        for (int j = 0; j < tasks.size(); j++) {
                            if (numaConfig->threadIdToNumaDict[i] != numaConfig->threadIdToNumaDict[j]) {
                                continue;
                            }
                            if (tasks[j].size() > maxS) {
                                maxS = tasks[j].size();
                                sel = j;
                            }
                        }
                        if (sel != -1) {
                            auto *task = tasks[sel].back();
                            tasks[sel].pop_back();
                            pool->PushOp(i, task);
                            pushTasks++;
                        }
                    }
                }
            }
        }
        
        // 等待所有任务完成
        for (int i = 0; i < tasks.size(); i++) {
            pool->Wait(i);
        }
        
        // 清理
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            for (auto* op : ops[nid]) {
                delete op;
            }
        }
    } */

    template<typename OpType>
    struct WorkStealingOp : MultiThreadBaseOp {
        struct alignas(64) TaskState {
            std::atomic<int> curr;
            int end;
            std::vector<OpType*> tasks;
            std::atomic<bool> completed;
        };
        
        int threadId;
        int numaId;
        std::vector<TaskState*>* allStates;
        TaskState* myState;
        NumaConfig* numaConfig;
        
        WorkStealingOp(int tid, int nid, std::vector<TaskState*>* states, 
                      TaskState* state, NumaConfig* config) 
            : threadId(tid), numaId(nid), allStates(states), 
              myState(state), numaConfig(config) {}
        
        void Run() override {
            // 首先执行自己的任务
            processOwnTasks();
            
            // 然后从同一NUMA节点的其他线程偷取任务
            stealFromSameNuma();
            
            // 标记完成
            myState->completed.store(true, std::memory_order_release);
        }
        
    private:
        void processOwnTasks() {
            while (true) {
                int taskId = myState->curr.fetch_add(1, std::memory_order_acq_rel);
                if (taskId >= myState->end) {
                    break;
                }
                if (taskId < myState->tasks.size()) {
                    myState->tasks[taskId]->Run();
                }
            }
        }
        
        void stealFromSameNuma() {
            // 获取同一NUMA节点的所有线程
            auto& numaThreads = numaConfig->numaToCpuDict[numaId];
            
            // 利用连续性计算位置：当前线程ID - NUMA节点第一个线程ID
            int numaStartThread = numaThreads[0].first;
            int myPos = threadId - numaStartThread;
            
            // 从当前线程开始，环形遍历其他线程
            for (int offset = 1; offset < numaThreads.size(); offset++) {
                int targetPos = (myPos + offset) % numaThreads.size();
                int tid = numaThreads[targetPos].first;
                
                TaskState* otherState = (*allStates)[tid];
                if (otherState == nullptr) continue;
                
                // 检查是否还有任务可偷
                while (true) {
                    int taskId = otherState->curr.fetch_add(1, std::memory_order_acq_rel);
                    if (taskId >= otherState->end) {
                        break;
                    }
                    if (taskId < otherState->tasks.size()) {
                        otherState->tasks[taskId]->Run();
                    }
                }
            }
        }
    };
    
    // 重构的动态任务调度函数，支持work-stealing
    template<typename OpType>
    void DynamicScheduleTasks(std::vector<std::vector<OpType*>>& ops) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        
        // 创建任务状态数组
        using TaskState = typename WorkStealingOp<OpType>::TaskState;
        std::vector<TaskState*> taskStates(numaConfig->threads, nullptr);
        
        // 为每个线程分配任务状态
        for (int i = 0; i < numaConfig->threads; i++) {
            taskStates[i] = new (std::align_val_t{64}) TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 分配任务到各个线程
        int totalOps = 0;
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            totalOps += ops[nid].size();
            
            if (ops[nid].empty()) continue;
            
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            if (threadNum == 0) continue;
            
            // 计算每个线程的任务数量
            int tasksPerThread = ops[nid].size() / threadNum;
            int remainingTasks = ops[nid].size() % threadNum;
            
            int taskIndex = 0;
            for (int i = 0; i < threadNum; i++) {
                int tid = numaConfig->numaToCpuDict[nid][i].first;
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                if (numTasks > 0) {
                    // 分配任务到该线程
                    taskStates[tid]->tasks.clear();
                    taskStates[tid]->tasks.reserve(numTasks);
                    
                    for (int j = 0; j < numTasks && taskIndex < ops[nid].size(); j++) {
                        taskStates[tid]->tasks.push_back(ops[nid][taskIndex++]);
                    }
                    
                    taskStates[tid]->curr.store(0, std::memory_order_relaxed);
                    taskStates[tid]->end = taskStates[tid]->tasks.size();
                } else {
                    taskStates[tid]->end = 0;
                }
            }
        }
        
        // 创建work-stealing ops并提交到线程池
        std::vector<WorkStealingOp<OpType>*> wsOps(numaConfig->threads);
        for (int i = 0; i < numaConfig->threads; i++) {
            int numaId = numaConfig->threadIdToNumaDict[i];
            wsOps[i] = new WorkStealingOp<OpType>(
                i, numaId, &taskStates, taskStates[i], numaConfig
            );
            
            // 只有有任务的线程才启动
            if (taskStates[i] != nullptr && taskStates[i]->end > 0) {
                pool->PushOp(i, wsOps[i]);
            } else {
                // 没有任务的线程也要启动，以便参与work-stealing
                taskStates[i]->completed.store(true, std::memory_order_release);
                pool->PushOp(i, wsOps[i]);
            }
        }
        // 等待所有线程完成
        for (int i = 0; i < numaConfig->threads; i++) {
            pool->Wait(i);
        }
        
        // 清理资源
        for (int i = 0; i < numaConfig->threads; i++) {
            delete wsOps[i];
            if (taskStates[i] != nullptr) {
                taskStates[i]->~TaskState();
                #if __cpp_aligned_new >= 201606
                    operator delete(taskStates[i], std::align_val_t{64});
                #else
                    free_aligned(taskStates[i], sizeof(TaskState));
                #endif
            }
        }
        
        // 删除原始ops
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            for (auto* op : ops[nid]) {
                delete op;
            }
        }
    }
    
    // 支持K-batch的work-stealing调度函数
    template<typename OpType>
    void DynamicScheduleKBatchTasks(std::vector<std::vector<OpType*>>& ops, int k) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        
        using TaskState = typename WorkStealingOp<OpType>::TaskState;
        std::vector<TaskState*> taskStates(numaConfig->threads, nullptr);
        
        // 初始化任务状态
        for (int i = 0; i < numaConfig->threads; i++) {
            taskStates[i] = new (std::align_val_t{64}) TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 按NUMA节点分配K-batch任务
        int totalBatches = 0;
        for (auto& nodeOps : ops) {
            totalBatches += (nodeOps.size() + k - 1) / k;  // 向上取整
        }
        
        int batchesPerNode = totalBatches / numaConfig->numaCnt;
        int remainingBatches = totalBatches % numaConfig->numaCnt;
        
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            if (nid >= ops.size() || ops[nid].empty()) continue;
            
            int nodeBatches = batchesPerNode + (nid < remainingBatches ? 1 : 0);
            int totalNodeTasks = nodeBatches * k;
            
            // 限制为实际任务数
            totalNodeTasks = std::min(totalNodeTasks, (int)ops[nid].size());
            
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            if (threadNum == 0) continue;
            
            int tasksPerThread = totalNodeTasks / threadNum;
            int remainingTasks = totalNodeTasks % threadNum;
            
            int taskIndex = 0;
            for (int i = 0; i < threadNum && taskIndex < ops[nid].size(); i++) {
                int tid = numaConfig->numaToCpuDict[nid][i].first;
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                taskStates[tid]->tasks.clear();
                taskStates[tid]->tasks.reserve(numTasks);
                
                for (int j = 0; j < numTasks && taskIndex < ops[nid].size(); j++) {
                    taskStates[tid]->tasks.push_back(ops[nid][taskIndex++]);
                }
                
                taskStates[tid]->curr.store(0, std::memory_order_relaxed);
                taskStates[tid]->end = taskStates[tid]->tasks.size();
            }
        }
        
        // 提交work-stealing任务
        std::vector<WorkStealingOp<OpType>*> wsOps(numaConfig->threads);
        for (int i = 0; i < numaConfig->threads; i++) {
            int numaId = numaConfig->threadIdToNumaDict[i];
            wsOps[i] = new WorkStealingOp<OpType>(
                i, numaId, &taskStates, taskStates[i], numaConfig
            );
            pool->PushOp(i, wsOps[i]);
        }
        
        // 等待完成
        for (int i = 0; i < numaConfig->threads; i++) {
            pool->Wait(i);
        }
        
        // 清理
        for (int i = 0; i < numaConfig->threads; i++) {
            delete wsOps[i];
            if (taskStates[i] != nullptr) {
                taskStates[i]->~TaskState();
                #if __cpp_aligned_new >= 201606
                    operator delete(taskStates[i], std::align_val_t{64});
                #else
                    free_aligned(taskStates[i], sizeof(TaskState));
                #endif
            }
        }
        
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            for (auto* op : ops[nid]) {
                delete op;
            }
        }
    }

    struct BF16ToFP32Manager {
        float dict[65536];

        BF16ToFP32Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                uint32_t x = (i << 16);
                dict[i] = *((float*)&x);
            }
        }
    };

    static void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        exit(0);
    }

    enum DataType {
        FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6, FLOAT16 = 7,
        INT4_NOZERO = 8, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale
        INT4_GROUP = 9, // 不用zeroPoint的int4, floatValue = min + uint4Value * scale, 且使用分组量化
        FP8_E4M3 = 10,
        INT2_GROUP = 11, // 不用zeroPoint的int2, floatValue = min + uint2Value * scale, 且使用分组量化
        BASE3_GROUP = 12, // 三元量化，-1 0 1
        INT32PARAM = 100, // int32的参数，这种类型的数据永远存在CPU上
        FP8_E4M3_BLOCK_128 = 1000, // fp8e4m3, block = 128
        AWQ_4BIT_128 = 1001, // awq, bits = 4, group = 128
        DATA_GGUF_FORMAT = 9999, DATA_GGUF_FORMAT_END = 19999, // [DATA_GGUF_FORMAT, DATA_GGUF_FORMAT_END]之间为GGUF格式的数据，ggml_type = type - DATA_FFUF_FORMAT
        DATA_AUTO_NONE = 99999, DATA_AUTO_LINEAR, DATA_AUTO_EMBEDDING, DATA_AUTO_CONV
    };

    static std::map <DataType, std::vector <std::string> > dataTypeNames = {
        {DataType::FLOAT32, {"float32", "fp32"}}, {DataType::BFLOAT16, {"bfloat16", "bf16"}}, {DataType::INT16, {"int16"}}, 
        {DataType::INT8, {"int8"}}, {DataType::INT4, {"int4o"}}, {DataType::INT2, {"int2"}}, {DataType::BIT, {"bit"}}, 
        {DataType::FLOAT16, {"float16", "fp16", "half"}}, {DataType::INT4_NOZERO, {"int4"}}, {DataType::INT4_GROUP, {"int4g"}},
        {DataType::FP8_E4M3, {"float8", "fp8", "fp8_e4m3"}}, {DataType::INT2_GROUP, {"int2g"}}, {DataType::BASE3_GROUP, {"base3g"}}
    };

    std::string GetDataTypeName(DataType type);

    struct FastllmLinearWeight {
        int batch = 1; // 一般都是1,多专家时有时候 > 1
        int k, m; // 数据排布成[k, m]
        std::vector <std::vector <void*> > datas;
        DataType dataType;

        void *scales, *zeros;
        int block = 0; // 分组量化每组的长度

        void Init(int batch, int k, int m, void *data, DataType dataType);

        void Init(int batch, int k, int m, std::vector <std::vector <uint8_t> > &datas, DataType dataType);
        
        FastllmLinearWeight (int batch, int k, int m, void *data, DataType dataType);

        FastllmLinearWeight (int batch, int k, int m, void *data, DataType dataType, 
            void *scales, void *zeros, int blockK, int blockM);
    };

    size_t GetDataBytes(DataType type, size_t rows, size_t columns);

    void ConvertToFloat32(const void *srcData, DataType srcDataType, float *floatData, size_t rows, size_t columns);

    void ConvertFromFloat32(void *dstData, DataType dstDataType, const float *floatData, size_t rows, size_t columns);

    // 对应的多线程运行函数
    void RunMultiThreadConvertFromFloat32(void *dstData, DataType dstDataType, 
                                            const float *floatData, size_t rows, 
                                            size_t columns, AliveThreadPool *pool);

    void FastllmGemm (int n, int m, int k, 
        const void *A, long lda, // A [n * m], lda = bytes for 1 row in A
        const void *B, long ldb, // B [k * m], ldb = bytes for 1 row in B
        void *C, long ldc, // C[n * k], ldc = bytes for 1 row in C
        int st, int end, // calc C[0 : n, st : end]
        DataType AType, DataType BType, DataType CType
    );
}

#endif