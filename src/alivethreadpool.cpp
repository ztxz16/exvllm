// Some code about numa setting, copy from: https://github.com/guqiong96/Lvllm

#include "alivethreadpool.h"

#include <numa.h>
#include <numaif.h>
#include <iostream>

namespace fastllm {
    struct NumaDetector {
        bool canUseNuma = true;

        NumaDetector () {
            if (numa_available() != -1) {
                if (numa_run_on_node(0) == -1) {
                    std::cerr << "Warning: NUMA node binding failed (non-privileged mode?)" << std::endl;
                    canUseNuma = false;
                }
            } else {
                canUseNuma = false;
            }
        }
    } numaDetector;
    
    void bind_to_cpu(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
            perror("sched_setaffinity failed");
            exit(EXIT_FAILURE);
        }
    }

    void bind_to_numa_node(int node_id) { 
        struct bitmask *node_cpus = numa_allocate_cpumask();
        if (numa_node_to_cpus(node_id, node_cpus) != 0) {
            perror("Failed to get NUMA node CPUs");
            numa_free_cpumask(node_cpus);
            numaDetector.canUseNuma = false;
            return;
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
    
        for (unsigned int i = 0; i < node_cpus->size; ++i) {
            if (numa_bitmask_isbitset(node_cpus, i)) {
                CPU_SET(i, &cpuset);
            }
        }
        numa_free_cpumask(node_cpus);
    
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            perror("sched_setaffinity failed"); 
        }
    }
    
    void set_numa_mempolicy(int node_id) {
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, node_id);
    
        int policy = MPOL_BIND;

        if (set_mempolicy(policy, mask->maskp, mask->size) == -1) {
            std::cerr << "set_mempolicy failed for node " << node_id 
                    << ": " << errno << " (" << strerror(errno) << ")\n";
            numaDetector.canUseNuma = false;
            return;
        }
        numa_free_nodemask(mask);
    }

    void* allocate_aligned(size_t size) {
        const size_t alignment = 64; 
        size_t total_size = size + alignment + sizeof(void*);
        void* raw_ptr = malloc(total_size);
        if (!raw_ptr) return nullptr;
        
        uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + sizeof(void*) + alignment - 1) & ~(alignment - 1);
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    
        void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
        *prev_ptr = raw_ptr;

        return aligned_ptr;
    } 

    void free_aligned(void* aligned_ptr, size_t size) {
        if (!aligned_ptr) return; 
        void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
        void* raw_ptr = *prev_ptr;
        free(raw_ptr);
    }

    void* allocate_aligned_numa(size_t size, int node) { 
        if (!numaDetector.canUseNuma) {
            return allocate_aligned(size);
        }
        
        size_t alignment = 64;
        size_t total_size = size + alignment - 1;
        void* raw_ptr = numa_alloc_onnode(total_size, node);
        if (!raw_ptr) {
            std::cerr << "Failed to allocate " << size << " bytes on NUMA node " << node << std::endl;
            return nullptr;
        }
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
        uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
        return reinterpret_cast<void*>(aligned_addr);
    }

    void free_aligned_numa(void* aligned_ptr, size_t size) {
        if (!numaDetector.canUseNuma) {
            free_aligned(aligned_ptr, size);
            return;
        }
        uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
        void* raw_ptr = reinterpret_cast<void*>(addr & ~(63));
        numa_free(raw_ptr, size);
    }

    struct BindCPUOp : MultiThreadBaseOp {
        int cpuId, numaId;

        BindCPUOp (int cpuId, int numaId) : cpuId(cpuId), numaId(numaId) {}

        void Run() {
            try {
                // 尝试绑定到NUMA节点
                try {
                    bind_to_numa_node(numaId);
                    set_numa_mempolicy(numaId);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to bind to NUMA node " << numaId 
                            << ": " << e.what() << std::endl;
                    std::cerr << "Continuing without NUMA binding (may affect performance)" << std::endl;
                }
                
                // 尝试设置CPU亲和性
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(this->cpuId, &cpuset);
                if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
                    // 检查是否是权限问题
                    if (errno == EPERM || errno == EACCES) {
                        std::cerr << "Warning: Failed to set CPU affinity to CPU " << this->cpuId 
                                << ": " << strerror(errno) << std::endl;
                        std::cerr << "Running without CPU pinning (requires privileged mode or CAP_SYS_NICE)" 
                                << std::endl;
                        std::cerr << "Consider running with --privileged or --cap-add=SYS_NICE" << std::endl;
                    } else {
                        // 其他错误可能更严重，抛出异常
                        throw std::runtime_error(std::string("sched_setaffinity failed: ") + strerror(errno));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in Run(): " << e.what() << std::endl;
                // 根据需求决定是否继续执行或退出
                // 如果想要继续执行：不做任何事
                // 如果想要退出：throw;
            }
        }
    };

    NumaConfig::NumaConfig (int threads, AliveThreadPool *pool, MachineNumaInfo *machineNumaInfo) {
        this->threads = threads;
        this->numaCnt = machineNumaInfo->numaCnt;

        this->numaToCpuDict.resize(this->numaCnt);
        int per = this->threads / this->numaCnt;
        this->threads = per * this->numaCnt;
        this->threadIdToNumaDict.resize(this->threads);
        int threadIdx = 0;

        for (int i = 0; i < this->numaCnt; i++) {
            for (int j = 0; j < per && j < machineNumaInfo->cpuIds[i].size(); j++) {
                this->threadIdToNumaDict[threadIdx] = i;
                this->numaToCpuDict[i].push_back(std::make_pair(threadIdx++, machineNumaInfo->cpuIds[i][j]));
                
                printf("threadIdx: %d, use cpu %d, bind to numa %d\n", threadIdx - 1, machineNumaInfo->cpuIds[i][j], i);
            }
        }

        std::vector<fastllm::BindCPUOp*> ops;
        ops.resize(this->threads);
        for (int i = 0; i < this->numaCnt; i++) {
            for (int j = 0; j < this->numaToCpuDict[i].size(); j++) {
                ops[this->numaToCpuDict[i][j].first] = new BindCPUOp(this->numaToCpuDict[i][j].second, i);
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
}