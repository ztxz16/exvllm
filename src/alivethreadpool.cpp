// Some code about numa setting, copy from: https://github.com/guqiong96/Lvllm

#include "alivethreadpool.h"

#include <numa.h>
#include <numaif.h>
#include <iostream>

namespace fastllm {
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
            std::abort();
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
            std::abort();
        }
        numa_free_nodemask(mask);
    }

    void* allocate_aligned_numa(size_t size, int node) { 
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
        uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
        void* raw_ptr = reinterpret_cast<void*>(addr & ~(63));
        numa_free(raw_ptr, size);
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

    struct BindCPUOp : MultiThreadBaseOp {
        int cpuId, numaId;

        BindCPUOp (int cpuId, int numaId) : cpuId(cpuId), numaId(numaId) {}

        void Run() {
            bind_to_numa_node(numaId);
            set_numa_mempolicy(numaId);
            
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(this->cpuId, &cpuset);
            if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
                perror("sched_setaffinity failed");
                exit(EXIT_FAILURE);
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