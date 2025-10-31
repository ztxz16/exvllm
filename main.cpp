// test_moe.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <memory>
#include "moe.h"

namespace fastllm {
    struct MOEConfig {
        int expert_num;
        int routed_expert_num;
        int hidden_size;
        int intermediate_size;
        void* gate_proj;
        void* up_proj;
        void* down_proj;
        DataType gate_type;
        DataType up_type;
        DataType down_type;
        DataType hidden_type;

        void *gate_scales;
        void *up_scales;
        void *down_scales;
        void *gate_zeros;
        void *up_zeros;
        void *down_zeros;
        int block_gate_0, block_gate_1;
        int block_up_0, block_up_1;
        int block_down_0, block_down_1;

        MOEConfig() {}

        MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, 
                void* gate_proj, void* up_proj, void* down_proj, 
                DataType gate_type, DataType up_type, DataType down_type, DataType hidden_type)
            : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), 
            gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), 
            gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
        
        MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, 
                void* gate_proj, void* up_proj, void* down_proj, 
                void* gate_scales, void* up_scales, void *down_scales,
                int block_gate_0, int block_gate_1,
                int block_up_0, int block_up_1,
                int block_down_0, int block_down_1,
                DataType gate_type, DataType up_type, DataType down_type, DataType hidden_type)
            : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), 
            gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), 
            gate_scales(gate_scales), up_scales(up_scales), down_scales(down_scales),
            block_gate_0(block_gate_0), block_gate_1(block_gate_1), 
            block_up_0(block_up_0), block_up_1(block_up_1),
            block_down_0(block_down_0), block_down_1(block_down_1),
            gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
        
        MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, 
                void* gate_proj, void* up_proj, void* down_proj, 
                void* gate_scales, void* up_scales, void *down_scales,
                void* gate_zeros, void* up_zeros, void *down_zeros,
                int block_gate_1, int block_up_1, int block_down_1,
                DataType gate_type, DataType up_type, DataType down_type, DataType hidden_type)
            : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), 
            gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), 
            gate_scales(gate_scales), up_scales(up_scales), down_scales(down_scales),
            gate_zeros(gate_zeros), up_zeros(up_zeros), down_zeros(down_zeros),
            block_gate_1(block_gate_1), block_up_1(block_up_1), block_down_1(block_down_1),
            gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
    };
}

// 朴素的MOE实现，用于验证正确性
class NaiveMOE {
private:
    fastllm::MOEConfig config_;
    std::vector<float> gate_proj_fp32_;
    std::vector<float> up_proj_fp32_;
    std::vector<float> down_proj_fp32_;
    
    // SiLU激活函数
    float silu(float x) {
        return x / (1.0f + expf(-x));
    }
public:
    NaiveMOE(const fastllm::MOEConfig& config) : config_(config) {
        int gate_size = config_.expert_num * config_.intermediate_size * config_.hidden_size;
        int up_size = config_.expert_num * config_.intermediate_size * config_.hidden_size;
        int down_size = config_.expert_num * config_.hidden_size * config_.intermediate_size;
        
        gate_proj_fp32_.resize(gate_size);
        up_proj_fp32_.resize(up_size);
        down_proj_fp32_.resize(down_size);
        
        fastllm::ConvertToFloat32(config_.gate_proj, config.gate_type,
             gate_proj_fp32_.data(), config_.expert_num * config_.intermediate_size, config_.hidden_size);
        fastllm::ConvertToFloat32(config_.up_proj, config.up_type,
             up_proj_fp32_.data(), config_.expert_num * config_.intermediate_size, config_.hidden_size);
        fastllm::ConvertToFloat32(config_.down_proj, config.down_type,
             down_proj_fp32_.data(), config_.expert_num * config_.hidden_size, config_.intermediate_size);
    }
    
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, 
                 const void* input, void* output) {
        std::vector <float> fp32InputVector, fp32OutputVector;
        fp32InputVector.resize(qlen * this->config_.hidden_size);
        fp32OutputVector.resize(qlen * this->config_.hidden_size);

        fastllm::ConvertToFloat32(input, this->config_.hidden_type, fp32InputVector.data(), qlen, this->config_.hidden_size);
        const float* input_fp32 = fp32InputVector.data();
        float* output_fp32 = fp32OutputVector.data();
        
        // 初始化输出为0
        memset(output_fp32, 0, qlen * config_.hidden_size * sizeof(float));
        
        // 对每个token进行处理
        for (int token_idx = 0; token_idx < qlen; token_idx++) {
            const float* token_input = input_fp32 + token_idx * config_.hidden_size;
            float* token_output = output_fp32 + token_idx * config_.hidden_size;
            
            // 对每个选中的专家
            for (int expert_idx = 0; expert_idx < k; expert_idx++) {
                uint64_t expert_id = expert_ids[token_idx * k + expert_idx];
                float weight = weights[token_idx * k + expert_idx];
                
                // Gate projection
                std::vector<float> gate_out(config_.intermediate_size, 0.0f);
                const float* gate_weight = gate_proj_fp32_.data() + 
                    expert_id * config_.intermediate_size * config_.hidden_size;
                
                for (int i = 0; i < config_.intermediate_size; i++) {
                    for (int j = 0; j < config_.hidden_size; j++) {
                        gate_out[i] += token_input[j] * 
                            gate_weight[i * config_.hidden_size + j];
                    }
                }
                
                // Up projection
                std::vector<float> up_out(config_.intermediate_size, 0.0f);
                const float* up_weight = up_proj_fp32_.data() + 
                    expert_id * config_.intermediate_size * config_.hidden_size;
                
                for (int i = 0; i < config_.intermediate_size; i++) {
                    for (int j = 0; j < config_.hidden_size; j++) {
                        up_out[i] += token_input[j] * 
                            up_weight[i * config_.hidden_size + j];
                    }
                }
                
                // SiLU activation and element-wise multiplication
                std::vector<float> activated(config_.intermediate_size);
                for (int i = 0; i < config_.intermediate_size; i++) {
                    activated[i] = silu(gate_out[i]) * up_out[i];
                }
                
                // Down projection
                const float* down_weight = down_proj_fp32_.data() + 
                    expert_id * config_.hidden_size * config_.intermediate_size;
                
                for (int i = 0; i < config_.hidden_size; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < config_.intermediate_size; j++) {
                        sum += activated[j] * 
                            down_weight[i * config_.intermediate_size + j];
                    }
                    token_output[i] += weight * sum;
                }
            }
        }

        fastllm::ConvertFromFloat32(output, this->config_.hidden_type, output_fp32, qlen, this->config_.hidden_size);
    }
};

// 初始化随机权重
void initialize_random_weights(void* ptr, size_t rows, size_t columns, fastllm::DataType type) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.01f);

    std::vector <float> fp32Value;
    fp32Value.resize(rows * columns);
    for (int i = 0; i < rows * columns; i++) {
        fp32Value[i] = dis(gen);
    }
    fastllm::ConvertFromFloat32(ptr, type, fp32Value.data(), rows, columns);
}

// 比较两个输出的相似度
float compute_similarity(const float* output1, const float* output2, size_t size) {
    float max_diff = 0.0f;
    float total_diff = 0.0f;
    float total_abs = 0.0f;
    
    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(output1[i] - output2[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
        total_abs += std::abs(output1[i]) + std::abs(output2[i]);
    }
    
    float avg_diff = total_diff / size;
    float relative_error = total_diff / (total_abs + 1e-6f);
    
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Average difference: " << avg_diff << std::endl;
    std::cout << "Relative error: " << relative_error << std::endl;
    
    return relative_error;
}

// MOE层的封装
struct MOELayer {
    void* gate_proj = nullptr;
    void* up_proj = nullptr;
    void* down_proj = nullptr;
    std::unique_ptr<fastllm::FastllmMoe> moe;
    std::vector<uint64_t> expert_ids;
    std::vector<float> weights;
    bool owns_memory = true;  // 是否拥有权重内存
    
    // 移动构造函数
    MOELayer() = default;
    MOELayer(MOELayer&& other) noexcept 
        : gate_proj(other.gate_proj)
        , up_proj(other.up_proj)
        , down_proj(other.down_proj)
        , moe(std::move(other.moe))
        , expert_ids(std::move(other.expert_ids))
        , weights(std::move(other.weights))
        , owns_memory(other.owns_memory) {
        other.gate_proj = nullptr;
        other.up_proj = nullptr;
        other.down_proj = nullptr;
        other.owns_memory = false;
    }
    
    // 移动赋值运算符
    MOELayer& operator=(MOELayer&& other) noexcept {
        if (this != &other) {
            if (owns_memory) {
                if (gate_proj) free(gate_proj);
                if (up_proj) free(up_proj);
                if (down_proj) free(down_proj);
            }
            
            gate_proj = other.gate_proj;
            up_proj = other.up_proj;
            down_proj = other.down_proj;
            moe = std::move(other.moe);
            expert_ids = std::move(other.expert_ids);
            weights = std::move(other.weights);
            owns_memory = other.owns_memory;
            
            other.gate_proj = nullptr;
            other.up_proj = nullptr;
            other.down_proj = nullptr;
            other.owns_memory = false;
        }
        return *this;
    }
    
    // 禁用拷贝
    MOELayer(const MOELayer&) = delete;
    MOELayer& operator=(const MOELayer&) = delete;
    
    ~MOELayer() {
        if (owns_memory) {
            if (gate_proj) free(gate_proj);
            if (up_proj) free(up_proj);
            if (down_proj) free(down_proj);
        }
    }
};

// 性能测试函数 - 多层版本
void benchmark_moe_layers(std::vector<MOELayer>& layers, int qlen, int k, 
                          void* input, void* output1, void* output2,
                          int num_iterations = 30) {
    // Warm up
    for (int warm = 0; warm < 5; warm++) {
        void* current_input = input;
        void* current_output = output1;
        
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].moe->forward(qlen, k, layers[i].expert_ids.data(), 
                                   layers[i].weights.data(), current_input, current_output);
            layers[i].moe->sync();
            
            // 交换输入输出缓冲区
            if (i < layers.size() - 1) {
                current_input = current_output;
                current_output = (current_output == output1) ? output2 : output1;
            }
        }
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < num_iterations; iter++) {
        void* current_input = input;
        void* current_output = output1;
        
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].moe->forward(qlen, k, layers[i].expert_ids.data(), 
                                   layers[i].weights.data(), current_input, current_output);
            
            // 交换输入输出缓冲区
            if (i < layers.size() - 1) {
                current_input = current_output;
                current_output = (current_output == output1) ? output2 : output1;
            }
        }
        
        // 同步所有层
        for (auto& layer : layers) {
            layer.moe->sync();
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    float avg_time = duration / (float)num_iterations / 1000.0f; // ms
    float throughput = qlen / (avg_time / 1000.0f); // tokens per second
    
    std::cout << "Average time for " << layers.size() << " layers: " << avg_time << " ms" << std::endl;
    std::cout << "Average time per layer: " << avg_time / layers.size() << " ms" << std::endl;
    std::cout << "Throughput: " << throughput << " tokens/s" << std::endl;
    
    // Calculate FLOPS
    if (!layers.empty()) {
        int64_t flops_per_token = 2LL * k * (
            layers[0].moe->hiddenSize * layers[0].moe->intermediateSize +  // gate projection
            layers[0].moe->hiddenSize * layers[0].moe->intermediateSize +  // up projection
            layers[0].moe->hiddenSize * layers[0].moe->intermediateSize    // down projection
        );
        int64_t total_flops = flops_per_token * qlen * layers.size();
        float gflops = total_flops / (avg_time / 1000.0f) / 1e9f;
        
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }
}

int main() {
    // 设置MOE配置参数
    int expert_num = 128;
    int routed_expert_num = 8;
    int hidden_size = 2048;
    int intermediate_size = 768;
    int stride = 32;
    int group_min_len = 1;
    int group_max_len = 4096;
    int num_layers = 12;  // 层数
    
    // 测试类型
    fastllm::DataType weight_type = fastllm::DataType::AWQ_4BIT_128;
    fastllm::DataType gate_type = weight_type;
    fastllm::DataType up_type = weight_type;
    fastllm::DataType down_type = weight_type;
    fastllm::DataType hidden_type = fastllm::DataType::BFLOAT16;
    
    // 创建多个MOE层
    std::cout << "Creating " << num_layers << " MOE layers..." << std::endl;
    std::vector<MOELayer> layers;
    layers.reserve(num_layers);
    
    // 第一层：分配并初始化权重
    std::cout << "Creating first layer with random weights..." << std::endl;
    {
        MOELayer layer;
        
        // 分配权重内存
        size_t gate_bytes = fastllm::GetDataBytes(gate_type, expert_num * intermediate_size, hidden_size);
        size_t up_bytes = fastllm::GetDataBytes(up_type, expert_num * intermediate_size, hidden_size);
        size_t down_bytes = fastllm::GetDataBytes(down_type, expert_num * hidden_size, intermediate_size);
        
        layer.gate_proj = aligned_alloc(64, gate_bytes);
        layer.up_proj = aligned_alloc(64, up_bytes);
        layer.down_proj = aligned_alloc(64, down_bytes);
        layer.owns_memory = true;
        
        // 初始化随机权重
        initialize_random_weights(layer.gate_proj, expert_num * intermediate_size, hidden_size, gate_type);
        initialize_random_weights(layer.up_proj, expert_num * intermediate_size, hidden_size, up_type);
        initialize_random_weights(layer.down_proj, expert_num * hidden_size, intermediate_size, down_type);
        
        // 创建MOE配置
        fastllm::MOEConfig config(expert_num, routed_expert_num, hidden_size, intermediate_size,
            layer.gate_proj, layer.up_proj, layer.down_proj, gate_type, up_type, down_type, hidden_type);

        // 创建MOE实例
        layer.moe = std::make_unique<fastllm::FastllmMoe> (
            expert_num, routed_expert_num, hidden_size, intermediate_size, hidden_type, 
            new fastllm::FastllmLinearWeight(expert_num, intermediate_size, hidden_size, layer.gate_proj, gate_type), 
            new fastllm::FastllmLinearWeight(expert_num, intermediate_size, hidden_size, layer.up_proj, up_type), 
            new fastllm::FastllmLinearWeight(expert_num, hidden_size, intermediate_size, layer.down_proj, down_type)
        );
        
        layers.push_back(std::move(layer));
    }
    
    // 后续层：共享第一层的权重
    std::cout << "Creating remaining layers (sharing weights with first layer)..." << std::endl;
    for (int layer_idx = 1; layer_idx < num_layers; layer_idx++) {
        MOELayer layer;
        
        // 共享第一层的权重，不需要重新分配
        layer.gate_proj = layers[0].gate_proj;
        layer.up_proj = layers[0].up_proj;
        layer.down_proj = layers[0].down_proj;
        layer.owns_memory = false;  // 标记为不拥有内存
        
        // 创建MOE配置，使用第一层的权重
        fastllm::MOEConfig config(expert_num, routed_expert_num, hidden_size, intermediate_size,
            layer.gate_proj, layer.up_proj, layer.down_proj, gate_type, up_type, down_type, hidden_type);
        
        // 创建MOE实例
        layer.moe = std::make_unique<fastllm::FastllmMoe> (
            expert_num, routed_expert_num, hidden_size, intermediate_size, hidden_type, 
            new fastllm::FastllmLinearWeight(expert_num, intermediate_size, hidden_size, layer.gate_proj, gate_type), 
            new fastllm::FastllmLinearWeight(expert_num, intermediate_size, hidden_size, layer.up_proj, up_type), 
            new fastllm::FastllmLinearWeight(expert_num, hidden_size, intermediate_size, layer.down_proj, down_type)
        );
        
        layers.push_back(std::move(layer));
        
        if ((layer_idx + 1) % 10 == 0) {
            std::cout << "  Created " << (layer_idx + 1) << " layers..." << std::endl;
        }
    }
    
    // 热身所有层
    std::cout << "Warming up all layers..." << std::endl;
    for (auto& layer : layers) {
        layer.moe->warm_up();
    }
    
    // 正确性测试（只测试第一层）
    std::cout << "\n=== Correctness Test (First Layer) ===" << std::endl;
    int qlen = 128;
    int k = routed_expert_num;
    
    // 分配测试数据
    size_t real_hidden_bytes = fastllm::GetDataBytes(hidden_type, qlen, hidden_size);
    void* real_input = aligned_alloc(64, real_hidden_bytes);
    void* real_output = aligned_alloc(64, real_hidden_bytes);
    void* real_naive_output = aligned_alloc(64, real_hidden_bytes);
    
    std::vector<float> input(qlen * hidden_size);
    std::vector<float> output(qlen * hidden_size);
    std::vector<float> naive_output(qlen * hidden_size);
    
    // 初始化测试数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    std::uniform_int_distribution<uint64_t> expert_dis(0, expert_num - 1);
    
    for (auto& val : input) {
        val = dis(gen);
    }
    fastllm::ConvertFromFloat32(real_input, hidden_type, input.data(), qlen, hidden_size);
    
    // 为每层生成专家选择和权重（每层使用不同的专家选择）
    for (auto& layer : layers) {
        layer.expert_ids.resize(qlen * k);
        layer.weights.resize(qlen * k);
        
        for (int i = 0; i < qlen; i++) {
            // 为每个token随机选择k个不同的专家
            std::vector<uint64_t> selected;
            while (selected.size() < k) {
                uint64_t expert = expert_dis(gen);
                if (std::find(selected.begin(), selected.end(), expert) == selected.end()) {
                    selected.push_back(expert);
                }
            }
            
            float weight_sum = 0.0f;
            for (int j = 0; j < k; j++) {
                layer.expert_ids[i * k + j] = selected[j];
                layer.weights[i * k + j] = std::abs(dis(gen));
                weight_sum += layer.weights[i * k + j];
            }
            
            // 归一化权重
            for (int j = 0; j < k; j++) {
                layer.weights[i * k + j] /= weight_sum;
            }
        }
    }
    
    // 测试第一层的正确性
    fastllm::MOEConfig first_config(expert_num, routed_expert_num, hidden_size, intermediate_size,
        layers[0].gate_proj, layers[0].up_proj, layers[0].down_proj, 
        gate_type, up_type, down_type, hidden_type);
    NaiveMOE naive_moe(first_config);
    
    std::cout << "Running optimized MOE..." << std::endl;
    layers[0].moe->forward(qlen, k, layers[0].expert_ids.data(), layers[0].weights.data(), 
                          real_input, real_output);
    layers[0].moe->sync();
    
    std::cout << "Running naive MOE..." << std::endl;
    naive_moe.forward(qlen, k, layers[0].expert_ids.data(), layers[0].weights.data(),
                     real_input, real_naive_output);
    
    std::cout << "\nComparing outputs..." << std::endl;
    fastllm::ConvertToFloat32(real_output, hidden_type, output.data(), qlen, hidden_size);
    fastllm::ConvertToFloat32(real_naive_output, hidden_type, naive_output.data(), qlen, hidden_size);
    float error = compute_similarity(output.data(), naive_output.data(), qlen * hidden_size);
    
    if (error < 1e-2) {
        std::cout << "✓ Correctness test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Correctness test FAILED!" << std::endl;
        return 1;
    }
    
    // 性能测试
    std::cout << "\n=== Performance Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Number of layers: " << num_layers << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Intermediate size: " << intermediate_size << std::endl;
    std::cout << "  Number of experts: " << expert_num << std::endl;
    std::cout << "  Experts per token: " << k << std::endl;
    std::cout << "  Note: All layers share the same weights (except expert selection)\n" << std::endl;
    
    // 测试不同序列长度的性能
    // std::vector<int> test_lengths = {1};
    std::vector<int> test_lengths = {1, 1024};
    
    for (int test_qlen : test_lengths) {
        if (test_qlen > group_max_len) continue;
        
        std::cout << "\nTesting qlen = " << test_qlen << std::endl;
        
        // 准备测试数据
        size_t test_bytes = fastllm::GetDataBytes(hidden_type, test_qlen, hidden_size);
        void* test_input = aligned_alloc(64, test_bytes);
        void* test_output1 = aligned_alloc(64, test_bytes);
        void* test_output2 = aligned_alloc(64, test_bytes);
        
        std::vector<float> test_input_fp32(test_qlen * hidden_size);
        for (auto& val : test_input_fp32) {
            val = dis(gen);
        }
        fastllm::ConvertFromFloat32(test_input, hidden_type, test_input_fp32.data(), test_qlen, hidden_size);
        
        // 更新每层的expert_ids和weights以匹配新的序列长度
        for (auto& layer : layers) {
            layer.expert_ids.resize(test_qlen * k);
            layer.weights.resize(test_qlen * k);
            
            for (int i = 0; i < test_qlen * k; i++) {
                layer.expert_ids[i] = expert_dis(gen);
                layer.weights[i] = 1.0f / k;  // 均匀权重
            }
        }

        int times = 1;
        if (test_qlen == 1) {
            times = 30;
        }
        
        benchmark_moe_layers(layers, test_qlen, k, test_input, 
                           test_output1, test_output2, times);
        
        free(test_input);
        free(test_output1);
        free(test_output2);
    }
    
    // 清理内存
    free(real_input);
    free(real_output);
    free(real_naive_output);
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    
    return 0;
}
