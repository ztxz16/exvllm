#include "moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(ft_kernel, m) {
    m.doc() = "MOE (Mixture of Experts) bindings";

    // FastllmLinearWeight 类绑定
    py::class_<fastllm::FastllmLinearWeight>(m, "FastllmLinearWeight")
        .def(py::init([](int batch, int k, int m, intptr_t data, int dataType) {
            return new fastllm::FastllmLinearWeight(
                batch, k, m, 
                (void*)data, 
                (fastllm::DataType)dataType
            );
        }))
        .def(py::init([](int batch, int k, int m, intptr_t data, int dataType,
                        intptr_t scales, intptr_t zeros, int blockK, int blockM) {
            return new fastllm::FastllmLinearWeight(
                batch, k, m, 
                (void*)data, 
                (fastllm::DataType)dataType,
                (void*)scales, 
                (void*)zeros, 
                blockK, blockM
            );
        }))
        .def_readwrite("batch", &fastllm::FastllmLinearWeight::batch)
        .def_readwrite("k", &fastllm::FastllmLinearWeight::k)
        .def_readwrite("m", &fastllm::FastllmLinearWeight::m)
        .def_readwrite("block", &fastllm::FastllmLinearWeight::block)
        .def_readonly("dataType", &fastllm::FastllmLinearWeight::dataType);
    
    // FastllmMoe 类绑定
    py::class_<fastllm::FastllmMoe>(m, "FastllmMoe")
        .def(py::init([](int expertNum, int routedExpertNum, int hiddenSize, 
                        int intermediateSize, int hiddenType,
                        fastllm::FastllmLinearWeight* gate, 
                        fastllm::FastllmLinearWeight* up, 
                        fastllm::FastllmLinearWeight* down) {
            return new fastllm::FastllmMoe(
                expertNum, routedExpertNum, hiddenSize, intermediateSize,
                (fastllm::DataType)hiddenType,
                gate, up, down
            );
        }), py::keep_alive<1, 7>(), py::keep_alive<1, 8>(), py::keep_alive<1, 9>())
        .def_readonly("expertNum", &fastllm::FastllmMoe::expertNum)
        .def_readonly("routedExpertNum", &fastllm::FastllmMoe::routedExpertNum)
        .def_readonly("hiddenSize", &fastllm::FastllmMoe::hiddenSize)
        .def_readonly("intermediateSize", &fastllm::FastllmMoe::intermediateSize)
        .def_readonly("hiddenType", &fastllm::FastllmMoe::hiddenType)
        .def("warm_up", &fastllm::FastllmMoe::warm_up)
        .def("sync_with_cuda_stream",
            [](fastllm::FastllmMoe& self, intptr_t user_cuda_stream) {
                self.sync_with_cuda_stream(user_cuda_stream);
            })
        .def("submit_with_cuda_stream", 
            [](fastllm::FastllmMoe& self, intptr_t user_cuda_stream, int qlen, int k, intptr_t expert_ids, 
            intptr_t weights, intptr_t input, intptr_t output, intptr_t batch_size_tensor) {
                self.submit_with_cuda_stream(
                    user_cuda_stream, 
                    qlen, 
                    k, 
                    (const uint64_t*)expert_ids, 
                    (const float*)weights, 
                    (const void*)input, 
                    (void*)output,
                    (int *)batch_size_tensor
                );
            }
        )
        .def("forward",
            [](fastllm::FastllmMoe& self,
               int qlen,
               int k,
               intptr_t expert_ids,
               intptr_t weights,
               intptr_t input,
               intptr_t output,
               intptr_t batch_size_tensor) {
                 
                
                self.forward(
                    qlen,
                    k,
                    (const uint64_t *)expert_ids,
                    (const float *)weights,
                    (const void *)input,
                    (void *)output
                );
        });
}