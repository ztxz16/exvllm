cmake_minimum_required(VERSION 3.12)
project(fsg)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Position independent code")

set(CMAKE_BUILD_TYPE "Release")

if (MAKE_WHL_X86)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -mavx -mavx2 -mf16c -mfma -static-libstdc++ -static-libgcc -fPIC -Wl,-Bsymbolic -Wl,--exclude-libs,ALL")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (CXX_MAJOR_VERSION GREATER_EQUAL 10 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")    
    set_source_files_properties(
            "src/x86/avx512f.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl"
    )
        
    set_source_files_properties(
            "src/x86/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512bf16;-mavx512f;-mavx512bw;-mavx512vl"
    )
    
    set_source_files_properties(
            "src/x86/avx512vnni.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512vnni;-mavx512f"
    )
endif()

include_directories(src)

file(GLOB BASE_OP_FILES "src/baseop/*.cpp")
file(GLOB X86_FILES "src/x86/*.cpp")

set(MAIN_SRC
  "main.cpp"
  "src/alivethreadpool.cpp"
  "src/fastllm.cpp"
  "src/moe.cpp"

  ${BASE_OP_FILES}
  ${X86_FILES}
)

set(FT_KERNEL_SRC
  "src/ft_kernel.cpp"
  "src/alivethreadpool.cpp"
  "src/fastllm.cpp"
  "src/moe.cpp"  

  ${BASE_OP_FILES}
  ${X86_FILES}
)

add_library(ft_kernel SHARED ${FT_KERNEL_SRC})
target_link_libraries(ft_kernel PRIVATE numa)
target_link_libraries(ft_kernel PRIVATE CUDA::cudart CUDA::cuda_driver)

add_executable(main ${MAIN_SRC})
target_link_libraries(main PRIVATE numa)
target_link_libraries(main PRIVATE CUDA::cudart CUDA::cuda_driver)

if (${CMAKE_HOST_WIN32})
    add_custom_command(
            TARGET ft_kernel
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory python
            COMMAND ${CMAKE_COMMAND} -E make_directory python/exvllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/exvllm ${CMAKE_BINARY_DIR}/python/exvllm/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/python/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/$(Configuration)/ft_kernel.dll ${CMAKE_BINARY_DIR}/python/exvllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/$(Configuration)/ft_kernel.dll
    )
else()
    add_custom_command(
            TARGET ft_kernel
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory python
            COMMAND ${CMAKE_COMMAND} -E make_directory python/exvllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/exvllm ${CMAKE_BINARY_DIR}/python/exvllm/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_BINARY_DIR}/python/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libft_kernel.* ${CMAKE_BINARY_DIR}/python/exvllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libft_kernel.*
    )
endif()