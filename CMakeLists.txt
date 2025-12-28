cmake_minimum_required(VERSION 3.16)

project(FlashTensor VERSION 0.1.0 LANGUAGES CXX)

# ---- 基本设置 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 异构后端开关（先只放开关，后面再慢慢填）
option(FLASHTENSOR_ENABLE_CUDA "Enable CUDA backend" OFF)
option(FLASHTENSOR_ENABLE_HIP  "Enable HIP backend"  OFF)

# ---- 定义库：先按“编译型库”写（你也可以改成 header-only）----
add_library(FlashTensor
    src/tensor.cpp
)

add_library(FlashTensor::FlashTensor ALIAS FlashTensor)

target_include_directories(FlashTensor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# 统一用 target 的方式写编译特性/警告/宏
target_compile_features(FlashTensor PUBLIC cxx_std_17)

# 你可以用宏让代码里做条件编译（比如 #ifdef FLASHTENSOR_USE_CUDA）
target_compile_definitions(FlashTensor PUBLIC
    FLASHTENSOR_VERSION_MAJOR=0
)

# ---- 可选：CUDA 后端（只示范怎么“挂接”，不需要你马上写完整）----
if(FLASHTENSOR_ENABLE_CUDA)
    enable_language(CUDA)
    # 你也可以 set(CMAKE_CUDA_STANDARD 17)（通常 CUDA 代码 host 侧也要 C++17）
    target_compile_definitions(FlashTensor PUBLIC FLASHTENSOR_USE_CUDA=1)
    # CUDA 源文件示例：
    # target_sources(FlashTensor PRIVATE src/cuda/kernels.cu)
    # set_target_properties(FlashTensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---- 可选：HIP 后端（同理）----
if(FLASHTENSOR_ENABLE_HIP)
    target_compile_definitions(FlashTensor PUBLIC FLASHTENSOR_USE_HIP=1)
    # 通常你会在这里 find_package(hip) 或者用 ROCm 的工具链
endif()