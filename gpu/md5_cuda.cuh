#ifndef MD5_CUDA_CUH
#define MD5_CUDA_CUH

#include <cstdint>
#include <vector>
#include <string>

// ===================================================================
// GPU MD5 批量哈希接口
//
// 每个 CUDA thread 处理一条口令，实现大规模并行 MD5 计算。
// CPU 侧负责口令生成和结果验证，GPU 侧负责哈希计算。
// ===================================================================

// 单条口令最大长度（字符数）
// 55 字符以内为单 MD5 block（512-bit），覆盖绝大多数口令
// 超过 55 字符最多支持 120 字符（2 blocks）
#define GPU_MAX_PW_LEN 120

// 批量 MD5 哈希
// @param passwords  输入口令列表
// @param hashes_out 输出哈希值数组 [passwords.size() * 4]，调用者分配内存
// @param kernel_ms  输出 GPU kernel 执行时间（毫秒）
// @return cudaSuccess 或 CUDA 错误码
cudaError_t cudaMD5HashBatch(
    const std::vector<std::string>& passwords,
    uint32_t* hashes_out,
    float* kernel_ms
);

// 初始化 GPU 资源（首次调用时自动调用）
void cudaMD5Init();

// 释放 GPU 资源
void cudaMD5Cleanup();

#endif // MD5_CUDA_CUH
