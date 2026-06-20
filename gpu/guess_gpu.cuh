#ifndef GUESS_GPU_CUH
#define GUESS_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>

// ===================================================================
// GPU 口令生成接口
//
// 将 PCFG 口令生成的两个循环（segment 值展开 → 口令拼接）
// 通过 CUDA kernel 进行 GPU 并行化。
//
// 设计：
//   - 每个 CUDA thread 处理一条口令（prefix + value 拼接）
//   - 支持批量多任务（多个 PT 的生成任务合并为一次 kernel 启动）
//   - 输出使用固定 stride 布局，与 md5_cuda 模块兼容
// ===================================================================

// 单条口令最大长度，与 md5_cuda.cuh 中的 GPU_MAX_PW_LEN 保持一致
#ifndef GPU_MAX_PW_LEN
#define GPU_MAX_PW_LEN 120
#endif

// ===================================================================
// 批量 GPU 口令生成
//
// 将多个生成任务合并为一次 GPU kernel 调用。
// 每个生成任务 = 一个 prefix + 若干 segment values → 若干候选口令。
//
// 参数：
//   @param prefixes          每个任务的 prefix 字符串，size = num_tasks
//   @param all_values        所有任务的 segment values 拼接在一起
//   @param task_value_counts 每个任务包含的 value 数量，size = num_tasks
//                            sum(task_value_counts) = all_values.size()
//   @param out_passwords     输出：生成的口令列表，将被 resize 并填充
//   @param kernel_ms         输出：GPU kernel 耗时（毫秒），可为 nullptr
//
// 返回：
//   cudaSuccess 或 CUDA 错误码
// ===================================================================
cudaError_t gpuPasswordGenBatch(
    const std::vector<std::string>& prefixes,
    const std::vector<std::string>& all_values,
    const std::vector<int>& task_value_counts,
    std::vector<std::string>& out_passwords,
    float* kernel_ms
);

// 初始化 / 释放 GPU 资源
void gpuPasswordGenInit();
void gpuPasswordGenCleanup();

#endif // GUESS_GPU_CUH
