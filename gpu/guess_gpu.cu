#include "guess_gpu.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ===================================================================
// GPU Kernel：批量口令生成
//
// 每个 thread 负责生成一条口令：
//   1. 根据 thread ID 找到所属的 task 和 value index
//   2. 将 prefix + value 拼接写入输出缓冲区
//
// Grid:  (num_passwords + BLOCK_SIZE - 1) / BLOCK_SIZE
// Block: BLOCK_SIZE (256)
// ===================================================================

// ---- 辅助：设备端字符串拷贝 ----
__device__ __forceinline__ int gpu_strcpy(char* dst, const char* src, int max_len)
{
    int i = 0;
    while (i < max_len && src[i] != '\0') {
        dst[i] = src[i];
        i++;
    }
    return i;  // 返回拷贝的字符数
}

// ===================================================================
// 主 Kernel
//
// 输入数据布局 (所有字符串数组均为 row-major, fixed stride = GPU_MAX_PW_LEN):
//   d_prefixes:       [num_tasks][GPU_MAX_PW_LEN]
//   d_prefix_lens:    [num_tasks]  — 实际长度
//   d_values:         [total_values][GPU_MAX_PW_LEN]
//   d_value_lens:     [total_values] — 实际长度
//   d_task_vstart:    [num_tasks] — 每个 task 在 d_values 中的起始下标
//   d_pw_task:        [num_passwords] — 每条口令属于哪个 task
//   d_pw_voffset:     [num_passwords] — 每条口令对应 task 内的第几个 value
//   stride:           GPU_MAX_PW_LEN
//
// 输出:
//   d_output:         [num_passwords][GPU_MAX_PW_LEN]
//   d_output_lens:    [num_passwords] — 实际长度
// ===================================================================
__global__ void password_gen_kernel(
    const char* __restrict__ d_prefixes,
    const int*  __restrict__ d_prefix_lens,
    const char* __restrict__ d_values,
    const int*  __restrict__ d_value_lens,
    const int*  __restrict__ d_task_vstart,
    const int*  __restrict__ d_pw_task,
    const int*  __restrict__ d_pw_voffset,
    int         stride,
    char*       __restrict__ d_output,
    int*        __restrict__ d_output_lens,
    int         num_passwords)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_passwords) return;

    // 查找这条口令属于哪个 task
    int task = d_pw_task[tid];
    int v_off = d_pw_voffset[tid];

    // 定位 prefix 和 value 字符串
    const char* prefix = d_prefixes + (size_t)task * stride;
    int prefix_len = d_prefix_lens[task];

    int global_vidx = d_task_vstart[task] + v_off;
    const char* value = d_values + (size_t)global_vidx * stride;
    int value_len = d_value_lens[global_vidx];

    // 拼接: output = prefix + value
    char* out = d_output + (size_t)tid * stride;
    int pos = 0;

    // 拷贝 prefix
    for (int i = 0; i < prefix_len && pos < stride - 1; i++) {
        out[pos++] = prefix[i];
    }
    // 拷贝 value
    for (int i = 0; i < value_len && pos < stride - 1; i++) {
        out[pos++] = value[i];
    }
    // null terminator
    out[pos] = '\0';
    d_output_lens[tid] = pos;
}

// ===================================================================
// 静态 GPU 资源（避免重复分配）
// ===================================================================
static char*     d_prefixes    = nullptr;
static int*      d_prefix_lens = nullptr;
static char*     d_values      = nullptr;
static int*      d_value_lens  = nullptr;
static int*      d_task_vstart = nullptr;
static int*      d_pw_task     = nullptr;
static int*      d_pw_voffset  = nullptr;
static char*     d_output      = nullptr;
static int*      d_output_lens = nullptr;

static size_t    d_max_tasks      = 0;  // 当前分配的最大 task 数
static size_t    d_max_values     = 0;  // 当前分配的最大 value 数
static size_t    d_max_passwords  = 0;  // 当前分配的最大 password 数

static bool      gen_initialized  = false;

// ---- 主机端打包缓冲区 ----
static std::vector<char>  h_prefixes_buf;
static std::vector<int>   h_prefix_lens_arr;
static std::vector<char>  h_values_buf;
static std::vector<int>   h_value_lens_arr;
static std::vector<int>   h_task_vstart_arr;
static std::vector<int>   h_pw_task_arr;
static std::vector<int>   h_pw_voffset_arr;

void gpuPasswordGenInit()
{
    if (!gen_initialized) {
        cudaFree(0);  // 预热 CUDA context
        gen_initialized = true;
    }
}

void gpuPasswordGenCleanup()
{
    if (d_prefixes)    { cudaFree(d_prefixes);    d_prefixes    = nullptr; }
    if (d_prefix_lens) { cudaFree(d_prefix_lens); d_prefix_lens = nullptr; }
    if (d_values)      { cudaFree(d_values);      d_values      = nullptr; }
    if (d_value_lens)  { cudaFree(d_value_lens);  d_value_lens  = nullptr; }
    if (d_task_vstart) { cudaFree(d_task_vstart); d_task_vstart = nullptr; }
    if (d_pw_task)     { cudaFree(d_pw_task);     d_pw_task     = nullptr; }
    if (d_pw_voffset)  { cudaFree(d_pw_voffset);  d_pw_voffset  = nullptr; }
    if (d_output)      { cudaFree(d_output);      d_output      = nullptr; }
    if (d_output_lens) { cudaFree(d_output_lens); d_output_lens = nullptr; }

    d_max_tasks     = 0;
    d_max_values    = 0;
    d_max_passwords = 0;
    gen_initialized = false;
}

// ---- 确保设备缓冲区容量 ----
static cudaError_t ensure_device_capacity(
    size_t num_tasks,
    size_t num_values,
    size_t num_passwords)
{
    cudaError_t err;

    if (num_tasks > d_max_tasks) {
        size_t new_cap = num_tasks * 2;

        if (d_prefixes)    cudaFree(d_prefixes);
        if (d_prefix_lens) cudaFree(d_prefix_lens);
        if (d_task_vstart) cudaFree(d_task_vstart);

        err = cudaMalloc(&d_prefixes,    new_cap * GPU_MAX_PW_LEN);
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_prefix_lens, new_cap * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_task_vstart, new_cap * sizeof(int));
        if (err != cudaSuccess) return err;

        d_max_tasks = new_cap;
    }

    if (num_values > d_max_values) {
        size_t new_cap = num_values * 2;

        if (d_values)     cudaFree(d_values);
        if (d_value_lens) cudaFree(d_value_lens);

        err = cudaMalloc(&d_values,     new_cap * GPU_MAX_PW_LEN);
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_value_lens, new_cap * sizeof(int));
        if (err != cudaSuccess) return err;

        d_max_values = new_cap;
    }

    if (num_passwords > d_max_passwords) {
        size_t new_cap = num_passwords * 2;

        if (d_pw_task)    cudaFree(d_pw_task);
        if (d_pw_voffset) cudaFree(d_pw_voffset);
        if (d_output)     cudaFree(d_output);
        if (d_output_lens)cudaFree(d_output_lens);

        err = cudaMalloc(&d_pw_task,    new_cap * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_pw_voffset, new_cap * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_output,     new_cap * GPU_MAX_PW_LEN);
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_output_lens,new_cap * sizeof(int));
        if (err != cudaSuccess) return err;

        d_max_passwords = new_cap;
    }

    return cudaSuccess;
}

// ===================================================================
// 主机端批量口令生成
// ===================================================================
cudaError_t gpuPasswordGenBatch(
    const std::vector<std::string>& prefixes,
    const std::vector<std::string>& all_values,
    const std::vector<int>& task_value_counts,
    std::vector<std::string>& out_passwords,
    float* kernel_ms)
{
    int num_tasks     = static_cast<int>(prefixes.size());
    int num_values    = static_cast<int>(all_values.size());
    int num_passwords = num_values;  // 每个 value 对应一条口令

    if (num_passwords == 0) {
        if (kernel_ms) *kernel_ms = 0.0f;
        out_passwords.clear();
        return cudaSuccess;
    }

    cudaError_t err;

    // ---- Step 1: 确保设备缓冲区容量 ----
    err = ensure_device_capacity(
        static_cast<size_t>(num_tasks),
        static_cast<size_t>(num_values),
        static_cast<size_t>(num_passwords));
    if (err != cudaSuccess) return err;

    // ---- Step 2: 打包 prefix 数据 ----
    {
        size_t bytes = (size_t)num_tasks * GPU_MAX_PW_LEN;
        if (h_prefixes_buf.size() < bytes) h_prefixes_buf.resize(bytes);
        if (h_prefix_lens_arr.size() < (size_t)num_tasks)
            h_prefix_lens_arr.resize(num_tasks);

        for (int i = 0; i < num_tasks; i++) {
            const std::string& p = prefixes[i];
            int plen = static_cast<int>(p.size());
            if (plen > GPU_MAX_PW_LEN - 1) plen = GPU_MAX_PW_LEN - 1;

            h_prefix_lens_arr[i] = plen;
            char* dst = h_prefixes_buf.data() + (size_t)i * GPU_MAX_PW_LEN;
            memcpy(dst, p.c_str(), plen);
            if (plen < GPU_MAX_PW_LEN) {
                memset(dst + plen, 0, GPU_MAX_PW_LEN - plen);
            }
        }
    }

    // ---- Step 3: 打包 value 数据 + 计算 task 偏移 ----
    {
        size_t bytes = (size_t)num_values * GPU_MAX_PW_LEN;
        if (h_values_buf.size() < bytes) h_values_buf.resize(bytes);
        if (h_value_lens_arr.size() < (size_t)num_values)
            h_value_lens_arr.resize(num_values);
        if (h_task_vstart_arr.size() < (size_t)num_tasks)
            h_task_vstart_arr.resize(num_tasks);

        int v_cursor = 0;
        for (int t = 0; t < num_tasks; t++) {
            h_task_vstart_arr[t] = v_cursor;
            for (int j = 0; j < task_value_counts[t]; j++) {
                int vidx = v_cursor + j;
                const std::string& v = all_values[vidx];
                int vlen = static_cast<int>(v.size());
                if (vlen > GPU_MAX_PW_LEN - 1) vlen = GPU_MAX_PW_LEN - 1;

                h_value_lens_arr[vidx] = vlen;
                char* dst = h_values_buf.data() + (size_t)vidx * GPU_MAX_PW_LEN;
                memcpy(dst, v.c_str(), vlen);
                if (vlen < GPU_MAX_PW_LEN) {
                    memset(dst + vlen, 0, GPU_MAX_PW_LEN - vlen);
                }
            }
            v_cursor += task_value_counts[t];
        }
    }

    // ---- Step 4: 构建 password→task/value 映射 ----
    {
        if (h_pw_task_arr.size() < (size_t)num_passwords)
            h_pw_task_arr.resize(num_passwords);
        if (h_pw_voffset_arr.size() < (size_t)num_passwords)
            h_pw_voffset_arr.resize(num_passwords);

        int pw_idx = 0;
        for (int t = 0; t < num_tasks; t++) {
            for (int j = 0; j < task_value_counts[t]; j++) {
                h_pw_task_arr[pw_idx]    = t;
                h_pw_voffset_arr[pw_idx] = j;
                pw_idx++;
            }
        }
    }

    // ---- Step 5: 数据传输 Host→Device ----
    err = cudaMemcpy(d_prefixes,    h_prefixes_buf.data(),
                     (size_t)num_tasks * GPU_MAX_PW_LEN, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_prefix_lens, h_prefix_lens_arr.data(),
                     (size_t)num_tasks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_values,      h_values_buf.data(),
                     (size_t)num_values * GPU_MAX_PW_LEN, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_value_lens,  h_value_lens_arr.data(),
                     (size_t)num_values * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_task_vstart, h_task_vstart_arr.data(),
                     (size_t)num_tasks * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_pw_task,     h_pw_task_arr.data(),
                     (size_t)num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_pw_voffset,  h_pw_voffset_arr.data(),
                     (size_t)num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // ---- Step 6: 启动 GPU Kernel ----
    const int BLOCK_SIZE = 256;
    int grid_size = (num_passwords + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev, 0);

    password_gen_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_prefixes, d_prefix_lens,
        d_values, d_value_lens,
        d_task_vstart,
        d_pw_task, d_pw_voffset,
        GPU_MAX_PW_LEN,
        d_output, d_output_lens,
        num_passwords);

    cudaEventRecord(stop_ev, 0);
    cudaEventSynchronize(stop_ev);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    if (kernel_ms) *kernel_ms = elapsed_ms;

    // 检查 kernel 执行错误
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    // ---- Step 7: 结果回传 Device→Host ----
    std::vector<char> h_output_buf((size_t)num_passwords * GPU_MAX_PW_LEN);
    std::vector<int>  h_output_lens_arr(num_passwords);

    err = cudaMemcpy(h_output_buf.data(), d_output,
                     (size_t)num_passwords * GPU_MAX_PW_LEN, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(h_output_lens_arr.data(), d_output_lens,
                     (size_t)num_passwords * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    // ---- Step 8: 转换为 std::vector<std::string> ----
    out_passwords.resize(num_passwords);
    for (int i = 0; i < num_passwords; i++) {
        int len = h_output_lens_arr[i];
        if (len < 0) len = 0;
        if (len > GPU_MAX_PW_LEN) len = GPU_MAX_PW_LEN;
        const char* src = h_output_buf.data() + (size_t)i * GPU_MAX_PW_LEN;
        out_passwords[i].assign(src, static_cast<size_t>(len));
    }

    return cudaSuccess;
}
