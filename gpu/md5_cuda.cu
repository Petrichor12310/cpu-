#include "md5_cuda.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

// ===================================================================
// CUDA MD5 内核实现
//
// 设计要点：
//   - 每个 thread 处理一条口令（one-thread-per-password）
//   - 口令数据以固定 stride 存储，便于 coalesced 访问
//   - 大多数口令 <= 55 字符，单 512-bit block 即可完成
//   - 长口令（56-120 字符）需要 2 个 block
//   - kernel 内部完成 StringProcess（padding）+ MD5 64轮 + 字节序翻转
// ===================================================================

// ---- MD5 常量 (与 md5.h 一致) ----
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

// ---- 设备端 MD5 辅助函数 (内联以消除函数调用开销) ----
__device__ __forceinline__ uint32_t md5_F(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

__device__ __forceinline__ uint32_t md5_G(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (y & ~z);
}

__device__ __forceinline__ uint32_t md5_H(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

__device__ __forceinline__ uint32_t md5_I(uint32_t x, uint32_t y, uint32_t z) {
    return y ^ (x | ~z);
}

__device__ __forceinline__ uint32_t md5_ROTL(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ void md5_FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                                       uint32_t x, int s, uint32_t ac) {
    a += md5_F(b, c, d) + x + ac;
    a = md5_ROTL(a, s);
    a += b;
}

__device__ __forceinline__ void md5_GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                                       uint32_t x, int s, uint32_t ac) {
    a += md5_G(b, c, d) + x + ac;
    a = md5_ROTL(a, s);
    a += b;
}

__device__ __forceinline__ void md5_HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                                       uint32_t x, int s, uint32_t ac) {
    a += md5_H(b, c, d) + x + ac;
    a = md5_ROTL(a, s);
    a += b;
}

__device__ __forceinline__ void md5_II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d,
                                       uint32_t x, int s, uint32_t ac) {
    a += md5_I(b, c, d) + x + ac;
    a = md5_ROTL(a, s);
    a += b;
}

// ---- 单条口令的 MD5 计算（设备端，处理单个 512-bit block）----
__device__ void md5_single_block(const uint8_t* block64, uint32_t* state)
{
    // 将 64-byte block 解析为 16 个 32-bit 小端字
    uint32_t x[16];
    for (int i = 0; i < 16; ++i) {
        x[i] = (uint32_t)block64[i * 4] |
               ((uint32_t)block64[i * 4 + 1] << 8) |
               ((uint32_t)block64[i * 4 + 2] << 16) |
               ((uint32_t)block64[i * 4 + 3] << 24);
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];

    /* Round 1 */
    md5_FF(a, b, c, d, x[ 0], S11, 0xd76aa478);
    md5_FF(d, a, b, c, x[ 1], S12, 0xe8c7b756);
    md5_FF(c, d, a, b, x[ 2], S13, 0x242070db);
    md5_FF(b, c, d, a, x[ 3], S14, 0xc1bdceee);
    md5_FF(a, b, c, d, x[ 4], S11, 0xf57c0faf);
    md5_FF(d, a, b, c, x[ 5], S12, 0x4787c62a);
    md5_FF(c, d, a, b, x[ 6], S13, 0xa8304613);
    md5_FF(b, c, d, a, x[ 7], S14, 0xfd469501);
    md5_FF(a, b, c, d, x[ 8], S11, 0x698098d8);
    md5_FF(d, a, b, c, x[ 9], S12, 0x8b44f7af);
    md5_FF(c, d, a, b, x[10], S13, 0xffff5bb1);
    md5_FF(b, c, d, a, x[11], S14, 0x895cd7be);
    md5_FF(a, b, c, d, x[12], S11, 0x6b901122);
    md5_FF(d, a, b, c, x[13], S12, 0xfd987193);
    md5_FF(c, d, a, b, x[14], S13, 0xa679438e);
    md5_FF(b, c, d, a, x[15], S14, 0x49b40821);

    /* Round 2 */
    md5_GG(a, b, c, d, x[ 1], S21, 0xf61e2562);
    md5_GG(d, a, b, c, x[ 6], S22, 0xc040b340);
    md5_GG(c, d, a, b, x[11], S23, 0x265e5a51);
    md5_GG(b, c, d, a, x[ 0], S24, 0xe9b6c7aa);
    md5_GG(a, b, c, d, x[ 5], S21, 0xd62f105d);
    md5_GG(d, a, b, c, x[10], S22, 0x2441453);
    md5_GG(c, d, a, b, x[15], S23, 0xd8a1e681);
    md5_GG(b, c, d, a, x[ 4], S24, 0xe7d3fbc8);
    md5_GG(a, b, c, d, x[ 9], S21, 0x21e1cde6);
    md5_GG(d, a, b, c, x[14], S22, 0xc33707d6);
    md5_GG(c, d, a, b, x[ 3], S23, 0xf4d50d87);
    md5_GG(b, c, d, a, x[ 8], S24, 0x455a14ed);
    md5_GG(a, b, c, d, x[13], S21, 0xa9e3e905);
    md5_GG(d, a, b, c, x[ 2], S22, 0xfcefa3f8);
    md5_GG(c, d, a, b, x[ 7], S23, 0x676f02d9);
    md5_GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);

    /* Round 3 */
    md5_HH(a, b, c, d, x[ 5], S31, 0xfffa3942);
    md5_HH(d, a, b, c, x[ 8], S32, 0x8771f681);
    md5_HH(c, d, a, b, x[11], S33, 0x6d9d6122);
    md5_HH(b, c, d, a, x[14], S34, 0xfde5380c);
    md5_HH(a, b, c, d, x[ 1], S31, 0xa4beea44);
    md5_HH(d, a, b, c, x[ 4], S32, 0x4bdecfa9);
    md5_HH(c, d, a, b, x[ 7], S33, 0xf6bb4b60);
    md5_HH(b, c, d, a, x[10], S34, 0xbebfbc70);
    md5_HH(a, b, c, d, x[13], S31, 0x289b7ec6);
    md5_HH(d, a, b, c, x[ 0], S32, 0xeaa127fa);
    md5_HH(c, d, a, b, x[ 3], S33, 0xd4ef3085);
    md5_HH(b, c, d, a, x[ 6], S34, 0x4881d05);
    md5_HH(a, b, c, d, x[ 9], S31, 0xd9d4d039);
    md5_HH(d, a, b, c, x[12], S32, 0xe6db99e5);
    md5_HH(c, d, a, b, x[15], S33, 0x1fa27cf8);
    md5_HH(b, c, d, a, x[ 2], S34, 0xc4ac5665);

    /* Round 4 */
    md5_II(a, b, c, d, x[ 0], S41, 0xf4292244);
    md5_II(d, a, b, c, x[ 7], S42, 0x432aff97);
    md5_II(c, d, a, b, x[14], S43, 0xab9423a7);
    md5_II(b, c, d, a, x[ 5], S44, 0xfc93a039);
    md5_II(a, b, c, d, x[12], S41, 0x655b59c3);
    md5_II(d, a, b, c, x[ 3], S42, 0x8f0ccc92);
    md5_II(c, d, a, b, x[10], S43, 0xffeff47d);
    md5_II(b, c, d, a, x[ 1], S44, 0x85845dd1);
    md5_II(a, b, c, d, x[ 8], S41, 0x6fa87e4f);
    md5_II(d, a, b, c, x[15], S42, 0xfe2ce6e0);
    md5_II(c, d, a, b, x[ 6], S43, 0xa3014314);
    md5_II(b, c, d, a, x[13], S44, 0x4e0811a1);
    md5_II(a, b, c, d, x[ 4], S41, 0xf7537e82);
    md5_II(d, a, b, c, x[11], S42, 0xbd3af235);
    md5_II(c, d, a, b, x[ 2], S43, 0x2ad7d2bb);
    md5_II(b, c, d, a, x[ 9], S44, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

// ---- 设备端字节翻转 ----
__device__ __forceinline__ uint32_t md5_byteswap(uint32_t v) {
    return ((v & 0xffu) << 24) |
           ((v & 0xff00u) << 8) |
           ((v & 0xff0000u) >> 8) |
           ((v & 0xff000000u) >> 24);
}

// ===================================================================
// 主 Kernel：批量 MD5 哈希
//
// Grid:  (num_passwords + BLOCK_SIZE - 1) / BLOCK_SIZE
// Block: BLOCK_SIZE (256)
//
// 每个 thread 处理一条口令：
//   1. 读取口令字符串（固定 stride 布局）
//   2. StringProcess：构建 padded message
//   3. 逐 block 运行 MD5 64 轮
//   4. 字节翻转后写入输出
// ===================================================================
__global__ void md5_batch_kernel(
    const char* __restrict__ pw_data,       // [numPws][GPU_MAX_PW_LEN], row-major
    const int*  __restrict__ pw_lengths,    // [numPws]
    int         pw_stride,                  // = GPU_MAX_PW_LEN
    uint32_t*   __restrict__ hashes,        // [numPws][4]
    int         num_passwords)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_passwords) return;

    // ---- Step 1: 读取口令数据 ----
    const char* my_pw = pw_data + (size_t)tid * pw_stride;
    int len = pw_lengths[tid];
    if (len < 0) len = 0;
    if (len > GPU_MAX_PW_LEN) len = GPU_MAX_PW_LEN;

    // ---- Step 2: StringProcess（padding）----
    // 计算所需的 block 数量
    int bit_len = len * 8;
    int padding_bits = bit_len % 512;
    if (padding_bits > 448)
        padding_bits = 512 - (padding_bits - 448);
    else if (padding_bits < 448)
        padding_bits = 448 - padding_bits;
    else
        padding_bits = 512;

    int padding_bytes = padding_bits / 8;
    int total_bytes = len + padding_bytes + 8;       // 总是 64 的倍数
    int n_blocks = total_bytes / 64;

    // 在本地 memory 构建 padded message（最多 2 blocks = 128 bytes）
    uint8_t padded[128] = {0};

    // 复制原始口令
    for (int i = 0; i < len; ++i) {
        padded[i] = (uint8_t)my_pw[i];
    }
    // 添加 0x80 padding
    padded[len] = 0x80;
    // 剩余 padding 字节已由初始化置零

    // 添加 64-bit 原始消息长度（小端序）
    uint64_t original_bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; ++i) {
        padded[len + padding_bytes + i] = (uint8_t)((original_bit_len >> (i * 8)) & 0xFF);
    }

    // ---- Step 3: MD5 初始化 ----
    uint32_t state[4];
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // ---- Step 4: 逐 block 运行 MD5 ----
    for (int blk = 0; blk < n_blocks; ++blk) {
        md5_single_block(padded + blk * 64, state);
    }

    // ---- Step 5: 字节翻转后写入输出 ----
    uint32_t* out = hashes + (size_t)tid * 4;
    out[0] = md5_byteswap(state[0]);
    out[1] = md5_byteswap(state[1]);
    out[2] = md5_byteswap(state[2]);
    out[3] = md5_byteswap(state[3]);
}

// ===================================================================
// 主机端包装函数
// ===================================================================

// 静态资源（避免重复分配）
static char*     d_pw_data    = nullptr;
static int*      d_pw_lengths = nullptr;
static uint32_t* d_hashes     = nullptr;
static size_t    d_capacity   = 0;        // 当前分配的容量（口令条数）
static bool      initialized  = false;

void cudaMD5Init()
{
    if (!initialized) {
        // 预热 CUDA context
        cudaFree(0);
        initialized = true;
    }
}

void cudaMD5Cleanup()
{
    if (d_pw_data)    { cudaFree(d_pw_data);    d_pw_data    = nullptr; }
    if (d_pw_lengths) { cudaFree(d_pw_lengths); d_pw_lengths = nullptr; }
    if (d_hashes)     { cudaFree(d_hashes);     d_hashes     = nullptr; }
    d_capacity = 0;
    initialized = false;
}

cudaError_t cudaMD5HashBatch(
    const std::vector<std::string>& passwords,
    uint32_t* hashes_out,
    float* kernel_ms)
{
    int count = static_cast<int>(passwords.size());
    if (count == 0) {
        if (kernel_ms) *kernel_ms = 0.0f;
        return cudaSuccess;
    }

    cudaError_t err;

    // ---- Step 1: 准备主机端数据（打包成固定 stride 布局）----
    // 使用静态缓冲区避免每次 resize 带来的分配开销
    static std::vector<char>     h_pw_data;
    static std::vector<int>      h_pw_lengths;

    size_t needed_bytes = (size_t)count * GPU_MAX_PW_LEN;
    if (h_pw_data.size() < needed_bytes) {
        h_pw_data.resize(needed_bytes);
    }
    if (h_pw_lengths.size() < (size_t)count) {
        h_pw_lengths.resize(count);
    }

    for (int i = 0; i < count; ++i) {
        const std::string& pw = passwords[i];
        int pw_len = static_cast<int>(pw.size());
        if (pw_len > GPU_MAX_PW_LEN) pw_len = GPU_MAX_PW_LEN;

        h_pw_lengths[i] = pw_len;
        // 复制口令字符串（含 \0）
        char* dst = h_pw_data.data() + (size_t)i * GPU_MAX_PW_LEN;
        memcpy(dst, pw.c_str(), pw_len);
        // 剩余部分置零（确保 padding 正确）
        if (pw_len < GPU_MAX_PW_LEN) {
            memset(dst + pw_len, 0, GPU_MAX_PW_LEN - pw_len);
        }
    }

    // ---- Step 2: 分配/扩展设备内存 ----
    if ((size_t)count > d_capacity) {
        if (d_pw_data)    cudaFree(d_pw_data);
        if (d_pw_lengths) cudaFree(d_pw_lengths);
        if (d_hashes)     cudaFree(d_hashes);

        size_t new_cap = (size_t)count * 2;  // 预留空间
        err = cudaMalloc(&d_pw_data,    new_cap * GPU_MAX_PW_LEN);
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_pw_lengths, new_cap * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&d_hashes,     new_cap * 4 * sizeof(uint32_t));
        if (err != cudaSuccess) return err;
        d_capacity = new_cap;
    }

    // ---- Step 3: 数据传输 Host→Device ----
    err = cudaMemcpy(d_pw_data, h_pw_data.data(),
                     (size_t)count * GPU_MAX_PW_LEN, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;
    err = cudaMemcpy(d_pw_lengths, h_pw_lengths.data(),
                     (size_t)count * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return err;

    // ---- Step 4: 启动 Kernel ----
    const int BLOCK_SIZE = 256;
    int grid_size = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev, 0);

    md5_batch_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_pw_data, d_pw_lengths, GPU_MAX_PW_LEN, d_hashes, count);

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

    // ---- Step 5: 结果传回 Device→Host ----
    err = cudaMemcpy(hashes_out, d_hashes,
                     (size_t)count * 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}
