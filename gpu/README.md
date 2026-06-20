# PCFG 口令猜测 — 并行程序设计 Lab5：GPU 编程

## 项目简介

本项目实现了基于 **PCFG（Probabilistic Context-Free Grammar，概率上下文无关文法）** 的口令猜测系统。通过对口令数据集进行统计建模，按照概率降序生成候选口令，并结合 MD5 哈希验证进行口令破解测试。

### 实验要求与完成情况

| 要求 | 实现 | 文件 |
|------|------|------|
| **基础要求** | CUDA GPU 批量 MD5 哈希加速 | `md5_cuda.cuh` / `md5_cuda.cu` / `main_gpu.cu` |
| 基础要求(MPI) | MPI Master-Worker 单 PT 级并行 | `main_mpi_basic.cpp` |
| 进阶要求(MPI) | MPI 批量 PT 级并行 | `main_mpi_advanced.cpp` |

### 核心文件说明

| 文件 | 功能 |
|------|------|
| `PCFG.h` | 核心数据结构（segment、PT、model、PriorityQueue） |
| `train.cpp` | 模型训练：读取口令集，解析统计 segment/PT |
| `guessing.cpp` | 口令生成：pthread/OpenMP 多线程 |
| `md5.h` / `md5.cpp` | CPU 端 MD5（含 ARM NEON SIMD 和 x86 回退） |
| **`md5_cuda.cuh`** | GPU MD5 批量哈希接口声明 |
| **`md5_cuda.cu`** | CUDA MD5 kernel + 主机端包装函数 |
| **`main_gpu.cu`** | **GPU 基础要求主程序** |
| `main.cpp` | 串行版本（CPU SIMD/标量） |
| `main_mpi_basic.cpp` | MPI 基础版本 |
| `main_mpi_advanced.cpp` | MPI 进阶版本 |
| `correctness.cpp` | MD5 正确性验证 |
| `correctness_guess.cpp` | 串行+破解率统计 |

---

## GPU 加速设计

### 数据流架构

```
CPU 负责:                           GPU 负责:
┌─────────────────┐                ┌─────────────────────┐
│ 1. 模型训练      │                │                     │
│ 2. 优先队列管理   │  口令批量传输   │  MD5 Hash Kernel    │
│ 3. 口令生成      │ ─────────────> │  (256 threads/block)│
│ 4. 破解结果统计   │ <───────────── │                     │
│ 5. 性能报告      │  哈希结果回传   │  每条口令 1 thread   │
└─────────────────┘                └─────────────────────┘
```

### CUDA Kernel 设计

- **并行粒度**：每条口令由一个 CUDA thread 独立处理（one-thread-per-password）
- **Block 大小**：256 threads
- **Grid 大小**：`ceil(num_passwords / 256)`
- **单 kernel 完成**：StringProcess（padding）→ 64 轮 MD5 → byte swap
- **内存布局**：口令以固定 stride（120 bytes）存储，确保 coalesced global memory 访问
- **支持口令长度**：≤120 字符（覆盖绝大多数口令），自动处理 1-2 个 MD5 block

---

## 环境搭建

### 本机没有 GPU？四种运行方案

由于你的电脑没有 NVIDIA GPU，以下是四种可选的运行方案：

| 方案 | 说明 | 难度 |
|------|------|------|
| **A. 仅编译验证** | 安装 CUDA Toolkit 编译 `.cu` 文件，但不能运行 | ⭐ 简单 |
| **B. 学校机房集群** | 提交 PBS 作业到有 GPU 的节点 | ⭐⭐ 中等 |
| **C. Google Colab** | 免费云端 T4 GPU，适合测试 | ⭐ 简单 |
| **D. GPGPU-Sim** | GPU 模拟器，在 CPU 上模拟 GPU 执行 | ⭐⭐⭐ 复杂 |

### 方案 A：仅编译（验证代码正确性）

即使没有 GPU，**CUDA Toolkit 仍可安装用于编译**：

1. **下载 CUDA Toolkit 11.8**（推荐，兼容性好）：
   https://developer.nvidia.com/cuda-11-8-0-download-archive
   - 选择：Windows → x86_64 → exe(local)
   - 安装时选择「自定义」，只需安装 **CUDA → Compiler (nvcc)**  和 **CUDA Runtime**

2. **验证安装**：
   ```powershell
   nvcc --version
   # 应输出: Cuda compilation tools, release 11.8, V11.8.89
   ```

3. **仅编译不运行**：
   ```powershell
   cd c:\Users\18695\Desktop\guess
   nvcc main_gpu.cu md5_cuda.cu train.cpp guessing.cpp md5.cpp -o build\main_gpu.exe -O2
   ```
   编译成功即证明 GPU 代码语法和 API 调用正确。

### 方案 B：学校机房集群（推荐方案）

学校通常有带 GPU 的集群节点。查看集群 GPU 资源：

```bash
# 登录集群后，查看 GPU 节点
qstat -Q  # 查看可用队列
pbsnodes -a  # 查看节点列表（含 GPU 信息）
nvidia-smi    # 查看 GPU 型号和显存
```

**创建 PBS 提交脚本** `qsub_gpu.sh`：

```bash
#!/bin/sh
#PBS -N pcfg_gpu
#PBS -e gpu.e
#PBS -o gpu.o
#PBS -l nodes=1:ppn=8:gpus=1
# 如果集群用 slurm 则改为:
#SBATCH --job-name=pcfg_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

cd /home/${USER}/guess

# 编译
nvcc main_gpu.cu md5_cuda.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 -arch=sm_60

# 运行
./main_gpu
```

提交：`qsub qsub_gpu.sh`

### 方案 C：Google Colab（免费 GPU）

1. 打开 https://colab.research.google.com/
2. 新建 Notebook，选择 Runtime → Change runtime type → T4 GPU
3. 上传代码或从 GitHub 克隆
4. 在 Notebook 中编译运行：

```python
# Colab cell 1: 安装编译工具
!apt-get install -y g++

# Colab cell 2: 编译
!nvcc main_gpu.cu md5_cuda.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 -std=c++11

# Colab cell 3: 上传数据后运行
!./main_gpu
```

> ⚠️ Colab 的 T4 GPU 显存有限（16GB），口令批处理大小可能需要调小（见常见问题）。

### 方案 D：GPGPU-Sim（CPU 模拟 GPU）

GPGPU-Sim 可以在 CPU 上模拟 NVIDIA GPU 的执行。需要 Linux 环境：

```bash
# 克隆 GPGPU-Sim
git clone https://github.com/gpgpu-sim/gpgpu-sim_distribution.git
cd gpgpu-sim_distribution

# 编译
source setup_environment release
make -j8

# 用 GPGPU-Sim 运行（会模拟 GPU 执行）
# 设置环境变量指向 GPGPU-Sim 的 libcudart.so
export LD_LIBRARY_PATH=/path/to/gpgpu-sim_distribution/lib/release:$LD_LIBRARY_PATH
./main_gpu
```

---

## 编译步骤

### GPU 版本（CUDA）

**在集群或有 GPU 的 Linux 机器上**：

```bash
# 查看 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 编译（根据 GPU 计算能力调整 -arch 参数）
# Tesla P100: sm_60, Tesla V100: sm_70, RTX 2080: sm_75, RTX 3080: sm_86, T4: sm_75
nvcc main_gpu.cu md5_cuda.cu guess_gpu.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 -std=c++11 -arch=sm_60

# 如果 nvcc 不识别 C++11 的某些特性，用 --std c++11
nvcc main_gpu.cu md5_cuda.cu guess_gpu.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 --std c++11 -arch=sm_60
```

### CPU 串行版本

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main_simd -O2 -std=c++11
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main_serial -O2 -std=c++11 -DMD5_SERIAL_BASELINE=1
g++ correctness_guess.cpp train.cpp guessing.cpp md5.cpp -o correctness_guess -O2 -std=c++11
```

### MPI 版本

```bash
mpic++ main_mpi_basic.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_basic -O2 -std=c++11
mpic++ main_mpi_advanced.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_advanced -O2 -std=c++11
```

---

## 运行步骤

### 运行前：修改数据路径

所有源文件中训练和测试数据路径硬编码为 `/guessdata/Rockyou-singleLined-full.txt`。

**两种处理方式**：

**方式一（推荐）：编译时定义宏**
```bash
# GPU 版本
nvcc main_gpu.cu md5_cuda.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 -arch=sm_60 \
    -DDATA_PATH='"./data/rockyou.txt"'

# 串行版本
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main_simd -O2 \
    -DDATA_PATH='"./data/rockyou.txt"'

# MPI 版本
mpic++ main_mpi_basic.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_basic -O2 \
    -DDATA_PATH='"./data/rockyou.txt"'
```

**方式二：直接修改源文件中的路径**（`/guessdata/...` → 你的本地路径）

### 运行 GPU 版本（基础要求）

```bash
# 直接运行（使用默认 GPU 0）
./main_gpu

# 指定 GPU 设备
CUDA_VISIBLE_DEVICES=0 ./main_gpu

# 使用 nvprof 分析 GPU 性能
nvprof ./main_gpu
```

**预期输出**：
```
Testing MD5Hash correctness...
MD5Hash test passed!
Training...
Training phase 1: reading and parsing passwords...
Lines processed: 10000
...
total pts: xxx
Ordering letters
Ordering digits
ordering symbols
GPU initialized.
Verifying GPU MD5 correctness...
GPU MD5 test passed!
here
Guesses generated: 1000000
Guesses generated: 2000000
...
Guesses generated: 10000000

========== Results ==========
Total guesses: 10000000
Cracked: xxxxx (rate: xx.xxx%)
Train time:xxxseconds               ← 请不要修改这一行
Guess time:xxxseconds               ← 请不要修改这一行
Hash time:xxxseconds                ← 请不要修改这一行
GPU kernel time:xxxseconds

=== Correctness Verification ===
Expected guesses: 10000000
Actual guesses:   10000000
Difference:       +0 (0%)
VERDICT: PASS — totals are consistent within 5% tolerance.

=== GPU Performance Note ===
GPU kernel accumulated time: x.xx s
Total hash time (kernel+transfer+check): x.xx s
Effective hash rate: xx.x M pws/s (total)
Kernel-only hash rate: xx.x M pws/s (kernel)
```

### 运行 MPI 版本

```bash
mpiexec -n 4 ./main_mpi_basic
mpiexec -n 8 ./main_mpi_advanced
```

### 在集群提交 PBS 作业

```bash
# MPI 作业（CPU 节点）
qsub qsub_mpi.sh

# GPU 作业
qsub qsub_gpu.sh
```

---

## 性能数据收集

### 核心性能指标

程序输出中有四行关键时间（标注了 "请不要修改这一行"）：

```
Train time:xxxseconds    ← 模型训练时间
Guess time:xxxseconds    ← 猜测生成时间（不含哈希）
Hash time:xxxseconds     ← MD5 哈希总时间（GPU kernel + 数据传输 + 破解检查）
GPU kernel time:xxxseconds ← 纯 GPU kernel 时间
```

### 建议的实验方案

#### 实验一：CPU vs GPU 性能对比

| 版本 | 命令 | 说明 |
|------|------|------|
| CPU 串行-标量 | `./main_serial` | MD5 逐条串行（基线） |
| CPU 串行-SIMD | `./main_simd` | ARM NEON x4 SIMD（x86 上=标量） |
| MPI CPU 8进程 | `mpiexec -n 8 ./main_mpi_basic` | CPU 多进程并行 |
| **GPU CUDA** | `./main_gpu` | GPU 批量并行（**基础要求**） |

#### 实验二：GPU 批量大小对性能的影响

修改 `main_gpu.cu` 中的 `GPU_BATCH_N`（默认 1000000），测试不同批量大小：
- 100000（小批量）
- 500000（中等批量）
- 1000000（大批量）
- 2000000（超大批量）

记录 Hash time 和 GPU kernel time 的变化。

#### 实验三：口令长度对 GPU 哈希速率的影响

在代码中过滤不同长度范围的口令，观察 GPU 哈希速率（M pws/s）的变化。

### 数据记录模板

| 版本 | 进程/设备 | Train(s) | Guess(s) | Hash(s) | GPU kernel(s) | Cracked | 破解率 | 哈希速率(M/s) |
|------|----------|----------|----------|---------|---------------|---------|--------|--------------|
| CPU-标量 | 1CPU | | | | N/A | | | |
| CPU-SIMD | 1CPU | | | | N/A | | | |
| MPI-基础 | 2CPU | | | | N/A | | | |
| MPI-基础 | 4CPU | | | | N/A | | | |
| MPI-基础 | 8CPU | | | | N/A | | | |
| **GPU-CUDA** | 1GPU | | | | | | | |

计算：
- CPU→GPU 加速比 = CPU Hash Time / GPU Hash Time
- GPU kernel 占比 = GPU kernel Time / GPU Hash Time（反映传输开销）

---

## 常见问题

### Q: 没有 GPU 怎么编译？
A: 安装 CUDA Toolkit 即可（不需要 GPU 硬件）。nvcc 编译器可以在任何 x86 机器上编译 `.cu` 文件。

### Q: 编译报错 "Unsupported gpu architecture"
A: 去掉或修改 `-arch` 参数。查看你的 GPU 计算能力：`nvidia-smi --query-gpu=compute_cap --format=csv`，然后设置对应的 `-arch=sm_XX`。

### Q: 运行报错 "no CUDA-capable device is detected"
A: 确认机器有 NVIDIA GPU 且驱动已安装。运行 `nvidia-smi` 检查。如果是集群环境，确认作业请求了 GPU 资源（`#PBS -l nodes=1:gpus=1`）。

### Q: 内存不足 (out of memory)
A: 减小 `main_gpu.cu` 中的 `GPU_BATCH_N` 值（如改为 500000 或 250000），单次传输数据量更小。

### Q: GPU MD5 结果与 CPU 不一致
A: 程序启动时会自动验证 GPU MD5 正确性（输出 "GPU MD5 test passed!"）。如果验证失败，请检查 CUDA 编译选项和 GPU 架构是否匹配。

### Q: 找不到数据集文件
A: 
- 集群上通常挂载在 `/guessdata/Rockyou-singleLined-full.txt`
- 本地测试需下载 RockYou 并修改路径（推荐编译时加 `-DDATA_PATH=...`）
- RockYou 下载：https://github.com/danielmiessler/SecLists

### Q: 编译报错找不到 omp.h
A: 移除 `PCFG.h` 第9行的 `#include <omp.h>`，或添加 `-fopenmp` 编译选项。

### Q: MPI 程序启动后立即退出
A: 检查所有 rank 的数据路径是否可访问，MPI 多进程中所有节点都要能访问到数据文件。

### Q: 想在本机（Windows）做 MPI 实验
A: 安装 Microsoft MPI：
1. 下载：https://www.microsoft.com/en-us/download/details.aspx?id=100593
2. 安装 SDK + Runtime
3. 将 `C:\Program Files\Microsoft MPI\Bin` 加入 PATH
4. 使用 `mpiexec -n 4 main_mpi_basic.exe` 运行

---

## 项目文件清单

```
guess/
├── README.md                 ← 本文件
├── PCFG.h                    ← 核心数据结构
├── train.cpp                 ← 模型训练
├── guessing.cpp              ← 口令生成（多线程）
├── md5.h                     ← MD5 接口
├── md5.cpp                   ← MD5 CPU 实现（ARM NEON SIMD + x86 回退）
├── md5_cuda.cuh              ← GPU MD5 批量哈希接口 ★
├── md5_cuda.cu               ← CUDA MD5 kernel 实现 ★
├── guess_gpu.cuh             ← GPU 口令生成接口 ★
├── guess_gpu.cu              ← CUDA 口令生成 kernel ★
├── main_gpu.cu               ← GPU 基础要求主程序 ★
├── main.cpp                  ← 串行版本
├── main_mpi_basic.cpp        ← MPI 基础版本
├── main_mpi_advanced.cpp     ← MPI 进阶版本
├── correctness.cpp           ← MD5 正确性验证
├── correctness_guess.cpp     ← 串行+破解率
├── qsub.sh                   ← PBS 串行提交脚本
├── qsub_mpi.sh               ← PBS MPI 提交脚本
├── qsub_gpu.sh               ← PBS GPU 提交脚本（需自行创建）
└── data/
    └── rockyou.txt           ← 口令数据集（需自行准备）
```

---

## 参考资料

- CUDA Toolkit 下载：https://developer.nvidia.com/cuda-downloads
- CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- nvprof 性能分析：https://docs.nvidia.com/cuda/profiler-users-guide/
- GPGPU-Sim 模拟器：https://github.com/gpgpu-sim/gpgpu-sim_distribution
- Google Colab (免费 GPU)：https://colab.research.google.com/
- RockYou 数据集：https://github.com/danielmiessler/SecLists
- PCFG 论文：Weir et al., "Password Cracking Using Probabilistic Context-Free Grammars", IEEE S&P 2009
