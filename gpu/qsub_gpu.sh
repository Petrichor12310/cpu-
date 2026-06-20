#!/bin/sh
#PBS -N pcfg_gpu
#PBS -e gpu.e
#PBS -o gpu.o
#PBS -l nodes=1:ppn=8:gpus=1
# 注意：不同集群的 GPU 资源请求语法可能不同
# 如果是 slurm 集群，把上面 #PBS 行改为:
#   #SBATCH --job-name=pcfg_gpu
#   #SBATCH --gres=gpu:1
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=8

cd /home/${USER}/guess

echo "=== GPU Info ==="
nvidia-smi

echo "=== Compiling ==="
# -arch 参数根据 GPU 型号选择:
#   Tesla K80: sm_37, Tesla P100: sm_60, Tesla V100: sm_70
#   Tesla T4: sm_75, RTX 2080: sm_75, RTX 3080: sm_86
nvcc main_gpu.cu md5_cuda.cu train.cpp guessing.cpp md5.cpp \
    -o main_gpu -O2 --std c++11 -arch=sm_60

echo "=== Running GPU Password Guessing ==="
./main_gpu

echo "=== Done ==="
