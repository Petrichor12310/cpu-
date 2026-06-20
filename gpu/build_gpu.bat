@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
cd /d "D:\缪臻\日常应用\guess"
nvcc main_gpu.cu md5_cuda.cu guess_gpu.cu train.cpp guessing.cpp md5.cpp -o main_gpu_O2.exe -O2 -std=c++11 -arch=sm_89 "-DDATA_PATH=\"D:/缪臻/日常应用/guess/Rockyou-singleLined-full.txt\""
