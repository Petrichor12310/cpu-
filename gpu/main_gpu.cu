#include "PCFG.h"
#include "guess_gpu.cuh"
#include "md5_cuda.cuh"
#include "md5.h"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <iomanip>
#include <chrono>
#include <cstring>

using namespace std;
using namespace chrono;

// ===================================================================
// GPU 基础要求：CUDA 加速的 PCFG 口令猜测
//
// 流水线：
//   CPU: 模型训练 → 优先队列维护（PopMax / NewPTs / Push）
//   GPU: 口令生成（password_gen_kernel）+ 批量 MD5 哈希（md5_batch_kernel）
//   CPU: 破解率统计 + 结果输出
//
// 编译 (nvcc):
//   nvcc main_gpu.cu md5_cuda.cu guess_gpu.cu train.cpp guessing.cpp md5.cpp -o main_gpu -O2
//
// 运行:
//   ./main_gpu
// ===================================================================

// ---- 可配置参数 ----
#ifndef DATA_PATH
#define DATA_PATH "/guessdata/Rockyou-singleLined-full.txt"
#endif

#define GENERATE_N    10000000   // 总猜测数
#define GPU_BATCH_N   1000000    // 每次 GPU 批处理的口令数
#define TEST_SET_N    1000000    // 测试集大小

// ===================================================================
// 生成任务：记录一个 PT 的口令生成参数
// ===================================================================
struct GenTask {
    string prefix;              // prefix 字符串（单 segment PT 为空串）
    const segment* model_seg;   // 指向模型中 segment 的指针（只读，含 ordered_values）
};

int main()
{
    // ================================================================
    // Phase 1: MD5 正确性测试
    // ================================================================
    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty",
                          "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e",
        "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad",
        "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b",
        "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055",
        "96e79218965eb72c92a549dd5a330112"
    };
    for (int i = 0; i < 8; i++) {
        bit32 state[4];
        MD5Hash(test_pws[i], state);
        stringstream ss;
        ss << setw(8) << setfill('0') << hex << state[0]
           << setw(8) << setfill('0') << hex << state[1]
           << setw(8) << setfill('0') << hex << state[2]
           << setw(8) << setfill('0') << hex << state[3];
        if (ss.str() != test_hashes[i]) {
            cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
            cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
            return 1;
        }
    }
    cout << "MD5Hash test passed!" << endl; //请不要修改这一行

    // ================================================================
    // Phase 2: 模型训练
    // ================================================================
    PriorityQueue q;
    auto t_train0 = system_clock::now();
    q.m.train(DATA_PATH);
    q.m.order();
    auto t_train1 = system_clock::now();
    double time_train = double(
        duration_cast<microseconds>(t_train1 - t_train0).count()
    ) * microseconds::period::num / microseconds::period::den;

    // ================================================================
    // Phase 3: 加载测试集
    // ================================================================
    unordered_set<string> test_set;
    {
        ifstream test_data(DATA_PATH);
        int test_count = 0;
        string pw;
        while (test_data >> pw) {
            test_count++;
            test_set.insert(pw);
            if (test_count >= TEST_SET_N) break;
        }
        cout << "Test set loaded: " << test_set.size() << " passwords" << endl;
    }

    // ================================================================
    // Phase 4: 初始化 GPU 资源
    // ================================================================
    gpuPasswordGenInit();
    cudaMD5Init();
    cout << "GPU initialized." << endl;

    // 验证 GPU MD5 正确性
    {
        cout << "Verifying GPU MD5 correctness..." << endl;
        vector<string> gpu_test_pws;
        for (int i = 0; i < 8; i++) gpu_test_pws.push_back(test_pws[i]);
        vector<uint32_t> gpu_hashes(8 * 4);
        float gpu_ms = 0;
        cudaError_t err = cudaMD5HashBatch(gpu_test_pws, gpu_hashes.data(), &gpu_ms);
        if (err != cudaSuccess) {
            cout << "GPU MD5 batch failed: " << cudaGetErrorString(err) << endl;
            return 1;
        }
        for (int i = 0; i < 8; i++) {
            stringstream ss;
            for (int j = 0; j < 4; j++)
                ss << setw(8) << setfill('0') << hex << gpu_hashes[i * 4 + j];
            if (ss.str() != test_hashes[i]) {
                cout << "GPU MD5 mismatch for " << test_pws[i] << "!" << endl;
                cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
                return 1;
            }
        }
        cout << "GPU MD5 test passed!" << endl;
    }

    // ---- 初始化优先队列 ----
    q.init();
    cout << "here" << endl;

    // ---- 辅助 lambda：在模型中定位 segment ----
    auto find_model_seg = [&](const segment& content_seg) -> const segment* {
        if (content_seg.type == 1) {
            int idx = q.m.FindLetter(content_seg);
            return &q.m.letters[idx];
        }
        if (content_seg.type == 2) {
            int idx = q.m.FindDigit(content_seg);
            return &q.m.digits[idx];
        }
        if (content_seg.type == 3) {
            int idx = q.m.FindSymbol(content_seg);
            return &q.m.symbols[idx];
        }
        return nullptr;
    };

    // ================================================================
    // Phase 5: 猜测生成 + GPU 哈希主循环
    //
    // 新的流水线：
    //   CPU: PopMax → 提取生成参数 → NewPTs → Push
    //   积累足够多任务后:
    //   GPU: password_gen_kernel（口令生成）
    //   GPU: md5_batch_kernel（MD5 哈希）
    //   CPU: 破解检查
    // ================================================================
    double time_gpu_kernel    = 0.0;   // GPU MD5 kernel 累计时间
    double time_gpu_gen       = 0.0;   // GPU 口令生成 kernel 累计时间
    double time_guess         = 0.0;   // 猜测总时间(含 queue ops + GPU gen + hash)
    double time_hash          = 0.0;   // 哈希阶段总时间（GPU gen + MD5 + transfer + check）
    int history               = 0;     // 已处理（已哈希）的猜测总数
    int cracked               = 0;     // 破解数
    int total_gen             = 0;     // 当前批次累计的生成口令数
    int last_report           = 0;     // 上次报告的累计猜测数

    vector<GenTask> gen_tasks;         // 当前批次的生成任务列表
    gen_tasks.reserve(10000);

    auto t_guess0 = system_clock::now();

    while (true)
    {
        // ---- Step A: 从优先队列取 PT，收集生成参数 ----
        while (q.HasWork() && total_gen < GPU_BATCH_N
               && history + total_gen < GENERATE_N)
        {
            PT current;
            if (!q.priority.PopMax(current)) break;

            // ---- 对应 Generate() 的逻辑：提取 prefix + 目标 segment ----
            q.CalProb(current);

            string        prefix;
            const segment* model_seg = nullptr;
            int           count = 0;

            if (current.content.size() == 1)
            {
                // 单 segment PT（如 L6、D3、S2）：prefix 为空，展开所有 values
                prefix = "";
                model_seg = find_model_seg(current.content[0]);
                count = current.max_indices[0];
            }
            else
            {
                // 多 segment PT（如 L6D1）：从 curr_indices 构建 prefix
                // prefix = ordered_values[idx] 逐段拼接（不含最后一段）
                int seg_idx = 0;
                for (int idx : current.curr_indices)
                {
                    const segment* ms = find_model_seg(current.content[seg_idx]);
                    if (ms && idx >= 0 && idx < static_cast<int>(ms->ordered_values.size()))
                        prefix += ms->ordered_values[idx];

                    seg_idx++;
                    if (seg_idx == static_cast<int>(current.content.size()) - 1)
                        break;
                }
                // 最后一个 segment 是需要展开的目标
                int last = static_cast<int>(current.content.size()) - 1;
                model_seg = find_model_seg(current.content[last]);
                count = current.max_indices[last];
            }

            // 记录生成任务（延迟到 GPU 批量生成）
            if (model_seg && count > 0)
            {
                gen_tasks.push_back({prefix, model_seg});
                total_gen += count;
            }

            // ---- 生成后继 PT 并推入优先队列 ----
            vector<PT> new_pts = current.NewPTs();
            for (PT& pt : new_pts)
            {
                q.CalProb(pt);
                q.priority.Push(pt);
            }
        }

        // ---- 进度报告 ----
        int current_total = history + total_gen;
        if (current_total - last_report >= 100000)
        {
            cout << "Guesses generated: " << current_total << endl;
            last_report = current_total;
        }

        // ---- Step B: 触发 GPU 批处理 ----
        bool batch_full  = (total_gen >= GPU_BATCH_N);
        bool queue_empty = (!q.HasWork() && total_gen > 0);
        bool reached_end = (history + total_gen >= GENERATE_N);

        if (batch_full || queue_empty || reached_end)
        {
            auto t_hash0 = system_clock::now();

            // ---- B1: 展平生成任务数据 ----
            vector<string> all_prefixes;
            vector<string> all_values;
            vector<int>    value_counts;

            int needed = GENERATE_N - history;
            if (needed <= 0) break;   // 已生成足够

            int accumulated = 0;
            for (size_t t = 0; t < gen_tasks.size(); t++)
            {
                const GenTask& task = gen_tasks[t];
                int n = static_cast<int>(task.model_seg->ordered_values.size());

                // 最后一批可能需要截断
                if (accumulated + n > needed)
                    n = needed - accumulated;
                if (n <= 0) break;

                all_prefixes.push_back(task.prefix);
                value_counts.push_back(n);
                for (int j = 0; j < n; j++)
                    all_values.push_back(task.model_seg->ordered_values[j]);

                accumulated += n;
                if (accumulated >= needed) break;
            }

            int batch_count = accumulated;
            if (batch_count == 0) break;

            // ---- B2: GPU 口令生成 ----
            vector<string> generated_pws;
            float gen_kernel_ms = 0.0f;
            cudaError_t err = gpuPasswordGenBatch(
                all_prefixes, all_values, value_counts,
                generated_pws, &gen_kernel_ms);
            if (err != cudaSuccess)
            {
                cout << "GPU password generation failed: " << cudaGetErrorString(err) << endl;
                return 1;
            }
            time_gpu_gen += gen_kernel_ms / 1000.0;

            // ---- B3: GPU MD5 哈希 ----
            vector<uint32_t> batch_hashes(static_cast<size_t>(batch_count) * 4);
            float md5_kernel_ms = 0.0f;
            err = cudaMD5HashBatch(generated_pws, batch_hashes.data(), &md5_kernel_ms);
            if (err != cudaSuccess)
            {
                cout << "CUDA MD5 error: " << cudaGetErrorString(err) << endl;
                return 1;
            }
            time_gpu_kernel += md5_kernel_ms / 1000.0;

            // ---- B4: CPU 破解检查 ----
            for (int i = 0; i < batch_count; i++)
            {
                if (test_set.find(generated_pws[i]) != test_set.end())
                    cracked++;
            }

            auto t_hash1 = system_clock::now();
            time_hash += double(
                duration_cast<microseconds>(t_hash1 - t_hash0).count()
            ) * microseconds::period::num / microseconds::period::den;

            // ---- B5: 更新状态 ----
            history += batch_count;
            total_gen = 0;
            gen_tasks.clear();

            // 记录 guess 总时间（在最后一批哈希完成后）
            if (history >= GENERATE_N && time_guess == 0.0)
            {
                time_guess = double(
                    duration_cast<microseconds>(system_clock::now() - t_guess0).count()
                ) * microseconds::period::num / microseconds::period::den;
            }

            if (history >= GENERATE_N) break;
        }
    }

    // ================================================================
    // Phase 6: 输出结果
    // ================================================================
    cout << "\n========== Results ==========" << endl;
    cout << "Total guesses: " << history << endl;
    cout << "Cracked: " << cracked
         << " (rate: " << (100.0 * cracked / history) << "%)" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;               //请不要修改这一行
    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;   //请不要修改这一行
    cout << "Hash time:" << time_hash << "seconds" << endl;                  //请不要修改这一行
    cout << "GPU kernel time:" << time_gpu_kernel << "seconds" << endl;
    cout << "GPU gen kernel time:" << time_gpu_gen << "seconds" << endl;

    // ---- 正确性验证 ----
    cout << "\n=== Correctness Verification ===" << endl;
    int expected = GENERATE_N;
    int delta = history - expected;
    double err_pct = 100.0 * abs(delta) / expected;
    cout << "Expected guesses: " << expected << endl;
    cout << "Actual guesses:   " << history << endl;
    cout << "Difference:       " << (delta >= 0 ? "+" : "") << delta
         << " (" << err_pct << "%)" << endl;
    if (err_pct < 5.0) {
        cout << "VERDICT: PASS — totals are consistent within 5% tolerance." << endl;
    } else {
        cout << "VERDICT: FAIL — totals differ by more than 5%. Check generation logic." << endl;
    }

    // ---- GPU vs CPU 性能对比提示 ----
    cout << "\n=== GPU Performance Note ===" << endl;
    cout << "GPU password gen kernel time: " << time_gpu_gen << " s" << endl;
    cout << "GPU MD5 kernel accumulated time: " << time_gpu_kernel << " s" << endl;
    cout << "Total hash time (gen+md5+transfer+check): " << time_hash << " s" << endl;
    cout << "Effective hash rate: "
         << (history / time_hash / 1000000.0) << " M pws/s (total)" << endl;

    // ================================================================
    // Phase 7: 清理
    // ================================================================
    gpuPasswordGenCleanup();
    cudaMD5Cleanup();

    return 0;
}
