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
// GPU 基础要求：单 PT 逐个 GPU 口令生成 + 批量 MD5 哈希
//
// 与进阶要求（main_gpu.cu）的关键区别：
//   基础：每 Pop 一个 PT，立即单独提交 GPU kernel 生成口令
//   进阶：连续 Pop 多个 PT，合并后一次 GPU kernel 批量生成
//
// 编译 (nvcc):
//   nvcc main_gpu_basic.cu md5_cuda.cu guess_gpu.cu train.cpp guessing.cpp md5.cpp -o main_gpu_basic.exe -O2 -std=c++17 -arch=sm_89
// ===================================================================

#ifndef DATA_PATH
#define DATA_PATH "/guessdata/Rockyou-singleLined-full.txt"
#endif

#define GENERATE_N    10000000
#define HASH_BATCH_N  1000000    // 积累够 100 万条后批量 GPU 哈希
#define TEST_SET_N    1000000

struct GenTask {
    string prefix;
    const segment* model_seg;
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
    // Phase 4: 初始化 GPU
    // ================================================================
    gpuPasswordGenInit();
    cudaMD5Init();
    cout << "GPU initialized." << endl;

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
                return 1;
            }
        }
        cout << "GPU MD5 test passed!" << endl;
    }

    q.init();
    cout << "here" << endl;

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
    // Phase 5: 基础要求主循环
    //
    // 每次只处理一个 PT：
    //   PopMax → 单个 PT 的 GPU 口令生成 → 积累口令
    //   → 够 100 万条后 GPU 批量哈希 → 破解检查
    //
    // 与进阶要求 (main_gpu.cu) 的关键区别：
    //   每次 GPU 口令生成只含一个 PT，而不是多个 PT 合并。
    //   这导致更多的 kernel 启动次数和更低的 PCIe 带宽利用率。
    // ================================================================
    double time_gpu_hash_ks   = 0.0;   // GPU MD5 kernel 累计
    double time_gpu_gen_ks    = 0.0;   // GPU 口令生成 kernel 累计
    double time_guess         = 0.0;
    double time_hash          = 0.0;
    int history               = 0;
    int cracked               = 0;
    int batch_gen             = 0;
    int last_report           = 0;
    int pt_count              = 0;
    int kernel_calls          = 0;     // GPU 口令生成 kernel 调用次数

    vector<string> batch_passwords;    // 等待哈希的口令缓冲

    auto t_guess0 = system_clock::now();

    while (q.HasWork() && history < GENERATE_N)
    {
        // ---- Step A: 取一个 PT，单独提交 GPU 生成口令 ----
        PT current;
        if (!q.priority.PopMax(current)) break;

        q.CalProb(current);

        string        prefix;
        const segment* model_seg = nullptr;
        int           count = 0;

        if (current.content.size() == 1)
        {
            prefix = "";
            model_seg = find_model_seg(current.content[0]);
            count = current.max_indices[0];
        }
        else
        {
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
            int last = static_cast<int>(current.content.size()) - 1;
            model_seg = find_model_seg(current.content[last]);
            count = current.max_indices[last];
        }

        if (model_seg && count > 0)
        {
            int remaining = GENERATE_N - (history + batch_gen);
            if (count > remaining) count = remaining;

            if (count > 0)
            {
                // ====================================================
                // 基础要求核心：单个 PT 单独提交 GPU kernel
                // （进阶版在这里会积累多个 PT 后合并提交）
                // ====================================================
                vector<string> prefixes(1, prefix);
                vector<string> all_values(model_seg->ordered_values.begin(),
                                          model_seg->ordered_values.begin() + count);
                vector<int> value_counts(1, count);
                vector<string> generated_pws;
                float gen_kernel_ms = 0.0f;

                cudaError_t err = gpuPasswordGenBatch(
                    prefixes, all_values, value_counts,
                    generated_pws, &gen_kernel_ms);
                if (err != cudaSuccess)
                {
                    cout << "GPU password generation failed: " << cudaGetErrorString(err) << endl;
                    return 1;
                }
                time_gpu_gen_ks += gen_kernel_ms / 1000.0;
                kernel_calls++;

                for (const string& pw : generated_pws)
                    batch_passwords.push_back(pw);
                batch_gen += count;
                pt_count++;
            }
        }

        // ---- 生成后继 PT ----
        vector<PT> new_pts = current.NewPTs();
        for (PT& pt : new_pts)
        {
            q.CalProb(pt);
            q.priority.Push(pt);
        }

        // ---- 进度报告 ----
        int current_total = history + batch_gen;
        if (current_total - last_report >= 100000)
        {
            cout << "Guesses generated: " << current_total
                 << " (PTs: " << pt_count << ", kernels: " << kernel_calls << ")" << endl;
            last_report = current_total;
        }

        // ---- Step B: 积累够后批量 GPU 哈希 ----
        bool batch_full  = (batch_gen >= HASH_BATCH_N);
        bool queue_empty = (!q.HasWork() && batch_gen > 0);
        bool reached_end = (history + batch_gen >= GENERATE_N);

        if (batch_full || queue_empty || reached_end)
        {
            if (reached_end && time_guess == 0.0)
            {
                time_guess = double(
                    duration_cast<microseconds>(system_clock::now() - t_guess0).count()
                ) * microseconds::period::num / microseconds::period::den;
            }

            auto t_hash0 = system_clock::now();

            vector<uint32_t> batch_hashes(static_cast<size_t>(batch_gen) * 4);
            float md5_kernel_ms = 0.0f;
            cudaError_t err = cudaMD5HashBatch(batch_passwords, batch_hashes.data(), &md5_kernel_ms);
            if (err != cudaSuccess)
            {
                cout << "CUDA MD5 error: " << cudaGetErrorString(err) << endl;
                return 1;
            }
            time_gpu_hash_ks += md5_kernel_ms / 1000.0;

            for (int i = 0; i < batch_gen; i++)
            {
                if (test_set.find(batch_passwords[i]) != test_set.end())
                    cracked++;
            }

            auto t_hash1 = system_clock::now();
            time_hash += double(
                duration_cast<microseconds>(t_hash1 - t_hash0).count()
            ) * microseconds::period::num / microseconds::period::den;

            history += batch_gen;
            batch_gen = 0;
            batch_passwords.clear();

            if (history >= GENERATE_N) break;
        }
    }

    // ================================================================
    // Phase 6: 结果输出
    // ================================================================
    cout << "\n========== Results (Basic: Per-PT GPU Gen) ==========" << endl;
    cout << "Total guesses: " << history << endl;
    cout << "PTs processed: " << pt_count << endl;
    cout << "Gen kernel calls: " << kernel_calls << endl;
    cout << "Cracked: " << cracked
         << " (rate: " << (history > 0 ? 100.0 * cracked / history : 0.0) << "%)" << endl;
    cout << "Train time:" << time_train << "seconds" << endl;               //请不要修改这一行
    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;   //请不要修改这一行
    cout << "Hash time:" << time_hash << "seconds" << endl;                  //请不要修改这一行
    cout << "GPU gen kernel time (sum):" << time_gpu_gen_ks << "seconds" << endl;
    cout << "GPU MD5 kernel time:" << time_gpu_hash_ks << "seconds" << endl;

    cout << "\n=== Correctness Verification ===" << endl;
    int expected = GENERATE_N;
    int delta = history - expected;
    double err_pct = 100.0 * abs(delta) / expected;
    cout << "Expected guesses: " << expected << endl;
    cout << "Actual guesses:   " << history << endl;
    cout << "Difference:       " << (delta >= 0 ? "+" : "") << delta
         << " (" << err_pct << "%)" << endl;
    if (err_pct < 5.0)
        cout << "VERDICT: PASS" << endl;
    else
        cout << "VERDICT: FAIL" << endl;

    gpuPasswordGenCleanup();
    cudaMD5Cleanup();

    return 0;
}
