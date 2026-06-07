#include "PCFG.h"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <cstring>
using namespace std;

// ===================================================================
// MPI 基础要求：单 PT 层面的 MPI 并行
//
// Master-Worker 架构：
//   Rank 0 (Master): 维护优先队列，PopNext → Bcast 给所有 rank →
//                    所有 rank 并行 GenerateChunk → NewPTs → Push
//   Rank 1..P-1 (Workers): 接收 Bcast → GenerateChunk → SIMD hash
//
// 编译：
//   mpic++ main_mpi_basic.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_basic -O2
// ===================================================================

#ifndef MD5_SERIAL_BASELINE
#define MD5_SERIAL_BASELINE 0
#endif

int main(int argc, char* argv[])
{
    // ---- MPI init ----
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double time_hash  = 0.0;  // MD5 哈希时间
    double time_guess = 0.0;  // 猜测总时长（不含训练）
    double time_train = 0.0;  // 训练时长
    PriorityQueue q;

    // ================================================================
    // Phase 1: Training (rank 0 only)
    // ================================================================
    if (rank == 0) {
        // --- MD5 correctness test ---
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
            for (int i1 = 0; i1 < 4; i1 += 1)
                ss << setw(8) << setfill('0') << hex << state[i1];
            if (ss.str() != test_hashes[i]) {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        cout << "MD5Hash test passed!" << endl; //请不要修改这一行

        double t0 = MPI_Wtime();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        double t1 = MPI_Wtime();
        time_train = t1 - t0;
    }

    // ================================================================
    // Phase 2: Broadcast model to all ranks
    // ================================================================
    vector<char> model_buf;
    int buf_size = 0;
    if (rank == 0) {
        model_buf = q.m.serialize();
        buf_size = static_cast<int>(model_buf.size());
    }
    MPI_Bcast(&buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) model_buf.resize(static_cast<size_t>(buf_size));
    MPI_Bcast(model_buf.data(), buf_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) q.m.deserialize(model_buf);
    model_buf.clear();
    model_buf.shrink_to_fit();

    // ================================================================
    // Phase 2b: Load test set for cracking rate (each rank independently)
    // ================================================================
    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    int test_count = 0;
    string pw_line;
    while (test_data >> pw_line) {
        test_set.insert(pw_line);
        test_count++;
        if (test_count >= 1000000) break;
    }
    if (rank == 0) {
        cout << "Test set loaded: " << test_set.size() << " passwords" << endl;
    }

    // ================================================================
    // Phase 3: Rank 0 init simple queue
    // ================================================================
    if (rank == 0) {
        q.init_simple();
        cout << "here" << endl;
    }

    // Per-rank guess buffers
    vector<string> local_guesses;
    int local_buf_count = 0;
    int local_history   = 0;
    int local_cracked   = 0;    // 破解计数
    const int GENERATE_N = 10000000;
    const int LOCAL_N     = GENERATE_N / nprocs;

    bool stop_all = false;
    double t_guess_start = 0.0;
    if (rank == 0) t_guess_start = MPI_Wtime();

    // ================================================================
    // Phase 4: Master-Worker loop
    // ================================================================
    while (!stop_all)
    {
        // msg[0]=stop, [1]=seg_type, [2]=seg_length, [3]=N, [4]=prefix_len
        int msg[5] = {0, 0, 0, 0, 0};
        string prefix;

        if (rank == 0)
        {
            if (q.EmptySimple()) {
                msg[0] = 1;
            } else {
                PT current;
                q.PopMaxSimple(current);

                int seg_type = 0, seg_length = 0, N = 0;

                if (current.content.size() == 1)
                {
                    seg_type   = current.content[0].type;
                    seg_length = current.content[0].length;
                    N          = current.max_indices[0];
                    prefix     = "";
                }
                else
                {
                    int seg_idx = 0;
                    for (int idx : current.curr_indices)
                    {
                        if (current.content[seg_idx].type == 1)
                            prefix += q.m.letters[static_cast<size_t>(
                                q.m.FindLetter(current.content[seg_idx]))].ordered_values[idx];
                        else if (current.content[seg_idx].type == 2)
                            prefix += q.m.digits[static_cast<size_t>(
                                q.m.FindDigit(current.content[seg_idx]))].ordered_values[idx];
                        else if (current.content[seg_idx].type == 3)
                            prefix += q.m.symbols[static_cast<size_t>(
                                q.m.FindSymbol(current.content[seg_idx]))].ordered_values[idx];
                        seg_idx++;
                        if (seg_idx == static_cast<int>(current.content.size()) - 1) break;
                    }
                    int last = static_cast<int>(current.content.size()) - 1;
                    seg_type   = current.content[last].type;
                    seg_length = current.content[last].length;
                    N          = current.max_indices[last];
                }

                msg[1] = seg_type;
                msg[2] = seg_length;
                msg[3] = N;
                msg[4] = static_cast<int>(prefix.size());

                // NewPTs + push derivatives
                vector<PT> new_pts = current.NewPTs();
                for (PT& pt : new_pts) {
                    q.CalProb(pt);
                    q.PushSimple(pt);
                }

                if (local_history + local_buf_count >= LOCAL_N) {
                    msg[0] = 1;
                }
            }
        }

        // Broadcast to all ranks
        MPI_Bcast(msg, 5, MPI_INT, 0, MPI_COMM_WORLD);
        if (msg[0] != 0) { stop_all = true; break; }

        int seg_type   = msg[1];
        int seg_length = msg[2];
        int N          = msg[3];
        int prefix_len = msg[4];

        if (prefix_len > 0) {
            if (rank != 0) prefix.resize(static_cast<size_t>(prefix_len));
            MPI_Bcast(&prefix[0], prefix_len, MPI_CHAR, 0, MPI_COMM_WORLD);
        }

        // Locate segment in local model
        const segment* seg_ptr = nullptr;
        segment search_seg(seg_type, seg_length);
        if (seg_type == 1)
            seg_ptr = &q.m.letters[static_cast<size_t>(q.m.FindLetter(search_seg))];
        else if (seg_type == 2)
            seg_ptr = &q.m.digits[static_cast<size_t>(q.m.FindDigit(search_seg))];
        else if (seg_type == 3)
            seg_ptr = &q.m.symbols[static_cast<size_t>(q.m.FindSymbol(search_seg))];

        // Compute chunk and generate
        int chunk = N / nprocs;
        int rem   = N % nprocs;
        int start = rank * chunk + (rank < rem ? rank : rem);
        int end   = start + chunk + (rank < rem ? 1 : 0);

        q.GenerateChunk(prefix, seg_ptr, start, end, local_guesses, local_buf_count);

        // Hash batch when buffer is full
        if (local_buf_count > 1000000)
        {
            double t_hash0 = MPI_Wtime();
#if MD5_SERIAL_BASELINE
            for (const string &pw : local_guesses) {
                bit32 state[4];
                MD5Hash(pw, state);
                if (test_set.find(pw) != test_set.end()) local_cracked++;
            }
#else
            bit32 states[4][4];
            const string *guessData = local_guesses.data();
            int total = static_cast<int>(local_guesses.size());
            int i = 0;
            for (; i + 4 <= total; i += 4) {
                MD5HashSIMD4(guessData + i, states, 4);
                for (int k = 0; k < 4; k++) {
                    if (test_set.find(guessData[i + k]) != test_set.end()) local_cracked++;
                }
            }
            if (i < total) {
                string tail[4];
                int remain = total - i;
                for (int k = 0; k < remain; ++k) tail[k] = guessData[i + k];
                MD5HashSIMD4(tail, states, remain);
                for (int k = 0; k < remain; k++) {
                    if (test_set.find(guessData[i + k]) != test_set.end()) local_cracked++;
                }
            }
#endif
            double t_hash1 = MPI_Wtime();
            time_hash += (t_hash1 - t_hash0);

            local_history += local_buf_count;
            local_buf_count = 0;
            local_guesses.clear();
        }
    }

    // ================================================================
    // Phase 5: Hash remaining guesses
    // ================================================================
    if (!local_guesses.empty())
    {
        double t_hash0 = MPI_Wtime();
#if MD5_SERIAL_BASELINE
        for (const string &pw : local_guesses) {
            bit32 state[4];
            MD5Hash(pw, state);
            if (test_set.find(pw) != test_set.end()) local_cracked++;
        }
#else
        bit32 states[4][4];
        const string *guessData = local_guesses.data();
        int total = static_cast<int>(local_guesses.size());
        int i = 0;
        for (; i + 4 <= total; i += 4) {
            MD5HashSIMD4(guessData + i, states, 4);
            for (int k = 0; k < 4; k++) {
                if (test_set.find(guessData[i + k]) != test_set.end()) local_cracked++;
            }
        }
        if (i < total) {
            string tail[4];
            int remain = total - i;
            for (int k = 0; k < remain; ++k) tail[k] = guessData[i + k];
            MD5HashSIMD4(tail, states, remain);
            for (int k = 0; k < remain; k++) {
                if (test_set.find(guessData[i + k]) != test_set.end()) local_cracked++;
            }
        }
#endif
        double t_hash1 = MPI_Wtime();
        time_hash += (t_hash1 - t_hash0);
        local_history += local_buf_count;
    }

    double t_guess_end = 0.0;
    if (rank == 0) {
        t_guess_end = MPI_Wtime();
        time_guess = t_guess_end - t_guess_start;
    }

    // ================================================================
    // Phase 6: Aggregate results
    // ================================================================
    int local_final = local_history;
    int global_total = 0;
    int global_cracked = 0;
    MPI_Reduce(&local_final, &global_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double send_t[3] = {time_hash, time_guess, time_train};
    double recv_t[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(send_t, recv_t, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Guesses generated: " << global_total << endl;
        cout << "Cracked: " << global_cracked << " (rate: "
             << (100.0 * global_cracked / global_total) << "%)" << endl;
        cout << "Guess time:" << recv_t[1] - recv_t[0] << "seconds" << endl; //请不要修改这一行
        cout << "Hash time:" << recv_t[0] << "seconds" << endl;              //请不要修改这一行
        cout << "Train time:" << recv_t[2] << "seconds" << endl;            //请不要修改这一行

        // ---- Correctness verification ----
        // 验证各 rank 工作分配之和正确（总猜测数应在目标值附近）
        int expected = GENERATE_N;
        int delta = global_total - expected;
        double err_pct = 100.0 * abs(delta) / expected;
        cout << "\n=== Correctness Verification ===" << endl;
        cout << "Expected guesses: " << expected << endl;
        cout << "Actual guesses:   " << global_total << endl;
        cout << "Difference:       " << (delta >= 0 ? "+" : "") << delta
             << " (" << err_pct << "%)" << endl;
        if (err_pct < 5.0) {
            cout << "VERDICT: PASS — totals are consistent within 5% tolerance." << endl;
        } else {
            cout << "VERDICT: FAIL — totals differ by more than 5%. Possible work duplication." << endl;
        }

        // 额外检查：多进程不应该产生远多于 GENERATE_N 的猜测（否则说明有重复）
        if (global_total > expected * 2) {
            cout << "WARNING: global_total > 2x expected — likely duplicated work across ranks!" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
