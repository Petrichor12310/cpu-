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
// MPI 进阶要求：PT 层面的批量并行
//
// 每轮从优先队列中一次性取出 batch_size 个 PT，分发给各 rank。
// 每个 rank 独立完成自己 PT 的全量 Generate + NewPTs。
// 所有 rank 完成后，收集所有新 PT 放回队列，下一轮。
// 停止条件：MPI_Allreduce 全局计数（每 ~100k/rank 同步一次）。
//
// 编译：
//   mpic++ main_mpi_advanced.cpp train.cpp guessing.cpp md5.cpp -o main_mpi_advanced -O2
// ===================================================================

#ifndef MD5_SERIAL_BASELINE
#define MD5_SERIAL_BASELINE 0
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double time_hash  = 0.0;
    double time_guess = 0.0;
    double time_train = 0.0;
    PriorityQueue q;

    // ================================================================
    // Phase 1: Training (rank 0 only)
    // ================================================================
    if (rank == 0) {
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
    // Phase 2: Broadcast model
    // ================================================================
    vector<char> model_buf;
    int buf_size = 0;
    if (rank == 0) { model_buf = q.m.serialize(); buf_size = static_cast<int>(model_buf.size()); }
    MPI_Bcast(&buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) model_buf.resize(static_cast<size_t>(buf_size));
    MPI_Bcast(model_buf.data(), buf_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) q.m.deserialize(model_buf);
    model_buf.clear(); model_buf.shrink_to_fit();

    // ================================================================
    // Phase 2b: Load test set (each rank independently)
    // ================================================================
    unordered_set<string> test_set;
    { ifstream f("/guessdata/Rockyou-singleLined-full.txt");
      int n = 0; string s;
      while (f >> s) { test_set.insert(s); if (++n >= 1000000) break; } }
    if (rank == 0) cout << "Test set loaded: " << test_set.size() << " passwords" << endl;

    // ================================================================
    // Phase 3: Rank 0 init simple queue
    // ================================================================
    if (rank == 0) { q.init_simple(); cout << "here" << endl; }

    // Per-rank state
    vector<string> local_guesses;
    int local_buf_count = 0;
    int local_history   = 0;
    int local_cracked   = 0;
    const int GENERATE_N = 10000000;

    bool stop_all = false;

    double t_guess_start = 0.0;
    if (rank == 0) t_guess_start = MPI_Wtime();

    // ================================================================
    // Phase 4: Batch PT processing loop
    // ================================================================
    int round_count = 0;
    const int SYNC_INTERVAL = 50;  // Allreduce only every 50 rounds to cut overhead
    while (!stop_all)
    {
        // ---- Step A: Rank 0 pops a batch of PTs, serializes them ----
        int batch_count = 0;
        vector<char> pt_buf;

        if (rank == 0) {
            if (!q.EmptySimple() && !stop_all) {
                vector<PT> batch;
                for (int i = 0; i < nprocs; ++i) {
                    if (q.EmptySimple()) break;
                    PT pt; q.PopMaxSimple(pt);
                    batch.push_back(pt);
                }
                batch_count = static_cast<int>(batch.size());
                for (const PT& pt : batch) PT_Serialize(pt, pt_buf);
            }
        }

        // ---- Step B: Bcast batch_count; 0 = stop ----
        MPI_Bcast(&batch_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (batch_count == 0) { stop_all = true; break; }

        // ---- Step C: Bcast serialized PTs ----
        int pt_buf_size = static_cast<int>(pt_buf.size());
        MPI_Bcast(&pt_buf_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) pt_buf.resize(static_cast<size_t>(pt_buf_size));
        MPI_Bcast(pt_buf.data(), pt_buf_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        // ---- Step D: Each rank deserializes its PT and generates fully ----
        vector<char> local_new_buf;

        if (rank < batch_count) {
            const char* p = pt_buf.data();
            for (int i = 0; i < rank; ++i) { PT_Deserialize(p); }
            PT my_pt = PT_Deserialize(p);

            q.GenerateLocal(my_pt, local_guesses, local_buf_count);

            vector<PT> new_pts = my_pt.NewPTs();
            for (PT& npt : new_pts) {
                q.CalProb(npt);
                PT_Serialize(npt, local_new_buf);
            }
        }

        // ---- Step E: Gatherv new PTs back to Rank 0 ----
        int local_new_sz = static_cast<int>(local_new_buf.size());
        vector<int> all_sizes(nprocs, 0);
        MPI_Gather(&local_new_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        vector<int> displs(nprocs, 0);
        int total_new_sz = 0;
        if (rank == 0) {
            for (int i = 0; i < nprocs; ++i) { displs[i] = total_new_sz; total_new_sz += all_sizes[i]; }
        }
        vector<char> all_new_data(total_new_sz);
        MPI_Gatherv(local_new_buf.data(), local_new_sz, MPI_BYTE,
                    all_new_data.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const char* p = all_new_data.data();
            const char* e = p + total_new_sz;
            while (p < e) { PT npt = PT_Deserialize(p); q.PushSimple(npt); }
        }

        // ---- Step F: Periodic global stop check (Allreduce every SYNC_INTERVAL rounds) ----
        round_count++;
        if (round_count % SYNC_INTERVAL == 0) {
            int local_sofar = local_history + local_buf_count;
            int global_sofar = 0;
            MPI_Allreduce(&local_sofar, &global_sofar, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (global_sofar >= GENERATE_N) stop_all = true;
        }

        // ---- Step G: Hash when local buffer is full ----
        if (local_buf_count > 1000000) {
            double t0 = MPI_Wtime();
            bit32 states[4][4];
            const string *gd = local_guesses.data();
            int tot = static_cast<int>(local_guesses.size());
            int i = 0;
            for (; i + 4 <= tot; i += 4) {
                MD5HashSIMD4(gd + i, states, 4);
                for (int k = 0; k < 4; k++)
                    if (test_set.find(gd[i+k]) != test_set.end()) local_cracked++;
            }
            if (i < tot) {
                string tail[4]; int rem = tot - i;
                for (int k = 0; k < rem; ++k) tail[k] = gd[i+k];
                MD5HashSIMD4(tail, states, rem);
                for (int k = 0; k < rem; k++)
                    if (test_set.find(gd[i+k]) != test_set.end()) local_cracked++;
            }
            time_hash += (MPI_Wtime() - t0);
            local_history += local_buf_count;
            local_buf_count = 0;
            local_guesses.clear();
        }
    }

    // ================================================================
    // Phase 5: Hash remaining guesses
    // ================================================================
    if (!local_guesses.empty()) {
        double t0 = MPI_Wtime();
        bit32 states[4][4];
        const string *gd = local_guesses.data();
        int tot = static_cast<int>(local_guesses.size());
        int i = 0;
        for (; i + 4 <= tot; i += 4) {
            MD5HashSIMD4(gd + i, states, 4);
            for (int k = 0; k < 4; k++)
                if (test_set.find(gd[i+k]) != test_set.end()) local_cracked++;
        }
        if (i < tot) {
            string tail[4]; int rem = tot - i;
            for (int k = 0; k < rem; ++k) tail[k] = gd[i+k];
            MD5HashSIMD4(tail, states, rem);
            for (int k = 0; k < rem; k++)
                if (test_set.find(gd[i+k]) != test_set.end()) local_cracked++;
        }
        time_hash += (MPI_Wtime() - t0);
        local_history += local_buf_count;
    }

    // ================================================================
    // Phase 6: Aggregate results
    // ================================================================
    int local_final = local_history;
    int global_total = 0, global_cracked = 0;
    MPI_Reduce(&local_final,  &global_total,  1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &global_cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double t_guess_end = 0.0;
    if (rank == 0) { t_guess_end = MPI_Wtime(); time_guess = t_guess_end - t_guess_start; }

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

        int expected = GENERATE_N;
        int delta = global_total - expected;
        double err_pct = 100.0 * abs(delta) / expected;
        cout << "\n=== Correctness Verification ===" << endl;
        cout << "Expected guesses: " << expected << endl;
        cout << "Actual guesses:   " << global_total << endl;
        cout << "Difference:       " << (delta >= 0 ? "+" : "") << delta
             << " (" << err_pct << "%)" << endl;
        cout << "VERDICT: " << (err_pct < 5.0 ? "PASS" : "FAIL") << endl;
    }

    MPI_Finalize();
    return 0;
}
