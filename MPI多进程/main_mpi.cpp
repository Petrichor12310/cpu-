#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

#ifndef MD5_SERIAL_BASELINE
#define MD5_SERIAL_BASELINE 0
#endif

// 编译指令如下
// mpic++ main_mpi.cpp train.cpp guessing.cpp md5.cpp -o main_mpi -O2

int main(int argc, char* argv[])
{
    // ---- MPI init ----
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    double time_hash  = 0;   // MD5哈希的时间
    double time_guess = 0;   // 哈希和猜测的总时长
    double time_train = 0;   // 模型训练的总时长
    PriorityQueue q;

    // ================================================================
    // Phase 1: Training (rank 0 only)
    // ================================================================
    if (rank == 0) {
        // ---- MD5 correctness test ----
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
            for (int i1 = 0; i1 < 4; i1 += 1) {
                ss << std::setw(8) << std::setfill('0') << hex << state[i1];
            }
            if (ss.str() != test_hashes[i]) {
                cout << "MD5Hash test failed for " << test_pws[i] << "!" << endl;
                cout << "Expected: " << test_hashes[i] << "\nGot:      " << ss.str() << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        cout << "MD5Hash test passed!" << endl; //请不要修改这一行

        // ---- train ----
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) *
                     microseconds::period::num / microseconds::period::den;
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
    if (rank != 0) {
        model_buf.resize(static_cast<size_t>(buf_size));
    }
    MPI_Bcast(model_buf.data(), buf_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        q.m.deserialize(model_buf);
    }
    // Free the serialization buffer (model data is now in q.m)
    model_buf.clear();
    model_buf.shrink_to_fit();

    // ================================================================
    // Phase 3: Initialize queue (each rank gets a disjoint subset of PTs)
    // ================================================================
    q.init_mpi(rank, nprocs);
    if (rank == 0) cout << "here" << endl;

    // ================================================================
    // Phase 4: Guessing loop (independent per rank, no communication)
    //
    // Each rank works on its own disjoint subset of initial PTs.
    // Since NewPTs() derivatives stay local to the rank that owns the
    // parent PT, the work stays partitioned with zero MPI during the
    // inner loop.  Each rank stops when its LOCAL share is done.
    // ================================================================
    int curr_num  = 0;       // guesses in current buffer
    int history   = 0;       // guesses already hashed and cleared
    auto start = system_clock::now();
    const int GENERATE_N = 10000000;            // global guess limit
    const int LOCAL_N = GENERATE_N / nprocs;    // per-rank share
    bool local_done = false;

    while (q.HasWork() && !local_done)
    {
        q.PopNext();
        q.total_guesses = static_cast<int>(q.guesses.size());

        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;

            if (history + curr_num >= LOCAL_N)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) *
                             microseconds::period::num / microseconds::period::den;
                local_done = true;
                // don't break yet — still need to hash the current buffer below
            }
        }

        // --- Hash batch when buffer is full ---
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
#if MD5_SERIAL_BASELINE
            for (const string &pw : q.guesses)
            {
                bit32 state[4];
                MD5Hash(pw, state);
            }
#else
            bit32 states[4][4];
            const string *guessData = q.guesses.data();
            int total = static_cast<int>(q.guesses.size());
            int i = 0;
            for (; i + 4 <= total; i += 4)
            {
                MD5HashSIMD4(guessData + i, states, 4);
            }
            if (i < total)
            {
                string tail[4];
                int remain = total - i;
                for (int k = 0; k < remain; ++k)
                {
                    tail[k] = guessData[i + k];
                }
                MD5HashSIMD4(tail, states, remain);
            }
#endif
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) *
                         microseconds::period::num / microseconds::period::den;

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    // Record final local guess count
    int local_final = history + curr_num;

    // ================================================================
    // Phase 5: Aggregate results
    // ================================================================
    int global_total = 0;
    MPI_Reduce(&local_final, &global_total, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // Timing: use MAX reduction (wall-clock = slowest rank)
    double send[3] = {time_hash, time_guess, time_train};
    double recv[3] = {0, 0, 0};
    MPI_Reduce(send, recv, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Guesses generated: " << global_total << endl;
        cout << "Guess time:" << recv[1] - recv[0] << "seconds" << endl; //请不要修改这一行
        cout << "Hash time:" << recv[0] << "seconds" << endl;            //请不要修改这一行
        cout << "Train time:" << recv[2] << "seconds" << endl;          //请不要修改这一行
    }

    MPI_Finalize();
    return 0;
}
