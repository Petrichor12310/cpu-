#include "PCFG.h"
#include <pthread.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
using namespace std;

namespace {
struct GuessTask
{
    const segment *seg;
    const string *prefix;
    vector<string> *out;
    size_t base;
    int start;
    int end;
};

int GetDefaultThreadCount()
{
    int max_threads = 1;
    const char *env = std::getenv("GUESS_THREADS");
    if (env != nullptr)
    {
        int value = std::atoi(env);
        if (value > 0)
        {
            max_threads = value;
        }
    }
    else
    {
        unsigned int hc = std::thread::hardware_concurrency();
        max_threads = (hc == 0) ? 4 : static_cast<int>(hc);
        if (max_threads > 8)
        {
            max_threads = 8;
        }
    }

    if (max_threads < 1)
    {
        max_threads = 1;
    }
    return max_threads;
}

int GetThreadCount(int work_items)
{
    if (work_items <= 0)
    {
        return 1;
    }

    int max_threads = GetDefaultThreadCount();
    if (max_threads > work_items)
    {
        max_threads = work_items;
    }
    if (max_threads < 1)
    {
        max_threads = 1;
    }
    return max_threads;
}

int GetQueueThreadCount()
{
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return GetDefaultThreadCount();
#endif
}

int GetQueueCount()
{
    const int kQueueFactor = 4;
    int threads = GetQueueThreadCount();
    if (threads < 1)
    {
        threads = 1;
    }
    int queues = threads * kQueueFactor;
    if (queues < 4)
    {
        queues = 4;
    }
    return queues;
}

int PickQueueIndex(int queue_count)
{
    thread_local std::mt19937 rng(static_cast<unsigned int>(std::random_device{}()));
    std::uniform_int_distribution<int> dist(0, queue_count - 1);
    return dist(rng);
}

void FillGuessRange(const GuessTask &task)
{
    for (int i = task.start; i < task.end; i += 1)
    {
        (*task.out)[task.base + i] = *task.prefix + task.seg->ordered_values[i];
    }
}

void *GuessWorker(void *arg)
{
    GuessTask *task = static_cast<GuessTask *>(arg);
    FillGuessRange(*task);
    return nullptr;
}

void AppendGuesses(const segment *seg, const string &prefix, int count, vector<string> &guesses, int &total_guesses)
{
    if (count <= 0)
    {
        return;
    }

    int thread_count = 1;
#if defined(PCFG_USE_PTHREAD)
    thread_count = GetThreadCount(count);
#elif defined(_OPENMP)
    thread_count = omp_get_max_threads();
#endif
    const int kMinParallelWork = 4096;
    if (count < kMinParallelWork || thread_count <= 1)
    {
        size_t base = guesses.size();
        guesses.resize(base + static_cast<size_t>(count));
        FillGuessRange(GuessTask{seg, &prefix, &guesses, base, 0, count});
        total_guesses = static_cast<int>(guesses.size());
        return;
    }

    size_t base = guesses.size();
    guesses.resize(base + static_cast<size_t>(count));

#if defined(PCFG_USE_PTHREAD)
    if (thread_count <= 1)
    {
        FillGuessRange(GuessTask{seg, &prefix, &guesses, base, 0, count});
    }
    else
    {
        int chunk = (count + thread_count - 1) / thread_count;
        int worker_count = thread_count - 1;
        vector<pthread_t> threads(static_cast<size_t>(worker_count));
        vector<GuessTask> tasks(static_cast<size_t>(thread_count));
        int launched = 0;

        for (int t = 0; t < worker_count; t += 1)
        {
            int start = t * chunk;
            int end = start + chunk;
            if (end > count)
            {
                end = count;
            }
            if (start >= end)
            {
                break;
            }
            tasks[t] = GuessTask{seg, &prefix, &guesses, base, start, end};
            if (pthread_create(&threads[t], nullptr, GuessWorker, &tasks[t]) == 0)
            {
                launched += 1;
            }
        }

        int main_start = worker_count * chunk;
        if (main_start < count)
        {
            tasks[worker_count] = GuessTask{seg, &prefix, &guesses, base, main_start, count};
            FillGuessRange(tasks[worker_count]);
        }

        for (int t = 0; t < launched; t += 1)
        {
            pthread_join(threads[t], nullptr);
        }
    }
#elif defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; i += 1)
    {
        guesses[base + static_cast<size_t>(i)] = prefix + seg->ordered_values[i];
    }
#else
    for (int i = 0; i < count; i += 1)
    {
        guesses[base + static_cast<size_t>(i)] = prefix + seg->ordered_values[i];
    }
#endif

    total_guesses = static_cast<int>(guesses.size());
}
} // namespace

MultiQueue::MultiQueue() : size(0)
{
}

void MultiQueue::Init(int queue_count)
{
    if (queue_count < 1)
    {
        queue_count = 1;
    }

    buckets.clear();
    buckets.reserve(static_cast<size_t>(queue_count));
    for (int i = 0; i < queue_count; i += 1)
    {
        buckets.emplace_back(std::make_unique<Bucket>());
    }
    size.store(0, std::memory_order_relaxed);
}

void MultiQueue::Clear()
{
    for (auto &bucket_ptr : buckets)
    {
        if (bucket_ptr)
        {
            bucket_ptr->heap.clear();
        }
    }
    size.store(0, std::memory_order_relaxed);
}

void MultiQueue::Push(const PT &pt)
{
    if (buckets.empty())
    {
        Init(1);
    }

    int queue_count = static_cast<int>(buckets.size());
    for (;;)
    {
        int idx = PickQueueIndex(queue_count);
        Bucket &bucket = *buckets[static_cast<size_t>(idx)];
        if (bucket.lock.try_lock())
        {
            bucket.heap.push_back(pt);
            std::push_heap(bucket.heap.begin(), bucket.heap.end(), PTProbLess());
            bucket.lock.unlock();
            size.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

bool MultiQueue::PopMax(PT &out)
{
    if (size.load(std::memory_order_relaxed) == 0)
    {
        return false;
    }

    int queue_count = static_cast<int>(buckets.size());
    if (queue_count == 0)
    {
        return false;
    }

    const int max_attempts = queue_count * 2;
    for (int attempt = 0; attempt < max_attempts; attempt += 1)
    {
        int i = PickQueueIndex(queue_count);
        int j = PickQueueIndex(queue_count);
        if (i == j)
        {
            continue;
        }

        int a = (i < j) ? i : j;
        int b = (i < j) ? j : i;
        Bucket &bucket_a = *buckets[static_cast<size_t>(a)];
        Bucket &bucket_b = *buckets[static_cast<size_t>(b)];
        if (!bucket_a.lock.try_lock())
        {
            continue;
        }
        if (!bucket_b.lock.try_lock())
        {
            bucket_a.lock.unlock();
            continue;
        }

        bool has_a = !bucket_a.heap.empty();
        bool has_b = !bucket_b.heap.empty();
        if (!has_a && !has_b)
        {
            bucket_b.lock.unlock();
            bucket_a.lock.unlock();
            continue;
        }

        Bucket *pick = nullptr;
        if (!has_b)
        {
            pick = &bucket_a;
        }
        else if (!has_a)
        {
            pick = &bucket_b;
        }
        else
        {
            pick = (bucket_a.heap.front().prob >= bucket_b.heap.front().prob) ? &bucket_a : &bucket_b;
        }

        std::pop_heap(pick->heap.begin(), pick->heap.end(), PTProbLess());
        out = pick->heap.back();
        pick->heap.pop_back();

        bucket_b.lock.unlock();
        bucket_a.lock.unlock();
        size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    for (auto &bucket_ptr : buckets)
    {
        if (!bucket_ptr)
        {
            continue;
        }
        Bucket &bucket = *bucket_ptr;
        if (!bucket.lock.try_lock())
        {
            continue;
        }
        if (bucket.heap.empty())
        {
            bucket.lock.unlock();
            continue;
        }
        std::pop_heap(bucket.heap.begin(), bucket.heap.end(), PTProbLess());
        out = bucket.heap.back();
        bucket.heap.pop_back();
        bucket.lock.unlock();
        size.fetch_sub(1, std::memory_order_relaxed);
        return true;
    }

    return false;
}

bool MultiQueue::Empty() const
{
    return size.load(std::memory_order_relaxed) == 0;
}

size_t MultiQueue::Size() const
{
    return size.load(std::memory_order_relaxed);
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    priority.Init(GetQueueCount());
	for (PT pt : m.ordered_pts)
	{
		for (segment seg : pt.content)
		{
			if (seg.type == 1)
			{
				pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
			}
			if (seg.type == 2)
			{
				pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
			}
			if (seg.type == 3)
			{
				pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
			}
		}
		pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
		CalProb(pt);
        priority.Push(pt);
	}
	// cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{
    PT current;
    if (!priority.PopMax(current))
    {
        return;
    }

    Generate(current);

    vector<PT> new_pts = current.NewPTs();
	for (PT pt : new_pts)
	{
		CalProb(pt);
        priority.Push(pt);
	}
}

bool PriorityQueue::HasWork() const
{
    return !priority.Empty();
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        string prefix;
        AppendGuesses(a, prefix, pt.max_indices[0], guesses, total_guesses);
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }
        
        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        AppendGuesses(a, guess, pt.max_indices[pt.content.size() - 1], guesses, total_guesses);
    }
}

// ============================================================
// MPI Serialization helpers (binary protocol)
// ============================================================
namespace {

void write_i32(std::vector<char>& buf, int32_t v) {
    buf.insert(buf.end(), reinterpret_cast<const char*>(&v),
               reinterpret_cast<const char*>(&v) + sizeof(v));
}
void write_f32(std::vector<char>& buf, float v) {
    buf.insert(buf.end(), reinterpret_cast<const char*>(&v),
               reinterpret_cast<const char*>(&v) + sizeof(v));
}
void write_str(std::vector<char>& buf, const std::string& s) {
    write_i32(buf, static_cast<int32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

int32_t read_i32(const char*& p) {
    int32_t v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}
float read_f32(const char*& p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    return v;
}
std::string read_str(const char*& p) {
    int32_t len = read_i32(p);
    std::string s(p, static_cast<size_t>(len));
    p += len;
    return s;
}

} // namespace

// ============================================================
// Serialize the full model into a byte buffer (called by rank 0)
// ============================================================
std::vector<char> model::serialize() const
{
    std::vector<char> buf;

    // --- scalars ---
    write_i32(buf, total_preterm);
    write_i32(buf, preterm_id);
    write_i32(buf, letters_id);
    write_i32(buf, digits_id);
    write_i32(buf, symbols_id);

    // --- helper: serialize one segment ---
    auto write_segment = [&](const segment& seg) {
        write_i32(buf, seg.type);
        write_i32(buf, seg.length);
        // ordered_values
        write_i32(buf, static_cast<int32_t>(seg.ordered_values.size()));
        for (const std::string& v : seg.ordered_values) write_str(buf, v);
        // ordered_freqs
        write_i32(buf, static_cast<int32_t>(seg.ordered_freqs.size()));
        for (int f : seg.ordered_freqs) write_i32(buf, f);
        // total_freq
        write_i32(buf, seg.total_freq);
        // values map  (string -> id)
        write_i32(buf, static_cast<int32_t>(seg.values.size()));
        for (const auto& kv : seg.values) {
            write_str(buf, kv.first);
            write_i32(buf, kv.second);
        }
        // freqs map  (id -> freq)
        write_i32(buf, static_cast<int32_t>(seg.freqs.size()));
        for (const auto& kv : seg.freqs) {
            write_i32(buf, kv.first);
            write_i32(buf, kv.second);
        }
    };

    // --- helper: serialize a PT (structure only, no deep segment data) ---
    auto write_pt = [&](const PT& pt) {
        // content
        write_i32(buf, static_cast<int32_t>(pt.content.size()));
        for (const segment& seg : pt.content) {
            write_i32(buf, seg.type);
            write_i32(buf, seg.length);
        }
        // pivot
        write_i32(buf, pt.pivot);
        // curr_indices
        write_i32(buf, static_cast<int32_t>(pt.curr_indices.size()));
        for (int idx : pt.curr_indices) write_i32(buf, idx);
        // max_indices
        write_i32(buf, static_cast<int32_t>(pt.max_indices.size()));
        for (int mx : pt.max_indices) write_i32(buf, mx);
        // preterm_prob, prob
        write_f32(buf, pt.preterm_prob);
        write_f32(buf, pt.prob);
    };

    // --- letters ---
    write_i32(buf, static_cast<int32_t>(letters.size()));
    for (const segment& seg : letters) write_segment(seg);
    // --- digits ---
    write_i32(buf, static_cast<int32_t>(digits.size()));
    for (const segment& seg : digits) write_segment(seg);
    // --- symbols ---
    write_i32(buf, static_cast<int32_t>(symbols.size()));
    for (const segment& seg : symbols) write_segment(seg);

    // --- preterminals ---
    write_i32(buf, static_cast<int32_t>(preterminals.size()));
    for (const PT& pt : preterminals) write_pt(pt);

    // --- ordered_pts ---
    write_i32(buf, static_cast<int32_t>(ordered_pts.size()));
    for (const PT& pt : ordered_pts) write_pt(pt);

    // --- frequency maps ---
    auto write_imap = [&](const std::unordered_map<int, int>& m) {
        write_i32(buf, static_cast<int32_t>(m.size()));
        for (const auto& kv : m) {
            write_i32(buf, kv.first);
            write_i32(buf, kv.second);
        }
    };
    write_imap(preterm_freq);
    write_imap(letters_freq);
    write_imap(digits_freq);
    write_imap(symbols_freq);

    return buf;
}

// ============================================================
// Deserialize the model from a byte buffer (called by all ranks)
// ============================================================
void model::deserialize(const std::vector<char>& buffer)
{
    const char* p = buffer.data();
    const char* end = p + buffer.size();
    (void)end;  // silence unused warning; we trust the buffer

    // --- scalars ---
    total_preterm = read_i32(p);
    preterm_id    = read_i32(p);
    letters_id    = read_i32(p);
    digits_id     = read_i32(p);
    symbols_id    = read_i32(p);

    // --- helper: deserialize one segment ---
    auto read_segment = [&]() -> segment {
        int type = read_i32(p);
        int len  = read_i32(p);
        segment seg(type, len);
        // ordered_values
        int nv = read_i32(p);
        seg.ordered_values.reserve(static_cast<size_t>(nv));
        for (int i = 0; i < nv; ++i) seg.ordered_values.push_back(read_str(p));
        // ordered_freqs
        int nf = read_i32(p);
        seg.ordered_freqs.reserve(static_cast<size_t>(nf));
        for (int i = 0; i < nf; ++i) seg.ordered_freqs.push_back(read_i32(p));
        // total_freq
        seg.total_freq = read_i32(p);
        // values map
        int nvm = read_i32(p);
        for (int i = 0; i < nvm; ++i) {
            std::string key = read_str(p);
            int id = read_i32(p);
            seg.values[key] = id;
        }
        // freqs map
        int nfm = read_i32(p);
        for (int i = 0; i < nfm; ++i) {
            int id = read_i32(p);
            int freq = read_i32(p);
            seg.freqs[id] = freq;
        }
        return seg;
    };

    // --- letters ---
    int nl = read_i32(p);
    letters.clear();
    letters.reserve(static_cast<size_t>(nl));
    for (int i = 0; i < nl; ++i) letters.push_back(read_segment());

    // --- digits ---
    int nd = read_i32(p);
    digits.clear();
    digits.reserve(static_cast<size_t>(nd));
    for (int i = 0; i < nd; ++i) digits.push_back(read_segment());

    // --- symbols ---
    int ns = read_i32(p);
    symbols.clear();
    symbols.reserve(static_cast<size_t>(ns));
    for (int i = 0; i < ns; ++i) symbols.push_back(read_segment());

    // --- helper: deserialize a PT ---
    auto read_pt = [&]() -> PT {
        PT pt;
        int nc = read_i32(p);
        for (int i = 0; i < nc; ++i) {
            int type = read_i32(p);
            int len  = read_i32(p);
            pt.content.emplace_back(type, len);
        }
        pt.pivot = read_i32(p);
        int nci = read_i32(p);
        pt.curr_indices.reserve(static_cast<size_t>(nci));
        for (int i = 0; i < nci; ++i) pt.curr_indices.push_back(read_i32(p));
        int nmi = read_i32(p);
        pt.max_indices.reserve(static_cast<size_t>(nmi));
        for (int i = 0; i < nmi; ++i) pt.max_indices.push_back(read_i32(p));
        pt.preterm_prob = read_f32(p);
        pt.prob = read_f32(p);
        return pt;
    };

    // --- preterminals ---
    int npt = read_i32(p);
    preterminals.clear();
    preterminals.reserve(static_cast<size_t>(npt));
    for (int i = 0; i < npt; ++i) preterminals.push_back(read_pt());

    // --- ordered_pts ---
    int nop = read_i32(p);
    ordered_pts.clear();
    ordered_pts.reserve(static_cast<size_t>(nop));
    for (int i = 0; i < nop; ++i) ordered_pts.push_back(read_pt());

    // --- frequency maps ---
    auto read_imap = [&](std::unordered_map<int, int>& m) {
        m.clear();
        int n = read_i32(p);
        for (int i = 0; i < n; ++i) {
            int key = read_i32(p);
            int val = read_i32(p);
            m[key] = val;
        }
    };
    read_imap(preterm_freq);
    read_imap(letters_freq);
    read_imap(digits_freq);
    read_imap(symbols_freq);
}

// ============================================================
// MPI-aware init: each rank seeds its queue with a round-robin
// subset of ordered_pts so every PT is owned by exactly one rank.
// ============================================================
void PriorityQueue::init_mpi(int rank, int nprocs)
{
    priority.Init(GetQueueCount());
    int total = static_cast<int>(m.ordered_pts.size());
    for (int i = rank; i < total; i += nprocs)
    {
        PT pt = m.ordered_pts[i];
        for (const segment& seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.push_back(
                    static_cast<int>(m.letters[static_cast<size_t>(m.FindLetter(seg))]
                                         .ordered_values.size()));
            else if (seg.type == 2)
                pt.max_indices.push_back(
                    static_cast<int>(m.digits[static_cast<size_t>(m.FindDigit(seg))]
                                         .ordered_values.size()));
            else if (seg.type == 3)
                pt.max_indices.push_back(
                    static_cast<int>(m.symbols[static_cast<size_t>(m.FindSymbol(seg))]
                                         .ordered_values.size()));
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.Push(pt);
    }
}

// ============================================================
// Simple priority queue (no MultiQueue, no threads)
// Used by MPI basic: rank 0 only
// ============================================================
void PriorityQueue::init_simple()
{
    // 清理旧队列
    while (!simple_heap.empty()) simple_heap.pop();

    for (PT pt : m.ordered_pts)
    {
        for (const segment& seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.push_back(
                    static_cast<int>(m.letters[static_cast<size_t>(m.FindLetter(seg))]
                                         .ordered_values.size()));
            else if (seg.type == 2)
                pt.max_indices.push_back(
                    static_cast<int>(m.digits[static_cast<size_t>(m.FindDigit(seg))]
                                         .ordered_values.size()));
            else if (seg.type == 3)
                pt.max_indices.push_back(
                    static_cast<int>(m.symbols[static_cast<size_t>(m.FindSymbol(seg))]
                                         .ordered_values.size()));
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        simple_heap.push(pt);
    }
}

bool PriorityQueue::PopMaxSimple(PT &out)
{
    if (simple_heap.empty()) return false;
    out = simple_heap.top();
    simple_heap.pop();
    return true;
}

void PriorityQueue::PushSimple(const PT &pt)
{
    simple_heap.push(pt);
}

bool PriorityQueue::EmptySimple() const
{
    return simple_heap.empty();
}

// ============================================================
// GenerateChunk: expand values[start..end) with prefix
// Used by all MPI ranks to process their share of a segment
// ============================================================
void PriorityQueue::GenerateChunk(const std::string& prefix, const segment* seg,
                                   int start, int end,
                                   std::vector<std::string>& out, int& out_count)
{
    if (start >= end) return;
    int count = end - start;
    size_t base = out.size();
    out.resize(base + static_cast<size_t>(count));
    for (int i = 0; i < count; i += 1)
    {
        out[base + static_cast<size_t>(i)] = prefix + seg->ordered_values[start + i];
    }
    out_count = static_cast<int>(out.size());
}

// ============================================================
// PT Serialization for MPI transfer
// ============================================================

// Write helpers (append to buffer)
static void pt_write_i32(std::vector<char>& b, int32_t v) {
    b.insert(b.end(), reinterpret_cast<const char*>(&v),
             reinterpret_cast<const char*>(&v) + sizeof(v));
}
static void pt_write_f32(std::vector<char>& b, float v) {
    b.insert(b.end(), reinterpret_cast<const char*>(&v),
             reinterpret_cast<const char*>(&v) + sizeof(v));
}

// Read helpers (advance pointer)
static int32_t pt_read_i32(const char*& p) {
    int32_t v; std::memcpy(&v, p, sizeof(v)); p += sizeof(v); return v;
}
static float pt_read_f32(const char*& p) {
    float v; std::memcpy(&v, p, sizeof(v)); p += sizeof(v); return v;
}

// Serialize a single PT into a flat buffer
void PT_Serialize(const PT& pt, std::vector<char>& buf)
{
    // content: [count] + [type, len] per segment
    pt_write_i32(buf, static_cast<int32_t>(pt.content.size()));
    for (const segment& seg : pt.content) {
        pt_write_i32(buf, seg.type);
        pt_write_i32(buf, seg.length);
    }
    // pivot
    pt_write_i32(buf, pt.pivot);
    // curr_indices
    pt_write_i32(buf, static_cast<int32_t>(pt.curr_indices.size()));
    for (int v : pt.curr_indices) pt_write_i32(buf, v);
    // max_indices
    pt_write_i32(buf, static_cast<int32_t>(pt.max_indices.size()));
    for (int v : pt.max_indices) pt_write_i32(buf, v);
    // probabilities
    pt_write_f32(buf, pt.preterm_prob);
    pt_write_f32(buf, pt.prob);
}

// Deserialize a single PT from a buffer pointer (advances p)
PT PT_Deserialize(const char*& p)
{
    PT pt;
    int nc = pt_read_i32(p);
    for (int i = 0; i < nc; ++i) {
        int type = pt_read_i32(p);
        int len  = pt_read_i32(p);
        pt.content.emplace_back(type, len);
    }
    pt.pivot = pt_read_i32(p);
    int nci = pt_read_i32(p);
    for (int i = 0; i < nci; ++i) pt.curr_indices.push_back(pt_read_i32(p));
    int nmi = pt_read_i32(p);
    for (int i = 0; i < nmi; ++i) pt.max_indices.push_back(pt_read_i32(p));
    pt.preterm_prob = pt_read_f32(p);
    pt.prob        = pt_read_f32(p);
    return pt;
}

// ============================================================
// GenerateFull: expand ALL values of a PT into a local buffer
// Used by MPI advanced: each rank fully processes its assigned PT
// ============================================================
void PriorityQueue::GenerateLocal(PT pt, std::vector<std::string>& out, int& out_count)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // Single-segment PT: expand all values directly
        segment* a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int N = pt.max_indices[0];
        size_t base = out.size();
        out.resize(base + static_cast<size_t>(N));
        for (int i = 0; i < N; ++i)
            out[base + i] = a->ordered_values[i];
        out_count = static_cast<int>(out.size());
    }
    else
    {
        // Multi-segment PT: build prefix, then expand last segment
        std::string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == static_cast<int>(pt.content.size()) - 1) break;
        }

        int last = static_cast<int>(pt.content.size()) - 1;
        segment* a = nullptr;
        if (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[last])];

        int N = pt.max_indices[last];
        size_t base = out.size();
        out.resize(base + static_cast<size_t>(N));
        for (int i = 0; i < N; ++i)
            out[base + i] = prefix + a->ordered_values[i];
        out_count = static_cast<int>(out.size());
    }
}