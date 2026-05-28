#include "PCFG.h"
#include <pthread.h>
#include <algorithm>
#include <cstdlib>
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

struct PTProbLess
{
    bool operator()(const PT &a, const PT &b) const
    {
        return a.prob < b.prob;
    }
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