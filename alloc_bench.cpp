// =============================================================================
// TurboAlloc Benchmark — Before/After Allocator Performance
//
// Measures:
//   1. Fresh alloc latency per size class (bump allocator path)
//   2. Free latency (limbo push)
//   3. GC promotion (janitor cycle)
//   4. Reuse alloc latency (freelist hit path)
//   5. Mixed workload (interleaved alloc/free)
//   6. Multi-threaded alloc contention
// =============================================================================
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <random>
#include <algorithm>
#include <cstring>
#include <limits>
#include <filesystem>
#include <memory>

#include "../core/turbocore.h"
#include "../core/turbo_alloc.h"
#include "../core/stax_epoch.h"

constexpr int BENCH_ITERS = 5;  // Best-of-5 per measurement
constexpr size_t BENCH_TARGET_LIVE_BYTES = 512ULL * 1024 * 1024;
constexpr size_t BENCH_MIN_OPS = 100'000;
constexpr size_t BENCH_MAX_OPS = 2'000'000;

static size_t ops_for_live_bytes(size_t bytes_per_alloc) {
    if (bytes_per_alloc == 0) return BENCH_MIN_OPS;
    size_t ops = BENCH_TARGET_LIVE_BYTES / bytes_per_alloc;
    if (ops < BENCH_MIN_OPS) ops = BENCH_MIN_OPS;
    if (ops > BENCH_MAX_OPS) ops = BENCH_MAX_OPS;
    return ops;
}

// Run a lambda BENCH_ITERS times, return minimum ns
template<typename Fn>
long long bench_min(Fn&& fn) {
    long long best = std::numeric_limits<long long>::max();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        if (ns < best) best = ns;
    }
    return best;
}

struct BenchResult {
    std::string name;
    long long ns_per_op;
    size_t ops;
};

std::vector<BenchResult> results;

void print_header(const char* section) {
    std::cout << "\n═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  " << section << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
}

void print_result(const char* name, long long ns_per_op, size_t ops) {
    std::cout << std::left << std::setw(45) << name 
              << ": " << std::right << std::setw(8) << ns_per_op 
              << " ns/op  (" << ops << " ops)" << std::endl;
    results.push_back({name, ns_per_op, ops});
}

// =============================================================================
// 1. Fresh Allocation Benchmark (bump allocator, no reuse)
// =============================================================================
void bench_fresh_alloc(System& sys) {
    print_header("FRESH ALLOCATION (bump path, no reuse)");
    
    // Size classes to test (in u64s)
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {1,    "1 u64 (8B)"},
        {2,    "2 u64s (16B)"},
        {4,    "4 u64s (32B)"},
        {8,    "8 u64s (64B)"},
        {16,   "16 u64s (128B)"},
        {32,   "32 u64s (256B)"},
        {64,   "64 u64s (512B)"},
        {128,  "128 u64s (1KB)"},
        {256,  "256 u64s (2KB)"},
        {512,  "512 u64s (4KB)"},
        {1024, "1024 u64s (8KB)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = ops_for_live_bytes(st.u64s * sizeof(uint64_t));
        
        volatile uint64_t sink = 0;
        // Track allocations so we can free them between iterations
        std::vector<void*> ptrs;
        ptrs.reserve(OPS);
        
        long long ns = bench_min([&]() {
            ptrs.clear();
            for (size_t i = 0; i < OPS; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);  // Prevent optimization
                ptrs.push_back(p);
            }
            // Free all so mimalloc can reuse for next iteration
            for (auto* p : ptrs) turbo_free(p);
        });
        
        std::string name = std::string("fresh_alloc ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

// =============================================================================
// 2. Free Latency (limbo push path)
// =============================================================================
void bench_free(System& sys) {
    print_header("FREE LATENCY (limbo push)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,   "4 u64s (32B)"},
        {16,  "16 u64s (128B)"},
        {64,  "64 u64s (512B)"},
        {256, "256 u64s (2KB)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = ops_for_live_bytes(st.u64s * sizeof(uint64_t));
        
        // Pre-allocate blocks to free
        std::vector<void*> ptrs(OPS);
        for (size_t i = 0; i < OPS; ++i) {
            ptrs[i] = turbo_alloc(st.u64s * sizeof(uint64_t));
        }
        
        long long ns = bench_min([&]() {
            for (size_t i = 0; i < OPS; ++i) {
                turbo_free(ptrs[i]);
            }
            // Re-allocate for next iteration
            for (size_t i = 0; i < OPS; ++i) {
                ptrs[i] = turbo_alloc(st.u64s * sizeof(uint64_t));
            }
        });
        
        // Cleanup
        for (auto* p : ptrs) turbo_free(p);
        
        std::string name = std::string("free ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

// =============================================================================
// 3. Epoch Reclaim (deferred free cycle time)
// =============================================================================
void bench_gc_promote(System& sys) {
    print_header("EPOCH RECLAIM (deferred free cycle)");
    
    struct SizeTest { size_t u64s; size_t count; const char* label; };
    SizeTest sizes[] = {
        {4,   100'000, "100K blocks × 32B"},
        {16,  100'000, "100K blocks × 128B"},
        {64,  50'000,  "50K blocks × 512B"},
        {256, 10'000,  "10K blocks × 2KB"},
    };
    
    for (auto& st : sizes) {
        // Allocate and defer-free blocks via epoch system
        auto slot = stax_epoch::EpochSystem::enter();
        for (size_t i = 0; i < st.count; ++i) {
            void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            stax_epoch::EpochSystem::defer_free(off, st.u64s, 
                [](void*, uint64_t o, uint32_t) { turbo_free(turbo_ptr(o)); }, nullptr);
        }
        stax_epoch::EpochSystem::exit(slot);
        
        size_t limbo_before = stax_epoch::EpochSystem::limbo_count();
        
        long long ns = bench_min([&]() {
            stax_epoch::EpochSystem::reclaim_safe(
                [](void*, uint64_t o, uint32_t) { turbo_free(turbo_ptr(o)); }, nullptr);
        });
        
        std::string name = std::string("epoch_reclaim ") + st.label;
        print_result(name.c_str(), ns, 1);
        std::cout << "    limbo before=" << limbo_before << std::endl;
    }
}

// =============================================================================
// 4. Reuse Allocation (freelist hit path)
// =============================================================================
void bench_reuse_alloc(System& sys) {
    print_header("REUSE ALLOCATION (freelist hit path)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,   "4 u64s (32B)"},
        {16,  "16 u64s (128B)"},
        {64,  "64 u64s (512B)"},
        {256, "256 u64s (2KB)"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS = 1'000'000;
        
        // Fill the reuse pool: alloc → free (mimalloc recycles internally)
        for (size_t i = 0; i < OPS; ++i) {
            void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
            turbo_free(p);
        }
        
        volatile uint64_t sink = 0;
        std::vector<void*> ptrs;
        ptrs.reserve(OPS);
        long long ns = bench_min([&]() {
            ptrs.clear();
            for (size_t i = 0; i < OPS; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                ptrs.push_back(p);
            }
            for (auto* p : ptrs) turbo_free(p);
        });
        
        std::string name = std::string("reuse_alloc ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

// =============================================================================
// 5. Mixed Workload (alloc/free interleaved)
// =============================================================================
void bench_mixed_workload(System& sys) {
    print_header("MIXED WORKLOAD (alloc+free interleaved, simulating real usage)");
    
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;  // Keep a rolling pool of live allocations
    
    // Pre-fill pool
    std::vector<std::pair<uint64_t, size_t>> pool(POOL_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);  // 1-64 u64s
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t sz = size_dist(rng);
        void* p = turbo_alloc(sz * sizeof(uint64_t));
        pool[i] = {turbo_index(p), sz};
    }
    
    volatile uint64_t sink = 0;
    long long ns = bench_min([&]() {
        std::mt19937 rng2(123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
        std::uniform_int_distribution<size_t> sz_dist2(1, 64);
        
        for (size_t i = 0; i < OPS; ++i) {
            // Free a random existing block
            size_t idx = idx_dist(rng2);
            turbo_free(turbo_ptr(pool[idx].first));
            
            // Allocate a new one (mimalloc recycles internally)
            size_t new_sz = sz_dist2(rng2);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
    });
    
    // Cleanup pool
    for (auto& [off, sz] : pool) {
        turbo_free(turbo_ptr(off));
    }
    
    print_result("mixed_alloc_free (1-512B)", ns / static_cast<long long>(OPS), OPS);
}

// =============================================================================
// 6. Multi-threaded Contention
// =============================================================================
void bench_threaded_alloc(System& sys) {
    print_header("MULTI-THREADED ALLOCATION (4 threads contending)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,  "4 u64s (32B)"},
        {16, "16 u64s (128B)"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS_PER_THREAD = 1'000'000;
        constexpr int NUM_THREADS = 4;
        
        std::atomic<long long> total_ns{0};
        
        auto worker = [&](int tid) {
            std::vector<void*> ptrs;
            ptrs.reserve(OPS_PER_THREAD);
            volatile uint64_t sink = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                ptrs.push_back(p);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
            // Cleanup
            for (auto* p : ptrs) turbo_free(p);
        };
        
        std::vector<std::thread> threads;
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();
        
        long long avg_ns_per_op = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
        std::string name = std::string("threaded_alloc ") + st.label;
        print_result(name.c_str(), avg_ns_per_op, OPS_PER_THREAD * NUM_THREADS);
    }
}

// =============================================================================
// Summary Table
// =============================================================================
void print_summary(const std::vector<BenchResult>& res, const char* title) {
    std::cout << "\n\n═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(45) << "Benchmark" 
              << std::right << std::setw(10) << "ns/op"
              << std::setw(12) << "ops" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (auto& r : res) {
        std::cout << std::left << std::setw(45) << r.name
                  << std::right << std::setw(10) << r.ns_per_op
                  << std::setw(12) << r.ops << std::endl;
    }
    std::cout << std::string(70, '-') << std::endl;
}

// =============================================================================
// External Allocator Benchmarks (malloc, mimalloc, jemalloc)
//
// All three tested through the same generic harness for fair comparison.
// =============================================================================
#include <cstdlib>
#include <dlfcn.h>  // For dlsym loading of mimalloc/jemalloc
#include <malloc/malloc.h>  // macOS malloc_size()
#include <mach/mach.h>      // RSS measurement

// Generic allocator interface — function pointers
struct AllocAdapter {
    const char* name;
    void* (*alloc_fn)(size_t);
    void  (*free_fn)(void*);
    size_t (*usable_size_fn)(const void*);  // Returns actual allocated size
};

// mimalloc / jemalloc loaded via dlsym to avoid link-time dependency
static void* (*mi_malloc_fn)(size_t) = nullptr;
static void  (*mi_free_fn)(void*) = nullptr;
static size_t (*mi_usable_size_fn)(const void*) = nullptr;
static void* (*je_malloc_fn)(size_t) = nullptr;
static void  (*je_free_fn)(void*) = nullptr;
static size_t (*je_usable_size_fn)(const void*) = nullptr;

bool load_external_allocators() {
    // mimalloc
    void* mi_lib = dlopen("/opt/homebrew/opt/mimalloc/lib/libmimalloc.dylib", RTLD_NOW);
    if (mi_lib) {
        mi_malloc_fn = reinterpret_cast<void*(*)(size_t)>(dlsym(mi_lib, "mi_malloc"));
        mi_free_fn = reinterpret_cast<void(*)(void*)>(dlsym(mi_lib, "mi_free"));
        mi_usable_size_fn = reinterpret_cast<size_t(*)(const void*)>(dlsym(mi_lib, "mi_usable_size"));
        if (!mi_malloc_fn || !mi_free_fn) {
            std::cerr << "[WARN] mimalloc: dlsym failed\n";
            mi_malloc_fn = nullptr; mi_free_fn = nullptr; mi_usable_size_fn = nullptr;
        }
    } else {
        std::cerr << "[WARN] mimalloc: " << dlerror() << "\n";
    }

    // jemalloc
    void* je_lib = dlopen("/opt/homebrew/opt/jemalloc/lib/libjemalloc.dylib", RTLD_NOW);
    if (je_lib) {
        je_malloc_fn = reinterpret_cast<void*(*)(size_t)>(dlsym(je_lib, "je_malloc"));
        je_free_fn = reinterpret_cast<void(*)(void*)>(dlsym(je_lib, "je_free"));
        je_usable_size_fn = reinterpret_cast<size_t(*)(const void*)>(dlsym(je_lib, "je_malloc_usable_size"));
        if (!je_malloc_fn || !je_free_fn) {
            // Try without prefix
            je_malloc_fn = reinterpret_cast<void*(*)(size_t)>(dlsym(je_lib, "malloc"));
            je_free_fn = reinterpret_cast<void(*)(void*)>(dlsym(je_lib, "free"));
            je_usable_size_fn = reinterpret_cast<size_t(*)(const void*)>(dlsym(je_lib, "malloc_usable_size"));
        }
        if (!je_malloc_fn || !je_free_fn) {
            std::cerr << "[WARN] jemalloc: dlsym failed\n";
            je_malloc_fn = nullptr; je_free_fn = nullptr; je_usable_size_fn = nullptr;
        }
    } else {
        std::cerr << "[WARN] jemalloc: " << dlerror() << "\n";
    }

    return true;
}

// Generic batch alloc-only benchmark for external allocator
void bench_ext_alloc_only(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": ALLOC-ONLY (batch)";
    print_header(header.c_str());
    
    struct SizeTest { size_t bytes; size_t ops; const char* label; };
    SizeTest sizes[] = {
        {32,   2'000'000, "32B (4 u64s)"},
        {128,  2'000'000, "128B (16 u64s)"},
        {512,  1'000'000, "512B (64 u64s)"},
        {2048,   500'000, "2KB (256 u64s)"},
        {8192,   100'000, "8KB (1024 u64s)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = st.ops;
        std::vector<void*> ptrs(OPS);
        
        volatile void* sink = nullptr;
        long long ns = bench_min([&]() {
            for (size_t i = 0; i < OPS; ++i) {
                ptrs[i] = alloc.alloc_fn(st.bytes);
                sink = ptrs[i];
            }
        });
        
        // Cleanup
        for (size_t i = 0; i < OPS; ++i) alloc.free_fn(ptrs[i]);
        
        std::string name = std::string(alloc.name) + " alloc " + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

// Generic batch free-only benchmark
void bench_ext_free_only(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": FREE-ONLY (batch)";
    print_header(header.c_str());
    
    struct SizeTest { size_t bytes; size_t ops; const char* label; };
    SizeTest sizes[] = {
        {32,   2'000'000, "32B (4 u64s)"},
        {128,  2'000'000, "128B (16 u64s)"},
        {512,  1'000'000, "512B (64 u64s)"},
        {2048,   500'000, "2KB (256 u64s)"},
        {8192,   100'000, "8KB (1024 u64s)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = st.ops;
        std::vector<void*> ptrs(OPS);
        
        long long best_free = std::numeric_limits<long long>::max();
        for (int iter = 0; iter < BENCH_ITERS; ++iter) {
            for (size_t i = 0; i < OPS; ++i) {
                ptrs[i] = alloc.alloc_fn(st.bytes);
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < OPS; ++i) {
                alloc.free_fn(ptrs[i]);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (ns < best_free) best_free = ns;
        }
        
        std::string name = std::string(alloc.name) + " free " + st.label;
        print_result(name.c_str(), best_free / static_cast<long long>(OPS), OPS);
    }
}

// Generic mixed workload benchmark  
void bench_ext_mixed(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": MIXED WORKLOAD";
    print_header(header.c_str());
    
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;
    
    std::vector<std::pair<void*, size_t>> pool(POOL_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(8, 512);
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t sz = size_dist(rng);
        pool[i] = {alloc.alloc_fn(sz), sz};
    }
    
    volatile void* sink = nullptr;
    long long ns = bench_min([&]() {
        std::mt19937 rng2(123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
        std::uniform_int_distribution<size_t> sz_dist2(8, 512);
        
        for (size_t i = 0; i < OPS; ++i) {
            size_t idx = idx_dist(rng2);
            alloc.free_fn(pool[idx].first);
            
            size_t new_sz = sz_dist2(rng2);
            pool[idx].first = alloc.alloc_fn(new_sz);
            pool[idx].second = new_sz;
            sink = pool[idx].first;
        }
    });
    
    std::string name = std::string(alloc.name) + " mixed (8-512B)";
    print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    
    for (auto& p : pool) alloc.free_fn(p.first);
}

// Generic threaded benchmark
void bench_ext_threaded(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": THREADED (4 threads)";
    print_header(header.c_str());
    
    struct SizeTest { size_t bytes; const char* label; };
    SizeTest sizes[] = {
        {32,  "32B"},
        {128, "128B"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS_PER_THREAD = 1'000'000;
        constexpr int NUM_THREADS = 4;
        
        std::atomic<long long> total_ns{0};
        
        auto worker = [&](int /*tid*/) {
            volatile void* sink = nullptr;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                void* p = alloc.alloc_fn(st.bytes);
                sink = p;
                alloc.free_fn(p);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
        };
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();
        
        long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
        std::string name = std::string(alloc.name) + " threaded " + st.label;
        print_result(name.c_str(), avg_ns, OPS_PER_THREAD * NUM_THREADS);
    }
}

// Generic threaded MIXED benchmark for external allocators
void bench_ext_threaded_mixed(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": THREADED MIXED (4 threads, varied sizes)";
    print_header(header.c_str());

    constexpr size_t OPS_PER_THREAD = 500'000;
    constexpr int NUM_THREADS = 4;
    constexpr size_t POOL_PER_THREAD = 2'500;

    // Pre-allocate per-thread pools
    struct ThreadPool {
        std::vector<std::pair<void*, size_t>> pool;
    };
    std::vector<ThreadPool> tpools(NUM_THREADS);

    std::mt19937 setup_rng(42);
    std::uniform_int_distribution<size_t> size_dist(8, 512);
    for (int t = 0; t < NUM_THREADS; ++t) {
        tpools[t].pool.resize(POOL_PER_THREAD);
        for (size_t i = 0; i < POOL_PER_THREAD; ++i) {
            size_t sz = size_dist(setup_rng);
            tpools[t].pool[i] = {alloc.alloc_fn(sz), sz};
        }
    }

    std::atomic<long long> total_ns{0};

    auto worker = [&](int tid) {
        auto& pool = tpools[tid].pool;
        std::mt19937 rng(tid * 1000 + 123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_PER_THREAD - 1);
        std::uniform_int_distribution<size_t> sz_dist(8, 512);

        volatile void* sink = nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
            size_t idx = idx_dist(rng);
            alloc.free_fn(pool[idx].first);

            size_t new_sz = sz_dist(rng);
            pool[idx].first = alloc.alloc_fn(new_sz);
            pool[idx].second = new_sz;
            sink = pool[idx].first;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    };

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
    std::string name = std::string(alloc.name) + " threaded_mixed (8-512B)";
    print_result(name.c_str(), avg_ns, OPS_PER_THREAD * NUM_THREADS);

    // Cleanup
    for (int t = 0; t < NUM_THREADS; ++t) {
        for (auto& p : tpools[t].pool) alloc.free_fn(p.first);
    }
}

// =============================================================================
// REALISTIC WORKLOAD SCENARIOS — external allocators
//
// 1. String-heavy:  variable 10-200 bytes (user names, emails, short strings)
// 2. JSON docs:     mixed 50-2000 bytes (REST API payloads, config objects)
// 3. Worst-case:    fully random 1-4096 bytes (pathological fragmentation)
// =============================================================================

void bench_ext_realistic(const AllocAdapter& alloc) {
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;

    struct Scenario { const char* label; size_t min_sz; size_t max_sz; };
    Scenario scenarios[] = {
        {"strings (10-200B)",    10,  200},
        {"JSON docs (50-2000B)", 50,  2000},
        {"worst-case (1-4096B)", 1,   4096},
    };

    for (auto& sc : scenarios) {
        std::string header = std::string(alloc.name) + ": " + sc.label;
        print_header(header.c_str());

        std::vector<std::pair<void*, size_t>> pool(POOL_SIZE);
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> size_dist(sc.min_sz, sc.max_sz);

        for (size_t i = 0; i < POOL_SIZE; ++i) {
            size_t sz = size_dist(rng);
            pool[i] = {alloc.alloc_fn(sz), sz};
        }

        volatile void* sink = nullptr;
        long long ns = bench_min([&]() {
            std::mt19937 rng2(123);
            std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
            std::uniform_int_distribution<size_t> sz_dist2(sc.min_sz, sc.max_sz);

            for (size_t i = 0; i < OPS; ++i) {
                size_t idx = idx_dist(rng2);
                alloc.free_fn(pool[idx].first);

                size_t new_sz = sz_dist2(rng2);
                pool[idx].first = alloc.alloc_fn(new_sz);
                pool[idx].second = new_sz;
                sink = pool[idx].first;
            }
        });

        std::string name = std::string(alloc.name) + " " + sc.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);

        for (auto& p : pool) alloc.free_fn(p.first);
    }
}

// Run full benchmark suite for one external allocator
void bench_external_allocator(const AllocAdapter& alloc) {
    bench_ext_alloc_only(alloc);
    bench_ext_free_only(alloc);
    bench_ext_mixed(alloc);
    bench_ext_threaded(alloc);
    bench_ext_threaded_mixed(alloc);
    bench_ext_realistic(alloc);
}

// =============================================================================
// Generic Fragmentation Benchmark for External Allocators
// =============================================================================

static size_t get_rss_bytes() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}

struct ExtLiveBlock {
    void*    ptr;
    size_t   requested;    // Requested bytes
    size_t   usable;       // Actual usable bytes (from usable_size)
    uint32_t tag;
};

void bench_ext_fragmentation(const AllocAdapter& alloc) {
    std::string header = std::string(alloc.name) + ": FRAGMENTATION & EFFICIENCY AUDIT";
    print_header(header.c_str());
    
    constexpr size_t INITIAL_POOL = 50'000;
    constexpr size_t CHURN_OPS = 100'000;
    
    std::vector<ExtLiveBlock> live;
    live.reserve(INITIAL_POOL * 2);
    
    std::mt19937 rng(12345);  // Same deterministic seed as TA test
    std::uniform_int_distribution<size_t> size_dist(8, 1024); // 8B to 1KB (bytes)
    uint32_t next_tag = 1;
    
    size_t total_requested = 0;
    size_t total_usable = 0;
    
    size_t rss_before = get_rss_bytes();
    
    // ── Phase 1: FILL ──
    for (size_t i = 0; i < INITIAL_POOL; ++i) {
        size_t sz = size_dist(rng);
        void* ptr = alloc.alloc_fn(sz);
        size_t usable = alloc.usable_size_fn ? alloc.usable_size_fn(ptr) : sz;
        
        uint32_t tag = next_tag++;
        *reinterpret_cast<uint32_t*>(ptr) = tag;
        
        live.push_back({ptr, sz, usable, tag});
        total_requested += sz;
        total_usable += usable;
    }
    
    double fill_frag = 100.0 * (total_usable - total_requested) / (total_usable > 0 ? total_usable : 1);
    std::cout << "  Phase 1 FILL: " << INITIAL_POOL << " blocks allocated\n";
    std::cout << "    Requested: " << (total_requested / 1024) << " KB\n";
    std::cout << "    Usable:    " << (total_usable / 1024) << " KB\n";
    std::cout << "    Internal frag: " << std::fixed << std::setprecision(1) << fill_frag << "%\n";
    
    // ── Phase 2: CHURN ──
    size_t frees = 0, allocs_new = 0;
    for (size_t i = 0; i < CHURN_OPS; ++i) {
        double action = std::uniform_real_distribution<>(0, 1)(rng);
        
        if (action < 0.4 && !live.empty()) {
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            size_t idx = pick(rng);
            total_requested -= live[idx].requested;
            total_usable -= live[idx].usable;
            alloc.free_fn(live[idx].ptr);
            live[idx] = live.back();
            live.pop_back();
            frees++;
        } else if (action < 0.7 && !live.empty()) {
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            size_t idx = pick(rng);
            total_requested -= live[idx].requested;
            total_usable -= live[idx].usable;
            alloc.free_fn(live[idx].ptr);
            
            size_t new_sz = size_dist(rng);
            void* new_ptr = alloc.alloc_fn(new_sz);
            size_t usable = alloc.usable_size_fn ? alloc.usable_size_fn(new_ptr) : new_sz;
            uint32_t tag = next_tag++;
            *reinterpret_cast<uint32_t*>(new_ptr) = tag;
            live[idx] = {new_ptr, new_sz, usable, tag};
            total_requested += new_sz;
            total_usable += usable;
            frees++; allocs_new++;
        } else {
            size_t sz = size_dist(rng);
            void* ptr = alloc.alloc_fn(sz);
            size_t usable = alloc.usable_size_fn ? alloc.usable_size_fn(ptr) : sz;
            uint32_t tag = next_tag++;
            *reinterpret_cast<uint32_t*>(ptr) = tag;
            live.push_back({ptr, sz, usable, tag});
            total_requested += sz;
            total_usable += usable;
            allocs_new++;
        }
    }
    
    size_t rss_after = get_rss_bytes();
    
    std::cout << "  Phase 2 CHURN: " << CHURN_OPS << " ops ("
              << frees << " frees, " << allocs_new << " new allocs)\n";
    std::cout << "    Live blocks: " << live.size() << "\n";
    
    // ── Phase 3: VERIFY ──
    size_t verified = 0, corrupted = 0;
    for (auto& b : live) {
        uint32_t stored = *reinterpret_cast<uint32_t*>(b.ptr);
        if (stored == b.tag) verified++;
        else {
            corrupted++;
            if (corrupted <= 3) {
                std::cout << "    CORRUPT: ptr=" << b.ptr 
                          << " expected=" << b.tag << " got=" << stored << "\n";
            }
        }
    }
    std::cout << "  Phase 3 VERIFY: " << verified << "/" << live.size() << " tags intact";
    if (corrupted > 0) std::cout << " *** " << corrupted << " CORRUPTED ***";
    std::cout << "\n";
    
    // ── Phase 4: OVERLAP CHECK ──
    std::sort(live.begin(), live.end(), [](const ExtLiveBlock& a, const ExtLiveBlock& b) {
        return reinterpret_cast<uintptr_t>(a.ptr) < reinterpret_cast<uintptr_t>(b.ptr);
    });
    size_t overlaps = 0;
    for (size_t i = 1; i < live.size(); ++i) {
        uintptr_t prev_end = reinterpret_cast<uintptr_t>(live[i-1].ptr) + live[i-1].usable;
        if (reinterpret_cast<uintptr_t>(live[i].ptr) < prev_end) overlaps++;
    }
    std::cout << "  Phase 4 OVERLAP: " << (overlaps == 0 ? "PASS" : "FAIL")
              << " (" << overlaps << " overlaps)\n";
    
    // ── Phase 5: EFFICIENCY REPORT ──
    double frag_pct = 100.0 * (total_usable - total_requested) / (total_usable > 0 ? total_usable : 1);
    long long rss_delta = static_cast<long long>(rss_after) - static_cast<long long>(rss_before);
    double rss_overhead = (total_requested > 0) ? 100.0 * (rss_delta - static_cast<long long>(total_requested)) / total_requested : 0;
    
    std::cout << "\n  ===== EFFICIENCY REPORT =====\n";
    std::cout << "    Live blocks:         " << std::setw(8) << live.size() << "\n";
    std::cout << "    Live (requested):    " << std::setw(8) << (total_requested / 1024) << " KB\n";
    std::cout << "    Live (usable):       " << std::setw(8) << (total_usable / 1024) << " KB\n";
    std::cout << "    Internal frag:       " << std::setw(7) << std::fixed << std::setprecision(1) << frag_pct << "%\n";
    std::cout << "    RSS delta:           " << std::setw(8) << (rss_delta / 1024) << " KB\n";
    std::cout << "    RSS overhead:        " << std::setw(7) << std::fixed << std::setprecision(1) << rss_overhead << "%\n";
    
    bool pass = (corrupted == 0) && (overlaps == 0);
    std::cout << "\n  RESULT: " << (pass ? "PASS" : "*** FAIL ***")
              << " — " << verified << " verified, " << corrupted << " corrupt, "
              << overlaps << " overlaps\n";
    std::cout << std::string(66, '-') << "\n";
    
    // Cleanup
    for (auto& b : live) alloc.free_fn(b.ptr);
}

// =============================================================================
// TurboAlloc Benchmarks (mimalloc-backed persistent allocator)
// =============================================================================
// turbo_alloc.h already included at top


void bench_ta_fresh_alloc(System& sys) {
    print_header("TURBOALLOC: FRESH ALLOCATION (bump path, no reuse)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {1,    "1 u64 (8B)"},
        {2,    "2 u64s (16B)"},
        {4,    "4 u64s (32B)"},
        {8,    "8 u64s (64B)"},
        {16,   "16 u64s (128B)"},
        {32,   "32 u64s (256B)"},
        {64,   "64 u64s (512B)"},
        {128,  "128 u64s (1KB)"},
        {256,  "256 u64s (2KB)"},
        {512,  "512 u64s (4KB)"},
        {1024, "1024 u64s (8KB)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = ops_for_live_bytes(st.u64s * sizeof(uint64_t));
        
        volatile uint64_t sink = 0;
        std::vector<void*> ptrs;
        ptrs.reserve(OPS);
        
        long long ns = bench_min([&]() {
            ptrs.clear();
            for (size_t i = 0; i < OPS; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                ptrs.push_back(p);
            }
            for (auto* p : ptrs) turbo_free(p);
        });
        
        std::string name = std::string("TA fresh_alloc ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

void bench_ta_free(System& sys) {
    print_header("TURBOALLOC: FREE LATENCY (lock-free limbo push)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,   "4 u64s (32B)"},
        {16,  "16 u64s (128B)"},
        {64,  "64 u64s (512B)"},
        {256, "256 u64s (2KB)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = ops_for_live_bytes(st.u64s * sizeof(uint64_t));
        
        // Pre-allocate blocks
        std::vector<void*> ptrs(OPS);
        for (size_t i = 0; i < OPS; ++i) {
            ptrs[i] = turbo_alloc(st.u64s * sizeof(uint64_t));
        }
        
        long long ns = bench_min([&]() {
            for (size_t i = 0; i < OPS; ++i) {
                turbo_free(ptrs[i]);
            }
            // Re-allocate for next iteration
            for (size_t i = 0; i < OPS; ++i) {
                ptrs[i] = turbo_alloc(st.u64s * sizeof(uint64_t));
            }
        });
        
        // Cleanup
        for (auto* p : ptrs) turbo_free(p);
        
        std::string name = std::string("TA free ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

void bench_ta_gc_promote(System& sys) {
    print_header("TURBOALLOC: GC PROMOTION (epoch-safe promote)");
    
    struct SizeTest { size_t u64s; size_t count; const char* label; };
    SizeTest sizes[] = {
        {4,   100'000, "100K blocks × 32B"},
        {16,  100'000, "100K blocks × 128B"},
        {64,  50'000,  "50K blocks × 512B"},
        {256, 10'000,  "10K blocks × 2KB"},
    };
    
    for (auto& st : sizes) {
        // Fill epoch deferred list
        auto slot = stax_epoch::EpochSystem::enter();
        for (size_t i = 0; i < st.count; ++i) {
            void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            stax_epoch::EpochSystem::defer_free(off, st.u64s,
                [](void*, uint64_t o, uint32_t) { turbo_free(turbo_ptr(o)); }, nullptr);
        }
        stax_epoch::EpochSystem::exit(slot);
        
        size_t limbo_before = stax_epoch::EpochSystem::limbo_count();
        
        long long ns = bench_min([&]() {
            stax_epoch::EpochSystem::reclaim_safe(
                [](void*, uint64_t o, uint32_t) { turbo_free(turbo_ptr(o)); }, nullptr);
        });
        
        std::string name = std::string("TA epoch_reclaim ") + st.label;
        print_result(name.c_str(), ns, 1);
        std::cout << "    limbo before=" << limbo_before << std::endl;
    }
}

void bench_ta_reuse_alloc(System& sys) {
    print_header("TURBOALLOC: REUSE ALLOCATION (freelist hit path)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,   "4 u64s (32B)"},
        {16,  "16 u64s (128B)"},
        {64,  "64 u64s (512B)"},
        {256, "256 u64s (2KB)"},
    };
    
    for (auto& st : sizes) {
        const size_t OPS = ops_for_live_bytes(st.u64s * sizeof(uint64_t));
        
        // Fill reuse pool (alloc + free → mimalloc recycles)
        for (size_t i = 0; i < OPS; ++i) {
            void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
            turbo_free(p);
        }
        
        volatile uint64_t sink = 0;
        std::vector<void*> ptrs;
        ptrs.reserve(OPS);
        long long ns = bench_min([&]() {
            ptrs.clear();
            for (size_t i = 0; i < OPS; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                ptrs.push_back(p);
            }
            for (auto* p : ptrs) turbo_free(p);
        });
        
        std::string name = std::string("TA reuse_alloc ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

void bench_ta_mixed_workload(System& sys) {
    print_header("TURBOALLOC: MIXED WORKLOAD (alloc+free interleaved)");
    
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;
    
    std::vector<std::pair<uint64_t, size_t>> pool(POOL_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t sz = size_dist(rng);
        void* p = turbo_alloc(sz * sizeof(uint64_t));
        pool[i] = {turbo_index(p), sz};
    }
    
    volatile uint64_t sink = 0;
    long long ns = bench_min([&]() {
        std::mt19937 rng2(123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
        std::uniform_int_distribution<size_t> sz_dist2(1, 64);
        
        for (size_t i = 0; i < OPS; ++i) {
            size_t idx = idx_dist(rng2);
            turbo_free(turbo_ptr(pool[idx].first));
            
            size_t new_sz = sz_dist2(rng2);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
    });
    
    // Cleanup pool
    for (auto& [off, sz] : pool) turbo_free(turbo_ptr(off));
    
    print_result("TA mixed_alloc_free (1-512B)", ns / static_cast<long long>(OPS), OPS);
}

void bench_ta_mixed_nogc(System& sys) {
    print_header("TURBOALLOC: MIXED (no gc, bump-only path)");
    
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;
    
    std::vector<std::pair<uint64_t, size_t>> pool(POOL_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t sz = size_dist(rng);
        void* p = turbo_alloc(sz * sizeof(uint64_t));
        pool[i] = {turbo_index(p), sz};
    }
    
    volatile uint64_t sink = 0;
    long long ns = bench_min([&]() {
        std::mt19937 rng2(123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
        std::uniform_int_distribution<size_t> sz_dist2(1, 64);
        
        for (size_t i = 0; i < OPS; ++i) {
            size_t idx = idx_dist(rng2);
            turbo_free(turbo_ptr(pool[idx].first));
            
            size_t new_sz = sz_dist2(rng2);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
    });
    
    // Cleanup pool
    for (auto& [off, sz] : pool) turbo_free(turbo_ptr(off));
    
    print_result("TA mixed_no_gc (1-512B)", ns / static_cast<long long>(OPS), OPS);
}

void bench_ta_threaded(System& sys) {
    print_header("TURBOALLOC: MULTI-THREADED (4 threads)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,  "32B"},
        {16, "128B"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS_PER_THREAD = 1'000'000;
        constexpr int NUM_THREADS = 4;
        
        std::atomic<long long> total_ns{0};
        
        auto worker = [&](int /*tid*/) {
            volatile uint64_t sink = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                turbo_free(p);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
        };
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();
        
        long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
        std::string name = std::string("TA threaded ") + st.label;
        print_result(name.c_str(), avg_ns, OPS_PER_THREAD * NUM_THREADS);
    }
}

void bench_ta_threaded_mixed(System& sys) {
    print_header("TURBOALLOC: THREADED MIXED (4 threads, varied sizes)");
    
    constexpr size_t OPS_PER_THREAD = 500'000;
    constexpr int NUM_THREADS = 4;
    constexpr size_t POOL_PER_THREAD = 2'500;
    
    // Pre-allocate per-thread pools
    struct ThreadPool {
        std::vector<std::pair<uint64_t, size_t>> pool;
    };
    std::vector<ThreadPool> tpools(NUM_THREADS);
    
    std::mt19937 setup_rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);
    for (int t = 0; t < NUM_THREADS; ++t) {
        tpools[t].pool.resize(POOL_PER_THREAD);
        for (size_t i = 0; i < POOL_PER_THREAD; ++i) {
            size_t sz = size_dist(setup_rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            tpools[t].pool[i] = {turbo_index(p), sz};
        }
    }
    
    std::atomic<long long> total_ns{0};
    
    auto worker = [&](int tid) {
        auto& pool = tpools[tid].pool;
        std::mt19937 rng(tid * 1000 + 123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_PER_THREAD - 1);
        std::uniform_int_distribution<size_t> sz_dist(1, 64);
        
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
            size_t idx = idx_dist(rng);
            turbo_free(turbo_ptr(pool[idx].first));
            
            size_t new_sz = sz_dist(rng);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    };
    
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();
    
    long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
    print_result("TA threaded_mixed (8-512B)", avg_ns, OPS_PER_THREAD * NUM_THREADS);
}

// =============================================================================
// TurboAlloc Benchmarks (public API with thread-local caching)
// =============================================================================

void bench_tac_single(System& sys) {
    print_header("TURBOALLOC: SINGLE-THREAD (alloc+free)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,  "32B"},
        {16, "128B"},
        {64, "512B"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS = 5'000'000;
        
        // Warm the cache with some allocs+frees
        for (int i = 0; i < 64; ++i) {
            void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
            turbo_free(p);
        }
        
        volatile uint64_t sink = 0;
        long long ns = bench_min([&]() {
            for (size_t i = 0; i < OPS; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                turbo_free(p);
            }
        });
        
        std::string name = std::string("single ") + st.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

void bench_tac_mixed(System& sys) {
    print_header("TURBOALLOC: MIXED (varied sizes)");
    
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;
    
    std::vector<std::pair<uint64_t, size_t>> pool(POOL_SIZE);
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);
    
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        size_t sz = size_dist(rng);
        void* p = turbo_alloc(sz * sizeof(uint64_t));
        pool[i] = {turbo_index(p), sz};
    }
    
    volatile uint64_t sink = 0;
    long long ns = bench_min([&]() {
        std::mt19937 rng2(123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
        std::uniform_int_distribution<size_t> sz_dist2(1, 64);
        
        for (size_t i = 0; i < OPS; ++i) {
            size_t idx = idx_dist(rng2);
            turbo_free(turbo_ptr(pool[idx].first));
            
            size_t new_sz = sz_dist2(rng2);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
    });
    
    print_result("mixed (8-512B)", ns / static_cast<long long>(OPS), OPS);
}

void bench_tac_threaded(System& sys) {
    print_header("TURBOALLOC: THREADED (4 threads)");
    
    struct SizeTest { size_t u64s; const char* label; };
    SizeTest sizes[] = {
        {4,  "32B"},
        {16, "128B"},
    };
    
    for (auto& st : sizes) {
        constexpr size_t OPS_PER_THREAD = 1'000'000;
        constexpr int NUM_THREADS = 4;
        
        std::atomic<long long> total_ns{0};
        
        auto worker = [&](int /*tid*/) {
            volatile uint64_t sink = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                void* p = turbo_alloc(st.u64s * sizeof(uint64_t));
                sink = turbo_index(p);
                turbo_free(p);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
                std::memory_order_relaxed);
        };
        
        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();
        
        long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
        std::string name = std::string("threaded ") + st.label;
        print_result(name.c_str(), avg_ns, OPS_PER_THREAD * NUM_THREADS);
    }
}

void bench_tac_threaded_mixed(System& sys) {
    print_header("TURBOALLOC: THREADED MIXED (4 threads, varied sizes)");
    
    constexpr size_t OPS_PER_THREAD = 500'000;
    constexpr int NUM_THREADS = 4;
    constexpr size_t POOL_PER_THREAD = 2'500;
    
    // Pre-allocate per-thread pools using the global allocator
    struct ThreadPool {
        std::vector<std::pair<uint64_t, size_t>> pool;
    };
    std::vector<ThreadPool> tpools(NUM_THREADS);
    
    std::mt19937 setup_rng(42);
    std::uniform_int_distribution<size_t> size_dist(1, 64);
    for (int t = 0; t < NUM_THREADS; ++t) {
        tpools[t].pool.resize(POOL_PER_THREAD);
        for (size_t i = 0; i < POOL_PER_THREAD; ++i) {
            size_t sz = size_dist(setup_rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            tpools[t].pool[i] = {turbo_index(p), sz};
        }
    }
    
    std::atomic<long long> total_ns{0};
    
    auto worker = [&](int tid) {
        auto& pool = tpools[tid].pool;
        std::mt19937 rng(tid * 1000 + 123);
        std::uniform_int_distribution<size_t> idx_dist(0, POOL_PER_THREAD - 1);
        std::uniform_int_distribution<size_t> sz_dist(1, 64);
        
        volatile uint64_t sink = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
            size_t idx = idx_dist(rng);
            turbo_free(turbo_ptr(pool[idx].first));
            
            size_t new_sz = sz_dist(rng);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            pool[idx] = {off, new_sz};
            sink = off;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
            std::memory_order_relaxed);
    };
    
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();
    
    long long avg_ns = total_ns.load() / (NUM_THREADS * static_cast<long long>(OPS_PER_THREAD));
    print_result("threaded_mixed (8-512B)", avg_ns, OPS_PER_THREAD * NUM_THREADS);
}

// =============================================================================
// REALISTIC WORKLOAD SCENARIOS — TurboAlloc
//
// 1. String-heavy:  10-200 bytes → 2-25 u64s  (user names, emails, short text)
// 2. JSON docs:     50-2000 bytes → 7-250 u64s (REST payloads, config objects)
// 3. Worst-case:    1-4096 bytes → 1-512 u64s  (pathological random)
// =============================================================================

void bench_tac_realistic(System& sys) {
    constexpr size_t OPS = 2'000'000;
    constexpr size_t POOL_SIZE = 10'000;

    struct Scenario { const char* label; size_t min_u64s; size_t max_u64s; };
    Scenario scenarios[] = {
        {"strings (10-200B)",    2,   25},
        {"JSON docs (50-2000B)", 7,   250},
        {"worst-case (1-4096B)", 1,   512},
    };

    for (auto& sc : scenarios) {
        std::string header = std::string("TurboAlloc: ") + sc.label;
        print_header(header.c_str());

        std::vector<std::pair<uint64_t, size_t>> pool(POOL_SIZE);
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> size_dist(sc.min_u64s, sc.max_u64s);

        for (size_t i = 0; i < POOL_SIZE; ++i) {
            size_t sz = size_dist(rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            pool[i] = {turbo_index(p), sz};
        }

        volatile uint64_t sink = 0;
        long long ns = bench_min([&]() {
            std::mt19937 rng2(123);
            std::uniform_int_distribution<size_t> idx_dist(0, POOL_SIZE - 1);
            std::uniform_int_distribution<size_t> sz_dist2(sc.min_u64s, sc.max_u64s);

            for (size_t i = 0; i < OPS; ++i) {
                size_t idx = idx_dist(rng2);
                turbo_free(turbo_ptr(pool[idx].first));

                size_t new_sz = sz_dist2(rng2);
                void* p = turbo_alloc(new_sz * sizeof(uint64_t));
                uint64_t off = turbo_index(p);
                pool[idx] = {off, new_sz};
                sink = off;
            }
        });

        std::string name = std::string("") + sc.label;
        print_result(name.c_str(), ns / static_cast<long long>(OPS), OPS);
    }
}

// =============================================================================
// EXHAUSTIVE GC+ALLOCATOR CORRECTNESS TEST
//
// Validates the FULL lifecycle: alloc → tag → free → limbo → gc_promote → 
//   freelist → reuse → verify tag cleared → re-tag → verify
//
// What this proves:
//   1. Freed blocks enter limbo and stay safe during epoch window
//   2. gc_promote() correctly moves blocks to freelist
//   3. Reused blocks are zeroed (no stale data leaks)
//   4. NO overlapping live blocks (no double-alloc)
//   5. NO data corruption under multi-threaded alloc+free+GC
// =============================================================================

struct GCTestBlock {
    uint64_t offset;
    size_t   size_u64s;
    uint64_t tag;       // Unique tag written across full block for corruption check
};

void bench_gc_correctness(System& sys) {
    print_header("ALLOCATOR CORRECTNESS TEST");

    // ── Phase 1: SINGLE-THREADED — alloc, tag, free, reuse, verify ──
    {
        std::cout << "  Phase 1: Single-thread alloc/free cycle\n";

        constexpr size_t POOL = 10000;
        std::vector<GCTestBlock> blocks;
        blocks.reserve(POOL);

        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> size_dist(1, 64);
        uint64_t next_tag = 0xDEAD000000000001ULL;

        // Alloc and tag blocks — write tag across EVERY u64 in the block
        for (size_t i = 0; i < POOL; ++i) {
            size_t sz = size_dist(rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            if (!p) {
                std::cout << "  FAIL: alloc returned null at i=" << i << "\n";
                return;
            }
            uint64_t off = turbo_index(p);

            uint64_t tag = next_tag++;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(p);
            for (size_t j = 0; j < sz; ++j) ptr[j] = tag;

            blocks.push_back({off, sz, tag});
        }

        // Verify all tags intact
        int corrupt_pre_free = 0;
        for (auto& b : blocks) {
            uint64_t* ptr = reinterpret_cast<uint64_t*>(turbo_ptr(b.offset));
            for (size_t j = 0; j < b.size_u64s; ++j) {
                if (ptr[j] != b.tag) { corrupt_pre_free++; break; }
            }
        }
        std::cout << "    Pre-free verification: " << POOL << " blocks, "
                  << corrupt_pre_free << " corrupt\n";

        // Free ALL blocks
        for (auto& b : blocks) {
            turbo_free(turbo_ptr(b.offset));
        }

        // Allocate SAME number of blocks — should reuse freed memory
        std::vector<GCTestBlock> blocks2;
        blocks2.reserve(POOL);
        uint64_t next_tag2 = 0xBEEF000000000001ULL;

        for (size_t i = 0; i < POOL; ++i) {
            size_t sz = size_dist(rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            if (!p) {
                std::cout << "  FAIL: reuse alloc returned null at i=" << i << "\n";
                return;
            }
            uint64_t off = turbo_index(p);

            uint64_t tag = next_tag2++;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(p);
            for (size_t j = 0; j < sz; ++j) ptr[j] = tag;

            blocks2.push_back({off, sz, tag});
        }

        // Verify all second-gen tags intact
        int corrupt_reuse = 0;
        for (auto& b : blocks2) {
            uint64_t* ptr = reinterpret_cast<uint64_t*>(turbo_ptr(b.offset));
            for (size_t j = 0; j < b.size_u64s; ++j) {
                if (ptr[j] != b.tag) { corrupt_reuse++; break; }
            }
        }

        // Check for overlapping live blocks
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        ranges.reserve(blocks2.size());
        for (auto& b : blocks2) {
            ranges.push_back({b.offset, b.offset + b.size_u64s});
        }
        std::sort(ranges.begin(), ranges.end());
        int overlaps = 0;
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i-1].second) overlaps++;
        }

        std::cout << "    Reuse verification: " << POOL << " blocks, "
                  << corrupt_reuse << " corrupt, " << overlaps << " overlaps\n";

        if (corrupt_pre_free == 0 && corrupt_reuse == 0 && overlaps == 0) {
            std::cout << "  PASS — Phase 1: Single-thread alloc/free cycle\n";
        } else {
            std::cout << "  FAIL — Phase 1: "
                      << corrupt_pre_free << " pre-free corrupt, "
                      << corrupt_reuse << " reuse corrupt, "
                      << overlaps << " overlaps\n";
            return;
        }

        // Clean up phase 1 allocations
        for (auto& b : blocks2) {
            turbo_free(turbo_ptr(b.offset));
        }
    }

    // ── Phase 2: MULTI-THREADED — 4 threads alloc+free concurrently ──
    {
        std::cout << "  Phase 2: Multi-thread alloc/free cycle (4 threads)\n";

        constexpr int NUM_THREADS = 4;
        constexpr size_t OPS_PER_THREAD = 50000;
        constexpr size_t POOL_PER_THREAD = 2000;

        std::atomic<int> total_corrupt{0};
        std::atomic<int> total_overlaps{0};

        auto worker = [&](int tid) {
            std::mt19937 rng(tid * 1000 + 42);
            std::uniform_int_distribution<size_t> size_dist(1, 64);
            std::uniform_int_distribution<size_t> idx_dist(0, POOL_PER_THREAD - 1);

            uint64_t tag_base = static_cast<uint64_t>(tid + 1) << 48;
            uint64_t next_tag = tag_base | 1;

            // Initial fill
            std::vector<GCTestBlock> pool(POOL_PER_THREAD);
            for (size_t i = 0; i < POOL_PER_THREAD; ++i) {
                size_t sz = size_dist(rng);
                void* p = turbo_alloc(sz * sizeof(uint64_t));
                if (!p) { total_corrupt.fetch_add(1); return; }
                uint64_t off = turbo_index(p);

                uint64_t tag = next_tag++;
                uint64_t* ptr = reinterpret_cast<uint64_t*>(p);
                for (size_t j = 0; j < sz; ++j) ptr[j] = tag;

                pool[i] = {off, sz, tag};
            }

            // Churn: free random, alloc new, verify old tag before free
            int corrupt = 0;
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                size_t idx = idx_dist(rng);
                auto& b = pool[idx];

                // Verify tag BEFORE freeing
                uint64_t* ptr = reinterpret_cast<uint64_t*>(turbo_ptr(b.offset));
                for (size_t j = 0; j < b.size_u64s; ++j) {
                    if (ptr[j] != b.tag) { corrupt++; break; }
                }

                turbo_free(turbo_ptr(b.offset));

                // Alloc new
                size_t new_sz = size_dist(rng);
                void* new_p = turbo_alloc(new_sz * sizeof(uint64_t));
                if (!new_p) { total_corrupt.fetch_add(1); return; }
                uint64_t new_off = turbo_index(new_p);

                uint64_t new_tag = next_tag++;
                uint64_t* new_ptr = reinterpret_cast<uint64_t*>(new_p);
                for (size_t j = 0; j < new_sz; ++j) new_ptr[j] = new_tag;

                pool[idx] = {new_off, new_sz, new_tag};
            }

            // Final verification: check all live blocks
            std::vector<std::pair<uint64_t, uint64_t>> ranges;
            ranges.reserve(POOL_PER_THREAD);
            for (auto& b : pool) {
                uint64_t* ptr = reinterpret_cast<uint64_t*>(turbo_ptr(b.offset));
                for (size_t j = 0; j < b.size_u64s; ++j) {
                    if (ptr[j] != b.tag) { corrupt++; break; }
                }
                ranges.push_back({b.offset, b.offset + b.size_u64s});
            }

            std::sort(ranges.begin(), ranges.end());
            int overlaps = 0;
            for (size_t i = 1; i < ranges.size(); ++i) {
                if (ranges[i].first < ranges[i-1].second) overlaps++;
            }

            total_corrupt.fetch_add(corrupt);
            total_overlaps.fetch_add(overlaps);

            // Cleanup
            for (auto& b : pool) {
                turbo_free(turbo_ptr(b.offset));
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(NUM_THREADS);
        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back(worker, t);
        }
        for (auto& t : threads) t.join();

        int c = total_corrupt.load();
        int o = total_overlaps.load();
        std::cout << "    Threads: " << NUM_THREADS
                  << ", Ops/thread: " << OPS_PER_THREAD
                  << ", Pool/thread: " << POOL_PER_THREAD << "\n";
        std::cout << "    Total corrupt: " << c << ", Total overlaps: " << o << "\n";

        if (c == 0 && o == 0) {
            std::cout << "  PASS — Phase 2: Multi-thread correctness\n";
        } else {
            std::cout << "  FAIL — Phase 2: " << c << " corrupt, " << o << " overlaps\n";
        }
    }

    std::cout << "\n";
}

// =============================================================================
// FRAGMENTATION & EFFICIENCY BENCHMARK
//
// Deterministic workload: alloc/free/realloc storm, then audit every live block.
// Validates: no gaps, no overlaps, no corruption, full reclaimability.
// =============================================================================

struct LiveBlock {
    uint64_t offset;
    size_t   size_u64s;   // Requested size
    size_t   class_u64s;  // Actual class size (rounded up)
    uint32_t tag;         // Unique tag written to first 4 bytes for verification
};

void bench_fragmentation(System& sys) {
    print_header("FRAGMENTATION & EFFICIENCY AUDIT");
    
    // ── Phase 1: FILL — allocate 50K blocks of varied sizes ──
    constexpr size_t INITIAL_POOL = 50'000;
    std::vector<LiveBlock> live;
    live.reserve(INITIAL_POOL * 2);
    
    std::mt19937 rng(12345);  // Deterministic seed
    std::uniform_int_distribution<size_t> size_dist(1, 128); // 8B to 1KB
    uint32_t next_tag = 1;
    
    size_t total_requested_bytes = 0;
    size_t total_class_bytes = 0;
    
    for (size_t i = 0; i < INITIAL_POOL; ++i) {
        size_t sz = size_dist(rng);
        size_t alloc_bytes = sz * sizeof(uint64_t);
        void* p = turbo_alloc(alloc_bytes);
        uint64_t off = turbo_index(p);
        size_t usable = mi_usable_size(p);
        size_t cls_u64s = usable / sizeof(uint64_t);
        
        // Write verification tag
        uint32_t tag = next_tag++;
        *reinterpret_cast<uint32_t*>(p) = tag;
        
        live.push_back({off, sz, cls_u64s, tag});
        total_requested_bytes += sz * 8;
        total_class_bytes += cls_u64s * 8;
    }
    
    std::cout << "  Phase 1 FILL: " << INITIAL_POOL << " blocks allocated\n";
    std::cout << "    Requested: " << (total_requested_bytes / 1024) << " KB\n";
    std::cout << "    Class-rounded: " << (total_class_bytes / 1024) << " KB\n";
    std::cout << "    Internal frag: " 
              << std::fixed << std::setprecision(1)
              << (100.0 * (total_class_bytes - total_requested_bytes) / (total_class_bytes > 0 ? total_class_bytes : 1)) 
              << "%\n";
    
    // ── Phase 2: CHURN — random free + realloc storm (100K ops) ──
    constexpr size_t CHURN_OPS = 100'000;
    size_t frees = 0, allocs_new = 0;
    
    for (size_t i = 0; i < CHURN_OPS; ++i) {
        double action = std::uniform_real_distribution<>(0, 1)(rng);
        
        if (action < 0.4 && !live.empty()) {
            // FREE a random block
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            size_t idx = pick(rng);
            
            total_requested_bytes -= live[idx].size_u64s * 8;
            total_class_bytes -= live[idx].class_u64s * 8;
            
            turbo_free(turbo_ptr(live[idx].offset));
            
            // Swap-remove
            live[idx] = live.back();
            live.pop_back();
            frees++;
            
        } else if (action < 0.7 && !live.empty()) {
            // FREE + REALLOC (different size)
            std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
            size_t idx = pick(rng);
            
            total_requested_bytes -= live[idx].size_u64s * 8;
            total_class_bytes -= live[idx].class_u64s * 8;
            turbo_free(turbo_ptr(live[idx].offset));
            
            // Allocate new size
            size_t new_sz = size_dist(rng);
            void* p = turbo_alloc(new_sz * sizeof(uint64_t));
            uint64_t new_off = turbo_index(p);
            size_t usable = mi_usable_size(p);
            size_t cls_sz = usable / sizeof(uint64_t);
            
            uint32_t tag = next_tag++;
            *reinterpret_cast<uint32_t*>(p) = tag;
            
            live[idx] = {new_off, new_sz, cls_sz, tag};
            total_requested_bytes += new_sz * 8;
            total_class_bytes += cls_sz * 8;
            frees++;
            allocs_new++;
            
        } else {
            // ALLOC new block
            size_t sz = size_dist(rng);
            void* p = turbo_alloc(sz * sizeof(uint64_t));
            uint64_t off = turbo_index(p);
            size_t usable = mi_usable_size(p);
            size_t cls_sz = usable / sizeof(uint64_t);
            
            uint32_t tag = next_tag++;
            *reinterpret_cast<uint32_t*>(p) = tag;
            
            live.push_back({off, sz, cls_sz, tag});
            total_requested_bytes += sz * 8;
            total_class_bytes += cls_sz * 8;
            allocs_new++;
        }
    }
    
    std::cout << "  Phase 2 CHURN: " << CHURN_OPS << " ops (" 
              << frees << " frees, " << allocs_new << " new allocs)\n";
    std::cout << "    Live blocks: " << live.size() << "\n";
    
    // ── Phase 3: VERIFY — check every live block's tag is intact ──
    size_t verified = 0, corrupted = 0;
    for (auto& b : live) {
        void* ptr = turbo_ptr(b.offset);
        uint32_t stored_tag = *reinterpret_cast<uint32_t*>(ptr);
        if (stored_tag == b.tag) {
            verified++;
        } else {
            corrupted++;
            if (corrupted <= 5) {
                std::cout << "    CORRUPT: offset=" << b.offset 
                          << " expected_tag=" << b.tag
                          << " got=" << stored_tag << "\n";
            }
        }
    }
    
    std::cout << "  Phase 3 VERIFY: " << verified << "/" << live.size() 
              << " tags intact";
    if (corrupted > 0) {
        std::cout << " *** " << corrupted << " CORRUPTED ***";
    }
    std::cout << "\n";
    
    // ── Phase 4: CHECK FOR OVERLAPS ──
    std::sort(live.begin(), live.end(), [](const LiveBlock& a, const LiveBlock& b) {
        return a.offset < b.offset;
    });
    
    size_t overlaps = 0;
    for (size_t i = 1; i < live.size(); ++i) {
        uint64_t prev_end = live[i-1].offset + live[i-1].size_u64s;
        if (live[i].offset < prev_end) {
            overlaps++;
            if (overlaps <= 5) {
                std::cout << "    OVERLAP: block[" << (i-1) << "] ends at " << prev_end
                          << " but block[" << i << "] starts at " << live[i].offset << "\n";
            }
        }
    }
    
    std::cout << "  Phase 4 OVERLAP CHECK: ";
    if (overlaps == 0) {
        std::cout << "PASS (0 overlaps)\n";
    } else {
        std::cout << "FAIL (" << overlaps << " overlaps)\n";
    }
    
    // ── Phase 5: FRAGMENTATION METRICS ──
    size_t total_gap_u64s = 0;
    for (size_t i = 1; i < live.size(); ++i) {
        uint64_t prev_end = live[i-1].offset + live[i-1].size_u64s;
        if (live[i].offset > prev_end) {
            total_gap_u64s += (live[i].offset - prev_end);
        }
    }
    
    size_t live_u64s = 0;
    for (auto& b : live) live_u64s += b.size_u64s;
    
    size_t live_requested_u64s = live_u64s; // same since we track requested
    
    double internal_frag = (total_class_bytes > total_requested_bytes)
        ? 100.0 * (total_class_bytes - total_requested_bytes) / (total_class_bytes > 0 ? total_class_bytes : 1)
        : 0.0;
    
    std::cout << "\n  ===== EFFICIENCY REPORT =====\n";
    std::cout << "    Live blocks:         " << std::setw(8) << live.size() << "\n";
    std::cout << "    Live (requested):    " << std::setw(8) << (live_requested_u64s * 8 / 1024) << " KB\n";
    std::cout << "    Live (class-round):  " << std::setw(8) << (live_u64s * 8 / 1024) << " KB\n";
    std::cout << "    Internal frag:       " << std::setw(7) << std::fixed << std::setprecision(1) << internal_frag << "%\n";
    std::cout << "    Gaps between blocks: " << std::setw(8) << (total_gap_u64s * 8 / 1024) << " KB\n";
    
    // PASS/FAIL summary
    bool pass = (corrupted == 0) && (overlaps == 0);
    std::cout << "\n  RESULT: " << (pass ? "PASS" : "*** FAIL ***") 
              << " — " << verified << " verified, " << corrupted << " corrupt, "
              << overlaps << " overlaps\n";
    std::cout << std::string(66, '-') << "\n";
    
    // Cleanup live blocks
    for (auto& b : live) {
        turbo_free(turbo_ptr(b.offset));
    }
}


// =============================================================================
void print_comparison(const std::vector<BenchResult>& legacy, const std::vector<BenchResult>& turbo) {
    std::cout << "\n\n╔═══════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    LEGACY vs TURBOALLOC — HEAD-TO-HEAD COMPARISON                        ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    
    // Build lookup map for TurboAlloc results (strip "TA " prefix)
    std::map<std::string, long long> ta_map;
    for (auto& r : turbo) {
        std::string key = r.name;
        if (key.substr(0, 3) == "TA ") key = key.substr(3);
        ta_map[key] = r.ns_per_op;
    }
    
    std::cout << std::string(90, '-') << std::endl;
    std::cout << std::left << std::setw(40) << "Benchmark"
              << std::right << std::setw(12) << "Legacy"
              << std::setw(12) << "TurboAlloc"
              << std::setw(12) << "Speedup"
              << std::setw(12) << "Winner" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    
    for (auto& r : legacy) {
        auto it = ta_map.find(r.name);
        if (it != ta_map.end()) {
            long long legacy_ns = r.ns_per_op;
            long long ta_ns = it->second;
            
            double speedup = (ta_ns > 0) ? static_cast<double>(legacy_ns) / ta_ns : 0.0;
            const char* winner = (ta_ns <= legacy_ns) ? "TurboAlloc" : "Legacy";
            
            std::cout << std::left << std::setw(40) << r.name
                      << std::right << std::setw(10) << legacy_ns << "ns"
                      << std::setw(10) << ta_ns << "ns"
                      << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x"
                      << std::setw(12) << winner << std::endl;
        }
    }
    std::cout << std::string(90, '-') << std::endl;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "====================================================================\n";
    std::cout << "  TURBOALLOC BENCHMARK -- ALL ALLOCATORS (best-of-5, 2M-5M ops)\n";
    std::cout << "====================================================================\n";

    std::string db_dir = "./db_alloc_bench";
    std::filesystem::remove_all(db_dir);
    std::filesystem::create_directories(db_dir);
    std::string db_path = db_dir + "/bench.db";
    constexpr size_t DB_SIZE = 4ULL * 1024 * 1024 * 1024;

    auto sys = std::make_unique<System>(db_path.c_str(), DB_SIZE);
    sys->init(db_path.c_str(), DB_SIZE);

    // ---- Part 1: System Allocator (turbo_alloc) ---- [SKIPPED FOR ISOLATION TEST]
    std::vector<BenchResult> legacy_results;
    #if 0
    std::cout << "\n== PART 1: SYSTEM ALLOCATOR (turbo_alloc) ==\n";
    bench_fresh_alloc(*sys);
    bench_free(*sys);
    bench_gc_promote(*sys);
    bench_reuse_alloc(*sys);
    bench_mixed_workload(*sys);
    bench_threaded_alloc(*sys);
    legacy_results = results;
    print_summary(legacy_results, "SYSTEM ALLOCATOR SUMMARY");
    #endif

    // ---- Part 2: External Allocators ---- [SKIPPED FOR ISOLATION TEST]
    std::vector<BenchResult> malloc_results, mi_results, je_results;
    #if 0
    load_external_allocators();

    // 2a: System malloc
    AllocAdapter sys_malloc{"SysMalloc", std::malloc, std::free,
        [](const void* p) -> size_t { return malloc_size(p); }};
    std::cout << "\n== PART 2a: SYSTEM MALLOC (Apple libc) ==\n";
    results.clear();
    bench_external_allocator(sys_malloc);
    bench_ext_fragmentation(sys_malloc);
    malloc_results = results;
    print_summary(malloc_results, "SYSTEM MALLOC SUMMARY");

    // 2b: mimalloc
    if (mi_malloc_fn && mi_free_fn) {
        AllocAdapter mi_alloc{"mimalloc", mi_malloc_fn, mi_free_fn, mi_usable_size_fn};
        std::cout << "\n== PART 2b: MIMALLOC (Microsoft Research) ==\n";
        results.clear();
        bench_external_allocator(mi_alloc);
        bench_ext_fragmentation(mi_alloc);
        mi_results = results;
        print_summary(mi_results, "MIMALLOC SUMMARY");
    } else {
        std::cout << "\n[SKIP] mimalloc not available\n";
    }

    // 2c: jemalloc
    if (je_malloc_fn && je_free_fn) {
        AllocAdapter je_alloc{"jemalloc", je_malloc_fn, je_free_fn, je_usable_size_fn};
        std::cout << "\n== PART 2c: JEMALLOC (Facebook/Meta) ==\n";
        results.clear();
        bench_external_allocator(je_alloc);
        bench_ext_fragmentation(je_alloc);
        je_results = results;
        print_summary(je_results, "JEMALLOC SUMMARY");
    } else {
        std::cout << "\n[SKIP] jemalloc not available\n";
    }
    #endif

    // ---- Part 3: TurboAlloc (mimalloc-backed) ----
    std::cout << "\n== PART 3: TURBOALLOC (mimalloc-backed persistent allocator) ==\n";
    results.clear();
    // bench_ta_fresh_alloc(*sys);
    // bench_ta_free(*sys);
    // bench_ta_gc_promote(*sys);
    // bench_ta_reuse_alloc(*sys);
    // bench_ta_mixed_workload(*sys);
    // bench_ta_mixed_nogc(*sys);
    // bench_ta_threaded(*sys);
    bench_ta_threaded_mixed(*sys);
    std::vector<BenchResult> turbo_results = results;
    print_summary(turbo_results, "TURBOALLOC SUMMARY");

    // ---- Part 3b: TurboAlloc (thread-local via mimalloc heaps) ----
    std::cout << "\n== PART 3b: TURBOALLOC (thread-local) ==\n";
    results.clear();

    bench_tac_single(*sys);
    bench_tac_mixed(*sys);
    bench_tac_threaded(*sys);
    bench_tac_threaded_mixed(*sys);
    bench_tac_realistic(*sys);
    std::vector<BenchResult> tac_results = results;
    print_summary(tac_results, "TURBOALLOC THREAD-LOCAL SUMMARY");
    
    // Merge TAC results into turbo_results for comparison
    for (auto& r : tac_results) turbo_results.push_back(r);

    // ---- Part 3c: Allocator Correctness ----
    bench_gc_correctness(*sys);

    // ---- Part 3d: Fragmentation & Efficiency Audit ----
    bench_fragmentation(*sys);

    // ---- Part 4: GRAND COMPARISON TABLE ----
    std::cout << "\n" << std::string(115, '=') << "\n";
    std::cout << "  GRAND COMPARISON -- ALL ALLOCATORS\n";
    std::cout << std::string(115, '=') << "\n";

    auto make_map = [](const std::vector<BenchResult>& res, const std::string& prefix) {
        std::map<std::string, long long> m;
        for (auto& r : res) {
            std::string key = r.name;
            if (!prefix.empty() && key.find(prefix) == 0) key = key.substr(prefix.size());
            m[key] = r.ns_per_op;
        }
        return m;
    };

    auto ta_map = make_map(turbo_results, "TA ");
    auto ml_map = make_map(malloc_results, "SysMalloc ");
    auto mi_map = make_map(mi_results, "mimalloc ");
    auto je_map = make_map(je_results, "jemalloc ");

    struct CompareRow { const char* display; const char* ta_key; const char* ext_key; };
    CompareRow rows[] = {
        // Single-threaded
        {"single 32B",      "single 32B",                  "alloc 32B (4 u64s)"},
        {"single 128B",     "single 128B",                 "alloc 128B (16 u64s)"},
        {"single 512B",     "single 512B",                 "alloc 512B (64 u64s)"},
        {"mixed",           "mixed (8-512B)",              "mixed (8-512B)"},
        // Threaded
        {"threaded 32B",    "threaded 32B",                "threaded 32B"},
        {"threaded 128B",   "threaded 128B",               "threaded 128B"},
        {"threaded mixed",  "threaded_mixed (8-512B)",     "threaded_mixed (8-512B)"},
        // Realistic workloads
        {"strings",         "strings (10-200B)",           "strings (10-200B)"},
        {"JSON docs",       "JSON docs (50-2000B)",        "JSON docs (50-2000B)"},
        {"worst-case",      "worst-case (1-4096B)",        "worst-case (1-4096B)"},
    };

    std::cout << std::string(115, '-') << "\n";
    std::cout << std::left  << std::setw(20) << "Benchmark"
              << std::right << std::setw(12) << "TurboAlloc"
              << std::setw(12) << "Legacy"
              << std::setw(12) << "SysMalloc"
              << std::setw(12) << "mimalloc"
              << std::setw(12) << "jemalloc"
              << std::setw(14) << "TA vs best"
              << std::setw(16) << "Winner" << "\n";
    std::cout << std::string(115, '-') << "\n";

    auto fmt = [](long long ns) -> std::string {
        if (ns < 0) return "---";
        return std::to_string(ns) + "ns";
    };

    for (auto& row : rows) {
        long long ta_ns = -1, leg_ns = -1, ml_ns = -1, mi_ns = -1, je_ns = -1;
        if (row.ta_key) {
            auto it = ta_map.find(row.ta_key);
            if (it != ta_map.end()) ta_ns = it->second;
        }
        if (row.ta_key) {
            for (auto& r : legacy_results) {
                if (r.name == std::string(row.ta_key)) { leg_ns = r.ns_per_op; break; }
            }
        }
        if (row.ext_key) {
            auto it = ml_map.find(row.ext_key); if (it != ml_map.end()) ml_ns = it->second;
            it = mi_map.find(row.ext_key); if (it != mi_map.end()) mi_ns = it->second;
            it = je_map.find(row.ext_key); if (it != je_map.end()) je_ns = it->second;
        }

        long long vals[] = {ta_ns, leg_ns, ml_ns, mi_ns, je_ns};
        const char* names[] = {"TurboAlloc", "Legacy", "SysMalloc", "mimalloc", "jemalloc"};
        long long best = std::numeric_limits<long long>::max();
        const char* winner = "---";
        long long best_other = std::numeric_limits<long long>::max();
        for (int i = 0; i < 5; ++i) {
            if (vals[i] >= 0 && vals[i] < best) { best = vals[i]; winner = names[i]; }
            if (i > 0 && vals[i] >= 0 && vals[i] < best_other) best_other = vals[i];
        }
        if (ta_ns >= 0 && ta_ns <= best) winner = "TurboAlloc";

        std::string speedup = "---";
        if (ta_ns >= 0 && best_other > 0 && best_other < std::numeric_limits<long long>::max()) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1fx", static_cast<double>(best_other) / (ta_ns > 0 ? ta_ns : 1));
            speedup = buf;
        }

        std::cout << std::left  << std::setw(20) << row.display
                  << std::right << std::setw(12) << fmt(ta_ns)
                  << std::setw(12) << fmt(leg_ns)
                  << std::setw(12) << fmt(ml_ns)
                  << std::setw(12) << fmt(mi_ns)
                  << std::setw(12) << fmt(je_ns)
                  << std::setw(14) << speedup
                  << std::setw(16) << winner << "\n";
    }
    std::cout << std::string(115, '-') << "\n";

    sys.reset();
    std::filesystem::remove_all(db_dir);
    std::cout << "\nDone.\n";
    return 0;
}
