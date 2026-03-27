// ═══════════════════════════════════════════════════════════════════════════
// V5 vs ANKERL — ULTIMATE MEGA BENCHMARK + CORRECTNESS SUITE
// 3 files: turbocorev5.h, unordered_dense.h, v5_mega_bench.cpp
// 20 benchmark concepts · 30+ correctness tests · 100K–1M scales
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <unordered_map>

#include "turbocorev5.h"
#include "unordered_dense.h"

using hrc = std::chrono::high_resolution_clock;
using ns  = std::chrono::nanoseconds;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::mt19937_64 g_rng(42);

static double elapsed_ns(hrc::time_point t0, hrc::time_point t1, uint32_t n) {
    return (double)std::chrono::duration_cast<ns>(t1 - t0).count() / n;
}

static std::string rand_string(uint32_t len) {
    static const char chars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRS"
                                "TUVWXYZ0123456789_-./:";
    std::string s(len, ' ');
    for (uint32_t i = 0; i < len; i++) s[i] = chars[g_rng() % (sizeof(chars) - 1)];
    return s;
}

static std::string make_path(int depth) {
    std::string p = "/";
    for (int i = 0; i < depth; i++) {
        uint32_t seg = 3 + (g_rng() % 12);
        for (uint32_t j = 0; j < seg; j++)
            p += 'a' + (g_rng() % 26);
        if (i < depth - 1) p += '/';
    }
    return p;
}

// ── Results table ────────────────────────────────────────────────────────────

struct BenchResult { double v5; double antl; };

static void print_row(const char *label, BenchResult r) {
    const char *winner = r.v5 < r.antl ? "\033[32mV5\033[0m" :
                         r.antl < r.v5 ? "\033[33mANTL\033[0m" : "TIE";
    double ratio = r.antl > 0 ? r.v5 / r.antl : 0;
    printf("  %-42s │ %7.1f │ %7.1f │ %5.2fx │ %s\n",
           label, r.v5, r.antl, ratio, winner);
}

// ── Generic bench templates ──────────────────────────────────────────────────

template <class Map, class Key>
double bench_insert(const std::vector<Key> &keys) {
    Map m;
    auto t0 = hrc::now();
    for (auto &k : keys) m[k] = 1;
    return elapsed_ns(t0, hrc::now(), keys.size());
}

template <class Map, class Key>
double bench_find_hit(const std::vector<Key> &keys) {
    Map m;
    for (auto &k : keys) m[k] = 1;
    volatile uint64_t sink = 0; uint64_t acc = 0;
    auto t0 = hrc::now();
    for (auto &k : keys) {
        auto it = m.find(k);
        if (it != m.end()) acc++;
    }
    sink = acc;
    return elapsed_ns(t0, hrc::now(), keys.size());
}

template <class Map, class Key>
double bench_erase(const std::vector<Key> &keys) {
    Map m;
    for (auto &k : keys) m[k] = 1;
    auto t0 = hrc::now();
    for (auto &k : keys) m.erase(k);
    return elapsed_ns(t0, hrc::now(), keys.size());
}

template <class Key>
BenchResult run_trio(const std::vector<Key> &keys, const char *op) {
    using V5 = turbo::TurboMap<Key, uint64_t>;
    using AL = ankerl::unordered_dense::map<Key, uint64_t>;
    BenchResult r;
    if (strcmp(op, "INSERT") == 0) {
        r.v5 = bench_insert<V5>(keys);
        r.antl = bench_insert<AL>(keys);
    } else if (strcmp(op, "FIND") == 0) {
        r.v5 = bench_find_hit<V5>(keys);
        r.antl = bench_find_hit<AL>(keys);
    } else {
        r.v5 = bench_erase<V5>(keys);
        r.antl = bench_erase<AL>(keys);
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: CORRECTNESS SUITE (V5 only — these validate V5 behavior)
// ═══════════════════════════════════════════════════════════════════════════

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  \033[31mFAIL\033[0m: %s\n", msg); g_fail++; } \
    else { g_pass++; } \
} while(0)

static void correctness_suite() {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  V5 CORRECTNESS SUITE                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // T1: Basic insert + find (uint64)
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 1000; i++) m[i] = i * 10;
        bool ok = true;
        for (uint64_t i = 0; i < 1000; i++) {
            auto it = m.find(i);
            if (it == m.end() || it->second != i * 10) { ok = false; break; }
        }
        CHECK(ok, "T01: uint64 insert 1000 + find all with exact values");
        CHECK(m.size() == 1000, "T01b: size == 1000");
    }

    // T2: Insert duplicates don't increase size
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 500; i++) m[i] = i;
        for (uint64_t i = 0; i < 500; i++) m[i] = i + 1000;
        CHECK(m.size() == 500, "T02: duplicate inserts don't grow size (500)");
        auto it = m.find(42);
        CHECK(it != m.end() && it->second == 1042, "T02b: duplicate overwrites value (42→1042)");
    }

    // T3: Erase + verify gone
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 1000; i++) m[i] = i;
        for (uint64_t i = 0; i < 500; i++) m.erase(i);
        CHECK(m.size() == 500, "T03: erase 500/1000 → size 500");
        bool erased_ok = true, remain_ok = true;
        for (uint64_t i = 0; i < 500; i++)
            if (m.find(i) != m.end()) { erased_ok = false; break; }
        for (uint64_t i = 500; i < 1000; i++)
            if (m.find(i) == m.end()) { remain_ok = false; break; }
        CHECK(erased_ok, "T03b: erased keys not found");
        CHECK(remain_ok, "T03c: remaining keys all found");
    }

    // T4: count() method
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        m[42] = 1; m[99] = 2;
        CHECK(m.count(42) == 1, "T04: count(42)==1");
        CHECK(m.count(99) == 1, "T04b: count(99)==1");
        CHECK(m.count(100) == 0, "T04c: count(100)==0");
    }

    // T5: clear()
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 100; i++) m[i] = i;
        CHECK(m.size() == 100, "T05: size 100 before clear");
        m.clear();
        CHECK(m.size() == 0, "T05b: size 0 after clear");
        CHECK(m.empty(), "T05c: empty() after clear");
        CHECK(m.find(42) == m.end(), "T05d: find fails after clear");
    }

    // T6: String keys
    {
        turbo::TurboMap<std::string, uint64_t> m;
        m["hello"] = 1; m["world"] = 2; m["foo"] = 3;
        CHECK(m.size() == 3, "T06: string map size 3");
        CHECK(m.find("hello") != m.end() && m.find("hello")->second == 1,
              "T06b: find 'hello'→1");
        CHECK(m.find("missing") == m.end(), "T06c: find 'missing' → end");
        m.erase("hello");
        CHECK(m.find("hello") == m.end(), "T06d: erase 'hello' → gone");
    }

    // T7: Empty key
    {
        turbo::TurboMap<std::string, uint64_t> m;
        m[""] = 42;
        CHECK(m.size() == 1, "T07: empty string key size 1");
        CHECK(m.find("") != m.end() && m.find("")->second == 42,
              "T07b: find ''→42");
    }

    // T8: Single-byte keys (all 256 values)
    {
        turbo::TurboMap<uint8_t, uint64_t> m;
        for (int i = 0; i < 256; i++) m[(uint8_t)i] = i;
        CHECK(m.size() == 256, "T08: 256 single-byte keys");
        bool ok = true;
        for (int i = 0; i < 256; i++) {
            auto it = m.find((uint8_t)i);
            if (it == m.end() || it->second != (uint64_t)i) { ok = false; break; }
        }
        CHECK(ok, "T08b: all 256 byte keys found with correct values");
    }

    // T9: Large-scale correctness (100K random uint64)
    {
        g_rng.seed(999);
        turbo::TurboMap<uint64_t, uint64_t> m;
        std::vector<uint64_t> keys(100000);
        for (auto &k : keys) { k = g_rng(); m[k] = k; }
        uint32_t found = 0;
        for (auto &k : keys) if (m.find(k) != m.end()) found++;
        CHECK(found == m.size(), "T09: 100K random uint64 find all");
    }

    // T10: Large-scale string correctness (10K paths)
    {
        g_rng.seed(888);
        turbo::TurboMap<std::string, uint64_t> m;
        std::vector<std::string> keys(10000);
        for (uint32_t i = 0; i < 10000; i++) {
            keys[i] = make_path(2 + (g_rng() % 4));
            m[keys[i]] = i;
        }
        uint32_t found = 0;
        for (auto &k : keys) if (m.find(k) != m.end()) found++;
        CHECK(found == m.size(), "T10: 10K path strings find all");
    }

    // T11: Cursor forward iteration count
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 500; i++) m[i * 3] = i;
        uint32_t count = 0;
        for (auto it = m.begin(); it != m.end(); ++it) count++;
        CHECK(count == 500, "T11: cursor forward count == 500");
    }

    // T12: insert() returns pair<iterator, bool>
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        auto [it1, ok1] = m.insert({42, 100});
        CHECK(ok1, "T12: first insert returns true");
        CHECK(it1->second == 100, "T12b: first insert value 100");
        auto [it2, ok2] = m.insert({42, 200});
        CHECK(!ok2, "T12c: duplicate insert returns false");
    }

    // T13: emplace
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        auto [it, ok] = m.emplace(42, 999);
        CHECK(ok, "T13: emplace returns true");
        CHECK(it->second == 999, "T13b: emplace value 999");
    }

    // T14: operator[] default-creates
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        m[42] = 10;
        uint64_t &ref = m[42];
        CHECK(ref == 10, "T14: operator[] returns ref to existing value");
        m[99]; // creates slot
        CHECK(m.size() == 2, "T14b: operator[] on new key increases size");
    }

    // T15: Very long string keys (1KB)
    {
        g_rng.seed(777);
        turbo::TurboMap<std::string, uint64_t> m;
        std::vector<std::string> keys(100);
        for (int i = 0; i < 100; i++) {
            keys[i] = rand_string(1024);
            m[keys[i]] = i;
        }
        bool ok = true;
        for (int i = 0; i < 100; i++) {
            auto it = m.find(keys[i]);
            if (it == m.end() || it->second != (uint64_t)i) { ok = false; break; }
        }
        CHECK(ok, "T15: 1KB string keys insert + find exact values");
    }

    // T16: uint32 keys
    {
        turbo::TurboMap<uint32_t, uint32_t> m;
        for (uint32_t i = 0; i < 1000; i++) m[i] = i * 7;
        bool ok = true;
        for (uint32_t i = 0; i < 1000; i++) {
            auto it = m.find(i);
            if (it == m.end() || it->second != i * 7) { ok = false; break; }
        }
        CHECK(ok, "T16: uint32 1000 keys with exact values");
    }

    // T17: Mixed insert, erase, re-insert (churn)
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 1000; i++) m[i] = i;
        for (uint64_t i = 0; i < 500; i++) m.erase(i);
        for (uint64_t i = 0; i < 500; i++) m[i] = i + 5000;
        CHECK(m.size() == 1000, "T17: churn size back to 1000");
        auto it = m.find(42);
        CHECK(it != m.end() && it->second == 5042, "T17b: re-inserted value 42→5042");
        auto it2 = m.find(750);
        CHECK(it2 != m.end() && it2->second == 750, "T17c: untouched value 750");
    }

    // T18: Common prefix keys
    {
        turbo::TurboMap<std::string, uint64_t> m;
        for (int i = 0; i < 1000; i++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "/api/v1/users/%04d/profile", i);
            m[buf] = i;
        }
        CHECK(m.size() == 1000, "T18: 1000 common-prefix keys");
        auto it = m.find("/api/v1/users/0042/profile");
        CHECK(it != m.end() && it->second == 42, "T18b: find exact path key");
    }

    // T19: Zero-value key
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        m[0] = 999;
        CHECK(m.find(0) != m.end() && m.find(0)->second == 999,
              "T19: zero key insert + find");
    }

    // T20: Max-value key
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        m[UINT64_MAX] = 777;
        CHECK(m.find(UINT64_MAX) != m.end() && m.find(UINT64_MAX)->second == 777,
              "T20: max uint64 key insert + find");
    }

    // T21: Erase non-existent key
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        m[1] = 1;
        size_t r = m.erase(999);
        CHECK(r == 0, "T21: erase non-existent returns 0");
        CHECK(m.size() == 1, "T21b: size unchanged");
    }

    // T22: Large struct values
    {
        struct BigVal { uint64_t data[8]; };
        turbo::TurboMap<uint64_t, BigVal> m;
        BigVal bv; memset(&bv, 0, sizeof(bv));
        bv.data[0] = 42; bv.data[7] = 99;
        m[1] = bv;
        auto it = m.find(1);
        CHECK(it != m.end() && it->second.data[0] == 42 && it->second.data[7] == 99,
              "T22: 64-byte struct value roundtrip");
    }

    // T23: Erase during iteration equivalent (erase all)
    {
        turbo::TurboMap<uint64_t, uint64_t> m;
        for (uint64_t i = 0; i < 500; i++) m[i] = i;
        for (uint64_t i = 0; i < 500; i++) m.erase(i);
        CHECK(m.size() == 0, "T23: erase all → size 0");
        CHECK(m.empty(), "T23b: empty after erase all");
    }

    // T24: 1M correctness
    {
        g_rng.seed(555);
        turbo::TurboMap<uint64_t, uint64_t> m;
        std::vector<uint64_t> keys(1000000);
        for (auto &k : keys) { k = g_rng(); m[k] = k; }
        uint32_t found = 0;
        for (auto &k : keys) if (m.find(k) != m.end()) found++;
        CHECK(found == m.size(), "T24: 1M random uint64 → all found");
    }

    printf("\n  ────────────────────────────\n");
    printf("  PASS: %d  FAIL: %d\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("  \033[31m*** CORRECTNESS FAILURES — BENCHMARK SKIPPED ***\033[0m\n");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: PERFORMANCE BENCHMARK (20 concepts)
// ═══════════════════════════════════════════════════════════════════════════

static void perf_suite() {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  V5 vs ANKERL — ULTIMATE PERFORMANCE SHOWDOWN (ns/op, best-of) ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    printf("  %-42s │ %7s │ %7s │ %6s │ %s\n",
           "BENCHMARK", "V5", "ANTL", "RATIO", "WINNER");
    printf("  ──────────────────────────────────────────┼─────────┼─────────┼────"
           "────┼────────\n");

    using V5u64 = turbo::TurboMap<uint64_t, uint64_t>;
    using ALu64 = ankerl::unordered_dense::map<uint64_t, uint64_t>;
    using V5u32 = turbo::TurboMap<uint32_t, uint32_t>;
    using ALu32 = ankerl::unordered_dense::map<uint32_t, uint32_t>;
    using V5str = turbo::TurboMap<std::string, uint64_t>;
    using ALstr = ankerl::unordered_dense::map<std::string, uint64_t>;

    auto best = [](auto fn, int iters) {
        double best = 1e9;
        for (int i = 0; i < iters; i++) {
            double d = fn();
            if (d < best) best = d;
        }
        return best;
    };

    // ════════════════════════════════════════════════════════════════
    // §1: uint64 Random — 100K
    // ════════════════════════════════════════════════════════════════
    {
        const uint32_t N = 100000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        print_row("§01 uint64 Random INSERT 100K",
            {best([&]{ return bench_insert<V5u64>(keys); }, 5),
             best([&]{ return bench_insert<ALu64>(keys); }, 5)});
        print_row("§01 uint64 Random FIND 100K",
            {best([&]{ return bench_find_hit<V5u64>(keys); }, 5),
             best([&]{ return bench_find_hit<ALu64>(keys); }, 5)});
        print_row("§01 uint64 Random ERASE 100K",
            {best([&]{ return bench_erase<V5u64>(keys); }, 5),
             best([&]{ return bench_erase<ALu64>(keys); }, 5)});
    }

    // §2: uint64 Random — 1M
    {
        const uint32_t N = 1000000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        print_row("§02 uint64 Random INSERT 1M",
            {best([&]{ return bench_insert<V5u64>(keys); }, 3),
             best([&]{ return bench_insert<ALu64>(keys); }, 3)});
        print_row("§02 uint64 Random FIND 1M",
            {best([&]{ return bench_find_hit<V5u64>(keys); }, 3),
             best([&]{ return bench_find_hit<ALu64>(keys); }, 3)});
        print_row("§02 uint64 Random ERASE 1M",
            {best([&]{ return bench_erase<V5u64>(keys); }, 3),
             best([&]{ return bench_erase<ALu64>(keys); }, 3)});
    }

    // §3: uint64 Sequential — 1M
    {
        const uint32_t N = 1000000;
        std::vector<uint64_t> keys(N);
        for (uint32_t i = 0; i < N; i++) keys[i] = i;
        print_row("§03 uint64 Sequential INSERT 1M",
            {best([&]{ return bench_insert<V5u64>(keys); }, 3),
             best([&]{ return bench_insert<ALu64>(keys); }, 3)});
        print_row("§03 uint64 Sequential FIND 1M",
            {best([&]{ return bench_find_hit<V5u64>(keys); }, 3),
             best([&]{ return bench_find_hit<ALu64>(keys); }, 3)});
    }

    // §4: uint32 Random — 1M
    {
        const uint32_t N = 1000000;
        g_rng.seed(42);
        std::vector<uint32_t> keys(N);
        for (auto &k : keys) k = (uint32_t)g_rng();
        print_row("§04 uint32 Random INSERT 1M",
            {best([&]{ return bench_insert<V5u32>(keys); }, 3),
             best([&]{ return bench_insert<ALu32>(keys); }, 3)});
        print_row("§04 uint32 Random FIND 1M",
            {best([&]{ return bench_find_hit<V5u32>(keys); }, 3),
             best([&]{ return bench_find_hit<ALu32>(keys); }, 3)});
    }

    // §5: String Short (8 bytes) — 500K
    {
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = rand_string(8);
        print_row("§05 String Short 8B INSERT 500K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§05 String Short 8B FIND 500K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §6: String Medium (32 bytes) — 500K
    {
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = rand_string(32);
        print_row("§06 String Medium 32B INSERT 500K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§06 String Medium 32B FIND 500K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §7: String Long (128 bytes) — 200K
    {
        const uint32_t N = 200000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = rand_string(128);
        print_row("§07 String Long 128B INSERT 200K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§07 String Long 128B FIND 200K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §8: String Variable Length (4-256 bytes) — 500K
    {
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = rand_string(4 + (g_rng() % 253));
        print_row("§08 String Variable 4-256B INSERT 500K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§08 String Variable 4-256B FIND 500K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §9: FIND MISS — 1M
    {
        const uint32_t N = 1000000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        V5u64 v5m; ALu64 alm;
        for (auto &k : keys) { v5m[k] = 1; alm[k] = 1; }
        // Miss keys
        std::vector<uint64_t> miss(N);
        for (auto &k : miss) k = g_rng();
        auto bfm = [&](auto &m) {
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &k : miss) { auto it = m.find(k); if (it != m.end()) a++; }
            s = a;
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§09 uint64 FIND MISS 1M",
            {best([&]{ return bfm(v5m); }, 3),
             best([&]{ return bfm(alm); }, 3)});
    }

    // §10: Path-like strings — 100K
    {
        const uint32_t N = 100000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = make_path(2 + (g_rng() % 5));
        print_row("§10 Path Strings INSERT 100K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§10 Path Strings FIND 100K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §11: INSERT duplicate overwrite — 1M
    {
        const uint32_t N = 1000000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        auto bench_dup_v5 = [&] {
            V5u64 m;
            for (auto &k : keys) m[k] = 1;
            auto t0 = hrc::now();
            for (auto &k : keys) m[k] = 2;
            return elapsed_ns(t0, hrc::now(), N);
        };
        auto bench_dup_al = [&] {
            ALu64 m;
            for (auto &k : keys) m[k] = 1;
            auto t0 = hrc::now();
            for (auto &k : keys) m[k] = 2;
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§11 uint64 Duplicate Overwrite 1M",
            {best(bench_dup_v5, 3), best(bench_dup_al, 3)});
    }

    // §12: Iteration — 1M
    {
        const uint32_t N = 1000000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        V5u64 v5m; ALu64 alm;
        for (auto &k : keys) { v5m[k] = 1; alm[k] = 1; }
        auto bench_iter_v5 = [&] {
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto it = v5m.begin(); it != v5m.end(); ++it) a += it->second;
            s = a;
            return elapsed_ns(t0, hrc::now(), N);
        };
        auto bench_iter_al = [&] {
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &[k,v] : alm) a += v;
            s = a;
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§12 Iteration 1M entries",
            {best(bench_iter_v5, 3), best(bench_iter_al, 3)});
    }

    // §13: Mixed INSERT + FIND interleaved — 500K each
    {
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N * 2);
        for (auto &k : keys) k = g_rng();
        auto bench_mixed = [&](auto make_map) {
            auto m = make_map();
            auto t0 = hrc::now();
            for (uint32_t i = 0; i < N; i++) {
                m[keys[i]] = i;
                if (i > 0) { auto it = m.find(keys[i/2]); (void)it; }
            }
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§13 Mixed INSERT+FIND 500K",
            {best([&]{ return bench_mixed([]{ return V5u64{}; }); }, 3),
             best([&]{ return bench_mixed([]{ return ALu64{}; }); }, 3)});
    }

    // §14: Churn (insert N, erase N/2, re-insert N/2) — 500K
    {
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        auto bench_churn = [&](auto make_map) {
            auto m = make_map();
            for (auto &k : keys) m[k] = 1;
            auto t0 = hrc::now();
            for (uint32_t i = 0; i < N/2; i++) m.erase(keys[i]);
            for (uint32_t i = 0; i < N/2; i++) m[keys[i]] = 2;
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§14 Churn erase+reinsert 500K",
            {best([&]{ return bench_churn([]{ return V5u64{}; }); }, 3),
             best([&]{ return bench_churn([]{ return ALu64{}; }); }, 3)});
    }

    // §15: Very Long Keys (1KB strings) — 50K
    {
        const uint32_t N = 50000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (auto &k : keys) k = rand_string(1024);
        print_row("§15 String 1KB INSERT 50K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§15 String 1KB FIND 50K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §16: Common-prefix keys (API paths) — 100K
    {
        const uint32_t N = 100000;
        g_rng.seed(42);
        std::vector<std::string> keys(N);
        for (uint32_t i = 0; i < N; i++) {
            char buf[128];
            snprintf(buf, sizeof(buf), "/api/v1/users/%06u/profile/settings", i);
            keys[i] = buf;
        }
        print_row("§16 Common-prefix paths INSERT 100K",
            {best([&]{ return bench_insert<V5str>(keys); }, 3),
             best([&]{ return bench_insert<ALstr>(keys); }, 3)});
        print_row("§16 Common-prefix paths FIND 100K",
            {best([&]{ return bench_find_hit<V5str>(keys); }, 3),
             best([&]{ return bench_find_hit<ALstr>(keys); }, 3)});
    }

    // §17: Zipf distribution (hot keys) — 1M ops, 100K universe
    {
        const uint32_t UNIVERSE = 100000;
        const uint32_t OPS = 1000000;
        g_rng.seed(42);
        // Zipf-like: key = universe * (rand^2)
        std::vector<uint64_t> init_keys(UNIVERSE);
        for (auto &k : init_keys) k = g_rng();
        std::vector<uint64_t> ops(OPS);
        for (auto &k : ops) {
            double r = (double)(g_rng() % 10000) / 10000.0;
            uint32_t idx = (uint32_t)(r * r * UNIVERSE);
            if (idx >= UNIVERSE) idx = UNIVERSE - 1;
            k = init_keys[idx];
        }
        V5u64 v5m; ALu64 alm;
        for (auto &k : init_keys) { v5m[k] = 1; alm[k] = 1; }
        auto bz = [&](auto &m) {
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &k : ops) { auto it = m.find(k); if (it!=m.end()) a++; }
            s = a;
            return elapsed_ns(t0, hrc::now(), OPS);
        };
        print_row("§17 Zipf FIND (hot keys) 1M ops",
            {best([&]{ return bz(v5m); }, 3),
             best([&]{ return bz(alm); }, 3)});
    }

    // §18: Reverse sequential — 1M
    {
        const uint32_t N = 1000000;
        std::vector<uint64_t> keys(N);
        for (uint32_t i = 0; i < N; i++) keys[i] = N - 1 - i;
        print_row("§18 uint64 Reverse Sequential INSERT 1M",
            {best([&]{ return bench_insert<V5u64>(keys); }, 3),
             best([&]{ return bench_insert<ALu64>(keys); }, 3)});
        print_row("§18 uint64 Reverse Sequential FIND 1M",
            {best([&]{ return bench_find_hit<V5u64>(keys); }, 3),
             best([&]{ return bench_find_hit<ALu64>(keys); }, 3)});
    }

    // §19: Large values (64-byte struct) — 500K
    {
        struct BigVal { uint64_t d[8] = {}; };
        using V5big = turbo::TurboMap<uint64_t, BigVal>;
        using ALbig = ankerl::unordered_dense::map<uint64_t, BigVal>;
        const uint32_t N = 500000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(N);
        for (auto &k : keys) k = g_rng();
        auto bench_ins_v5 = [&] {
            V5big m; BigVal bv{};
            auto t0 = hrc::now();
            for (auto &k : keys) { bv.d[0] = k; m[k] = bv; }
            return elapsed_ns(t0, hrc::now(), N);
        };
        auto bench_ins_al = [&] {
            ALbig m; BigVal bv{};
            auto t0 = hrc::now();
            for (auto &k : keys) { bv.d[0] = k; m[k] = bv; }
            return elapsed_ns(t0, hrc::now(), N);
        };
        auto bench_find_v5 = [&] {
            V5big m; BigVal bv{};
            for (auto &k : keys) { bv.d[0] = k; m[k] = bv; }
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &k : keys) { auto it = m.find(k); if(it!=m.end()) a++; }
            s = a;
            return elapsed_ns(t0, hrc::now(), N);
        };
        auto bench_find_al = [&] {
            ALbig m; BigVal bv{};
            for (auto &k : keys) { bv.d[0] = k; m[k] = bv; }
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &k : keys) { auto it = m.find(k); if(it!=m.end()) a++; }
            s = a;
            return elapsed_ns(t0, hrc::now(), N);
        };
        print_row("§19 Large Value 64B INSERT 500K",
            {best(bench_ins_v5, 3), best(bench_ins_al, 3)});
        print_row("§19 Large Value 64B FIND 500K",
            {best(bench_find_v5, 3), best(bench_find_al, 3)});
    }

    // §20: Small map hot (100 entries, 1M lookups)
    {
        const uint32_t MAP_SZ = 100;
        const uint32_t OPS = 1000000;
        g_rng.seed(42);
        std::vector<uint64_t> keys(MAP_SZ);
        for (auto &k : keys) k = g_rng();
        V5u64 v5m; ALu64 alm;
        for (auto &k : keys) { v5m[k] = 1; alm[k] = 1; }
        std::vector<uint64_t> ops(OPS);
        for (auto &k : ops) k = keys[g_rng() % MAP_SZ];
        auto bh = [&](auto &m) {
            volatile uint64_t s = 0; uint64_t a = 0;
            auto t0 = hrc::now();
            for (auto &k : ops) { auto it = m.find(k); if(it!=m.end()) a++; }
            s = a;
            return elapsed_ns(t0, hrc::now(), OPS);
        };
        print_row("§20 Small Map 100 entries, 1M FIND",
            {best([&]{ return bh(v5m); }, 5),
             best([&]{ return bh(alm); }, 5)});
    }

    printf("  ──────────────────────────────────────────┴─────────┴─────────┴────"
           "────┴────────\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    correctness_suite();
    if (g_fail > 0) return 1;
    perf_suite();
    return 0;
}
