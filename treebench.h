#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace treebench {

using Clock = std::chrono::steady_clock;

inline double ns_since(Clock::time_point start) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

template <typename Func>
inline double bench_op(int n, Func &&fn) {
  auto start = Clock::now();
  fn();
  return ns_since(start) / static_cast<double>(n);
}

inline std::string make_binary_key(const uint8_t *key, uint32_t key_length) {
  return std::string(reinterpret_cast<const char *>(key), key_length);
}

class UMapHarness {
 public:
  UMapHarness() = default;

  template <typename Ignored>
  explicit UMapHarness(Ignored &&) {}

  bool insert(const uint8_t *key, uint32_t key_length, uint64_t value) {
    std::string encoded = make_binary_key(key, key_length);
    auto it = map_.find(encoded);
    if (it == map_.end()) {
      map_.emplace(std::move(encoded), value);
      return true;
    }
    it->second = value;
    return false;
  }

  bool get(const uint8_t *key, uint32_t key_length, uint64_t &value) const {
    auto it = map_.find(make_binary_key(key, key_length));
    if (it == map_.end()) {
      return false;
    }
    value = it->second;
    return true;
  }

  size_t count() const {
    return map_.size();
  }

  size_t cursor_count() const {
    return map_.size();
  }

  void collect_sorted(std::vector<std::pair<std::string, uint64_t>> &out) const {
    out.clear();
    out.reserve(map_.size());
    for (const auto &entry : map_) {
      out.push_back(entry);
    }
    std::sort(out.begin(), out.end(), [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
  }

 private:
  std::unordered_map<std::string, uint64_t> map_;
};

class ConcurrentUMapHarness {
 public:
  bool insert(const uint8_t *key, uint32_t key_length, uint64_t value) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string encoded = make_binary_key(key, key_length);
    auto it = map_.find(encoded);
    if (it == map_.end()) {
      map_.emplace(std::move(encoded), value);
      return true;
    }
    it->second = value;
    return false;
  }

  bool get(const uint8_t *key, uint32_t key_length, uint64_t &value) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = map_.find(make_binary_key(key, key_length));
    if (it == map_.end()) {
      return false;
    }
    value = it->second;
    return true;
  }

  size_t count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.size();
  }

  size_t cursor_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.size();
  }

  void collect_sorted(std::vector<std::pair<std::string, uint64_t>> &out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    out.clear();
    out.reserve(map_.size());
    for (const auto &entry : map_) {
      out.push_back(entry);
    }
    std::sort(out.begin(), out.end(), [](const auto &left, const auto &right) {
      return left.first < right.first;
    });
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, uint64_t> map_;
};

struct Key8 {
  uint8_t b[8];
};

struct Key16 {
  uint8_t b[16];
};

inline uint64_t hash_mix(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

inline std::vector<Key8> make_seq_be(int n) {
  std::vector<Key8> keys(n);
  for (int i = 0; i < n; ++i) {
    uint64_t value = __builtin_bswap64(static_cast<uint64_t>(i));
    std::memcpy(keys[i].b, &value, sizeof(value));
  }
  return keys;
}

inline std::vector<Key8> make_random(int n) {
  std::vector<Key8> keys(n);
  for (int i = 0; i < n; ++i) {
    uint64_t value = hash_mix(static_cast<uint64_t>(i));
    std::memcpy(keys[i].b, &value, sizeof(value));
  }
  return keys;
}

inline std::vector<Key8> make_prefixed(int n) {
  std::vector<Key8> keys(n);
  for (int i = 0; i < n; ++i) {
    uint32_t prefix = static_cast<uint32_t>(i / 100);
    uint32_t suffix = static_cast<uint32_t>(i % 100);
    std::memcpy(keys[i].b, &prefix, sizeof(prefix));
    std::memcpy(keys[i].b + 4, &suffix, sizeof(suffix));
  }
  return keys;
}

inline std::vector<Key8> make_lofan7(int n) {
  std::vector<Key8> keys(n);
  std::mt19937 rng(777);
  std::uniform_int_distribution<uint8_t> dist(0, 6);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 8; ++j) {
      keys[i].b[j] = dist(rng);
    }
  }
  return keys;
}

inline std::vector<Key16> make_seq_be_16(int n) {
  std::vector<Key16> keys(n);
  for (int i = 0; i < n; ++i) {
    std::memset(keys[i].b, 0, 8);
    uint64_t value = __builtin_bswap64(static_cast<uint64_t>(i));
    std::memcpy(keys[i].b + 8, &value, sizeof(value));
  }
  return keys;
}

inline std::vector<Key16> make_hash_16(int n) {
  std::vector<Key16> keys(n);
  for (int i = 0; i < n; ++i) {
    uint64_t hi = hash_mix(static_cast<uint64_t>(i));
    uint64_t lo = hash_mix(hi);
    std::memcpy(keys[i].b, &hi, sizeof(hi));
    std::memcpy(keys[i].b + 8, &lo, sizeof(lo));
  }
  return keys;
}

struct TestSummary {
  int passed = 0;
  int failed = 0;
};

template <typename MapType>
inline bool expect_present(MapType &map,
                           const std::vector<uint8_t> &key,
                           uint64_t expected,
                           const char *label,
                           bool verbose,
                           TestSummary &summary,
                           FILE *out) {
  uint64_t found = 0;
  bool ok = map.get(key.data(), static_cast<uint32_t>(key.size()), found) && found == expected;
  if (!ok) {
    std::fprintf(out, "    FAIL: %s\n", label);
    summary.failed++;
    return false;
  }
  if (verbose) {
    std::fprintf(out, "    PASS: %s\n", label);
  }
  summary.passed++;
  return true;
}

template <typename MapType>
inline bool expect_absent(MapType &map,
                          const std::vector<uint8_t> &key,
                          const char *label,
                          bool verbose,
                          TestSummary &summary,
                          FILE *out) {
  uint64_t found = 0;
  bool ok = !map.get(key.data(), static_cast<uint32_t>(key.size()), found);
  if (!ok) {
    std::fprintf(out, "    FAIL: %s\n", label);
    summary.failed++;
    return false;
  }
  if (verbose) {
    std::fprintf(out, "    PASS: %s\n", label);
  }
  summary.passed++;
  return true;
}

template <typename MapType>
inline void insert_key(MapType &map, const std::vector<uint8_t> &key, uint64_t value) {
  map.insert(key.data(), static_cast<uint32_t>(key.size()), value);
}

inline std::vector<uint8_t> bytes_from_string(const std::string &value) {
  return std::vector<uint8_t>(value.begin(), value.end());
}

inline std::vector<uint8_t> repeated_byte(uint8_t byte, size_t count) {
  return std::vector<uint8_t>(count, byte);
}

template <typename MapType = UMapHarness>
inline TestSummary run_edge_case_suite(bool verbose = true, FILE *out = stdout) {
  TestSummary summary;

  std::fprintf(out, "================================================================\n");
  std::fprintf(out, "  Treebench Edge Case Suite (UMAP only)\n");
  std::fprintf(out, "================================================================\n");

  {
    std::fprintf(out, "\n  T1: prefix-of-another-key\n");
    MapType map;
    insert_key(map, bytes_from_string("abc"), 100);
    insert_key(map, bytes_from_string("abcd"), 200);
    insert_key(map, bytes_from_string("abcde"), 300);
    expect_present(map, bytes_from_string("abc"), 100, "T1: abc", verbose, summary, out);
    expect_present(map, bytes_from_string("abcd"), 200, "T1: abcd", verbose, summary, out);
    expect_present(map, bytes_from_string("abcde"), 300, "T1: abcde", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T2: all 256 single-byte keys\n");
    MapType map;
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>(i)};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>(i)};
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T2: 1-byte key", false, summary, out);
    }
    std::fprintf(out, "    PASS: all 256 single-byte keys verified\n");
  }

  {
    std::fprintf(out, "\n  T3: all 65536 two-byte combinations\n");
    MapType map;
    for (int hi = 0; hi < 256; ++hi) {
      for (int lo = 0; lo < 256; ++lo) {
        std::vector<uint8_t> key = {static_cast<uint8_t>(hi), static_cast<uint8_t>(lo)};
        insert_key(map, key, static_cast<uint64_t>((hi << 8) | lo));
      }
    }
    for (int hi = 0; hi < 256; ++hi) {
      for (int lo = 0; lo < 256; ++lo) {
        std::vector<uint8_t> key = {static_cast<uint8_t>(hi), static_cast<uint8_t>(lo)};
        expect_present(map, key, static_cast<uint64_t>((hi << 8) | lo), "T3: 2-byte key", false, summary, out);
      }
    }
    std::fprintf(out, "    PASS: all 65536 two-byte keys verified\n");
  }

  {
    std::fprintf(out, "\n  T4: long keys (128 bytes)\n");
    MapType map;
    auto a = repeated_byte('A', 128);
    auto b = a;
    auto c = a;
    b[127] = 'B';
    c[0] = 'C';
    insert_key(map, a, 1);
    insert_key(map, b, 2);
    insert_key(map, c, 3);
    expect_present(map, a, 1, "T4: A*128", verbose, summary, out);
    expect_present(map, b, 2, "T4: differ-last", verbose, summary, out);
    expect_present(map, c, 3, "T4: differ-first", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T5: binary keys with null bytes\n");
    MapType map;
    std::vector<std::vector<uint8_t>> keys = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}, {0, 0}};
    for (size_t i = 0; i < keys.size(); ++i) {
      insert_key(map, keys[i], static_cast<uint64_t>((i + 1) * 10));
    }
    for (size_t i = 0; i < keys.size(); ++i) {
      expect_present(map, keys[i], static_cast<uint64_t>((i + 1) * 10), "T5: binary null key", false, summary, out);
    }
    std::fprintf(out, "    PASS: null-byte keys verified\n");
  }

  {
    std::fprintf(out, "\n  T6: all-0xFF keys\n");
    MapType map;
    auto a = repeated_byte(0xFF, 4);
    auto b = repeated_byte(0xFF, 8);
    auto c = repeated_byte(0xFF, 1);
    insert_key(map, a, 111);
    insert_key(map, b, 222);
    insert_key(map, c, 333);
    expect_present(map, a, 111, "T6: 0xFFx4", verbose, summary, out);
    expect_present(map, b, 222, "T6: 0xFFx8", verbose, summary, out);
    expect_present(map, c, 333, "T6: 0xFFx1", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T7: duplicate insert overwrite\n");
    MapType map;
    auto key = bytes_from_string("dup-key");
    bool first = map.insert(key.data(), static_cast<uint32_t>(key.size()), 100);
    bool second = map.insert(key.data(), static_cast<uint32_t>(key.size()), 200);
    if (!first || second) {
      std::fprintf(out, "    FAIL: T7 insert return values\n");
      summary.failed++;
    } else {
      summary.passed++;
      if (verbose) {
        std::fprintf(out, "    PASS: T7 insert return values\n");
      }
    }
    expect_present(map, key, 200, "T7: overwrite value", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T8: forward prefix chain\n");
    MapType map;
    std::string seed = "abcdefgh";
    for (int len = 1; len <= 8; ++len) {
      insert_key(map, bytes_from_string(seed.substr(0, len)), static_cast<uint64_t>(len * 10));
    }
    for (int len = 1; len <= 8; ++len) {
      expect_present(map,
                     bytes_from_string(seed.substr(0, len)),
                     static_cast<uint64_t>(len * 10),
                     "T8: prefix chain",
                     false,
                     summary,
                     out);
    }
    std::fprintf(out, "    PASS: forward prefix chain verified\n");
  }

  {
    std::fprintf(out, "\n  T9: reverse prefix chain\n");
    MapType map;
    std::string seed = "zyxwvuts";
    for (int len = 1; len <= 8; ++len) {
      insert_key(map, bytes_from_string(seed.substr(0, len)), static_cast<uint64_t>(len * 10));
    }
    for (int len = 1; len <= 8; ++len) {
      expect_present(map,
                     bytes_from_string(seed.substr(0, len)),
                     static_cast<uint64_t>(len * 10),
                     "T9: reverse prefix chain",
                     false,
                     summary,
                     out);
    }
    std::fprintf(out, "    PASS: reverse prefix chain verified\n");
  }

  {
    std::fprintf(out, "\n  T10: 256 keys differing only in last byte\n");
    MapType map;
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {'A', 'A', 'A', 'A', 'A', 'A', 'A', static_cast<uint8_t>(i)};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {'A', 'A', 'A', 'A', 'A', 'A', 'A', static_cast<uint8_t>(i)};
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T10: last-byte", false, summary, out);
    }
    std::fprintf(out, "    PASS: last-byte fan-out verified\n");
  }

  {
    std::fprintf(out, "\n  T11: 256 keys differing only in first byte\n");
    MapType map;
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>(i), 'B', 'B', 'B', 'B', 'B', 'B', 'B'};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>(i), 'B', 'B', 'B', 'B', 'B', 'B', 'B'};
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T11: first-byte", false, summary, out);
    }
    std::fprintf(out, "    PASS: first-byte fan-out verified\n");
  }

  {
    std::fprintf(out, "\n  T12: mixed variable-length keys\n");
    MapType map;
    for (int len = 1; len <= 64; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), static_cast<uint8_t>(len));
      insert_key(map, key, static_cast<uint64_t>(len));
    }
    for (int len = 1; len <= 64; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), static_cast<uint8_t>(len));
      expect_present(map, key, static_cast<uint64_t>(len), "T12: variable-length", false, summary, out);
    }
    std::fprintf(out, "    PASS: variable-length keys verified\n");
  }

  {
    std::fprintf(out, "\n  T13: embedded nulls in middle of keys\n");
    MapType map;
    std::vector<std::vector<uint8_t>> keys = {
        {'h', 'e', 0, 'l', 'o'}, {'h', 'e', 0, 'l', 'p'}, {'h', 'e', 0, 0, 0}};
    insert_key(map, keys[0], 501);
    insert_key(map, keys[1], 502);
    insert_key(map, keys[2], 503);
    expect_present(map, keys[0], 501, "T13: he\\0lo", verbose, summary, out);
    expect_present(map, keys[1], 502, "T13: he\\0lp", verbose, summary, out);
    expect_present(map, keys[2], 503, "T13: he\\0\\0\\0", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T14: very long common prefix\n");
    MapType map;
    auto a = repeated_byte('P', 200);
    auto b = a;
    auto c = a;
    b[199] = 'Q';
    c[100] = 'R';
    insert_key(map, a, 1001);
    insert_key(map, b, 1002);
    insert_key(map, c, 1003);
    expect_present(map, a, 1001, "T14: same", verbose, summary, out);
    expect_present(map, b, 1002, "T14: differ@199", verbose, summary, out);
    expect_present(map, c, 1003, "T14: differ@100", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T15: interleaved insert/get\n");
    MapType map;
    for (int i = 0; i < 1000; ++i) {
      uint64_t value = hash_mix(static_cast<uint64_t>(i));
      std::vector<uint8_t> key(8);
      std::memcpy(key.data(), &value, sizeof(value));
      insert_key(map, key, static_cast<uint64_t>(i + 1));
      for (int j = 0; j <= i; ++j) {
        uint64_t probe = hash_mix(static_cast<uint64_t>(j));
        std::vector<uint8_t> probe_key(8);
        std::memcpy(probe_key.data(), &probe, sizeof(probe));
        expect_present(map, probe_key, static_cast<uint64_t>(j + 1), "T15: interleaved", false, summary, out);
      }
    }
    std::fprintf(out, "    PASS: interleaved insert/get verified\n");
  }

  {
    std::fprintf(out, "\n  T16: alternating key lengths 1-7\n");
    MapType map;
    for (int i = 0; i < 500; ++i) {
      int len = (i % 7) + 1;
      std::vector<uint8_t> key(static_cast<size_t>(len), static_cast<uint8_t>(i & 0xFF));
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 500; ++i) {
      int len = (i % 7) + 1;
      std::vector<uint8_t> key(static_cast<size_t>(len), static_cast<uint8_t>(i & 0xFF));
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T16: alternating-length", false, summary, out);
    }
    std::fprintf(out, "    PASS: alternating lengths verified\n");
  }

  {
    std::fprintf(out, "\n  T17: monotonically increasing shared prefix\n");
    MapType map;
    for (int len = 1; len <= 100; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), 'M');
      key.back() = static_cast<uint8_t>(len);
      insert_key(map, key, static_cast<uint64_t>(len));
    }
    for (int len = 1; len <= 100; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), 'M');
      key.back() = static_cast<uint8_t>(len);
      expect_present(map, key, static_cast<uint64_t>(len), "T17: monotone-length", false, summary, out);
    }
    std::fprintf(out, "    PASS: monotone shared-prefix keys verified\n");
  }

  {
    std::fprintf(out, "\n  T18: binary counter keys\n");
    MapType map;
    for (int i = 0; i < 4096; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>((i >> 8) & 0xFF), static_cast<uint8_t>(i & 0xFF)};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 4096; ++i) {
      std::vector<uint8_t> key = {static_cast<uint8_t>((i >> 8) & 0xFF), static_cast<uint8_t>(i & 0xFF)};
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T18: counter", false, summary, out);
    }
    std::fprintf(out, "    PASS: binary counter keys verified\n");
  }

  {
    std::fprintf(out, "\n  T19: miss keys\n");
    MapType map;
    auto exists = bytes_from_string("exists");
    insert_key(map, exists, 999);
    expect_present(map, exists, 999, "T19: exists", verbose, summary, out);
    expect_absent(map, bytes_from_string("exist"), "T19: miss prefix", verbose, summary, out);
    expect_absent(map, bytes_from_string("exists!"), "T19: miss extension", verbose, summary, out);
    expect_absent(map, bytes_from_string("different"), "T19: miss different", verbose, summary, out);
    expect_absent(map, bytes_from_string("EXISTS"), "T19: miss case-diff", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T20: stress 50K random-length binary keys\n");
    MapType map;
    std::unordered_map<std::string, uint64_t> expected;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> len_dist(1, 32);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    for (int i = 0; i < 50000; ++i) {
      int len = len_dist(rng);
      std::vector<uint8_t> key(static_cast<size_t>(len));
      for (int j = 0; j < len; ++j) {
        key[j] = static_cast<uint8_t>(byte_dist(rng));
      }
      uint64_t value = static_cast<uint64_t>(i) + 1;
      insert_key(map, key, value);
      expected[make_binary_key(key.data(), static_cast<uint32_t>(key.size()))] = value;
    }
    bool ok = true;
    for (const auto &entry : expected) {
      uint64_t found = 0;
      ok = map.get(reinterpret_cast<const uint8_t *>(entry.first.data()), static_cast<uint32_t>(entry.first.size()), found) && found == entry.second;
      if (!ok) {
        std::fprintf(out, "    FAIL: T20: stress verification\n");
        summary.failed++;
        break;
      }
    }
    if (ok) {
      summary.passed++;
      std::fprintf(out, "    PASS: T20: stress 50K — %zu/%zu unique keys verified\n", expected.size(), expected.size());
    }
  }

  {
    std::fprintf(out, "\n  T21: all-zero prefix chain\n");
    MapType map;
    for (int len = 1; len <= 32; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), 0);
      insert_key(map, key, static_cast<uint64_t>(len * 111));
    }
    for (int len = 1; len <= 32; ++len) {
      std::vector<uint8_t> key(static_cast<size_t>(len), 0);
      expect_present(map, key, static_cast<uint64_t>(len * 111), "T21: zero-prefix", false, summary, out);
    }
    std::fprintf(out, "    PASS: all-zero prefix chain verified\n");
  }

  {
    std::fprintf(out, "\n  T22: single-byte 0x00 vs multi-byte 0x00-prefix\n");
    MapType map;
    std::vector<std::pair<std::vector<uint8_t>, uint64_t>> cases = {
        {{0x00}, 10}, {{0x00, 0x00}, 20}, {{0x00, 0x00, 0x00}, 30},
        {{0x00, 0x01}, 40}, {{0x00, 0x00, 0x01}, 50}, {{0x01}, 60}};
    for (const auto &entry : cases) {
      insert_key(map, entry.first, entry.second);
    }
    for (const auto &entry : cases) {
      expect_present(map, entry.first, entry.second, "T22: null-prefix", false, summary, out);
    }
    std::fprintf(out, "    PASS: null-prefix fan-out verified\n");
  }

  {
    std::fprintf(out, "\n  T23: overwrite prefix key\n");
    MapType map;
    auto prefix = bytes_from_string("prefix");
    auto long_key = bytes_from_string("prefix-long-value");
    insert_key(map, prefix, 100);
    insert_key(map, long_key, 200);
    map.insert(prefix.data(), static_cast<uint32_t>(prefix.size()), 999);
    expect_present(map, long_key, 200, "T23: long key intact", verbose, summary, out);
    expect_present(map, prefix, 999, "T23: prefix overwritten", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T24: reverse keys\n");
    MapType map;
    std::vector<std::pair<std::vector<uint8_t>, uint64_t>> cases = {
        {{1, 2, 3, 4, 5}, 12345}, {{5, 4, 3, 2, 1}, 54321}, {{1, 2, 3, 4}, 1234}, {{4, 3, 2, 1}, 4321}};
    for (const auto &entry : cases) {
      insert_key(map, entry.first, entry.second);
    }
    for (const auto &entry : cases) {
      expect_present(map, entry.first, entry.second, "T24: reverse-key", false, summary, out);
    }
    std::fprintf(out, "    PASS: reverse-key set verified\n");
  }

  {
    std::fprintf(out, "\n  T25: alternating 0x00/0xFF patterns\n");
    MapType map;
    uint64_t value = 1;
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (int len = 1; len <= 16; ++len) {
        std::vector<uint8_t> key(static_cast<size_t>(len));
        for (int i = 0; i < len; ++i) {
          switch (pattern) {
            case 0: key[i] = (i & 1) ? 0xFF : 0x00; break;
            case 1: key[i] = (i & 1) ? 0x00 : 0xFF; break;
            case 2: key[i] = 0x00; break;
            default: key[i] = 0xFF; break;
          }
        }
        insert_key(map, key, value++);
      }
    }
    if (map.count() == 64) {
      summary.passed++;
      std::fprintf(out, "    PASS: all 64 alternating-pattern keys stored\n");
    } else {
      summary.failed++;
      std::fprintf(out, "    FAIL: alternating-pattern key count\n");
    }
  }

  {
    std::fprintf(out, "\n  T26: very long key with prefixes\n");
    MapType map;
    auto long_key = repeated_byte('L', 1000);
    insert_key(map, long_key, 77777);
    for (int len : {1, 100, 500, 999}) {
      insert_key(map, repeated_byte('L', static_cast<size_t>(len)), static_cast<uint64_t>(len));
    }
    expect_present(map, long_key, 77777, "T26: 1000-byte key", verbose, summary, out);
    expect_present(map, repeated_byte('L', 1), 1, "T26: prefix len=1", verbose, summary, out);
    expect_present(map, repeated_byte('L', 100), 100, "T26: prefix len=100", verbose, summary, out);
    expect_present(map, repeated_byte('L', 500), 500, "T26: prefix len=500", verbose, summary, out);
    expect_present(map, repeated_byte('L', 999), 999, "T26: prefix len=999", verbose, summary, out);
    auto miss = repeated_byte('L', 1000);
    miss.back() = 'M';
    expect_absent(map, miss, "T26: near-match miss", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T27: prefix fan-out\n");
    MapType map;
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {'A', 'A', 'A', 'A', static_cast<uint8_t>(i)};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    for (int i = 0; i < 256; ++i) {
      std::vector<uint8_t> key = {'A', 'A', 'A', 'A', static_cast<uint8_t>(i)};
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T27: prefix fan-out", false, summary, out);
    }
    std::fprintf(out, "    PASS: prefix + 256 suffix keys verified\n");
  }

  {
    std::fprintf(out, "\n  T28: insert return value and count accuracy\n");
    MapType map;
    auto k1 = bytes_from_string("k1");
    auto k2 = bytes_from_string("k2");
    auto k3 = bytes_from_string("k3");
    bool a = map.insert(k1.data(), static_cast<uint32_t>(k1.size()), 10);
    bool b = map.insert(k2.data(), static_cast<uint32_t>(k2.size()), 20);
    bool c = map.insert(k3.data(), static_cast<uint32_t>(k3.size()), 30);
    bool d = map.insert(k1.data(), static_cast<uint32_t>(k1.size()), 11);
    bool e = map.insert(k2.data(), static_cast<uint32_t>(k2.size()), 22);
    if (a && b && c && !d && !e && map.count() == 3) {
      summary.passed++;
      std::fprintf(out, "    PASS: T28 insert returns and count are correct\n");
    } else {
      summary.failed++;
      std::fprintf(out, "    FAIL: T28 insert returns or count incorrect\n");
    }
    expect_present(map, k1, 11, "T28: k1 overwritten", verbose, summary, out);
    expect_present(map, k2, 22, "T28: k2 overwritten", verbose, summary, out);
    expect_present(map, k3, 30, "T28: k3 intact", verbose, summary, out);
  }

  {
    std::fprintf(out, "\n  T29: cursor_count matches count\n");
    MapType map;
    for (int i = 0; i < 64; ++i) {
      std::vector<uint8_t> key = {'P', 'F', 'X', static_cast<uint8_t>(i)};
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    if (map.count() == 64 && map.cursor_count() == 64) {
      summary.passed++;
      std::fprintf(out, "    PASS: T29 count == cursor_count == 64\n");
    } else {
      summary.failed++;
      std::fprintf(out, "    FAIL: T29 cursor_count mismatch\n");
    }
  }

  {
    std::fprintf(out, "\n  T30: massive prefix chain variant\n");
    MapType map;
    std::vector<uint8_t> prefix(10, 'Z');
    insert_key(map, prefix, 99999);
    for (int i = 0; i < 300; ++i) {
      std::vector<uint8_t> key = prefix;
      key.push_back(static_cast<uint8_t>(i & 0xFF));
      if (i >= 256) {
        key.push_back(static_cast<uint8_t>(i >> 8));
      }
      insert_key(map, key, static_cast<uint64_t>(i + 1));
    }
    expect_present(map, prefix, 99999, "T30: prefix key", verbose, summary, out);
    for (int i = 0; i < 300; ++i) {
      std::vector<uint8_t> key = prefix;
      key.push_back(static_cast<uint8_t>(i & 0xFF));
      if (i >= 256) {
        key.push_back(static_cast<uint8_t>(i >> 8));
      }
      expect_present(map, key, static_cast<uint64_t>(i + 1), "T30: prefixed child", false, summary, out);
    }
    std::fprintf(out, "    PASS: prefix + 300 child keys verified\n");
  }

  std::fprintf(out, "\n  ── EDGE SUITE: %d PASS, %d FAIL ──\n\n", summary.passed, summary.failed);
  return summary;
}

template <typename KeyType>
inline void run_umap_bench(const char *name, const std::vector<KeyType> &keys, int runs, FILE *out) {
  const int n = static_cast<int>(keys.size());
  const uint32_t key_length = sizeof(keys[0].b);
  double best_insert = 1e18;
  double best_get = 1e18;
  int final_hits = 0;
  int final_miss = 0;

  for (int run = 0; run < runs; ++run) {
    UMapHarness map;
    double insert_ns = bench_op(n, [&] {
      for (int i = 0; i < n; ++i) {
        map.insert(keys[i].b, key_length, static_cast<uint64_t>(i) + 1);
      }
    });
    int hits = 0;
    int miss = 0;
    double get_ns = bench_op(n, [&] {
      for (int i = 0; i < n; ++i) {
        uint64_t found = 0;
        if (map.get(keys[i].b, key_length, found)) {
          ++hits;
        } else {
          ++miss;
        }
      }
    });
    best_insert = std::min(best_insert, insert_ns);
    if (get_ns < best_get) {
      best_get = get_ns;
      final_hits = hits;
      final_miss = miss;
    }
  }

  std::fprintf(out,
               "  %-20s INS: %6.1f ns  GET: %5.1f ns  hits=%d miss=%d\n",
               name,
               best_insert,
               best_get,
               final_hits,
               final_miss);
}

inline void run_reference_benchmarks(int n = 500000, int runs = 3, FILE *out = stdout) {
  std::fprintf(out, "================================================================\n");
  std::fprintf(out, "  Treebench Reference Benchmarks (UMAP only)\n");
  std::fprintf(out, "  %dK keys, best-of-%d\n", n / 1000, runs);
  std::fprintf(out, "================================================================\n");

  auto seq_keys = make_seq_be(n);
  auto hash_keys = make_random(n);
  auto prefixed_keys = make_prefixed(n);
  auto lofan_keys = make_lofan7(n);
  auto seq16_keys = make_seq_be_16(n);
  auto hash16_keys = make_hash_16(n);

  std::fprintf(out, "\n--- Seq BE 8B ---\n");
  run_umap_bench("umap", seq_keys, runs, out);
  std::fprintf(out, "\n--- Hash 8B ---\n");
  run_umap_bench("umap", hash_keys, runs, out);
  std::fprintf(out, "\n--- Prefixed 8B ---\n");
  run_umap_bench("umap", prefixed_keys, runs, out);
  std::fprintf(out, "\n--- LoFan7 8B ---\n");
  run_umap_bench("umap", lofan_keys, runs, out);
  std::fprintf(out, "\n--- Seq BE 16B ---\n");
  run_umap_bench("umap", seq16_keys, runs, out);
  std::fprintf(out, "\n--- Hash 16B ---\n");
  run_umap_bench("umap", hash16_keys, runs, out);

  std::fprintf(out, "\n================================================================\n\n");
}

inline void run_concurrent_benchmark(int n = 500000,
                                     int threads = 8,
                                     int runs = 3,
                                     FILE *out = stdout) {
  std::fprintf(out, "================================================================\n");
  std::fprintf(out, "  Treebench Concurrent Benchmark (UMAP only)\n");
  std::fprintf(out, "  %dK keys, %d threads, best-of-%d\n", n / 1000, threads, runs);
  std::fprintf(out, "================================================================\n");

  struct Pattern {
    const char *name;
    std::vector<Key8> keys;
  };

  std::vector<Pattern> patterns = {
      {"Seq BE 8B", make_seq_be(n)},
      {"Hash 8B", make_random(n)},
      {"Prefixed 8B", make_prefixed(n)},
      {"LoFan7 8B", make_lofan7(n)},
  };

  for (const auto &pattern : patterns) {
    double best_insert = 1e18;
    double best_get = 1e18;
    int final_hits = 0;
    int final_miss = 0;
    int final_insert_fail = 0;
    std::fprintf(out, "\n--- %s ---\n", pattern.name);

    for (int run = 0; run < runs; ++run) {
      ConcurrentUMapHarness map;
      std::atomic<int> insert_fail{0};
      int chunk = n / threads;
      auto insert_start = Clock::now();
      std::vector<std::thread> pool;
      for (int tid = 0; tid < threads; ++tid) {
        pool.emplace_back([&, tid] {
          int lo = tid * chunk;
          int hi = (tid == threads - 1) ? n : lo + chunk;
          int local_fail = 0;
          for (int i = lo; i < hi; ++i) {
            if (!map.insert(pattern.keys[i].b, 8, static_cast<uint64_t>(i) + 1)) {
              local_fail++;
            }
          }
          if (local_fail != 0) {
            insert_fail.fetch_add(local_fail, std::memory_order_relaxed);
          }
        });
      }
      for (auto &thread : pool) {
        thread.join();
      }
      double insert_ns = ns_since(insert_start) / static_cast<double>(n);

      int hits = 0;
      int miss = 0;
      double get_ns = bench_op(n, [&] {
        for (int i = 0; i < n; ++i) {
          uint64_t found = 0;
          if (map.get(pattern.keys[i].b, 8, found)) {
            ++hits;
          } else {
            ++miss;
          }
        }
      });

      best_insert = std::min(best_insert, insert_ns);
      if (get_ns < best_get) {
        best_get = get_ns;
        final_hits = hits;
        final_miss = miss;
        final_insert_fail = insert_fail.load(std::memory_order_relaxed);
      }
    }

    std::fprintf(out,
                 "  %-20s INS: %6.1f ns  GET: %5.1f ns  count=%d/%d  miss=%d  ins_fail=%d\n",
                 "umap",
                 best_insert,
                 best_get,
                 final_hits,
                 n,
                 final_miss,
                 final_insert_fail);
  }

  std::fprintf(out, "\n================================================================\n\n");
}

inline void run_cursor_benchmark(int n = 500000, int runs = 3, FILE *out = stdout) {
  std::fprintf(out, "================================================================\n");
  std::fprintf(out, "  Treebench Cursor Benchmark (UMAP only)\n");
  std::fprintf(out, "  %dK keys, best-of-%d\n", n / 1000, runs);
  std::fprintf(out, "================================================================\n");

  auto seq_keys = make_seq_be(n);
  UMapHarness map;
  for (int i = 0; i < n; ++i) {
    map.insert(seq_keys[i].b, 8, static_cast<uint64_t>(i) + 1);
  }

  double best_count = 1e18;
  double best_collect = 1e18;
  size_t counted = 0;
  size_t collected = 0;
  for (int run = 0; run < runs; ++run) {
    double count_ns = bench_op(n, [&] {
      counted = map.cursor_count();
    });
    std::vector<std::pair<std::string, uint64_t>> out_vec;
    double collect_ns = bench_op(n, [&] {
      map.collect_sorted(out_vec);
      collected = out_vec.size();
    });
    best_count = std::min(best_count, count_ns);
    best_collect = std::min(best_collect, collect_ns);
  }

  std::fprintf(out,
               "  %-20s COUNT: %6.1f ns  COLLECT: %6.1f ns  count=%zu collect=%zu\n",
               "umap",
               best_count,
               best_collect,
               counted,
               collected);
  std::fprintf(out, "\n================================================================\n\n");
}

inline void run_all(bool verbose = true, FILE *out = stdout) {
  run_edge_case_suite<UMapHarness>(verbose, out);
  run_reference_benchmarks(500000, 3, out);
  run_concurrent_benchmark(500000, 8, 3, out);
  run_cursor_benchmark(500000, 3, out);
}

}  // namespace treebench