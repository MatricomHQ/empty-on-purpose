// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_v7_bench.cpp — PureJIT Code Trie (Code IS the Data Structure)
//
// Dual-arch: ARM64 + x86-64, compile-time selection.
// Each trie node IS a block of executable code with inline header.
// NO AST. NO compact(). NO hashmap. ZERO alloc lookups.
// Insert writes directly into executable memory. Always lookup-ready.
//
// ARM64 child: CMP W3,#k + B.NE skip + B child (12 bytes, ±128MB range)
// x86-64 child: CMP CL,k + JE rel32 (9 bytes, ±2GB range)
//
// clang++ -O3 -std=c++17 -o jit_v7 src/stax/benches/jit_trie_v7_bench.cpp
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

// ═══════════════════════════════════════════════════════════════════════════
// Architecture-specific node layout constants
//
// ARM64 node layout (all 4-byte aligned):
//   [-4] NodeHeader: cap(1) count(1) pad(2)
//   [0]  CMP X1, X2          — length check
//   [4]  B.NE +5 (→[24])     — skip handle if depth != len
//   [8]  LDR X0, +2 (→[16])  — load handle from inline data
//   [12] RET
//   [16] <handle: 8 bytes>   — inline data (not executed)
//   [24] LDRB W3, [X0, X2]   — load key[depth]
//   [28] ADD X2, X2, #1      — depth++
//   [32+i*12] CMP W3, #k     — child entry (12 bytes each)
//   [32+i*12+4] B.NE +8      — skip to next child
//   [32+i*12+8] B child_node — unconditional jump (26-bit, ±128MB)
//   [32+count*12] MOV X0, #0 — miss
//   [32+count*12+4] RET
//
// x86-64 node layout (variable-length):
//   [-2] NodeHeader: cap(1) count(1)
//   [0]  CMP RSI, RDX (3)  — length check
//   [3]  JNE +11     (2)   — skip handle
//   [5]  MOV RAX, h  (10)  — handle
//   [15] RET          (1)
//   [16] MOVZX ECX, [RDI+RDX] (4) — load key[depth]
//   [20] INC RDX      (3)  — depth++
//   [23+i*9] CMP CL, k (3) + JE rel32 (6) — child (9 bytes each)
//   [23+count*9] XOR EAX,EAX (2) + RET (1) — miss
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__aarch64__) || defined(_M_ARM64)
static constexpr size_t HDR_SZ = 4;
static constexpr size_t PRE_SZ = 32;
static constexpr size_t CHILD_SZ = 12; // CMP(4)+B.NE(4)+B(4)
static constexpr size_t SUF_SZ = 8;
static constexpr size_t HNDL_OFF = 16;
#else
static constexpr size_t HDR_SZ = 2;
static constexpr size_t PRE_SZ = 23;
static constexpr size_t CHILD_SZ = 9;
static constexpr size_t SUF_SZ = 3;
static constexpr size_t HNDL_OFF = 7;
#endif

struct NodeHeader {
  uint8_t capacity, count;
};

// ═══════════════════════════════════════════════════════════════════════════
// PureJITTrie
// ═══════════════════════════════════════════════════════════════════════════

class PureJITTrie {
public:
  using LookupFn = uint64_t (*)(const char *, size_t, size_t);

  explicit PureJITTrie(size_t cap_mb) {
    mem_size_ = cap_mb * 1024 * 1024;
#ifdef __APPLE__
    mem_ = mmap(nullptr, mem_size_, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
#else
    mem_ = mmap(nullptr, mem_size_, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    cursor_ = static_cast<uint8_t *>(mem_);
    begin_write();
    root_je_ = cursor_;
#if defined(__aarch64__) || defined(_M_ARM64)
    // Root pointer: B (unconditional) with placeholder offset
    uint32_t b = 0x14000000;
    memcpy(cursor_, &b, 4);
    cursor_ += 4;
#else
    *cursor_++ = 0x0F;
    *cursor_++ = 0x84;
    memset(cursor_, 0, 4);
    cursor_ += 4;
#endif
    uint8_t *init = alloc_node(4);
    set_handle(init, 0);
    update_root(init);
    end_write();
  }

  ~PureJITTrie() {
    if (mem_)
      munmap(mem_, mem_size_);
  }

  void begin_batch() {
    begin_write();
    batch_ = true;
  }
  void end_batch() {
    batch_ = false;
    end_write();
  }

  void insert(const char *key, size_t len, uint64_t handle) {
    bool own = !batch_;
    if (own)
      begin_write();
    uint8_t *parent_je = root_je_;
    bool parent_is_root = true;
    uint8_t *node = root_target();

    for (size_t d = 0; d < len; d++) {
      uint8_t k = static_cast<uint8_t>(key[d]);
      auto *hdr = get_header(node);
      int ci = -1;
      for (int i = 0; i < hdr->count; i++)
        if (child_byte(node, i) == k) {
          ci = i;
          break;
        }

      if (ci == -1) {
        if (hdr->count == hdr->capacity) {
          node = realloc_node(node, parent_je, parent_is_root);
          hdr = get_header(node);
        }
        int ins = hdr->count;
        while (ins > 0 && child_byte(node, ins - 1) > k)
          ins--;
        uint8_t *dest = node + PRE_SZ + ins * CHILD_SZ;
        if (ins < hdr->count) {
          memmove(dest + CHILD_SZ, dest, (hdr->count - ins) * CHILD_SZ);
          for (int i = ins + 1; i <= hdr->count; i++)
            fixup_shifted_child(node, i);
        }
        uint8_t *cn = alloc_node(4);
        set_handle(cn, 0);
        emit_child(dest, k, cn);
        hdr->count++;
        emit_suffix(node, hdr->count);

        parent_je = child_b_ptr(dest);
        parent_is_root = false;
        node = cn;
      } else {
        uint8_t *slot = node + PRE_SZ + ci * CHILD_SZ;
        parent_je = child_b_ptr(slot);
        parent_is_root = false;
        node = child_target(slot);
      }
    }
    set_handle(node, handle);
    count_++;
    if (own)
      end_write();
  }

  uint64_t lookup(const char *key, size_t len) const {
    return reinterpret_cast<LookupFn>(root_target())(key, len, 0);
  }

  size_t count() const { return count_; }
  size_t memory_used() const { return cursor_ - static_cast<uint8_t *>(mem_); }

  void iterate(void (*cb)(const char *, size_t, uint64_t, void *),
               void *ctx) const {
    std::string key;
    iterate_node(root_target(), key, cb, ctx);
  }

private:
  void *mem_ = nullptr;
  size_t mem_size_ = 0;
  uint8_t *cursor_ = nullptr;
  uint8_t *root_je_ = nullptr;
  bool batch_ = false;
  size_t count_ = 0;

  NodeHeader *get_header(uint8_t *code) const {
    return reinterpret_cast<NodeHeader *>(code - HDR_SZ);
  }
  void set_handle(uint8_t *code, uint64_t h) { memcpy(code + HNDL_OFF, &h, 8); }
  uint64_t get_handle(uint8_t *code) const {
    uint64_t h;
    memcpy(&h, code + HNDL_OFF, 8);
    return h;
  }

#if defined(__aarch64__) || defined(_M_ARM64)
  // ── ARM64 ──

  // Root uses B (unconditional, 26-bit offset, ±128MB)
  uint8_t *root_target() const {
    uint32_t inst;
    memcpy(&inst, root_je_, 4);
    int32_t imm26 = static_cast<int32_t>(inst & 0x3FFFFFF);
    if (imm26 & 0x2000000)
      imm26 |= static_cast<int32_t>(0xFC000000);
    return root_je_ + imm26 * 4;
  }
  void update_root(uint8_t *target) {
    int32_t off = static_cast<int32_t>((target - root_je_)) / 4;
    uint32_t inst = 0x14000000 | (static_cast<uint32_t>(off) & 0x3FFFFFF);
    memcpy(root_je_, &inst, 4);
  }

  // Child uses B (unconditional, 26-bit) at slot+8
  uint8_t *child_b_ptr(uint8_t *slot) const { return slot + 8; }

  uint8_t *child_target(uint8_t *slot) const {
    uint8_t *bp = slot + 8;
    uint32_t inst;
    memcpy(&inst, bp, 4);
    int32_t imm26 = static_cast<int32_t>(inst & 0x3FFFFFF);
    if (imm26 & 0x2000000)
      imm26 |= static_cast<int32_t>(0xFC000000);
    return bp + imm26 * 4;
  }

  uint8_t child_byte(uint8_t *code, int idx) const {
    uint32_t inst;
    memcpy(&inst, code + PRE_SZ + idx * CHILD_SZ, 4);
    return static_cast<uint8_t>((inst >> 10) & 0xFF);
  }

  void emit_child(uint8_t *dest, uint8_t k, uint8_t *target) {
    uint32_t cmp = 0x7100007F | (static_cast<uint32_t>(k) << 10);
    uint32_t bne = 0x54000041; // B.NE +2 instructions (skip B)
    uint8_t *bp = dest + 8;
    int32_t off = static_cast<int32_t>((target - bp)) / 4;
    uint32_t b = 0x14000000 | (static_cast<uint32_t>(off) & 0x3FFFFFF);
    memcpy(dest, &cmp, 4);
    memcpy(dest + 4, &bne, 4);
    memcpy(dest + 8, &b, 4);
  }

  void fixup_shifted_child(uint8_t *code, int i) {
    uint8_t *bp = code + PRE_SZ + i * CHILD_SZ + 8;
    uint32_t inst;
    memcpy(&inst, bp, 4);
    int32_t imm26 = static_cast<int32_t>(inst & 0x3FFFFFF);
    if (imm26 & 0x2000000)
      imm26 |= static_cast<int32_t>(0xFC000000);
    imm26 -= 3; // moved 12 bytes = 3 instructions further
    inst = 0x14000000 | (static_cast<uint32_t>(imm26) & 0x3FFFFFF);
    memcpy(bp, &inst, 4);
  }

  void emit_suffix(uint8_t *code, int cnt) {
    uint32_t *s = reinterpret_cast<uint32_t *>(code + PRE_SZ + cnt * CHILD_SZ);
    s[0] = 0xD2800000;
    s[1] = 0xD65F03C0;
  }

  uint8_t *alloc_node(uint8_t cap) {
    auto *hdr = reinterpret_cast<NodeHeader *>(cursor_);
    hdr->capacity = cap;
    hdr->count = 0;
    cursor_ += HDR_SZ;
    uint8_t *code = cursor_;
    cursor_ += PRE_SZ + cap * CHILD_SZ + SUF_SZ;
    auto *p = reinterpret_cast<uint32_t *>(code);
    p[0] = 0xEB02003F; // CMP X1, X2
    p[1] = 0x540000A1; // B.NE +5 (→ offset 24)
    p[2] = 0x58000040; // LDR X0, +2 (→ offset 16)
    p[3] = 0xD65F03C0; // RET
    memset(code + 16, 0, 8);
    p[6] = 0x38626803; // LDRB W3, [X0, X2]
    p[7] = 0x91000442; // ADD X2, X2, #1
    emit_suffix(code, 0);
    return code;
  }

  uint8_t *realloc_node(uint8_t *oc, uint8_t *pje, bool parent_root) {
    auto *oh = get_header(oc);
    uint8_t nc = oh->capacity < 128 ? oh->capacity * 2 : 255;
    uint8_t *ncode = alloc_node(nc);
    auto *nh = get_header(ncode);
    nh->count = oh->count;
    set_handle(ncode, get_handle(oc));
    for (int i = 0; i < oh->count; i++) {
      uint8_t *src = oc + PRE_SZ + i * CHILD_SZ;
      uint8_t *dst = ncode + PRE_SZ + i * CHILD_SZ;
      memcpy(dst, src, CHILD_SZ);
      uint8_t *target = child_target(src);
      emit_child_b(dst + 8, target);
    }
    emit_suffix(ncode, oh->count);
    if (parent_root)
      update_root(ncode);
    else
      emit_child_b(pje, ncode);
    return ncode;
  }

  void emit_child_b(uint8_t *bp, uint8_t *target) {
    int32_t off = static_cast<int32_t>((target - bp)) / 4;
    uint32_t b = 0x14000000 | (static_cast<uint32_t>(off) & 0x3FFFFFF);
    memcpy(bp, &b, 4);
  }

#else
  // ── x86-64 ──

  uint8_t *root_target() const {
    int32_t rel;
    memcpy(&rel, root_je_ + 2, 4);
    return root_je_ + 6 + rel;
  }
  void update_root(uint8_t *target) {
    int32_t rel = static_cast<int32_t>(target - (root_je_ + 6));
    memcpy(root_je_ + 2, &rel, 4);
  }

  uint8_t *child_b_ptr(uint8_t *slot) const { return slot + 3; }
  uint8_t *child_target(uint8_t *slot) const {
    uint8_t *je = slot + 3;
    int32_t rel;
    memcpy(&rel, je + 2, 4);
    return je + 6 + rel;
  }

  uint8_t child_byte(uint8_t *code, int idx) const {
    return code[PRE_SZ + idx * CHILD_SZ + 2];
  }

  void emit_child(uint8_t *dest, uint8_t k, uint8_t *target) {
    dest[0] = 0x80;
    dest[1] = 0xF9;
    dest[2] = k;
    dest[3] = 0x0F;
    dest[4] = 0x84;
    int32_t rel = static_cast<int32_t>(target - (dest + 9));
    memcpy(dest + 5, &rel, 4);
  }

  void fixup_shifted_child(uint8_t *code, int i) {
    uint8_t *je = code + PRE_SZ + i * CHILD_SZ + 3;
    int32_t rel;
    memcpy(&rel, je + 2, 4);
    rel -= static_cast<int32_t>(CHILD_SZ);
    memcpy(je + 2, &rel, 4);
  }

  void emit_suffix(uint8_t *code, int cnt) {
    uint8_t *s = code + PRE_SZ + cnt * CHILD_SZ;
    s[0] = 0x31;
    s[1] = 0xC0;
    s[2] = 0xC3;
  }

  uint8_t *alloc_node(uint8_t cap) {
    auto *hdr = reinterpret_cast<NodeHeader *>(cursor_);
    hdr->capacity = cap;
    hdr->count = 0;
    cursor_ += HDR_SZ;
    uint8_t *code = cursor_;
    cursor_ += PRE_SZ + cap * CHILD_SZ + SUF_SZ;
    uint8_t *p = code;
    *p++ = 0x48;
    *p++ = 0x39;
    *p++ = 0xF2;
    *p++ = 0x75;
    *p++ = 0x0B;
    *p++ = 0x48;
    *p++ = 0xB8;
    memset(p, 0, 8);
    p += 8;
    *p++ = 0xC3;
    *p++ = 0x0F;
    *p++ = 0xB6;
    *p++ = 0x0C;
    *p++ = 0x17;
    *p++ = 0x48;
    *p++ = 0xFF;
    *p++ = 0xC2;
    emit_suffix(code, 0);
    return code;
  }

  uint8_t *realloc_node(uint8_t *oc, uint8_t *pje, bool parent_root) {
    auto *oh = get_header(oc);
    uint8_t nc = oh->capacity < 128 ? oh->capacity * 2 : 255;
    uint8_t *ncode = alloc_node(nc);
    auto *nh = get_header(ncode);
    nh->count = oh->count;
    set_handle(ncode, get_handle(oc));
    for (int i = 0; i < oh->count; i++) {
      uint8_t *src = oc + PRE_SZ + i * CHILD_SZ;
      uint8_t *dst = ncode + PRE_SZ + i * CHILD_SZ;
      memcpy(dst, src, CHILD_SZ);
      uint8_t *target = child_target(src);
      // Fix JE relative offset
      int32_t rel = static_cast<int32_t>(target - (dst + 9));
      memcpy(dst + 5, &rel, 4);
    }
    emit_suffix(ncode, oh->count);
    if (parent_root)
      update_root(ncode);
    else {
      int32_t rel = static_cast<int32_t>(ncode - (pje + 6));
      memcpy(pje + 2, &rel, 4);
    }
    return ncode;
  }
#endif

  void iterate_node(uint8_t *node, std::string &key,
                    void (*cb)(const char *, size_t, uint64_t, void *),
                    void *ctx) const {
    uint64_t h = get_handle(node);
    if (h != 0)
      cb(key.c_str(), key.size(), h, ctx);
    auto *hdr = get_header(node);
    for (int i = 0; i < hdr->count; i++) {
      uint8_t k = child_byte(node, i);
      key.push_back(static_cast<char>(k));
      uint8_t *slot = node + PRE_SZ + i * CHILD_SZ;
      iterate_node(child_target(slot), key, cb, ctx);
      key.pop_back();
    }
  }

  void begin_write() {
#ifdef __APPLE__
    pthread_jit_write_protect_np(0);
#endif
  }
  void end_write() {
#ifdef __APPLE__
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(mem_, cursor_ - static_cast<uint8_t *>(mem_));
#endif
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark
// ═══════════════════════════════════════════════════════════════════════════

static auto make_seq(size_t N) {
  std::vector<std::pair<std::string, uint64_t>> o;
  o.reserve(N);
  char buf[32];
  for (size_t i = 0; i < N; i++) {
    snprintf(buf, sizeof(buf), "k_%07zu", i);
    o.emplace_back(buf, uint64_t(i + 1));
  }
  o.emplace_back("k_00", 9999991);
  o.emplace_back("k_0000", 9999992);
  o.emplace_back("k", 9999993);
  std::sort(o.begin(), o.end());
  return o;
}

struct BR {
  double mean, p50, p95;
};
template <typename Fn> BR bench_fn(Fn &&fn, size_t batch, size_t samples) {
  for (size_t i = 0; i < 3000; i++)
    fn();
  std::vector<double> t(samples);
  for (size_t s = 0; s < samples; s++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < batch; i++)
      fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    t[s] = std::chrono::duration<double, std::nano>(t1 - t0).count() / batch;
  }
  std::sort(t.begin(), t.end());
  double sum = 0;
  for (auto v : t)
    sum += v;
  return {sum / samples, t[samples / 2],
          t[static_cast<size_t>(samples * 0.95)]};
}

struct CursorEntry {
  std::string key;
  uint64_t handle;
};
static void collect_cb(const char *k, size_t l, uint64_t h, void *ctx) {
  static_cast<std::vector<CursorEntry> *>(ctx)->push_back(
      {std::string(k, l), h});
}

static void run(const char *label,
                std::vector<std::pair<std::string, uint64_t>> &entries) {
  size_t N = entries.size();
  printf("\n═══════════════════════════════════════════════════════════════════"
         "\n");
  printf("  %s (%zu entries)\n", label, N);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  auto shuffled = entries;
  std::mt19937 g(1337);
  std::shuffle(shuffled.begin(), shuffled.end(), g);
  std::vector<const char *> ks;
  std::vector<size_t> ls;
  for (auto &e : entries) {
    ks.push_back(e.first.c_str());
    ls.push_back(e.first.size());
  }

  size_t ins_samples = N <= 100000 ? 10 : 5;
  std::vector<double> jit_ins(ins_samples), umap_ins(ins_samples);
  for (size_t s = 0; s < ins_samples; s++) {
    PureJITTrie t(256);
    auto t0 = std::chrono::high_resolution_clock::now();
    t.begin_batch();
    for (auto &e : shuffled)
      t.insert(e.first.c_str(), e.first.size(), e.second);
    t.end_batch();
    auto t1 = std::chrono::high_resolution_clock::now();
    jit_ins[s] = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;

    std::unordered_map<std::string, uint64_t> u;
    u.reserve(N);
    auto u0 = std::chrono::high_resolution_clock::now();
    for (auto &e : shuffled)
      u[e.first] = e.second;
    auto u1 = std::chrono::high_resolution_clock::now();
    umap_ins[s] = std::chrono::duration<double, std::nano>(u1 - u0).count() / N;
  }
  std::sort(jit_ins.begin(), jit_ins.end());
  std::sort(umap_ins.begin(), umap_ins.end());
  double ji50 = jit_ins[ins_samples / 2], ui50 = umap_ins[ins_samples / 2];

  PureJITTrie trie(256);
  trie.begin_batch();
  for (auto &e : shuffled)
    trie.insert(e.first.c_str(), e.first.size(), e.second);
  trie.end_batch();

  printf("  Insert: JIT=%.0fns umap=%.0fns | Mem: %.2f MB (%.0f B/entry)\n",
         ji50, ui50, trie.memory_used() / 1048576.0,
         static_cast<double>(trie.memory_used()) / N);

  bool ok = true;
  for (size_t i = 0; i < N && ok; i++) {
    uint64_t h = trie.lookup(entries[i].first.c_str(), entries[i].first.size());
    if (h != entries[i].second) {
      printf("  ❌ '%s' got %llu want %llu\n", entries[i].first.c_str(),
             static_cast<unsigned long long>(h),
             static_cast<unsigned long long>(entries[i].second));
      ok = false;
    }
  }
  if (trie.lookup("ZZZZZZZZZZZ", 11) != 0) {
    printf("  ❌ miss\n");
    ok = false;
  }
  if (!ok)
    return;
  printf("  ✅ ALL %zu verified + miss OK\n", N);

  std::vector<CursorEntry> cur;
  trie.iterate(collect_cb, &cur);
  bool cord = cur.size() == N;
  for (size_t i = 0; i < N && cord; i++)
    if (cur[i].key != entries[i].first || cur[i].handle != entries[i].second)
      cord = false;
  if (cord)
    printf("  ✅ Cursor order verified (%zu items)\n", N);
  else
    printf("  ❌ Cursor failed (%zu vs %zu)\n", cur.size(), N);

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  for (auto &e : entries)
    umap[e.first] = e.second;
  size_t ki = 0, bs = std::min(N, size_t(10000));
  ki = 0;
  auto ru = bench_fn(
      [&]() {
        DoNotOptimize(umap.find(std::string(ks[ki], ls[ki]))->second);
        ki = (ki + 1) % N;
      },
      bs, 500);
  ki = 0;
  auto rj = bench_fn(
      [&]() {
        DoNotOptimize(trie.lookup(ks[ki], ls[ki]));
        ki = (ki + 1) % N;
      },
      bs, 500);

  double lk = ru.p50 / rj.p50, is = ui50 / ji50;
  printf(
      "\n  "
      "╔═══════════════════════════╦══════════════════╦══════════════════╗\n");
  printf("  ║                           ║   INSERT (p50)   ║   LOOKUP (p50)   "
         "║\n");
  printf(
      "  "
      "╠═══════════════════════════╬══════════════════╬══════════════════╣\n");
  printf(
      "  ║ std::unordered_map        ║     %7.1fns     ║     %7.1fns     ║\n",
      ui50, ru.p50);
  printf(
      "  ║ ★ PureJIT V7              ║     %7.1fns     ║     %7.1fns     ║\n",
      ji50, rj.p50);
  printf(
      "  "
      "╠═══════════════════════════╬══════════════════╬══════════════════╣\n");
  printf("  ║ V7 vs umap               ║ %5.2fx %s    ║ %5.2fx %s    ║\n",
         is > 1 ? is : 1 / is, is > 1 ? "✅" : "❌", lk > 1 ? lk : 1 / lk,
         lk > 1 ? "✅" : "❌");
  printf(
      "  "
      "╚═══════════════════════════╩══════════════════╩══════════════════╝\n");
}

int main() {
#if defined(__aarch64__) || defined(_M_ARM64)
  const char *arch = "ARM64";
#elif defined(__x86_64__) || defined(_M_X64)
  const char *arch = "x86-64";
#endif
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║  ★ PureJIT Trie V7 — Code IS Data (%s)              ║\n", arch);
  printf("║  No AST, No compact, Zero-alloc lookup, Ordered cursor    ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 5000, 10000, 50000, 100000, 500000, 1000000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }
  return 0;
}
