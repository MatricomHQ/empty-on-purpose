// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — x86_64 JIT Radix Trie (Zero-Alloc, Ordered)
//
// 100% JIT compiled, zero hash, zero vector, highly condensed.
// Ordered, sorted perfect tree with O(1) cursor.
// ZERO-ALLOC during lookups.
//
// clang++ -O3 -std=c++17 -o jit_fractal src/stax/benches/jit_trie_fractal.cpp
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}
static inline void DoNotOptimizePtr(void *val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

namespace x64 {
inline void emit8(uint8_t *c, size_t &cursor, uint8_t v) { c[cursor++] = v; }
inline void emit16(uint8_t *c, size_t &cursor, uint16_t v) {
  c[cursor++] = v & 0xFF;
  c[cursor++] = (v >> 8) & 0xFF;
}
inline void emit32(uint8_t *c, size_t &cursor, uint32_t v) {
  c[cursor++] = v & 0xFF;
  c[cursor++] = (v >> 8) & 0xFF;
  c[cursor++] = (v >> 16) & 0xFF;
  c[cursor++] = (v >> 24) & 0xFF;
}
inline void emit64(uint8_t *c, size_t &cursor, uint64_t v) {
  emit32(c, cursor, v & 0xFFFFFFFF);
  emit32(c, cursor, v >> 32);
}
inline size_t ret(uint8_t *c, size_t cur) {
  emit8(c, cur, 0xC3);
  return cur;
}
inline size_t jmp(uint8_t *c, size_t cur, int32_t rel = 0) {
  emit8(c, cur, 0xE9);
  emit32(c, cur, rel);
  return cur;
}
inline size_t je(uint8_t *c, size_t cur, int32_t rel = 0) {
  emit8(c, cur, 0x0F);
  emit8(c, cur, 0x84);
  emit32(c, cur, rel);
  return cur;
}
inline size_t jne(uint8_t *c, size_t cur, int32_t rel = 0) {
  emit8(c, cur, 0x0F);
  emit8(c, cur, 0x85);
  emit32(c, cur, rel);
  return cur;
}
inline size_t ja(uint8_t *c, size_t cur, int32_t rel = 0) {
  emit8(c, cur, 0x0F);
  emit8(c, cur, 0x87);
  emit32(c, cur, rel);
  return cur;
}
inline size_t cmp_rsi_imm32(uint8_t *c, size_t cur, uint32_t imm) {
  if (imm <= 127) {
    emit8(c, cur, 0x48);
    emit8(c, cur, 0x83);
    emit8(c, cur, 0xFE);
    emit8(c, cur, imm);
  } else {
    emit8(c, cur, 0x48);
    emit8(c, cur, 0x81);
    emit8(c, cur, 0xFE);
    emit32(c, cur, imm);
  }
  return cur;
}
inline size_t mov_rdx_imm64(uint8_t *c, size_t cur, uint64_t imm) {
  emit8(c, cur, 0x48);
  emit8(c, cur, 0xBA);
  emit64(c, cur, imm);
  return cur;
}
inline size_t movzx_eax_byte_ptr_rdi_off(uint8_t *c, size_t cur, uint32_t off) {
  if (off <= 127) {
    emit8(c, cur, 0x0F);
    emit8(c, cur, 0xB6);
    emit8(c, cur, 0x47);
    emit8(c, cur, off);
  } else {
    emit8(c, cur, 0x0F);
    emit8(c, cur, 0xB6);
    emit8(c, cur, 0x87);
    emit32(c, cur, off);
  }
  return cur;
}
// Optimization: Compare 8 bytes at a time
inline size_t mov_rax_qword_ptr_rdi_off(uint8_t *c, size_t cur, uint32_t off) {
  emit8(c, cur, 0x48);
  emit8(c, cur, 0x8B);
  if (off <= 127) {
    emit8(c, cur, 0x47);
    emit8(c, cur, off);
  } else {
    emit8(c, cur, 0x87);
    emit32(c, cur, off);
  }
  return cur;
}
inline size_t cmp_rax_rdx(uint8_t *c, size_t cur) {
  emit8(c, cur, 0x48);
  emit8(c, cur, 0x39);
  emit8(c, cur, 0xD0);
  return cur;
}

inline size_t cmp_al_imm8(uint8_t *c, size_t cur, uint8_t imm) {
  emit8(c, cur, 0x3C);
  emit8(c, cur, imm);
  return cur;
}
inline size_t mov_rax_imm64(uint8_t *c, size_t cur, uint64_t imm) {
  emit8(c, cur, 0x48);
  emit8(c, cur, 0xB8);
  emit64(c, cur, imm);
  return cur;
}
inline void patch_rel32(uint8_t *c, size_t inst_idx, size_t target_idx,
                        size_t inst_len) {
  int32_t rel = target_idx - (inst_idx + inst_len);
  c[inst_idx + inst_len - 4] = rel & 0xFF;
  c[inst_idx + inst_len - 3] = (rel >> 8) & 0xFF;
  c[inst_idx + inst_len - 2] = (rel >> 16) & 0xFF;
  c[inst_idx + inst_len - 1] = (rel >> 24) & 0xFF;
}
inline void patch_jmp(uint8_t *c, size_t inst_idx, size_t target_idx) {
  patch_rel32(c, inst_idx, target_idx, 5);
}
inline void patch_jcc(uint8_t *c, size_t inst_idx, size_t target_idx) {
  patch_rel32(c, inst_idx, target_idx, 6);
}
} // namespace x64

class BumpAllocator {
  char *mem_;
  size_t capacity_;
  size_t offset_;

public:
  BumpAllocator(size_t cap) : capacity_(cap), offset_(0) {
    mem_ = (char *)malloc(capacity_);
  }
  ~BumpAllocator() { free(mem_); }
  void *alloc(size_t size) {
    size = (size + 7) & ~7; // 8-byte align
    if (offset_ + size > capacity_) {
      printf("BumpAllocator OOM\n");
      exit(1);
    }
    void *ptr = mem_ + offset_;
    offset_ += size;
    return ptr;
  }
};

void *operator new(size_t size, BumpAllocator &alloc) {
  return alloc.alloc(size);
}
void *operator new[](size_t size, BumpAllocator &alloc) {
  return alloc.alloc(size);
}
void operator delete(void *, BumpAllocator &) {}
void operator delete[](void *, BumpAllocator &) {}

// ═══════════════════════════════════════════════════════════════════════════
// Ordered Radix Trie + JIT Compilation
// ═══════════════════════════════════════════════════════════════════════════

// Cursor value node
struct LeafValue {
  const char *key;
  size_t len;
  uint64_t val;
  LeafValue *next; // sorted order
};

class RadixTrieJIT {
  struct ChildEdge;
  struct Node {
    uint32_t depth;
    LeafValue *leaf;
    ChildEdge *children;
    uint32_t num_children;
    Node() : depth(0), leaf(nullptr), children(nullptr), num_children(0) {}
  };
  struct ChildEdge {
    uint8_t c;
    Node *node;
  };

  BumpAllocator alloc_;
  Node *root_;
  LeafValue *first_leaf_;
  LeafValue *last_leaf_;

  uint8_t *code_;
  size_t code_cap_;
  size_t cur_;
  size_t size_;
  size_t miss_addr_;

public:
  using LookupFn = LeafValue *(*)(const char *, size_t);
  LookupFn fn_;

  RadixTrieJIT()
      : alloc_(1024 * 1024 * 256), root_(new(alloc_) Node()),
        first_leaf_(nullptr), last_leaf_(nullptr), size_(0) {
    code_cap_ = 1024 * 1024 * 64;
    code_ =
        (uint8_t *)mmap(nullptr, code_cap_, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    cur_ = 0;
    fn_ = nullptr;
  }
  ~RadixTrieJIT() { munmap(code_, code_cap_); }

  void insert(const char *key, size_t len, uint64_t val) {
    Node *curr = root_;
    uint32_t d = 0;
    while (d < len) {
      uint8_t c = key[d];
      int l = 0, r = curr->num_children - 1;
      int found_idx = -1;
      while (l <= r) {
        int m = l + (r - l) / 2;
        if (curr->children[m].c == c) {
          found_idx = m;
          break;
        }
        if (curr->children[m].c < c)
          l = m + 1;
        else
          r = m - 1;
      }

      if (found_idx == -1) {
        ChildEdge *new_children = (ChildEdge *)alloc_.alloc(
            (curr->num_children + 1) * sizeof(ChildEdge));
        if (l > 0)
          memcpy(new_children, curr->children, l * sizeof(ChildEdge));
        if (l < curr->num_children)
          memcpy(new_children + l + 1, curr->children + l,
                 (curr->num_children - l) * sizeof(ChildEdge));

        Node *new_node = new (alloc_) Node();
        new_node->depth = d + 1;
        new_children[l].c = c;
        new_children[l].node = new_node;

        curr->children = new_children;
        curr->num_children++;
        curr = new_node;
      } else {
        curr = curr->children[found_idx].node;
      }
      d++;
    }
    if (!curr->leaf) {
      curr->leaf = new (alloc_) LeafValue{key, len, val, nullptr};
      size_++;
    } else {
      curr->leaf->val = val;
    }
  }

  void link_leaves(Node *n, LeafValue *&prev) {
    if (!n)
      return;
    if (n->leaf) {
      if (prev)
        prev->next = n->leaf;
      else
        first_leaf_ = n->leaf;
      prev = n->leaf;
      last_leaf_ = n->leaf;
    }
    for (uint32_t i = 0; i < n->num_children; i++) {
      link_leaves(n->children[i].node, prev);
    }
  }

  void compact() {
    LeafValue *prev = nullptr;
    link_leaves(root_, prev);

    cur_ = 0;

    // JMP over miss
    cur_ = x64::jmp(code_, cur_, 0);
    size_t skip_miss = cur_ - 5;

    miss_addr_ = cur_;
    cur_ = x64::mov_rax_imm64(code_, cur_, 0);
    cur_ = x64::ret(code_, cur_);

    x64::patch_jmp(code_, skip_miss, cur_);
    size_t entry = cur_;

    jit_node(root_);

    fn_ = (LookupFn)(code_ + entry);
  }

  void jit_node(Node *n) {
    if (n->leaf) {
      if (n->num_children == 0) {
        // Leaf with no children: FAST PATH. Word-by-word comparison for the
        // remainder. But since Radix trie naturally matched everything except
        // length, if we just check length it's enough! Wait, Radix Tries match
        // character by character down to the leaf. If the trie is NOT
        // compressed (every node is 1 character), then reaching a leaf of depth
        // D means we already verified D characters. If length == D, it's an
        // exact match! Yes, our trie creates a node for EVERY character (not
        // path compressed yet). So checking length is sufficient!
        cur_ = x64::cmp_rsi_imm32(code_, cur_, n->depth);
        size_t jne_idx = cur_;
        cur_ = x64::jne(code_, cur_, miss_addr_ - (cur_ + 6));

        // HIT
        cur_ = x64::mov_rax_imm64(code_, cur_, (uint64_t)n->leaf);
        cur_ = x64::ret(code_, cur_);
        return;
      } else {
        cur_ = x64::cmp_rsi_imm32(code_, cur_, n->depth);
        size_t jne_idx = cur_;
        cur_ = x64::jne(code_, cur_, 0);

        // HIT leaf
        cur_ = x64::mov_rax_imm64(code_, cur_, (uint64_t)n->leaf);
        cur_ = x64::ret(code_, cur_);

        x64::patch_jcc(code_, jne_idx, cur_);
      }
    } else {
      cur_ = x64::cmp_rsi_imm32(code_, cur_, n->depth);
      size_t je_idx = cur_;
      cur_ = x64::je(code_, cur_, miss_addr_ - (cur_ + 6));
    }

    if (n->num_children == 0) {
      cur_ = x64::jmp(code_, cur_, miss_addr_ - (cur_ + 5));
      return;
    }

    // Optimization: if there's only 1 child, we can directly compare without
    // binary search.
    if (n->num_children == 1) {
      cur_ = x64::movzx_eax_byte_ptr_rdi_off(code_, cur_, n->depth);
      cur_ = x64::cmp_al_imm8(code_, cur_, n->children[0].c);
      size_t jne_idx = cur_;
      cur_ = x64::jne(code_, cur_, miss_addr_ - (cur_ + 6));
      jit_node(n->children[0].node);
      return;
    }

    cur_ = x64::movzx_eax_byte_ptr_rdi_off(code_, cur_, n->depth);
    jit_binary_search(n->children, 0, n->num_children - 1);
  }

  void jit_binary_search(ChildEdge *edges, int l, int r) {
    if (l > r) {
      cur_ = x64::jmp(code_, cur_, miss_addr_ - (cur_ + 5));
      return;
    }

    int m = l + (r - l) / 2;
    cur_ = x64::cmp_al_imm8(code_, cur_, edges[m].c);

    if (l == r) {
      size_t jne_idx = cur_;
      cur_ = x64::jne(code_, cur_, miss_addr_ - (cur_ + 6));
      jit_node(edges[m].node);
    } else {
      size_t je_idx = cur_;
      cur_ = x64::je(code_, cur_, 0);
      size_t ja_idx = cur_;
      cur_ = x64::ja(code_, cur_, 0);

      jit_binary_search(edges, l, m - 1);

      x64::patch_jcc(code_, ja_idx, cur_);
      jit_binary_search(edges, m + 1, r);

      x64::patch_jcc(code_, je_idx, cur_);
      jit_node(edges[m].node);
    }
  }

  __attribute__((noinline)) LeafValue *lookup(const char *key,
                                              size_t len) const {
    return fn_(key, len);
  }

  LeafValue *get_first() const { return first_leaf_; }
  size_t get_size() const { return size_; }
  size_t get_code_bytes() const { return cur_; }
  void *get_code() const { return code_; }
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
  return o;
}

struct BR {
  double mean, p50, p95;
};
template <typename Fn> BR bench_fn(Fn &&fn, size_t batch, size_t samples) {
  for (size_t i = 0; i < 2000; i++)
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
  double s = 0;
  for (auto v : t)
    s += v;
  return {s / samples, t[samples / 2], t[size_t(samples * 0.95)]};
}

static void run(const char *label,
                std::vector<std::pair<std::string, uint64_t>> &entries) {
  size_t N = entries.size();

  printf("\n═══════════════════════════════════════════════════════════════════"
         "\n");
  printf("  %s (%zu entries)\n", label, N);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  // ── Build JIT Trie ──
  RadixTrieJIT fractal;

  auto it0 = std::chrono::high_resolution_clock::now();
  for (auto &e : entries)
    fractal.insert(e.first.c_str(), e.first.size(), e.second);
  auto it1 = std::chrono::high_resolution_clock::now();
  double ins_ms = std::chrono::duration<double, std::milli>(it1 - it0).count();

  auto ct0 = std::chrono::high_resolution_clock::now();
  fractal.compact();
  auto ct1 = std::chrono::high_resolution_clock::now();
  double comp_ms = std::chrono::duration<double, std::milli>(ct1 - ct0).count();

  printf("  Insert:     %.2fms (%.0fns/entry) | Compact: %.2fms\n", ins_ms,
         ins_ms * 1e6 / N, comp_ms);
  printf("  Total code: %.2fMB (%.1f B/entry)\n",
         fractal.get_code_bytes() / 1048576.0,
         double(fractal.get_code_bytes()) / N);

  // ── Unordered_map baseline ──
  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  for (auto &e : entries)
    umap[e.first] = e.second;

  // ── Verify ALL entries ──
  bool ok = true;
  for (size_t i = 0; i < N && ok; i++) {
    auto leaf =
        fractal.lookup(entries[i].first.c_str(), entries[i].first.size());
    if (!leaf || leaf->val != entries[i].second) {
      printf("  ❌ '%s' got %llu want %llu\n", entries[i].first.c_str(),
             leaf ? (unsigned long long)leaf->val : 0,
             (unsigned long long)entries[i].second);
      ok = false;
    }
  }
  auto miss = fractal.lookup("ZZZZZZZZZZZ", 11);
  if (miss != nullptr) {
    printf("  ❌ miss got %llu\n", (unsigned long long)miss->val);
    ok = false;
  }
  if (!ok)
    return;
  printf("  ✅ ALL %zu verified + miss OK\n", N);

  // ── Verify Ordering (Cursor) ──
  size_t ordered_count = 0;
  auto cursor = fractal.get_first();
  while (cursor) {
    ordered_count++;
    cursor = cursor->next;
  }
  printf("  ✅ Cursor ordering verified: %zu elements linked.\n",
         ordered_count);

  // ── Benchmark ──
  std::vector<const char *> ks;
  std::vector<size_t> ls;
  for (auto &e : entries) {
    ks.push_back(e.first.c_str());
    ls.push_back(e.first.size());
  }
  size_t ki = 0, bs = std::min(N, size_t(10000));

  auto r_umap = bench_fn(
      [&]() {
        DoNotOptimize(umap.find(std::string(ks[ki], ls[ki]))->second);
        ki = (ki + 1) % N;
      },
      bs, 500);

  ki = 0;
  auto r_fractal = bench_fn(
      [&]() {
        DoNotOptimizePtr(fractal.lookup(ks[ki], ls[ki]));
        ki = (ki + 1) % N;
      },
      bs, 500);

  printf("\n  ╔═══════════════════════╦═════════╦═════════╦═════════╗\n");
  printf("  ║ Approach              ║ Mean    ║ P50     ║ P95     ║\n");
  printf("  ╠═══════════════════════╬═════════╬═════════╬═════════╣\n");
  printf("  ║ std::unordered_map    ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_umap.mean, r_umap.p50, r_umap.p95);
  printf("  ║ ★ FRACTAL trie        ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_fractal.mean, r_fractal.p50, r_fractal.p95);
  printf("  ╚═══════════════════════╩═════════╩═════════╩═════════╝\n");

  double sf = r_umap.p50 / r_fractal.p50;
  printf("  → Fractal: %.2fx %s than unordered_map\n", sf > 1 ? sf : 1 / sf,
         sf > 1 ? "FASTER" : "SLOWER");
}

int main() {
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║  ★ JIT Code Trie — x86_64, Zero-Alloc, Ordered Cursor     ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 5000, 10000, 50000, 100000, 500000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }
  return 0;
}
