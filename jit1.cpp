#include <algorithm> // only for std::min, std::max, std::swap
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

namespace x86 {
inline size_t emit_jbe_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0x0F;
  d[1] = 0x86;
  std::memcpy(d + 2, &rel, 4);
  return 6;
}
inline size_t emit_ret(uint8_t *d) {
  d[0] = 0xC3;
  return 1;
}
inline size_t emit_cmp_rsi_imm32(uint8_t *d, uint32_t imm) {
  d[0] = 0x48;
  d[1] = 0x81;
  d[2] = 0xFE;
  std::memcpy(d + 3, &imm, 4);
  return 7;
}
inline size_t emit_cmp_rsi_imm8(uint8_t *d, uint8_t imm) {
  d[0] = 0x48;
  d[1] = 0x83;
  d[2] = 0xFE;
  d[3] = imm;
  return 4;
}
inline size_t emit_je_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0x0F;
  d[1] = 0x84;
  std::memcpy(d + 2, &rel, 4);
  return 6;
}
inline size_t emit_jne_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0x0F;
  d[1] = 0x85;
  std::memcpy(d + 2, &rel, 4);
  return 6;
}
inline size_t emit_jb_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0x0F;
  d[1] = 0x82;
  std::memcpy(d + 2, &rel, 4);
  return 6;
}
inline size_t emit_ja_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0x0F;
  d[1] = 0x87;
  std::memcpy(d + 2, &rel, 4);
  return 6;
}
inline size_t emit_jmp_rel32(uint8_t *d, int32_t rel) {
  d[0] = 0xE9;
  std::memcpy(d + 1, &rel, 4);
  return 5;
}
inline size_t emit_return_zero(uint8_t *d) {
  d[0] = 0x31;
  d[1] = 0xC0;
  d[2] = 0xC3;
  return 3;
}
inline size_t emit_mov_rax_imm64(uint8_t *d, uint64_t imm) {
  d[0] = 0x48;
  d[1] = 0xB8;
  std::memcpy(d + 2, &imm, 8);
  return 10;
}
inline size_t emit_mov_rax_imm32(uint8_t *d, uint32_t imm) {
  d[0] = 0xB8;
  std::memcpy(d + 1, &imm, 4);
  return 5;
}

inline size_t emit_cmp_byte_at_offset(uint8_t *d, uint32_t offset,
                                      uint8_t val) {
  size_t n = 0;
  if (offset < 128) {
    d[n++] = 0x80;
    d[n++] = 0x7F;
    d[n++] = (uint8_t)offset;
    d[n++] = val;
  } else {
    d[n++] = 0x80;
    d[n++] = 0xBF;
    std::memcpy(d + n, &offset, 4);
    n += 4;
    d[n++] = val;
  }
  return n;
}

inline size_t emit_cmp_qword_at_offset(uint8_t *d, uint32_t offset,
                                       uint64_t val) {
  size_t n = 0;
  if (offset == 0) {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x0F;
  } else if (offset < 128) {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x4F;
    d[n++] = (uint8_t)offset;
  } else {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x8F;
    std::memcpy(d + n, &offset, 4);
    n += 4;
  }
  d[n++] = 0x48;
  d[n++] = 0xBA;
  std::memcpy(d + n, &val, 8);
  n += 8;
  d[n++] = 0x48;
  d[n++] = 0x39;
  d[n++] = 0xD1;
  return n;
}

inline size_t emit_cmp_dword_at_offset(uint8_t *d, uint32_t offset,
                                       uint32_t val) {
  size_t n = 0;
  d[n++] = 0x81;
  if (offset < 128) {
    d[n++] = 0x7F;
    d[n++] = (uint8_t)offset;
  } else {
    d[n++] = 0xBF;
    std::memcpy(d + n, &offset, 4);
    n += 4;
  }
  std::memcpy(d + n, &val, 4);
  n += 4;
  return n;
}

inline size_t emit_cmp_word_at_offset(uint8_t *d, uint32_t offset,
                                      uint16_t val) {
  size_t n = 0;
  d[n++] = 0x66;
  d[n++] = 0x81;
  if (offset < 128) {
    d[n++] = 0x7F;
    d[n++] = (uint8_t)offset;
  } else {
    d[n++] = 0xBF;
    std::memcpy(d + n, &offset, 4);
    n += 4;
  }
  std::memcpy(d + n, &val, 2);
  n += 2;
  return n;
}

inline void patch_jcc_rel32(uint8_t *d, size_t instr_offset,
                            size_t target_offset) {
  int32_t rel = (int32_t)target_offset - (int32_t)(instr_offset + 6);
  std::memcpy(d + instr_offset + 2, &rel, 4);
}
inline void patch_jmp_rel32(uint8_t *d, size_t instr_offset,
                            size_t target_offset) {
  int32_t rel = (int32_t)target_offset - (int32_t)(instr_offset + 5);
  std::memcpy(d + instr_offset + 1, &rel, 4);
}
} // namespace x86

class Arena {
  uint8_t *mem_;
  size_t cap_;
  size_t cur_ = 0;

public:
  Arena(size_t cap) : cap_(cap) { mem_ = (uint8_t *)malloc(cap); }
  ~Arena() { free(mem_); }
  void *alloc(size_t sz) {
    if (cur_ + sz > cap_) {
      printf("Arena OOM! Need %zu, cap %zu\n", cur_ + sz, cap_);
      return nullptr;
    }
    void *p = mem_ + cur_;
    cur_ += sz;
    return p;
  }
  void reset() { cur_ = 0; }
};

// We will implement a JIT Radix Trie where nodes are allocated in an Arena,
// keeping their children sorted as they are inserted. No std::vector!
struct TrieNode {
  uint32_t child_count = 0;

  // Terminal value at this exact node path (e.g. key length ends here)
  bool has_value = false;
  uint64_t handle = 0;
  const char *key_ptr = nullptr;
  size_t key_len = 0;

  struct Child {
    uint8_t bv;
    TrieNode *sub;
  };
  Child *children = nullptr;

  // Track depth relative to string to avoid passing it around everywhere if
  // needed, but passing depth is often easier during JIT generation.
};

class FractalJIT {
  Arena arena_{150 * 1024 * 1024}; // 150MB arena for nodes and strings
  TrieNode root_;
  void *mmap_mem_ = nullptr;
  size_t mmap_sz_ = 0;
  size_t code_bytes_ = 0;
  size_t count_ = 0;

  using Fn = uint64_t (*)(const char *, size_t);
  Fn fn_ = nullptr;

public:
  FractalJIT() {
    mmap_sz_ = 4096;
    mmap_mem_ = mmap(nullptr, mmap_sz_, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t *nc = (uint8_t *)mmap_mem_;
    x86::emit_return_zero(nc);
    fn_ = (Fn)nc;
  }

  ~FractalJIT() {
    if (mmap_mem_)
      munmap(mmap_mem_, mmap_sz_);
  }

  void insert(const std::string &key, uint64_t handle) {
    char *k = (char *)arena_.alloc(key.size() + 1);
    std::memcpy(k, key.c_str(), key.size() + 1);

    insert_recursive(&root_, k, key.size(), handle, 0);
    count_++;
  }

  bool insert_recursive(TrieNode *n, const char *key, size_t len, uint64_t h,
                        uint8_t depth) {
    if (depth == len) {
      n->has_value = true;
      n->handle = h;
      n->key_ptr = key;
      n->key_len = len;
      return true;
    }

    uint8_t bv = (uint8_t)key[depth];

    // Find child with this byte. Children are kept sorted.
    // Linear search is fine since max children = 256, usually very small.
    int insert_idx = n->child_count;
    for (uint32_t i = 0; i < n->child_count; i++) {
      if (n->children[i].bv == bv) {
        return insert_recursive(n->children[i].sub, key, len, h, depth + 1);
      }
      if (n->children[i].bv > bv) {
        insert_idx = i;
        break;
      }
    }

    // Need to insert new child at insert_idx.
    // Allocate new array and copy over to keep it sorted! No std::vector!
    TrieNode::Child *new_children = (TrieNode::Child *)arena_.alloc(
        sizeof(TrieNode::Child) * (n->child_count + 1));

    for (int i = 0; i < insert_idx; i++) {
      new_children[i] = n->children[i];
    }
    for (int i = insert_idx; i < (int)n->child_count; i++) {
      new_children[i + 1] = n->children[i];
    }

    TrieNode *new_node = (TrieNode *)arena_.alloc(sizeof(TrieNode));
    new_node->child_count = 0;
    new_node->has_value = false;

    new_children[insert_idx].bv = bv;
    new_children[insert_idx].sub = new_node;

    n->children = new_children;
    n->child_count++;

    return insert_recursive(new_node, key, len, h, depth + 1);
  }

  void compact() {
    if (count_ == 0)
      return;

    // Count instructions roughly based on node count to size mmap
    size_t new_sz = count_ * 128 + 4096;
    void *new_mem = mmap(nullptr, new_sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    uint8_t *nc = (uint8_t *)new_mem;
    size_t cur = 0;

    size_t miss = cur;
    cur += x86::emit_return_zero(&nc[cur]);

    size_t entry_jump = cur;
    nc[cur++] = 0xE9;
    int32_t zero = 0;
    std::memcpy(nc + cur, &zero, 4);
    cur += 4;

    size_t logic = cur;
    x86::patch_jmp_rel32(nc, entry_jump, logic);

    // Compile the tree into JIT instructions
    compile_node(&root_, nc, cur, miss, 0);

    if (mmap_mem_)
      munmap(mmap_mem_, mmap_sz_);
    mmap_mem_ = new_mem;
    mmap_sz_ = new_sz;
    code_bytes_ = cur;
    fn_ = (Fn)(nc + entry_jump);
  }

  void compile_node(TrieNode *n, uint8_t *nc, size_t &cur, size_t miss,
                    uint8_t depth) {
    if (n->has_value) {
      // If the key length strictly equals the depth, we found an exact match
      // We check this with a length check first, to differentiate it from
      // children extensions
      if (n->child_count > 0) {
        if (n->key_len < 128)
          cur += x86::emit_cmp_rsi_imm8(&nc[cur], n->key_len);
        else
          cur += x86::emit_cmp_rsi_imm32(&nc[cur], n->key_len);

        size_t jne = cur;
        cur += x86::emit_jne_rel32(&nc[cur], 0);

        if (n->handle <= 0xFFFFFFFF)
          cur += x86::emit_mov_rax_imm32(&nc[cur], (uint32_t)n->handle);
        else
          cur += x86::emit_mov_rax_imm64(&nc[cur], n->handle);
        cur += x86::emit_ret(&nc[cur]);

        x86::patch_jcc_rel32(
            nc, jne, cur); // if not length, fallthrough to children check
      } else {
        // Leaf node, no children. Verify remaining bytes and length, but since
        // it's the only path, we verify length first
        if (n->key_len < 128)
          cur += x86::emit_cmp_rsi_imm8(&nc[cur], n->key_len);
        else
          cur += x86::emit_cmp_rsi_imm32(&nc[cur], n->key_len);
        size_t jne_len = cur;
        cur += x86::emit_jne_rel32(&nc[cur], 0);
        x86::patch_jcc_rel32(nc, jne_len, miss);

        // We've matched up to 'depth', check remaining
        size_t start = depth;
        while (start + 8 <= n->key_len) {
          uint64_t expected;
          std::memcpy(&expected, n->key_ptr + start, 8);
          cur += x86::emit_cmp_qword_at_offset(&nc[cur], start, expected);
          size_t jne = cur;
          cur += x86::emit_jne_rel32(&nc[cur], 0);
          x86::patch_jcc_rel32(nc, jne, miss);
          start += 8;
        }
        while (start + 4 <= n->key_len) {
          uint32_t expected;
          std::memcpy(&expected, n->key_ptr + start, 4);
          cur += x86::emit_cmp_dword_at_offset(&nc[cur], start, expected);
          size_t jne = cur;
          cur += x86::emit_jne_rel32(&nc[cur], 0);
          x86::patch_jcc_rel32(nc, jne, miss);
          start += 4;
        }
        while (start + 2 <= n->key_len) {
          uint16_t expected;
          std::memcpy(&expected, n->key_ptr + start, 2);
          cur += x86::emit_cmp_word_at_offset(&nc[cur], start, expected);
          size_t jne = cur;
          cur += x86::emit_jne_rel32(&nc[cur], 0);
          x86::patch_jcc_rel32(nc, jne, miss);
          start += 2;
        }
        if (start < n->key_len) {
          cur +=
              x86::emit_cmp_byte_at_offset(&nc[cur], start, n->key_ptr[start]);
          size_t jne = cur;
          cur += x86::emit_jne_rel32(&nc[cur], 0);
          x86::patch_jcc_rel32(nc, jne, miss);
        }

        if (n->handle <= 0xFFFFFFFF)
          cur += x86::emit_mov_rax_imm32(&nc[cur], (uint32_t)n->handle);
        else
          cur += x86::emit_mov_rax_imm64(&nc[cur], n->handle);
        cur += x86::emit_ret(&nc[cur]);
        return; // done with this branch!
      }
    }

    if (n->child_count == 0) {
      cur += x86::emit_jmp_rel32(&nc[cur], miss);
      return;
    }

    // Length check: ensure string is at least long enough to read 'depth' byte
    // actually, we should do this if not done, but usually users pass correct
    // length. For safety, cmp rsi, depth; jle miss (since we read [depth],
    // length must be > depth)
    cur += x86::emit_cmp_rsi_imm32(&nc[cur], depth);
    size_t jbe = cur;
    cur += x86::emit_jbe_rel32(&nc[cur], 0);
    x86::patch_jcc_rel32(nc, jbe, miss);

    // Compress single-child chains (path compression in JIT without modifying
    // tree)
    TrieNode *curr = n;
    uint8_t current_depth = depth;

    while (curr->child_count == 1 && !curr->children[0].sub->has_value &&
           curr->children[0].sub->child_count == 1) {
      // Check character
      cur += x86::emit_cmp_byte_at_offset(&nc[cur], current_depth,
                                          curr->children[0].bv);
      size_t jne = cur;
      cur += x86::emit_jne_rel32(&nc[cur], 0);
      x86::patch_jcc_rel32(nc, jne, miss);

      curr = curr->children[0].sub;
      current_depth++;

      // Length check for next byte
      cur += x86::emit_cmp_rsi_imm32(&nc[cur], current_depth);
      size_t jbe2 = cur;
      cur += x86::emit_jbe_rel32(&nc[cur], 0);
      x86::patch_jcc_rel32(nc, jbe2, miss);
    }

    // Now emit binary search for curr's children at current_depth
    emit_binary_search(curr, nc, cur, miss, 0, curr->child_count,
                       current_depth);
  }

  void emit_binary_search(TrieNode *n, uint8_t *nc, size_t &cur, size_t miss,
                          size_t start, size_t end, uint8_t depth) {
    if (start >= end) {
      cur += x86::emit_jmp_rel32(&nc[cur], miss);
      return;
    }

    if (end - start == 1) {
      cur +=
          x86::emit_cmp_byte_at_offset(&nc[cur], depth, n->children[start].bv);
      size_t jne = cur;
      cur += x86::emit_jne_rel32(&nc[cur], 0);
      x86::patch_jcc_rel32(nc, jne, miss);

      compile_node(n->children[start].sub, nc, cur, miss, depth + 1);
      return;
    }

    size_t mid = start + (end - start) / 2;
    uint8_t pivot = n->children[mid].bv;

    cur += x86::emit_cmp_byte_at_offset(&nc[cur], depth, pivot);
    size_t jeq = cur;
    cur += x86::emit_je_rel32(&nc[cur], 0);
    size_t jb = cur;
    cur += x86::emit_jb_rel32(&nc[cur], 0);

    emit_binary_search(n, nc, cur, miss, mid + 1, end, depth); // >

    size_t match = cur;
    x86::patch_jcc_rel32(nc, jeq, match);
    compile_node(n->children[mid].sub, nc, cur, miss, depth + 1); // ==

    size_t less = cur;
    x86::patch_jcc_rel32(nc, jb, less);
    emit_binary_search(n, nc, cur, miss, start, mid, depth); // <
  }

  uint64_t lookup(const char *k, size_t l) const { return fn_(k, l); }
  size_t count() const { return count_; }
  size_t code_bytes() const { return code_bytes_; }

  // Stateful Cursor using fixed size array (stack) to avoid std::vector. Max
  // string depth ~2048 is more than enough for normal use.
  class Cursor {
    struct Frame {
      const TrieNode *node;
      uint32_t child_idx;
    };
    const FractalJIT *trie_;
    Frame stack_[4096];
    int top_ = -1;

    const TrieNode *current_value_node_ = nullptr;
    bool valid_ = false;

    void advance() {
      // DFS traversal
      valid_ = false;
      current_value_node_ = nullptr;

      while (top_ >= 0) {
        Frame &f = stack_[top_];
        if (f.child_idx < f.node->child_count) {
          const TrieNode *child_sub = f.node->children[f.child_idx].sub;
          f.child_idx++; // advance for next time we pop back

          // Push child
          top_++;
          stack_[top_].node = child_sub;
          stack_[top_].child_idx = 0;

          // Does child have a value? If so, we yield it
          if (child_sub->has_value) {
            current_value_node_ = child_sub;
            valid_ = true;
            return;
          }
        } else {
          // done with this node, pop
          top_--;
        }
      }
    }

  public:
    Cursor(const FractalJIT *trie) : trie_(trie) {}

    bool valid() const { return valid_; }

    void next() {
      if (valid_)
        advance();
    }

    void begin() {
      top_ = -1;
      valid_ = false;
      current_value_node_ = nullptr;

      if (trie_) {
        top_ = 0;
        stack_[top_].node = &trie_->root_;
        stack_[top_].child_idx = 0;

        if (trie_->root_.has_value) {
          current_value_node_ = &trie_->root_;
          valid_ = true;
        } else {
          advance();
        }
      }
    }

    const char *key() const {
      return valid_ ? current_value_node_->key_ptr : nullptr;
    }
    size_t key_len() const { return valid_ ? current_value_node_->key_len : 0; }
    uint64_t handle() const { return valid_ ? current_value_node_->handle : 0; }
  };

  Cursor cursor() const { return Cursor(this); }
};

#include <string>
#include <unordered_map>
#include <vector>

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
  printf(" %s (%zu entries)\n", label, N);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  FractalJIT fractal;

  auto it0 = std::chrono::high_resolution_clock::now();
  for (auto &e : entries)
    fractal.insert(e.first, e.second);
  auto it1 = std::chrono::high_resolution_clock::now();
  double ins_ms = std::chrono::duration<double, std::milli>(it1 - it0).count();

  auto ct0 = std::chrono::high_resolution_clock::now();
  fractal.compact();
  auto ct1 = std::chrono::high_resolution_clock::now();
  double comp_ms = std::chrono::duration<double, std::milli>(ct1 - ct0).count();

  printf(" Insert: %.2fms (%.0fns/entry) | Compact: %.2fms\n", ins_ms,
         ins_ms * 1e6 / N, comp_ms);
  printf(" Total code: %.2fMB (%.1f B/entry)\n",
         fractal.code_bytes() / 1048576.0, double(fractal.code_bytes()) / N);

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  for (auto &e : entries)
    umap[e.first] = e.second;

  bool ok = true;
  for (size_t i = 0; i < N && ok; i++) {
    uint64_t h =
        fractal.lookup(entries[i].first.c_str(), entries[i].first.size());
    if (h != entries[i].second) {
      printf(" ❌ '%s' got %llu want %llu\n", entries[i].first.c_str(),
             (unsigned long long)h, (unsigned long long)entries[i].second);
      ok = false;
    }
  }
  uint64_t miss = fractal.lookup("ZZZZZZZZZZZ", 11);
  if (miss != 0) {
    printf(" ❌ miss got %llu\n", (unsigned long long)miss);
    ok = false;
  }

  auto c = fractal.cursor();
  c.begin();
  size_t cid = 0;
  while (c.valid() && cid < N) {
    if (strncmp(c.key(), entries[cid].first.c_str(), c.key_len()) != 0) {
      printf(" ❌ Cursor mismatch at %zu: '%s' != '%s'\n", cid, c.key(),
             entries[cid].first.c_str());
      ok = false;
      break;
    }
    c.next();
    cid++;
  }
  if (cid != N) {
    printf(" ❌ Cursor returned %zu items, want %zu\n", cid, N);
    ok = false;
  }

  if (!ok)
    return;
  printf(" ✅ ALL %zu verified + cursor ordered correctly + miss OK\n", N);

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
        DoNotOptimize(fractal.lookup(ks[ki], ls[ki]));
        ki = (ki + 1) % N;
      },
      bs, 500);

  printf("\n ╔═══════════════════════╦═════════╦═════════╦═════════╗\n");
  printf(" ║ Approach              ║ Mean    ║ P50     ║ P95     ║\n");
  printf(" ╠═══════════════════════╬═════════╬═════════╬═════════╣\n");
  printf(" ║ std::unordered_map    ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_umap.mean, r_umap.p50, r_umap.p95);
  printf(" ║ ★ FRACTAL trie       ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_fractal.mean, r_fractal.p50, r_fractal.p95);
  printf(" ╚═══════════════════════╩═════════╩═════════╩═════════╝\n");

  double sf = r_umap.p50 / r_fractal.p50;
  printf(" → Fractal: %.2fx %s than unordered_map\n", sf > 1 ? sf : 1 / sf,
         sf > 1 ? "FASTER" : "SLOWER");
}
int main() {
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║ ★ FRACTAL JIT Code Trie — Tree of Trees                       ║\n");
  printf("║ Dynamic insert(). Zero-alloc lookup(). Ordered and Fast.      ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 10000, 100000, 1000000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }
  return 0;
}
