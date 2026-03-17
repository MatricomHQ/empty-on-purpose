// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — Fractal JIT Code Trie (Tree of Trees) x86_64
//
// Zero allocations in inner node logic. Ordered inserts via insertion-sort-like
// approach on inner arrays (small sizes usually, or perfect trees).
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

namespace x86 {
// Basic JIT Emitter for x86_64
struct Assembler {
  uint8_t *code;
  size_t size;
  size_t capacity;

  Assembler(uint8_t *buffer, size_t cap)
      : code(buffer), size(0), capacity(cap) {}

  void emit8(uint8_t v) { code[size++] = v; }
  void emit16(uint16_t v) {
    memcpy(code + size, &v, 2);
    size += 2;
  }
  void emit32(uint32_t v) {
    memcpy(code + size, &v, 4);
    size += 4;
  }
  void emit64(uint64_t v) {
    memcpy(code + size, &v, 8);
    size += 8;
  }

  // ret
  void ret() { emit8(0xC3); }

  // mov rax, imm64
  void mov_rax(uint64_t imm) {
    emit8(0x48);
    emit8(0xB8);
    emit64(imm);
  }

  // xor rax, rax
  void xor_rax_rax() {
    emit8(0x48);
    emit8(0x31);
    emit8(0xC0);
  }

  // cmp rsi, imm32  (rsi = len)
  void cmp_rsi(uint32_t imm) {
    if (imm <= 127) {
      emit8(0x48);
      emit8(0x83);
      emit8(0xFE);
      emit8((uint8_t)imm);
    } else {
      emit8(0x48);
      emit8(0x81);
      emit8(0xFE);
      emit32(imm);
    }
  }

  // cmp byte ptr [rdi + offset], imm8  (rdi = key)
  void cmp_rdi_off_byte(uint32_t offset, uint8_t imm) {
    if (offset == 0) {
      emit8(0x80);
      emit8(0x3F);
      emit8(imm);
    } else if (offset <= 127) {
      emit8(0x80);
      emit8(0x7F);
      emit8((uint8_t)offset);
      emit8(imm);
    } else {
      emit8(0x80);
      emit8(0xBF);
      emit32(offset);
      emit8(imm);
    }
  }

  void mov_r8(uint64_t imm) {
    emit8(0x49);
    emit8(0xB8);
    emit64(imm);
  }

  // cmp [rdi + offset], r8
  void cmp_rdi_off_64(uint32_t offset) {
    if (offset == 0) {
      emit8(0x4C);
      emit8(0x39);
      emit8(0x07);
    } else if (offset <= 127) {
      emit8(0x4C);
      emit8(0x39);
      emit8(0x47);
      emit8((uint8_t)offset);
    } else {
      emit8(0x4C);
      emit8(0x39);
      emit8(0x87);
      emit32(offset);
    }
  }

  // je rel32
  size_t je() {
    emit8(0x0F);
    emit8(0x84);
    size_t patch = size;
    emit32(0);
    return patch;
  }

  // jne rel32
  size_t jne() {
    emit8(0x0F);
    emit8(0x85);
    size_t patch = size;
    emit32(0);
    return patch;
  }

  // jmp rel32
  size_t jmp() {
    emit8(0xE9);
    size_t patch = size;
    emit32(0);
    return patch;
  }

  void patch_jmp(size_t offset, size_t target) {
    int64_t rel = (int64_t)target - (int64_t)(offset + 4);
    memcpy(code + offset, (uint32_t *)&rel, 4);
  }
};
} // namespace x86

// ═══════════════════════════════════════════════════════════════════════════
// Memory Arena for Zero-Alloc JIT Nodes
// ═══════════════════════════════════════════════════════════════════════════
class Arena {
public:
  uint8_t *mem;
  size_t cap;
  size_t offset;

  Arena(size_t size) {
    cap = size;
    offset = 0;
    mem = (uint8_t *)mmap(nullptr, cap, PROT_READ | PROT_WRITE | PROT_EXEC,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED)
      mem = nullptr;
  }

  ~Arena() {
    if (mem)
      munmap(mem, cap);
  }

  void *alloc(size_t sz) {
    if (offset + sz > cap)
      return nullptr;
    void *ptr = mem + offset;
    offset += sz;
    return ptr;
  }

  void reset() { offset = 0; }
};

// ═══════════════════════════════════════════════════════════════════════════
// JIT Trie
// ═══════════════════════════════════════════════════════════════════════════

struct TrieNode {
  uint8_t depth = 0;
  // Metadata for ordering and structure
  struct Child {
    uint8_t byte;
    TrieNode *sub;
    char *key; // kept for easy rebuild/compacting
    size_t key_len;
    uint64_t handle;
  };
  // Dynamic array for children, allocated from arena
  Child *children = nullptr;
  uint32_t num_children = 0;
  uint32_t cap_children = 0;
};

class JITCodeTrie {
  friend class JITCursor;

public:
  using LookupFn = uint64_t (*)(const char *, size_t);

  JITCodeTrie(Arena *arena) : arena_(arena) {
    root_ = (TrieNode *)arena_->alloc(sizeof(TrieNode));
    root_->depth = 0;

    // Initial dummy JIT function (returns 0)
    code_start_ = arena_->offset;
    x86::Assembler a(arena_->mem + arena_->offset, 128);
    a.xor_rax_rax();
    a.ret();
    arena_->offset += a.size;
    fn_ = (LookupFn)(arena_->mem + code_start_);
  }

  bool insert(const std::string &key, uint64_t handle) {
    bool inserted = do_insert(root_, key, handle, 0);
    if (inserted)
      count_++;
    return inserted;
  }

  void compact() {
    // Re-JIT the entire trie
    code_start_ = arena_->offset;
    x86::Assembler a(arena_->mem + arena_->offset,
                     arena_->cap - arena_->offset);

    size_t len_groups[128];
    size_t num_lens = 0;
    // Group by length
    collect_lens(root_, len_groups, num_lens);

    // Main entry: Check length
    struct Patch {
      size_t len;
      size_t patch_je;
    };
    Patch len_patches[128];

    for (size_t i = 0; i < num_lens; ++i) {
      a.cmp_rsi(len_groups[i]);
      size_t patch_je = a.je();
      len_patches[i] = {len_groups[i], patch_je};
    }

    // Miss! Return 0
    a.xor_rax_rax();
    a.ret();

    // Emit blocks for each length
    for (size_t i = 0; i < num_lens; ++i) {
      a.patch_jmp(len_patches[i].patch_je, a.size);
      emit_node(a, root_, len_patches[i].len);
    }

    arena_->offset += a.size;
    fn_ = (LookupFn)(arena_->mem + code_start_);
  }

  uint64_t lookup(const char *key, size_t len) const { return fn_(key, len); }

  size_t count() const { return count_; }

private:
  Arena *arena_;
  TrieNode *root_;
  size_t code_start_ = 0;
  LookupFn fn_ = nullptr;
  size_t count_ = 0;

  bool do_insert(TrieNode *node, const std::string &key, uint64_t handle,
                 uint8_t depth) {
    if (depth >= key.size())
      return false;
    uint8_t b = key[depth];

    // Find child or insertion point
    uint32_t pos = 0;
    while (pos < node->num_children && node->children[pos].byte < b) {
      pos++;
    }

    if (pos < node->num_children && node->children[pos].byte == b) {
      if (node->children[pos].sub) {
        auto &c = node->children[pos];
        size_t sub_depth = c.sub->depth;
        size_t match_len = node->depth + 1;
        while (match_len < sub_depth && match_len < key.size() &&
               match_len < c.key_len && c.key[match_len] == key[match_len]) {
          match_len++;
        }

        if (match_len < sub_depth) {
          TrieNode *inter = (TrieNode *)arena_->alloc(sizeof(TrieNode));
          inter->depth = match_len;
          inter->cap_children = 2;
          inter->num_children = 2;
          inter->children =
              (TrieNode::Child *)arena_->alloc(2 * sizeof(TrieNode::Child));

          uint8_t old_b = c.key[match_len];
          uint8_t new_b = key[match_len];

          char *new_k = (char *)arena_->alloc(key.size());
          memcpy(new_k, key.data(), key.size());

          if (old_b < new_b) {
            inter->children[0].key = c.key;
            inter->children[0].key_len = c.key_len;
            inter->children[0].byte = old_b;
            inter->children[0].sub = c.sub;
            inter->children[0].handle = 0;

            inter->children[1].key = new_k;
            inter->children[1].key_len = key.size();
            inter->children[1].byte = new_b;
            inter->children[1].sub = nullptr;
            inter->children[1].handle = handle;
          } else {
            inter->children[0].key = new_k;
            inter->children[0].key_len = key.size();
            inter->children[0].byte = new_b;
            inter->children[0].sub = nullptr;
            inter->children[0].handle = handle;

            inter->children[1].key = c.key;
            inter->children[1].key_len = c.key_len;
            inter->children[1].byte = old_b;
            inter->children[1].sub = c.sub;
            inter->children[1].handle = 0;
          }

          c.sub = inter;
          return true;
        } else {
          return do_insert(c.sub, key, handle, sub_depth);
        }
      } else {
        return split_leaf(node, pos, key, handle);
      }
    }

    // Insert new leaf at pos (keep sorted!)
    if (node->num_children == node->cap_children) {
      uint32_t new_cap = node->cap_children == 0 ? 4 : node->cap_children * 2;
      TrieNode::Child *new_children =
          (TrieNode::Child *)arena_->alloc(new_cap * sizeof(TrieNode::Child));
      if (node->num_children > 0) {
        memcpy(new_children, node->children,
               node->num_children * sizeof(TrieNode::Child));
      }
      node->children = new_children;
      node->cap_children = new_cap;
    }

    // Shift for sorted insertion
    for (uint32_t i = node->num_children; i > pos; --i) {
      node->children[i] = node->children[i - 1];
    }

    node->children[pos].byte = b;
    node->children[pos].sub = nullptr;
    node->children[pos].key = (char *)arena_->alloc(key.size());
    memcpy(node->children[pos].key, key.data(), key.size());
    node->children[pos].key_len = key.size();
    node->children[pos].handle = handle;
    node->num_children++;

    return true;
  }

  bool split_leaf(TrieNode *p, uint32_t child_idx, const std::string &nk,
                  uint64_t nh) {
    auto &oc = p->children[child_idx];

    uint8_t disc = p->depth + 1;
    while (disc < oc.key_len && disc < nk.size() && oc.key[disc] == nk[disc]) {
      disc++;
    }

    if (disc >= oc.key_len || disc >= nk.size())
      return false;

    TrieNode *sub = (TrieNode *)arena_->alloc(sizeof(TrieNode));
    sub->depth = disc;
    sub->cap_children = 2;
    sub->num_children = 2;
    sub->children =
        (TrieNode::Child *)arena_->alloc(2 * sizeof(TrieNode::Child));

    uint8_t ob = oc.key[disc];
    uint8_t nb = nk[disc];

    char *old_k = oc.key;
    size_t old_l = oc.key_len;

    char *new_k = (char *)arena_->alloc(nk.size());
    memcpy(new_k, nk.data(), nk.size());

    if (ob < nb) {
      sub->children[0].key = old_k;
      sub->children[0].key_len = old_l;
      sub->children[0].byte = ob;
      sub->children[0].sub = nullptr;
      sub->children[0].handle = oc.handle;

      sub->children[1].key = new_k;
      sub->children[1].key_len = nk.size();
      sub->children[1].byte = nb;
      sub->children[1].sub = nullptr;
      sub->children[1].handle = nh;
    } else {
      sub->children[0].key = new_k;
      sub->children[0].key_len = nk.size();
      sub->children[0].byte = nb;
      sub->children[0].sub = nullptr;
      sub->children[0].handle = nh;

      sub->children[1].key = old_k;
      sub->children[1].key_len = old_l;
      sub->children[1].byte = ob;
      sub->children[1].sub = nullptr;
      sub->children[1].handle = oc.handle;
    }

    oc.sub = sub;
    return true;
  }

  void collect_lens(TrieNode *node, size_t *lens, size_t &num_lens) {
    if (!node)
      return;
    for (uint32_t i = 0; i < node->num_children; ++i) {
      if (node->children[i].sub) {
        collect_lens(node->children[i].sub, lens, num_lens);
      } else {
        size_t l = node->children[i].key_len;
        bool found = false;
        for (size_t j = 0; j < num_lens; ++j) {
          if (lens[j] == l) {
            found = true;
            break;
          }
        }
        if (!found && num_lens < 128) {
          lens[num_lens++] = l;
        }
      }
    }
  }

  void emit_node(x86::Assembler &a, TrieNode *node, size_t target_len) {
    if (!node)
      return;

    // Linear search over children for this depth
    size_t match_patches[256];
    int num_matches = 0;

    for (uint32_t i = 0; i < node->num_children; ++i) {
      auto &c = node->children[i];

      // Is this branch relevant for target_len?
      bool relevant = false;
      if (c.sub) {
        relevant = has_len(c.sub, target_len);
      } else {
        relevant = (c.key_len == target_len);
      }

      if (!relevant)
        continue;

      a.cmp_rdi_off_byte(node->depth, c.byte);
      match_patches[num_matches++] = a.je();
    }

    // None matched -> jump to end of this node block (miss)
    size_t miss_jump = a.jmp();

    int match_idx = 0;
    for (uint32_t i = 0; i < node->num_children; ++i) {
      auto &c = node->children[i];
      bool relevant = false;
      if (c.sub)
        relevant = has_len(c.sub, target_len);
      else
        relevant = (c.key_len == target_len);

      if (!relevant)
        continue;

      a.patch_jmp(match_patches[match_idx++], a.size);

      if (c.sub) {
        emit_node(a, c.sub, target_len);
      } else {
        size_t bne_patches[256];
        int num_bnes = 0;

        for (size_t b = node->depth + 1; b < target_len; ++b) {
          a.cmp_rdi_off_byte((uint32_t)b, (uint8_t)c.key[b]);
          bne_patches[num_bnes++] = a.jne();
        }

        a.mov_rax(c.handle);
        a.ret();

        for (int j = 0; j < num_bnes; ++j) {
          a.patch_jmp(bne_patches[j], a.size);
        }
        a.xor_rax_rax();
        a.ret();
      }
    }

    a.patch_jmp(miss_jump, a.size);
    a.xor_rax_rax(); // catch fallthrough
    a.ret();
  }

  bool has_len(TrieNode *node, size_t len) {
    for (uint32_t i = 0; i < node->num_children; ++i) {
      if (node->children[i].sub) {
        if (has_len(node->children[i].sub, len))
          return true;
      } else {
        if (node->children[i].key_len == len)
          return true;
      }
    }
    return false;
  }
};

// ═══════════════════════════════════════════════════════════════════════════
// Cursor for Sorted Iteration (Zero Alloc, On-Demand)
// ═══════════════════════════════════════════════════════════════════════════
class JITCursor {
  struct Frame {
    TrieNode *node;
    uint32_t child_idx;
  };
  Frame stack_[128];
  int top_ = -1;
  std::string current_key_;
  bool valid_ = false;
  uint64_t current_val_ = 0;

  void advance() {
    while (top_ >= 0) {
      Frame &f = stack_[top_];
      if (f.child_idx < f.node->num_children) {
        auto &c = f.node->children[f.child_idx++];
        if (c.sub) {
          stack_[++top_] = {c.sub, 0};
        } else {
          current_key_ = std::string(c.key, c.key_len);
          current_val_ = c.handle;
          valid_ = true;
          return;
        }
      } else {
        top_--;
      }
    }
    valid_ = false;
  }

public:
  JITCursor(JITCodeTrie &trie) {
    if (trie.root_) {
      stack_[++top_] = {trie.root_, 0};
      advance();
    }
  }

  bool valid() const { return valid_; }
  void next() { advance(); }

  const std::string &key() const { return current_key_; }
  uint64_t value() const { return current_val_; }

  friend class JITCodeTrie;
};

// ═══════════════════════════════════════════════════════════════════════════
// Fractal JIT Trie Wrapper
// ═══════════════════════════════════════════════════════════════════════════

class FractalJITTrie {
public:
  FractalJITTrie(size_t prefix_start, size_t prefix_len, size_t sub_capacity)
      : pstart_(prefix_start), plen_(prefix_len), arena_(64 * 1024 * 1024),
        meta_(&arena_) {} // 64MB arena

  ~FractalJITTrie() {
    for (size_t i = 0; i < subs_count_; ++i) {
      subs_[i]->~JITCodeTrie();
    }
  }

  bool insert(const std::string &key, uint64_t handle) {
    if (key.size() < pstart_ + plen_)
      return false;

    std::string pfx = key.substr(pstart_, plen_);
    size_t bucket = 0;
    bool found = false;

    for (size_t i = 0; i < prefix_count_; ++i) {
      if (prefix_keys_[i] == pfx) {
        bucket = prefix_vals_[i];
        found = true;
        break;
      }
    }

    if (!found) {
      if (prefix_count_ == prefix_cap_) {
        size_t new_cap = prefix_cap_ == 0 ? 16 : prefix_cap_ * 2;
        std::string *new_keys =
            (std::string *)arena_.alloc(new_cap * sizeof(std::string));
        size_t *new_vals = (size_t *)arena_.alloc(new_cap * sizeof(size_t));
        if (prefix_count_ > 0) {
          for (size_t i = 0; i < prefix_count_; ++i) {
            new (&new_keys[i]) std::string(prefix_keys_[i]);
            new_vals[i] = prefix_vals_[i];
          }
        }
        prefix_keys_ = new_keys;
        prefix_vals_ = new_vals;
        prefix_cap_ = new_cap;
      }

      bucket = subs_count_;
      new (&prefix_keys_[prefix_count_]) std::string(pfx);
      prefix_vals_[prefix_count_] = bucket;
      prefix_count_++;

      if (subs_count_ == subs_cap_) {
        size_t new_cap = subs_cap_ == 0 ? 16 : subs_cap_ * 2;
        JITCodeTrie **new_subs =
            (JITCodeTrie **)arena_.alloc(new_cap * sizeof(JITCodeTrie *));
        if (subs_count_ > 0) {
          memcpy(new_subs, subs_, subs_count_ * sizeof(JITCodeTrie *));
        }
        subs_ = new_subs;
        subs_cap_ = new_cap;
      }

      void *mem = arena_.alloc(sizeof(JITCodeTrie));
      auto *st = new (mem) JITCodeTrie(&arena_);
      subs_[subs_count_++] = st;
    }

    bool ok = subs_[bucket]->insert(key, handle);
    if (ok)
      count_++;
    return ok;
  }

  void compact() {
    for (size_t i = 0; i < subs_count_; ++i) {
      subs_[i]->compact();
    }

    for (size_t i = 0; i < prefix_count_; ++i) {
      meta_.insert(prefix_keys_[i], prefix_vals_[i] + 1);
    }
    meta_.compact();
  }

  __attribute__((noinline)) uint64_t lookup(const char *key, size_t len) const {
    if (len < pstart_ + plen_)
      return 0;
    uint64_t bucket_id = meta_.lookup(key + pstart_, plen_);
    if (bucket_id == 0)
      return 0;
    return subs_[bucket_id - 1]->lookup(key, len);
  }

  size_t count() const { return count_; }
  size_t num_buckets() const { return subs_count_; }

  size_t memory_used() const { return arena_.offset; }

  JITCodeTrie *get_sub(size_t i) { return subs_[i]; }

  JITCodeTrie meta() { return meta_; } // Copy for access

private:
  size_t pstart_, plen_;
  size_t count_ = 0;
  Arena arena_;

  std::string *prefix_keys_ = nullptr;
  size_t *prefix_vals_ = nullptr;
  size_t prefix_count_ = 0;
  size_t prefix_cap_ = 0;

  JITCodeTrie meta_;

  JITCodeTrie **subs_ = nullptr;
  size_t subs_count_ = 0;
  size_t subs_cap_ = 0;
};

static std::tuple<size_t, size_t, size_t>
auto_prefix(std::vector<std::pair<std::string, uint64_t>> &entries,
            size_t target) {
  size_t start = 2;
  for (size_t plen = 1; plen <= 7; plen++) {
    std::unordered_map<std::string, size_t> counts;
    for (auto &e : entries) {
      if (e.first.size() < start + plen)
        continue;
      counts[e.first.substr(start, plen)]++;
    }
    size_t max_count = 0;
    for (auto &kv : counts)
      max_count = std::max(max_count, kv.second);
    printf(" auto_prefix: plen=%zu buckets=%zu max_bucket=%zu\n", plen,
           counts.size(), max_count);
    if (max_count <= target)
      return {start, plen, max_count};
  }
  return {start, 5, target};
}

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
  double sum = 0;
  for (auto v : t)
    sum += v;
  return {sum / samples, t[samples / 2], t[size_t(samples * 0.95)]};
}

static void run(const char *label,
                std::vector<std::pair<std::string, uint64_t>> &entries) {
  size_t N = entries.size();

  auto prefix_info = auto_prefix(entries, 1500);
  size_t pstart = std::get<0>(prefix_info);
  size_t plen = std::get<1>(prefix_info);
  size_t max_bucket = std::get<2>(prefix_info);

  printf("\n═══════════════════════════════════════════════════════════════════"
         "\n");
  printf(" %s (%zu entries, prefix=key[%zu..%zu], max_bucket=%zu)\n", label, N,
         pstart, pstart + plen - 1, max_bucket);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  FractalJITTrie fractal(pstart, plen, max_bucket + 128);

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
  printf(" Buckets: %zu\n", fractal.num_buckets());
  printf(" Memory used by Trie: %.2f KB\n", fractal.memory_used() / 1024.0);

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
  if (!ok)
    return;
  printf(" ✅ ALL %zu verified + miss OK\n", N);

  // Verify ordering via Cursor
  bool ordered = true;
  std::string last_key = "";
  for (size_t i = 0; i < fractal.num_buckets(); ++i) {
    JITCursor cursor(*fractal.get_sub(i));
    while (cursor.valid()) {
      if (cursor.key() < last_key && !last_key.empty()) {
        printf(" ❌ Ordering failed! %s comes after %s\n", cursor.key().c_str(),
               last_key.c_str());
        ordered = false;
        break;
      }
      last_key = cursor.key();
      cursor.next();
    }
    if (!ordered)
      break;
  }
  if (ordered)
    printf(" ✅ Iterator proven perfectly SORTED\n");

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
  printf(" ║ ★ FRACTAL trie      ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_fractal.mean, r_fractal.p50, r_fractal.p95);
  printf(" ╚═══════════════════════╩═════════╩═════════╩═════════╝\n");

  double sf = r_umap.p50 / r_fractal.p50;
  printf(" → Fractal: %.2fx %s than unordered_map\n", sf > 1 ? sf : 1 / sf,
         sf > 1 ? "FASTER" : "SLOWER");
}

int main() {
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║ ★ FRACTAL JIT Code Trie — Tree of Trees x86_64              ║\n");
  printf("║ Dynamic insert(). Zero-alloc lookup(). Auto-partition.        ║\n");
  printf("║ Ordered via perfectly sorted leaf insertion.                  ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 10000, 100000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }
  return 0;
}
