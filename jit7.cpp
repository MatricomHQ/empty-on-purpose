// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — x86-64 Pure JIT Adaptive Radix Trie
//
// 100% JIT, Zero Copy, Zero Alloc (Bump Arena), Ordered, No Hashmaps
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

namespace x64 {
inline void emit_cmp_rdx_rsi(uint8_t *code, int &c) {
  code[c++] = 0x48;
  code[c++] = 0x39;
  code[c++] = 0xF2;
}
inline int emit_je_dummy(uint8_t *code, int &c) {
  code[c++] = 0x0F;
  code[c++] = 0x84;
  int p = c;
  c += 4;
  return p;
}
inline int emit_jne_dummy(uint8_t *code, int &c) {
  code[c++] = 0x0F;
  code[c++] = 0x85;
  int p = c;
  c += 4;
  return p;
}
inline int emit_ja_dummy(uint8_t *code, int &c) {
  code[c++] = 0x0F;
  code[c++] = 0x87;
  int p = c;
  c += 4;
  return p;
}
inline int emit_jmp_dummy(uint8_t *code, int &c) {
  code[c++] = 0xE9;
  int p = c;
  c += 4;
  return p;
}
inline void patch_rel32(uint8_t *code, int patch_offset, int target_offset) {
  int32_t rel = target_offset - (patch_offset + 4);
  memcpy(code + patch_offset, &rel, 4);
}
inline void patch_jmp_rel32_addr(uint8_t *code, int patch_offset,
                                 uint8_t *target_addr) {
  int32_t rel = target_addr - (code + patch_offset + 4);
  memcpy(code + patch_offset, &rel, 4);
}
inline void emit_mov_rax(uint8_t *code, int &c, uint64_t val) {
  code[c++] = 0x48;
  code[c++] = 0xB8;
  memcpy(code + c, &val, 8);
  c += 8;
}
inline void emit_xor_eax_eax(uint8_t *code, int &c) {
  code[c++] = 0x31;
  code[c++] = 0xC0;
}
inline void emit_ret(uint8_t *code, int &c) { code[c++] = 0xC3; }
inline void emit_load_byte(uint8_t *code, int &c) {
  code[c++] = 0x0F;
  code[c++] = 0xB6;
  code[c++] = 0x04;
  code[c++] = 0x17;
}
inline void emit_inc_rdx(uint8_t *code, int &c) {
  code[c++] = 0x48;
  code[c++] = 0xFF;
  code[c++] = 0xC2;
}
inline void emit_dec_rdx(uint8_t *code, int &c) {
  code[c++] = 0x48;
  code[c++] = 0xFF;
  code[c++] = 0xCA;
}
inline void emit_cmp_al_imm(uint8_t *code, int &c, uint8_t val) {
  code[c++] = 0x3C;
  code[c++] = val;
}

inline void emit_jmp_table(uint8_t *code, int &c, int32_t *table_data) {
  code[c++] = 0x4C;
  code[c++] = 0x8D;
  code[c++] = 0x05;
  uint32_t rel = 9;
  memcpy(code + c, &rel, 4);
  c += 4;
  code[c++] = 0x4F;
  code[c++] = 0x63;
  code[c++] = 0x0C;
  code[c++] = 0x80;
  code[c++] = 0x4D;
  code[c++] = 0x01;
  code[c++] = 0xC8;
  code[c++] = 0x41;
  code[c++] = 0xFF;
  code[c++] = 0xE0;
  memcpy(code + c, table_data, 1024);
  c += 1024;
}
} // namespace x64

class JitTrie {
  uint8_t *mem_;
  size_t mem_size_;
  size_t cursor_;
  size_t count_;

public:
  enum class NodeType : uint16_t {
    LEAF = 0,
    NODE4 = 4,
    NODE16 = 16,
    NODE256 = 256
  };

  struct alignas(8) NodeHeader {
    uint64_t value;
    uint32_t prefix_offset;
    uint16_t prefix_len;
    uint16_t type;
    uint16_t count;

    uint8_t *keys() { return reinterpret_cast<uint8_t *>(this + 1); }
    uint32_t *children() {
      if (type == 256)
        return reinterpret_cast<uint32_t *>(this + 1);
      return reinterpret_cast<uint32_t *>(keys() + type);
    }
    uint8_t *jit_code() {
      uint8_t *base = reinterpret_cast<uint8_t *>(this + 1);
      if (type == 0)
        return base;
      if (type == 256)
        return base + 256 * 4;
      return base + type + type * 4;
    }
  };

  size_t get_max_jit_size(uint16_t type, uint16_t prefix_len) {
    size_t base = 128 + prefix_len * 24;
    if (type == 0)
      return base;
    if (type == 4)
      return base + 128;
    if (type == 16)
      return base + 512;
    if (type == 256)
      return base + 2048 + 128;
    return 0;
  }

  size_t get_node_alloc_size(uint16_t type, uint16_t prefix_len) {
    size_t sz = sizeof(NodeHeader);
    if (type == 4 || type == 16)
      sz += type + type * 4;
    else if (type == 256)
      sz += 256 * 4;
    sz += get_max_jit_size(type, prefix_len);
    return (sz + 7) & ~7;
  }

  uint32_t root_offset_;
  using LookupFn = uint64_t (*)(const char *, size_t, size_t);
  LookupFn fn_ = nullptr;

  JitTrie(size_t cap_bytes = 1024 * 1024 * 512)
      : mem_size_(cap_bytes), cursor_(0), count_(0), root_offset_(0) {
    // offset 0 is reserved to represent "NULL"
    mem_ =
        (uint8_t *)mmap(nullptr, mem_size_, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem_ == MAP_FAILED) {
      mem_ = nullptr;
      return;
    }
    alloc(8);
    root_offset_ = alloc_node(0, 0);
    jit_node(root_offset_);
    fn_ = reinterpret_cast<LookupFn>(node(root_offset_)->jit_code());
  }
  ~JitTrie() {
    if (mem_)
      munmap(mem_, mem_size_);
  }

  bool valid() const { return mem_ != nullptr; }
  size_t count() const { return count_; }
  size_t code_bytes() const { return cursor_; }

  uint32_t alloc(size_t sz) {
    sz = (sz + 7) & ~7;
    uint32_t res = cursor_;
    cursor_ += sz;
    return res;
  }
  uint32_t alloc_string(const char *str, size_t len) {
    if (len == 0)
      return 0;
    uint32_t res = alloc(len);
    memcpy(ptr(res), str, len);
    return res;
  }
  uint32_t alloc_node(uint16_t type, uint16_t prefix_len) {
    size_t sz = get_node_alloc_size(type, prefix_len);
    uint32_t res = alloc(sz);
    memset(ptr(res), 0, sz);
    NodeHeader *h = (NodeHeader *)ptr(res);
    h->type = type;
    h->prefix_len =
        prefix_len; // we set it here to remember allocation size base
    return res;
  }

  uint8_t *ptr(uint32_t offset) const { return mem_ + offset; }
  NodeHeader *node(uint32_t offset) const {
    return (NodeHeader *)(mem_ + offset);
  }

  void emit_bsearch2(uint8_t *code, int &c, uint8_t *keys, uint32_t *children,
                     int L, int R, std::vector<int> &misses) {
    if (L > R) {
      misses.push_back(x64::emit_jmp_dummy(code, c));
      return;
    }
    int mid = (L + R) / 2;
    x64::emit_cmp_al_imm(code, c, keys[mid]);
    int je_idx = x64::emit_je_dummy(code, c);

    if (L == R) {
      misses.push_back(x64::emit_jmp_dummy(code, c));
    } else {
      int ja_idx = x64::emit_ja_dummy(code, c);
      emit_bsearch2(code, c, keys, children, L, mid - 1, misses);
      x64::patch_rel32(code, ja_idx, c);
      emit_bsearch2(code, c, keys, children, mid + 1, R, misses);
    }

    x64::patch_rel32(code, je_idx, c);
    NodeHeader *ch = node(children[mid]);
    int jmp_idx = x64::emit_jmp_dummy(code, c);
    x64::patch_jmp_rel32_addr(code, jmp_idx, ch->jit_code());
  }

  void jit_node(uint32_t offset) {
    NodeHeader *n = node(offset);
    uint8_t *code = n->jit_code();
    int c = 0;

    std::vector<int> misses;

    // 1. Prefix check
    if (n->prefix_len > 0) {
      uint8_t *pfx = ptr(n->prefix_offset);
      for (int i = 0; i < n->prefix_len; i++) {
        x64::emit_cmp_rdx_rsi(code, c);
        misses.push_back(x64::emit_je_dummy(code, c));
        x64::emit_load_byte(code, c);
        x64::emit_inc_rdx(code, c);
        x64::emit_cmp_al_imm(code, c, pfx[i]);
        misses.push_back(x64::emit_jne_dummy(code, c));
      }
    }

    // 2. Value check (if terminal)
    if (n->value != 0) {
      x64::emit_cmp_rdx_rsi(code, c);
      int jne_load = x64::emit_jne_dummy(code, c);
      x64::emit_mov_rax(code, c, n->value);
      x64::emit_ret(code, c);
      x64::patch_rel32(code, jne_load, c);
    } else {
      x64::emit_cmp_rdx_rsi(code, c);
      misses.push_back(x64::emit_je_dummy(code, c));
    }

    // 3. Child dispatch
    if (n->count > 0) {
      x64::emit_load_byte(code, c);
      x64::emit_inc_rdx(code, c);

      uint8_t *keys = n->keys();
      uint32_t *children = n->children();

      if (n->type == 4) {
        int jump_targets[4];
        for (int i = 0; i < n->count; i++) {
          x64::emit_cmp_al_imm(code, c, keys[i]);
          jump_targets[i] = x64::emit_je_dummy(code, c);
        }
        misses.push_back(x64::emit_jmp_dummy(code, c));

        for (int i = 0; i < n->count; i++) {
          x64::patch_rel32(code, jump_targets[i], c);
          NodeHeader *ch = node(children[i]);
          int jmp_idx = x64::emit_jmp_dummy(code, c);
          x64::patch_jmp_rel32_addr(code, jmp_idx, ch->jit_code());
        }
      } else if (n->type == 16) {
        emit_bsearch2(code, c, keys, children, 0, n->count - 1, misses);
      } else if (n->type == 256) {
        int32_t table[256];
        int table_start = c + 17;
        for (int i = 0; i < 256; i++) {
          if (children[i]) {
            NodeHeader *ch = node(children[i]);
            uint8_t *tgt = ch->jit_code();
            uint8_t *table_addr = code + table_start;
            table[i] = (int32_t)(tgt - table_addr);
          } else {
            table[i] = 0;
          }
        }
        x64::emit_jmp_table(code, c, table);
        int miss_addr_c = c;
        for (int i = 0; i < 256; i++) {
          if (children[i] == 0) {
            int32_t rel_miss =
                (int32_t)((code + miss_addr_c) - (code + table_start));
            memcpy(code + table_start + i * 4, &rel_miss, 4);
          }
        }
      }
    }

    // 4. Miss block
    int miss_block_idx = c;
    x64::emit_xor_eax_eax(code, c);
    x64::emit_ret(code, c);

    for (int m : misses) {
      x64::patch_rel32(code, m, miss_block_idx);
    }

    if (n->type != 0 && n->type != 4 && n->type != 16 && n->type != 256) {
      printf("PANIC: CORRUPT NODE TYPE %d AT OFFSET %u\n", n->type, offset);
      exit(1);
    }
    if (c > get_max_jit_size(n->type, n->prefix_len)) {
      printf(
          "PANIC: JIT size %d exceeded max %zu for type %d (prefix_len=%d)\n",
          c, get_max_jit_size(n->type, n->prefix_len), n->type, n->prefix_len);
      exit(1);
    }
  }

  uint32_t find_child(NodeHeader *n, uint8_t key, int &pos) {
    if (n->count == 0)
      return 0;
    uint8_t *keys = n->keys();
    uint32_t *children = n->children();
    if (n->type == 256) {
      pos = key;
      return children[key];
    }
    int L = 0, R = n->count - 1;
    while (L <= R) {
      int m = (L + R) / 2;
      if (keys[m] == key) {
        pos = m;
        return children[m];
      }
      if (keys[m] < key)
        L = m + 1;
      else
        R = m - 1;
    }
    pos = L;
    return 0;
  }

  void add_child(uint32_t &node_off, uint8_t key, uint32_t child_off) {
    NodeHeader *n = node(node_off);
    if (n->type == 256) {
      n->children()[key] = child_off;
      n->count++;
      jit_node(node_off);
      return;
    }
    if (n->count < n->type) {
      int pos;
      find_child(n, key, pos);
      uint8_t *keys = n->keys();
      uint32_t *children = n->children();
      for (int i = n->count; i > pos; i--) {
        keys[i] = keys[i - 1];
        children[i] = children[i - 1];
      }
      keys[pos] = key;
      children[pos] = child_off;
      n->count++;
      jit_node(node_off);
    } else {
      uint16_t new_type = (n->type == 0) ? 4 : (n->type == 4) ? 16 : 256;
      uint32_t new_off = alloc_node(new_type, n->prefix_len);
      NodeHeader *nn = node(new_off);
      nn->value = n->value;
      nn->prefix_offset = n->prefix_offset;
      nn->prefix_len = n->prefix_len;

      uint8_t *ok = n->keys();
      uint32_t *oc = n->children();

      if (new_type == 256) {
        uint32_t *nc = nn->children();
        for (int i = 0; i < n->count; i++)
          nc[ok[i]] = oc[i];
        nn->count = n->count;
      } else {
        uint8_t *nk = nn->keys();
        uint32_t *nc = nn->children();
        for (int i = 0; i < n->count; i++) {
          nk[i] = ok[i];
          nc[i] = oc[i];
        }
        nn->count = n->count;
      }
      node_off = new_off;
      add_child(node_off, key, child_off);
    }
  }

  bool _insert_recursive(uint32_t &cur_off, const std::string &key,
                         uint64_t handle, size_t depth) {
    NodeHeader *n = node(cur_off);
    const char *k = key.c_str();
    size_t len = key.size();

    if (n->prefix_len > 0) {
      uint8_t *pfx = ptr(n->prefix_offset);
      int match_len = 0;
      while (match_len < n->prefix_len && depth + match_len < len &&
             pfx[match_len] == (uint8_t)k[depth + match_len]) {
        match_len++;
      }
      if (match_len < n->prefix_len) {
        // Split
        uint32_t split_off = alloc_node(4, match_len);
        NodeHeader *split = node(split_off);
        if (match_len > 0)
          split->prefix_offset = alloc_string((const char *)pfx, match_len);

        n->prefix_len -= (match_len + 1);
        uint8_t diff_byte = pfx[match_len];

        if (n->prefix_len > 0) {
          n->prefix_offset =
              alloc_string((const char *)pfx + match_len + 1, n->prefix_len);
        } else {
          n->prefix_offset = 0;
        }

        split->keys()[0] = diff_byte;
        split->children()[0] = cur_off;
        split->count = 1;

        uint32_t old_off = cur_off;
        cur_off = split_off;

        jit_node(old_off);
        jit_node(split_off);

        n = node(cur_off);
      }
      depth += match_len;
    }

    if (depth == len) {
      if (n->value == 0)
        count_++;
      n->value = handle;
      jit_node(cur_off);
      return true;
    }

    uint8_t byte = k[depth];
    int pos;
    uint32_t next = find_child(n, byte, pos);
    if (next == 0) {
      size_t rem = len - depth - 1;
      uint32_t leaf_off = alloc_node(0, rem);
      NodeHeader *leaf = node(leaf_off);
      leaf->value = handle;
      if (rem > 0) {
        leaf->prefix_offset = alloc_string(k + depth + 1, rem);
      }
      jit_node(leaf_off);
      add_child(cur_off, byte, leaf_off);
      count_++;
      return true;
    } else {
      uint32_t old_next = next;
      bool ok = _insert_recursive(next, key, handle, depth + 1);
      if (next != old_next) {
        if (n->type == 256) {
          n->children()[byte] = next;
        } else {
          n->children()[pos] = next;
        }
        jit_node(cur_off);
      }
      return ok;
    }
  }

  bool insert(const std::string &key, uint64_t handle) {
    if (count_ == 0) {
      uint32_t new_root = alloc_node(0, key.size());
      NodeHeader *r = node(new_root);
      r->value = handle;
      if (key.size() > 0)
        r->prefix_offset = alloc_string(key.c_str(), key.size());
      root_offset_ = new_root;
      jit_node(root_offset_);
      count_++;
      fn_ = reinterpret_cast<LookupFn>(node(root_offset_)->jit_code());
      return true;
    }

    uint32_t old_root = root_offset_;
    bool ok = _insert_recursive(root_offset_, key, handle, 0);
    if (root_offset_ != old_root || ok) {
      fn_ = reinterpret_cast<LookupFn>(node(root_offset_)->jit_code());
    }
    return ok;
  }

  uint64_t lookup(const char *key, size_t len) const {
    if (!fn_)
      return 0;
    return fn_(key, len, 0);
  }

  // Ordered cursor iteration
  class Cursor {
    struct State {
      uint32_t node_off;
      int child_idx; // currently processing child index
    };
    std::vector<State> stack;
    JitTrie *trie;
    std::string pfx;
    uint64_t val;
    bool valid;

    void advance() {
      while (!stack.empty()) {
        State &s = stack.back();
        NodeHeader *n = trie->node(s.node_off);

        if (s.child_idx == -1) {
          // Node just pushed. Push its prefix.
          if (n->prefix_len > 0) {
            uint8_t *p = trie->ptr(n->prefix_offset);
            pfx.append((char *)p, n->prefix_len);
          }
          s.child_idx = 0;
          if (n->value != 0) {
            val = n->value;
            return; // Yield this node
          }
        }

        // Process next child
        uint32_t next_child = 0;
        uint8_t next_byte = 0;

        if (n->type == 256) {
          while (s.child_idx < 256) {
            if (n->children()[s.child_idx]) {
              next_child = n->children()[s.child_idx];
              next_byte = s.child_idx;
              s.child_idx++;
              break;
            }
            s.child_idx++;
          }
        } else {
          if (s.child_idx < n->count) {
            next_child = n->children()[s.child_idx];
            next_byte = n->keys()[s.child_idx];
            s.child_idx++;
          }
        }

        if (next_child != 0) {
          pfx.push_back((char)next_byte);
          stack.push_back({next_child, -1});
        } else {
          // Done with this node, pop it
          if (n->prefix_len > 0) {
            pfx.erase(pfx.size() - n->prefix_len);
          }
          if (stack.size() >
              1) { // if not root, we also pushed the child routing byte
            pfx.pop_back();
          }
          stack.pop_back();
        }
      }
      valid = false;
    }

  public:
    Cursor(JitTrie *t) : trie(t), val(0), valid(false) {
      if (trie->count() > 0) {
        stack.push_back({trie->root_offset_, -1});
        valid = true;
        advance();
      }
    }
    bool is_valid() const { return valid; }
    std::string key() const { return pfx; }
    uint64_t value() const { return val; }
    void next() { advance(); }
  };

  Cursor get_cursor() { return Cursor(this); }
};

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

static void run(const char *label,
                std::vector<std::pair<std::string, uint64_t>> &entries) {
  size_t N = entries.size();
  printf("\n═══════════════════════════════════════════════════════════════════"
         "\n");
  printf("  %s (%zu entries)\n", label, N);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  JitTrie trie;

  auto it0 = std::chrono::high_resolution_clock::now();
  for (auto &e : entries)
    trie.insert(e.first, e.second);
  auto it1 = std::chrono::high_resolution_clock::now();
  double ins_ms = std::chrono::duration<double, std::milli>(it1 - it0).count();

  printf("  Insert:     %.2fms (%.0fns/entry)\n", ins_ms, ins_ms * 1e6 / N);
  printf("  Total code: %.2fMB (%.1f B/entry)\n", trie.code_bytes() / 1048576.0,
         double(trie.code_bytes()) / N);

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  for (auto &e : entries)
    umap[e.first] = e.second;

  bool ok = true;
  for (size_t i = 0; i < N && ok; i++) {
    uint64_t h = trie.lookup(entries[i].first.c_str(), entries[i].first.size());
    if (h != entries[i].second) {
      printf("  ❌ '%s' got %llu want %llu\n", entries[i].first.c_str(),
             (unsigned long long)h, (unsigned long long)entries[i].second);
      ok = false;
    }
  }
  if (!ok)
    return;
  printf("  ✅ ALL %zu verified\n", N);

  // Verify ordering
  auto cursor = trie.get_cursor();
  std::string last_key = "";
  bool ordered = true;
  size_t c_count = 0;
  while (cursor.is_valid()) {
    if (cursor.key() < last_key)
      ordered = false;
    last_key = cursor.key();
    cursor.next();
    c_count++;
  }
  if (ordered && c_count == N) {
    printf("  ✅ PERFECT ORDER VERIFIED (Cursor traversed %zu items)\n",
           c_count);
  } else {
    printf("  ❌ Cursor failed! count=%zu, ordered=%d\n", c_count, ordered);
  }

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
        DoNotOptimize(trie.lookup(ks[ki], ls[ki]));
        ki = (ki + 1) % N;
      },
      bs, 500);

  printf("\n  ╔═══════════════════════╦═════════╦═════════╦═════════╗\n");
  printf("  ║ Approach              ║ Mean    ║ P50     ║ P95     ║\n");
  printf("  ╠═══════════════════════╬═════════╬═════════╬═════════╣\n");
  printf("  ║ std::unordered_map    ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_umap.mean, r_umap.p50, r_umap.p95);
  printf("  ║ ★ PURE JIT TRIE       ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_fractal.mean, r_fractal.p50, r_fractal.p95);
  printf("  ╚═══════════════════════╩═════════╩═════════╩═════════╝\n");

  double sf = r_umap.p50 / r_fractal.p50;
  printf("  → PURE JIT: %.2fx %s than unordered_map\n", sf > 1 ? sf : 1 / sf,
         sf > 1 ? "FASTER" : "SLOWER");
}

int main() {
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║  ★ PURE JIT Adaptive Radix Trie                           ║\n");
  printf("║  Zero-alloc lookup(). Ordered Cursor. 100%% JIT.            ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 5000, 10000, 50000, 100000, 500000, 1000000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }
  return 0;
}
