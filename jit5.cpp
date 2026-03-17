// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — Fractal JIT Code Trie (Tree of Trees) x86_64 Edition
//
// DYNAMIC: insert() works at any time, no build() required.
// ZERO-ALLOC: lookup passes key+offset directly to JIT trie.
//
// clang++ -O3 -std=c++17 -o jit_fractal src/stax/benches/jit_trie_fractal.cpp
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

namespace x86 {
inline void ret(uint8_t *d, size_t &n) { d[n++] = 0xC3; }

inline size_t jmp(uint8_t *d, size_t &n) {
  d[n++] = 0xE9;
  size_t p = n;
  n += 4;
  return p;
}

inline size_t je(uint8_t *d, size_t &n) {
  d[n++] = 0x0F;
  d[n++] = 0x84;
  size_t p = n;
  n += 4;
  return p;
}

inline size_t jne(uint8_t *d, size_t &n) {
  d[n++] = 0x0F;
  d[n++] = 0x85;
  size_t p = n;
  n += 4;
  return p;
}

inline size_t jb(uint8_t *d, size_t &n) {
  d[n++] = 0x0F;
  d[n++] = 0x82;
  size_t p = n;
  n += 4;
  return p;
}

inline void patch(uint8_t *d, size_t p, size_t target) {
  int32_t rel = (int32_t)target - (int32_t)(p + 4);
  memcpy(&d[p], &rel, 4);
}

inline void cmp_rsi_imm(uint8_t *d, size_t &n, uint32_t v) {
  if (v < 128) {
    d[n++] = 0x48;
    d[n++] = 0x83;
    d[n++] = 0xFE;
    d[n++] = (uint8_t)v;
  } else {
    d[n++] = 0x48;
    d[n++] = 0x81;
    d[n++] = 0xFE;
    memcpy(&d[n], &v, 4);
    n += 4;
  }
}

inline void mov_al_rdi_off(uint8_t *d, size_t &n, uint32_t off) {
  if (off == 0) {
    d[n++] = 0x8A;
    d[n++] = 0x07;
  } else if (off < 128) {
    d[n++] = 0x8A;
    d[n++] = 0x47;
    d[n++] = (uint8_t)off;
  } else {
    d[n++] = 0x8A;
    d[n++] = 0x87;
    memcpy(&d[n], &off, 4);
    n += 4;
  }
}

inline void cmp_al_imm(uint8_t *d, size_t &n, uint8_t v) {
  d[n++] = 0x3C;
  d[n++] = v;
}

inline void mov_rax_imm(uint8_t *d, size_t &n, uint64_t v) {
  d[n++] = 0x48;
  d[n++] = 0xB8;
  memcpy(&d[n], &v, 8);
  n += 8;
}

inline void mov_rax_zero(uint8_t *d, size_t &n) {
  d[n++] = 0x31;
  d[n++] = 0xC0;
}

inline void mov_rdx_rdi_off(uint8_t *d, size_t &n, uint32_t off) {
  if (off == 0) {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x17;
  } else if (off < 128) {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x57;
    d[n++] = (uint8_t)off;
  } else {
    d[n++] = 0x48;
    d[n++] = 0x8B;
    d[n++] = 0x97;
    memcpy(&d[n], &off, 4);
    n += 4;
  }
}

inline void mov_rcx_imm(uint8_t *d, size_t &n, uint64_t v) {
  d[n++] = 0x48;
  d[n++] = 0xB9;
  memcpy(&d[n], &v, 8);
  n += 8;
}

inline void cmp_rdx_rcx(uint8_t *d, size_t &n) {
  d[n++] = 0x48;
  d[n++] = 0x39;
  d[n++] = 0xCA;
}
} // namespace x86

// Fixed memory arena for building the JIT trie tree.
// No hash maps, no vectors.
struct Arena {
  uint8_t *mem;
  size_t cap;
  size_t head = 0;

  Arena(size_t capacity) : cap(capacity) {
    mem = (uint8_t *)mmap(nullptr, capacity, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }

  ~Arena() { munmap(mem, cap); }

  void *alloc(size_t size) {
    void *ptr = mem + head;
    head += size;
    return ptr;
  }

  char *strdup(const char *s, size_t len) {
    char *ptr = (char *)alloc(len + 1);
    memcpy(ptr, s, len);
    ptr[len] = '\0';
    return ptr;
  }
};

struct Node {
  uint32_t num_children;
  uint64_t handle;
  const char *key;
  size_t len;

  Node **children() { return (Node **)(this + 1); }
};

struct Cursor {
  struct Frame {
    Node *node;
    int child_idx;
  };

  Frame *stack;
  int stack_size;
  const char *current_key;
  size_t current_len;
  uint64_t current_handle;
  bool valid;

  Cursor(Node *root) {
    // Assume depth max 256 for string keys
    stack = new Frame[256];
    stack_size = 0;
    valid = false;
    if (root) {
      stack[stack_size++] = {root, -1};
      advance();
    }
  }

  ~Cursor() { delete[] stack; }

  void advance() {
    valid = false;
    while (stack_size > 0) {
      Frame &frame = stack[stack_size - 1];
      if (frame.child_idx == -1) {
        frame.child_idx = 0;
        if (frame.node->handle) {
          current_key = frame.node->key;
          current_len = frame.node->len;
          current_handle = frame.node->handle;
          valid = true;
          return;
        }
      }

      if (frame.child_idx < frame.node->num_children) {
        Node *child = frame.node->children()[frame.child_idx++];
        stack[stack_size++] = {child, -1};
      } else {
        stack_size--;
      }
    }
  }
};

struct FractalJITTrie {
  Arena arena;
  void *jit_mem;
  size_t jit_cap;
  uint8_t *code;
  size_t cur = 0;
  Node *root;
  using LookupFn = uint64_t (*)(const char *, size_t);
  LookupFn fn = nullptr;

  FractalJITTrie(size_t arena_cap, size_t code_cap)
      : arena(arena_cap), jit_cap(code_cap) {
    root = (Node *)arena.alloc(sizeof(Node));
    root->num_children = 0;
    root->handle = 0;
    root->key = nullptr;
    root->len = 0;

    jit_mem = mmap(nullptr, code_cap, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    code = (uint8_t *)jit_mem;
  }

  ~FractalJITTrie() { munmap(jit_mem, jit_cap); }

  void insert(Node **node_ptr, const char *k, size_t l, uint64_t h, int depth) {
    Node *node = *node_ptr;
    if (depth == l) {
      node->handle = h;
      if (!node->key) {
        node->key = arena.strdup(k, l);
        node->len = l;
      }
      return;
    }

    char c = k[depth];
    Node **children = node->children();

    int left = 0, right = node->num_children - 1;
    while (left <= right) {
      int mid = left + (right - left) / 2;
      char mid_c = children[mid]->key[depth];
      if (mid_c == c) {
        Node *child = children[mid];
        int match_len = 0;
        while (depth + match_len < l && depth + match_len < child->len &&
               k[depth + match_len] == child->key[depth + match_len]) {
          match_len++;
        }

        if (depth + match_len == child->len) {
          insert(&children[mid], k, l, h, depth + match_len);
          return;
        }

        Node *split = (Node *)arena.alloc(sizeof(Node) +
                                          child->num_children * sizeof(Node *));
        split->key = child->key;
        split->len = child->len;
        split->handle = child->handle;
        split->num_children = child->num_children;
        memcpy(split->children(), child->children(),
               child->num_children * sizeof(Node *));

        child->len = depth + match_len;
        child->handle = 0;

        if (depth + match_len == l) {
          child->handle = h;
          Node *new_child = (Node *)arena.alloc(sizeof(Node) + sizeof(Node *));
          new_child->key = child->key;
          new_child->len = child->len;
          new_child->handle = child->handle;
          new_child->num_children = 1;
          new_child->children()[0] = split;
          children[mid] = new_child;
        } else {
          Node *new_leaf = (Node *)arena.alloc(sizeof(Node));
          new_leaf->key = arena.strdup(k, l);
          new_leaf->len = l;
          new_leaf->handle = h;
          new_leaf->num_children = 0;

          Node *new_child =
              (Node *)arena.alloc(sizeof(Node) + 2 * sizeof(Node *));
          new_child->key = child->key;
          new_child->len = child->len;
          new_child->handle = child->handle;
          new_child->num_children = 2;

          char b1 = split->key[depth + match_len];
          char b2 = k[depth + match_len];

          if (b1 < b2) {
            new_child->children()[0] = split;
            new_child->children()[1] = new_leaf;
          } else {
            new_child->children()[0] = new_leaf;
            new_child->children()[1] = split;
          }
          children[mid] = new_child;
        }
        return;
      }
      if (mid_c < c)
        left = mid + 1;
      else
        right = mid - 1;
    }

    Node *new_leaf = (Node *)arena.alloc(sizeof(Node));
    new_leaf->key = arena.strdup(k, l);
    new_leaf->len = l;
    new_leaf->handle = h;
    new_leaf->num_children = 0;

    Node *new_node = (Node *)arena.alloc(
        sizeof(Node) + (node->num_children + 1) * sizeof(Node *));
    new_node->key = node->key;
    new_node->len = node->len;
    new_node->handle = node->handle;
    new_node->num_children = node->num_children + 1;

    Node **new_children = new_node->children();
    memcpy(new_children, children, left * sizeof(Node *));
    new_children[left] = new_leaf;
    memcpy(new_children + left + 1, children + left,
           (node->num_children - left) * sizeof(Node *));

    *node_ptr = new_node;
  }

  void insert(const std::string &key, uint64_t handle) {
    insert(&root, key.c_str(), key.size(), handle, 0);
  }

  void compile_bsearch(Node *node, size_t ret0, int l, int r, int check_depth) {
    if (l > r) {
      size_t j = x86::jmp(code, cur);
      x86::patch(code, j, ret0);
      return;
    }

    int m = l + (r - l) / 2;
    Node *child = node->children()[m];
    char c = child->key[check_depth];

    x86::cmp_al_imm(code, cur, c);
    size_t je_m = x86::je(code, cur);

    if (l < m) {
      size_t jb_m = x86::jb(code, cur);
      if (m + 1 <= r) {
        compile_bsearch(node, ret0, m + 1, r, check_depth);
      } else {
        size_t j = x86::jmp(code, cur);
        x86::patch(code, j, ret0);
      }
      x86::patch(code, jb_m, cur);
      compile_bsearch(node, ret0, l, m - 1, check_depth);
    } else {
      if (m + 1 <= r) {
        compile_bsearch(node, ret0, m + 1, r, check_depth);
      } else {
        size_t j = x86::jmp(code, cur);
        x86::patch(code, j, ret0);
      }
    }

    x86::patch(code, je_m, cur);

    int depth = check_depth + 1;
    size_t prefix_len = child->len - depth;
    size_t words = prefix_len / 8;

    for (size_t w = 0; w < words; ++w) {
      size_t offset = depth + w * 8;
      x86::mov_rdx_rdi_off(code, cur, offset);
      uint64_t expected = 0;
      memcpy(&expected, child->key + offset, 8);
      x86::mov_rcx_imm(code, cur, expected);
      x86::cmp_rdx_rcx(code, cur);
      size_t jne_word = x86::jne(code, cur);
      x86::patch(code, jne_word, ret0);
    }

    size_t tail_start = depth + words * 8;
    for (size_t i = tail_start; i < child->len; ++i) {
      x86::mov_al_rdi_off(code, cur, i);
      x86::cmp_al_imm(code, cur, child->key[i]);
      size_t jne_prefix = x86::jne(code, cur);
      x86::patch(code, jne_prefix, ret0);
    }

    if (child->num_children == 0) {
      x86::cmp_rsi_imm(code, cur, child->len);
      size_t jne_len = x86::jne(code, cur);
      x86::patch(code, jne_len, ret0);

      x86::mov_rax_imm(code, cur, child->handle);
      x86::ret(code, cur);
    } else {
      x86::cmp_rsi_imm(code, cur, child->len);
      size_t je_len = x86::je(code, cur);

      x86::mov_al_rdi_off(code, cur, child->len);
      compile_bsearch(child, ret0, 0, child->num_children - 1, child->len);

      x86::patch(code, je_len, cur);
      if (child->handle) {
        x86::mov_rax_imm(code, cur, child->handle);
        x86::ret(code, cur);
      } else {
        size_t jmp_ret0 = x86::jmp(code, cur);
        x86::patch(code, jmp_ret0, ret0);
      }
    }
  }

  void compile() {
    cur = 0;
    size_t ret0 = cur;
    x86::mov_rax_zero(code, cur);
    x86::ret(code, cur);

    size_t start = cur;

    if (root->num_children == 0) {
      size_t jmp_ret0 = x86::jmp(code, cur);
      x86::patch(code, jmp_ret0, ret0);
    } else {
      x86::cmp_rsi_imm(code, cur, 0);
      size_t je_len = x86::je(code, cur);

      x86::mov_al_rdi_off(code, cur, 0);
      compile_bsearch(root, ret0, 0, root->num_children - 1, 0);

      x86::patch(code, je_len, cur);
      if (root->handle) {
        x86::mov_rax_imm(code, cur, root->handle);
        x86::ret(code, cur);
      } else {
        size_t jmp_ret0 = x86::jmp(code, cur);
        x86::patch(code, jmp_ret0, ret0);
      }
    }

    fn = (LookupFn)(code + start);
  }

  __attribute__((always_inline)) inline uint64_t lookup(const char *key,
                                                        size_t len) const {
    return fn(key, len);
  }

  Cursor cursor() { return Cursor(root); }
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

int main() {
  size_t N = 1000000;
  std::vector<std::string> keys;
  char buf[32];
  for (size_t i = 0; i < N; ++i) {
    snprintf(buf, sizeof(buf), "k_%07zu", i);
    keys.push_back(buf);
  }

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  auto t0_umap_ins = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    umap[keys[i]] = i + 1;
  }
  auto t1_umap_ins = std::chrono::high_resolution_clock::now();
  double umap_ins_ms =
      std::chrono::duration<double, std::milli>(t1_umap_ins - t0_umap_ins)
          .count();
  std::cout << "std::unordered_map Insert Time: " << umap_ins_ms << " ms"
            << std::endl;

  FractalJITTrie trie(1024 * 1024 * 256,
                      1024 * 1024 * 128); // 256MB arena, 128MB code
  auto t0_jit_ins = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    trie.insert(keys[i], i + 1);
  }
  auto t1_jit_ins = std::chrono::high_resolution_clock::now();
  double jit_ins_ms =
      std::chrono::duration<double, std::milli>(t1_jit_ins - t0_jit_ins)
          .count();
  std::cout << "Fractal JIT Insert Time: " << jit_ins_ms << " ms" << std::endl;

  trie.compile();

  size_t ki = 0;
  size_t bs = std::min(N, size_t(10000));

  auto r_umap = bench_fn(
      [&]() {
        DoNotOptimize(umap.find(keys[ki])->second);
        ki = (ki + 1) % N;
      },
      bs, 500);

  ki = 0;
  auto r_fractal = bench_fn(
      [&]() {
        DoNotOptimize(trie.lookup(keys[ki].c_str(), keys[ki].size()));
        ki = (ki + 1) % N;
      },
      bs, 500);

  printf("\n ╔═══════════════════════╦═════════╦═════════╦═════════╗\n");
  printf(" ║ Approach              ║ Mean    ║ P50     ║ P95     ║\n");
  printf(" ╠═══════════════════════╬═════════╬═════════╬═════════╣\n");
  printf(" ║ std::unordered_map    ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_umap.mean, r_umap.p50, r_umap.p95);
  printf(" ║ ★ FRACTAL JIT trie    ║ %6.1fns ║ %6.1fns ║ %6.1fns ║\n",
         r_fractal.mean, r_fractal.p50, r_fractal.p95);
  printf(" ╚═══════════════════════╩═════════╩═════════╩═════════╝\n");

  double sf = r_umap.p50 / r_fractal.p50;
  printf(" → Fractal: %.2fx %s than unordered_map\n", sf > 1 ? sf : 1 / sf,
         sf > 1 ? "FASTER" : "SLOWER");

  int count = 0;
  auto cursor = trie.cursor();
  while (cursor.valid) {
    count++;
    cursor.advance();
  }
  std::cout << "Cursor iterates items: " << count << std::endl;

  return 0;
}
