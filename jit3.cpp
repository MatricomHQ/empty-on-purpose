// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — Fractal JIT Code Trie (Tree of Trees)
//
// A JIT Trie with:
// 1. Extreme lookup performance via JIT assembly
// 2. High insert performance
// 3. Compact memory usage
// 4. Ordered iteration via Cursor
//
// g++ -O3 src/stax/benches/jit_trie_fractal.cpp -o
// src/stax/benches/jit_trie_fractal && ./src/stax/benches/jit_trie_fractal
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

struct NodeMeta {
  uint32_t count;
  uint32_t capacity;
};

class JITTrie {
  uint8_t *memory;
  size_t offset;
  uint8_t *root;

  uint8_t *arena_alloc(size_t size) {
    // align to 16 bytes for better instruction caching
    size_t align = (16 - (offset & 15)) & 15;
    offset += align;
    uint8_t *ptr = memory + offset;
    offset += size;
    return ptr;
  }

  uint8_t *allocate_node(int cap) {
    // Size computation:
    // Metadata: 8 bytes
    // Base logic (if len==0 ret handle, else load byte and loop): 25 bytes
    // Each branch: 8 bytes
    // Footer (ret 0): 3 bytes
    size_t size = 8 + 25 + cap * 8 + 3;
    uint8_t *ptr = arena_alloc(size);

    NodeMeta *meta = (NodeMeta *)ptr;
    meta->capacity = cap;
    meta->count = 0;

    uint8_t *node = ptr + 8;

    // test rsi, rsi
    node[0] = 0x48;
    node[1] = 0x85;
    node[2] = 0xF6;
    // jnz 11 (skip over return logic to movzx)
    node[3] = 0x75;
    node[4] = 0x0B;
    // mov rax, handle
    node[5] = 0x48;
    node[6] = 0xB8;
    uint64_t h0 = 0;
    memcpy(node + 7, &h0, 8);
    // ret
    node[15] = 0xC3;

    // movzx eax, byte ptr [rdi]
    node[16] = 0x0F;
    node[17] = 0xB6;
    node[18] = 0x07;
    // inc rdi
    node[19] = 0x48;
    node[20] = 0xFF;
    node[21] = 0xC7;
    // dec rsi
    node[22] = 0x48;
    node[23] = 0xFF;
    node[24] = 0xCE;

    // footer (return 0)
    node[25] = 0x31;
    node[26] = 0xC0;
    node[27] = 0xC3;

    return node;
  }

  uint8_t *create_chain(const char *key, size_t len, uint64_t handle) {
    if (len == 0) {
      uint8_t *node = allocate_node(0);
      memcpy(node + 7, &handle, 8);
      return node;
    }

    uint8_t *child = create_chain(key + 1, len - 1, handle);

    uint8_t *node = allocate_node(1);
    NodeMeta *meta = (NodeMeta *)(node - 8);
    meta->count = 1;

    uint8_t byte = key[0];
    uint8_t *b_ptr = node + 25;

    // cmp al, byte
    b_ptr[0] = 0x3C;
    b_ptr[1] = byte;
    // je child
    b_ptr[2] = 0x0F;
    b_ptr[3] = 0x84;

    int32_t rel32 = (int32_t)(child - (b_ptr + 8));
    memcpy(b_ptr + 4, &rel32, 4);

    uint8_t *end_ptr = node + 25 + 8;
    end_ptr[0] = 0x31;
    end_ptr[1] = 0xC0;
    end_ptr[2] = 0xC3;

    return node;
  }

  uint8_t *insert_into_node(uint8_t *node, const char *key, size_t len,
                            uint64_t handle) {
    if (len == 0) {
      memcpy(node + 7, &handle, 8);
      return node;
    }

    NodeMeta *meta = (NodeMeta *)(node - 8);
    uint8_t byte = key[0];

    int branch_idx = -1;
    int l = 0, r = meta->count - 1;

    // Binary search for large nodes, linear for small ones!
    if (meta->count > 8) {
      while (l <= r) {
        int m = l + (r - l) / 2;
        uint8_t b = node[25 + m * 8 + 1];
        if (b == byte) {
          branch_idx = m;
          break;
        } else if (b < byte) {
          l = m + 1;
        } else {
          r = m - 1;
        }
      }
    } else {
      for (int i = 0; i < meta->count; ++i) {
        uint8_t b = node[25 + i * 8 + 1];
        if (b == byte) {
          branch_idx = i;
          break;
        } else if (b > byte) {
          break;
        }
        l = i + 1;
      }
    }

    if (branch_idx != -1) {
      uint8_t *branch_ptr = node + 25 + branch_idx * 8;
      int32_t rel32;
      memcpy(&rel32, branch_ptr + 4, 4);
      uint8_t *child = branch_ptr + 8 + rel32;

      uint8_t *new_child = insert_into_node(child, key + 1, len - 1, handle);
      if (new_child != child) {
        int32_t new_rel32 = (int32_t)(new_child - (branch_ptr + 8));
        memcpy(branch_ptr + 4, &new_rel32, 4);
      }
      return node;
    }

    if (meta->count == meta->capacity) {
      // Faster capacity doubling
      int new_cap = meta->capacity == 0 ? 4 : meta->capacity * 2;

      uint8_t *new_node = allocate_node(new_cap);
      NodeMeta *new_meta = (NodeMeta *)(new_node - 8);
      new_meta->count = meta->count;

      memcpy(new_node, node, 25);

      for (int i = 0; i < meta->count; ++i) {
        uint8_t *old_b = node + 25 + i * 8;
        uint8_t *new_b = new_node + 25 + i * 8;
        memcpy(new_b, old_b, 8);

        int32_t old_rel32;
        memcpy(&old_rel32, old_b + 4, 4);

        uint8_t *target = old_b + 8 + old_rel32;
        int32_t new_rel32 = (int32_t)(target - (new_b + 8));
        memcpy(new_b + 4, &new_rel32, 4);
      }

      node = new_node;
      meta = new_meta;
    }

    int insert_idx = l;

    for (int i = meta->count - 1; i >= insert_idx; --i) {
      uint8_t *src = node + 25 + i * 8;
      uint8_t *dst = node + 25 + (i + 1) * 8;
      memcpy(dst, src, 8);

      int32_t rel32;
      memcpy(&rel32, dst + 4, 4);
      rel32 -= 8;
      memcpy(dst + 4, &rel32, 4);
    }

    uint8_t *child = create_chain(key + 1, len - 1, handle);

    uint8_t *b_ptr = node + 25 + insert_idx * 8;
    b_ptr[0] = 0x3C;
    b_ptr[1] = byte;
    b_ptr[2] = 0x0F;
    b_ptr[3] = 0x84;

    int32_t rel32 = (int32_t)(child - (b_ptr + 8));
    memcpy(b_ptr + 4, &rel32, 4);

    meta->count++;

    uint8_t *end_ptr = node + 25 + meta->count * 8;
    end_ptr[0] = 0x31;
    end_ptr[1] = 0xC0;
    end_ptr[2] = 0xC3;

    return node;
  }

public:
  JITTrie() {
    // Allocate 1GB for the arena. No more malloc calls!
    memory = (uint8_t *)mmap(nullptr, 1024 * 1024 * 1024,
                             PROT_READ | PROT_WRITE | PROT_EXEC,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    offset = 0;
    root = allocate_node(16);
  }

  ~JITTrie() { munmap(memory, 1024 * 1024 * 1024); }

  void insert(const std::string &key, uint64_t handle) {
    root = insert_into_node(root, key.c_str(), key.size(), handle);
  }

  __attribute__((noinline)) uint64_t lookup(const char *key, size_t len) {
    using Fn = uint64_t (*)(const char *, size_t);
    Fn fn = (Fn)root;
    return fn(key, len);
  }

  class Cursor {
    struct Frame {
      uint8_t *node;
      int branch_idx;
    };

    JITTrie *trie;
    Frame stack[1024];
    int depth;
    char key_buf[1024];
    int key_len;
    bool valid;
    uint64_t current_handle;

    void advance() {
      while (depth > 0) {
        Frame &frame = stack[depth - 1];
        NodeMeta *meta = (NodeMeta *)(frame.node - 8);

        if (frame.branch_idx < meta->count) {
          uint8_t byte = frame.node[25 + frame.branch_idx * 8 + 1];
          int32_t rel32;
          memcpy(&rel32, frame.node + 25 + frame.branch_idx * 8 + 4, 4);
          uint8_t *child = frame.node + 25 + frame.branch_idx * 8 + 8 + rel32;

          key_buf[depth - 1] = byte;
          frame.branch_idx++;

          stack[depth].node = child;
          stack[depth].branch_idx = 0;
          depth++;

          uint64_t handle = 0;
          memcpy(&handle, child + 7, 8);
          if (handle != 0) {
            key_len = depth - 1;
            current_handle = handle;
            return;
          }
        } else {
          depth--;
        }
      }
      valid = false;
    }

  public:
    Cursor(JITTrie *t) : trie(t), depth(0), key_len(0), valid(false) {
      uint64_t handle = 0;
      memcpy(&handle, trie->root + 7, 8);

      stack[0].node = trie->root;
      stack[0].branch_idx = 0;
      depth = 1;

      if (handle != 0) {
        key_len = 0;
        current_handle = handle;
        valid = true;
      } else {
        valid = true;
        advance();
      }
    }

    bool is_valid() const { return valid; }
    std::string key() const { return std::string(key_buf, key_len); }
    uint64_t handle() const { return current_handle; }
    void next() { advance(); }
  };

  Cursor cursor() { return Cursor(this); }
  size_t mem_used() const { return offset; }
};

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
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

int main() {
  printf(
      "\n╔═══════════════════════════════════════════════════════════════╗\n");
  printf("║ ★ JITTrie — Zero-Alloc, JIT-Compiled Trie for x86_64        ║\n");
  printf("║ Dynamic insert(), Ordered cursor(), Memory-compact          ║\n");
  printf(
      "╚═══════════════════════════════════════════════════════════════╝\n\n");

  size_t N = 100000;
  auto seq = make_seq(N);

  // We want to test sequential inserts as well.
  JITTrie trie;

  auto t0 = std::chrono::high_resolution_clock::now();
  for (auto &e : seq) {
    trie.insert(e.first, e.second);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ins_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("Trie Insert (Seq): %.2fms (%.0fns/entry)\n", ins_ms,
         ins_ms * 1e6 / N);

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);

  auto t2_um = std::chrono::high_resolution_clock::now();
  for (auto &e : seq) {
    umap[e.first] = e.second;
  }
  auto t3_um = std::chrono::high_resolution_clock::now();
  double um_ins_ms =
      std::chrono::duration<double, std::milli>(t3_um - t2_um).count();
  printf("UMap Insert (Seq): %.2fms (%.0fns/entry)\n", um_ins_ms,
         um_ins_ms * 1e6 / N);

  // Verify
  bool ok = true;
  for (size_t i = 0; i < std::min(N, size_t(10000)); i++) {
    auto &e = seq[i];
    if (trie.lookup(e.first.c_str(), e.first.size()) != e.second) {
      printf("Mismatch: %s\n", e.first.c_str());
      ok = false;
      break;
    }
  }
  if (ok)
    printf("Verification passed.\n\n");

  // Test Cursor Ordering
  JITTrie trie_order;
  trie_order.insert("banana", 1);
  trie_order.insert("apple", 2);
  trie_order.insert("cherry", 3);

  auto cursor = trie_order.cursor();
  std::cout << "Testing ordered iteration:\n";
  while (cursor.is_valid()) {
    std::cout << "  " << cursor.key() << ": " << cursor.handle() << "\n";
    cursor.next();
  }
  std::cout << "\n";

  std::vector<const char *> ks;
  std::vector<size_t> ls;
  for (auto &e : seq) {
    ks.push_back(e.first.c_str());
    ls.push_back(e.first.size());
  }

  size_t bs = std::min(N, size_t(10000));

  // Benchmarks
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 500; ++k) {
    for (size_t i = 0; i < bs; ++i) {
      DoNotOptimize(umap.find(std::string(ks[i], ls[i]))->second);
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double umap_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  auto t4 = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < 500; ++k) {
    for (size_t i = 0; i < bs; ++i) {
      DoNotOptimize(trie.lookup(ks[i], ls[i]));
    }
  }
  auto t5 = std::chrono::high_resolution_clock::now();
  double trie_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

  printf(" ╔═══════════════════════╦════════════════╗\n");
  printf(" ║ Approach              ║ Lookup speed   ║\n");
  printf(" ╠═══════════════════════╬════════════════╣\n");
  printf(" ║ std::unordered_map    ║ %8.2fns/op ║\n",
         umap_ms * 1e6 / (500 * bs));
  printf(" ║ JITTrie               ║ %8.2fns/op ║\n",
         trie_ms * 1e6 / (500 * bs));
  printf(" ╚═══════════════════════╩════════════════╝\n\n");

  printf("Memory Used: %zu bytes (%.2f bytes/entry)\n", trie.mem_used(),
         (double)trie.mem_used() / N);

  double speedup = umap_ms / trie_ms;
  printf("Speedup vs unordered_map: %.2fx\n", speedup);

  return 0;
}
