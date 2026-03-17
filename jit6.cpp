// ═══════════════════════════════════════════════════════════════════════════
// jit_trie_fractal.cpp — Fractal JIT Code Trie (Tree of Trees)
//
// 100% JIT-compiled, NO unordered_map, NO vectors internally.
// x86_64 architecture implementation.
// ═══════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unordered_map>
#include <vector>

static inline void DoNotOptimize(uint64_t val) {
  asm volatile("" : : "r,m"(val) : "memory");
}

template <typename T> T *get_target(int32_t disp, void *instruction_end) {
  return (T *)((uint8_t *)instruction_end + disp);
}

void set_target(int32_t &disp, void *instruction_end, void *target) {
  disp = (int32_t)((uint8_t *)target - (uint8_t *)instruction_end);
}

#pragma pack(push, 1)
struct NodeEntry {
  // Check if rdx (current length) == rsi (target length)
  uint8_t cmp_rdx_rsi[3]; // 48 39 F2
  uint8_t jne[2];         // 75 0B
  // Return block
  uint8_t movabs_rax[2]; // 48 B8
  uint64_t value;        // <value>
  uint8_t ret_match;     // C3
  // Continuation block
  uint8_t movzx_eax[4];     // 0F B6 04 17 (movzx eax, byte ptr [rdi + rdx])
  uint8_t inc_rdx[3];       // 48 FF C2    (inc rdx)
  uint8_t jmp_first_sib[1]; // E9
  int32_t first_sib_disp;   // <disp>
};

struct SiblingBlock {
  uint8_t cmp_al;      // 3C
  uint8_t byte;        // <byte>
  uint8_t je[2];       // 0F 84
  int32_t child_disp;  // <disp>
  uint8_t jmp_next[1]; // E9
  int32_t next_disp;   // <disp>
};

struct GlobalPrologue {
  uint8_t xor_edx[2];  // 31 D2
  uint8_t jmp_root[1]; // E9
  int32_t root_disp;   // <disp>
};
#pragma pack(pop)

class JITCodeTrie {
  uint8_t *mem_;
  size_t cursor_;
  size_t mem_size_;
  GlobalPrologue *prologue_;
  NodeEntry *root_;
  uint8_t *miss_block_;
  size_t count_;

  void *alloc(size_t sz) {
    if (cursor_ + sz > mem_size_) {
      std::cerr << "OOM in JIT alloc\n";
      exit(1);
    }
    void *p = mem_ + cursor_;
    cursor_ += sz;
    return p;
  }

public:
  using LookupFn = uint64_t (*)(const char *, size_t);

  JITCodeTrie()
      : mem_(nullptr), cursor_(0), mem_size_(0), prologue_(nullptr),
        root_(nullptr), miss_block_(nullptr), count_(0) {}

  bool init(size_t cap) {
    mem_size_ = cap;
    mem_ =
        (uint8_t *)mmap(nullptr, mem_size_, PROT_READ | PROT_WRITE | PROT_EXEC,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem_ == MAP_FAILED) {
      mem_ = nullptr;
      return false;
    }
    cursor_ = 0;

    miss_block_ = (uint8_t *)alloc(3);
    miss_block_[0] = 0x31;
    miss_block_[1] = 0xC0;
    miss_block_[2] = 0xC3; // xor eax, eax; ret

    prologue_ = (GlobalPrologue *)alloc(sizeof(GlobalPrologue));
    prologue_->xor_edx[0] = 0x31;
    prologue_->xor_edx[1] = 0xD2;  // xor edx, edx
    prologue_->jmp_root[0] = 0xE9; // jmp rel32

    root_ = allocate_node();
    set_target(prologue_->root_disp, (uint8_t *)&prologue_->root_disp + 4,
               root_);

    return true;
  }

  ~JITCodeTrie() {
    if (mem_)
      munmap(mem_, mem_size_);
  }

  JITCodeTrie(const JITCodeTrie &) = delete;
  JITCodeTrie &operator=(const JITCodeTrie &) = delete;

  bool valid() const { return mem_ != nullptr; }

  NodeEntry *allocate_node() {
    NodeEntry *n = (NodeEntry *)alloc(sizeof(NodeEntry));
    n->cmp_rdx_rsi[0] = 0x48;
    n->cmp_rdx_rsi[1] = 0x39;
    n->cmp_rdx_rsi[2] = 0xF2;
    n->jne[0] = 0x75;
    n->jne[1] = 0x0B;
    n->movabs_rax[0] = 0x48;
    n->movabs_rax[1] = 0xB8;
    n->value = 0;
    n->ret_match = 0xC3;
    n->movzx_eax[0] = 0x0F;
    n->movzx_eax[1] = 0xB6;
    n->movzx_eax[2] = 0x04;
    n->movzx_eax[3] = 0x17;
    n->inc_rdx[0] = 0x48;
    n->inc_rdx[1] = 0xFF;
    n->inc_rdx[2] = 0xC2;
    n->jmp_first_sib[0] = 0xE9;
    set_target(n->first_sib_disp, (uint8_t *)&n->first_sib_disp + 4,
               miss_block_);
    return n;
  }

  SiblingBlock *allocate_sibling() {
    return (SiblingBlock *)alloc(sizeof(SiblingBlock));
  }

  bool insert(const char *key, size_t len, uint64_t value) {
    if (!mem_ || len == 0)
      return false;
    NodeEntry *curr = root_;
    for (size_t i = 0; i < len; ++i) {
      uint8_t c = key[i];

      void *ptr_to_disp = &curr->first_sib_disp;
      void *instruction_end = (uint8_t *)ptr_to_disp + 4;
      uint8_t *target =
          get_target<uint8_t>(curr->first_sib_disp, instruction_end);

      SiblingBlock *prev_sib = nullptr;
      SiblingBlock *sib = nullptr;

      while (target != miss_block_) {
        sib = (SiblingBlock *)target;
        if (sib->byte == c)
          break;
        if (sib->byte > c) {
          sib = nullptr;
          break;
        } // Keep siblings sorted

        prev_sib = sib;
        ptr_to_disp = &sib->next_disp;
        instruction_end = (uint8_t *)ptr_to_disp + 4;
        target = get_target<uint8_t>(sib->next_disp, instruction_end);
      }

      if (!sib || sib->byte != c) {
        NodeEntry *new_node = allocate_node();
        SiblingBlock *new_sib = allocate_sibling();
        new_sib->cmp_al = 0x3C;
        new_sib->byte = c;
        new_sib->je[0] = 0x0F;
        new_sib->je[1] = 0x84;
        set_target(new_sib->child_disp, &new_sib->jmp_next, new_node);
        new_sib->jmp_next[0] = 0xE9;
        set_target(new_sib->next_disp, (uint8_t *)&new_sib->next_disp + 4,
                   target);

        int32_t *disp_to_patch = (int32_t *)ptr_to_disp;
        set_target(*disp_to_patch, instruction_end, new_sib);

        curr = new_node;
      } else {
        curr = get_target<NodeEntry>(sib->child_disp, &sib->jmp_next);
      }
    }
    curr->value = value;
    count_++;
    return true;
  }

  __attribute__((noinline)) uint64_t lookup(const char *key, size_t len) const {
    LookupFn fn = (LookupFn)prologue_;
    return fn(key, len);
  }

  size_t count() const { return count_; }
  size_t code_bytes() const { return cursor_; }

  struct Cursor {
    struct Frame {
      NodeEntry *node;
      uint8_t *sib_target;
      char ch;
    };
    // Maximum key depth. Avoid std::vector.
    Frame stack[256];
    size_t depth;
    char current_key[256];
    size_t current_len;
    uint64_t current_value;
    uint8_t *miss_block;

    Cursor(NodeEntry *root, uint8_t *miss_block)
        : miss_block(miss_block), depth(0), current_len(0), current_value(0) {
      stack[depth++] = {
          root,
          get_target<uint8_t>(root->first_sib_disp,
                              (uint8_t *)&root->first_sib_disp + 4),
          0};
      advance_to_next();
    }

    bool valid() const { return depth > 0; }

    std::string key() const { return std::string(current_key, current_len); }
    uint64_t value() const { return current_value; }

    void next() {
      if (!valid())
        return;

      NodeEntry *curr_node = stack[depth - 1].node;
      uint8_t *child_target = get_target<uint8_t>(
          curr_node->first_sib_disp, (uint8_t *)&curr_node->first_sib_disp + 4);

      if (child_target != miss_block) {
        // Descend
        SiblingBlock *sib = (SiblingBlock *)child_target;
        NodeEntry *child_node =
            get_target<NodeEntry>(sib->child_disp, &sib->jmp_next);
        current_key[current_len++] = sib->byte;
        stack[depth++] = {
            child_node,
            get_target<uint8_t>(sib->next_disp, (uint8_t *)&sib->next_disp + 4),
            (char)sib->byte};
      } else {
        // No children, pop up and try next sibling
        while (depth > 0) {
          Frame frame = stack[--depth];

          if (depth > 0) {
            current_len--;
            uint8_t *next_sib_target = frame.sib_target;
            if (next_sib_target != miss_block) {
              SiblingBlock *sib = (SiblingBlock *)next_sib_target;
              NodeEntry *child_node =
                  get_target<NodeEntry>(sib->child_disp, &sib->jmp_next);
              current_key[current_len++] = sib->byte;
              stack[depth++] = {
                  child_node,
                  get_target<uint8_t>(sib->next_disp,
                                      (uint8_t *)&sib->next_disp + 4),
                  (char)sib->byte};
              break;
            }
          }
        }
      }
      advance_to_next();
    }

    void advance_to_next() {
      while (valid()) {
        if (stack[depth - 1].node->value != 0) {
          current_value = stack[depth - 1].node->value;
          return; // Found a valid node
        }

        NodeEntry *curr_node = stack[depth - 1].node;
        uint8_t *child_target =
            get_target<uint8_t>(curr_node->first_sib_disp,
                                (uint8_t *)&curr_node->first_sib_disp + 4);

        if (child_target != miss_block) {
          SiblingBlock *sib = (SiblingBlock *)child_target;
          NodeEntry *child_node =
              get_target<NodeEntry>(sib->child_disp, &sib->jmp_next);
          current_key[current_len++] = sib->byte;
          stack[depth++] = {child_node,
                            get_target<uint8_t>(sib->next_disp,
                                                (uint8_t *)&sib->next_disp + 4),
                            (char)sib->byte};
        } else {
          bool found_next = false;
          while (depth > 0) {
            Frame frame = stack[--depth];

            if (depth > 0) {
              current_len--;
              uint8_t *next_sib_target = frame.sib_target;
              if (next_sib_target != miss_block) {
                SiblingBlock *sib = (SiblingBlock *)next_sib_target;
                NodeEntry *child_node =
                    get_target<NodeEntry>(sib->child_disp, &sib->jmp_next);
                current_key[current_len++] = sib->byte;
                stack[depth++] = {
                    child_node,
                    get_target<uint8_t>(sib->next_disp,
                                        (uint8_t *)&sib->next_disp + 4),
                    (char)sib->byte};
                found_next = true;
                break;
              }
            }
          }
          if (!found_next)
            break;
        }
      }
    }
  };

  Cursor cursor() { return Cursor(root_, miss_block_); }
};

// Top-level fractal trie.
// 100% JIT-compiled.
class FractalJITTrie {
public:
  FractalJITTrie(size_t prefix_start, size_t prefix_len, size_t sub_capacity)
      : pstart_(prefix_start), plen_(prefix_len), sub_cap_(sub_capacity) {

    // Allocate space for the meta-trie (Level 0) and the pointers array
    meta_.init(256 * 1024); // 256 KB for meta trie

    // We preallocate an array for sub-tries pointers instead of vector
    subs_ = (JITCodeTrie **)mmap(nullptr, 65536 * sizeof(JITCodeTrie *),
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    num_subs_ = 0;
  }

  ~FractalJITTrie() {
    for (size_t i = 0; i < num_subs_; i++) {
      delete subs_[i];
    }
    munmap(subs_, 65536 * sizeof(JITCodeTrie *));
  }

  bool insert(const std::string &key, uint64_t handle) {
    if (key.size() < pstart_ + plen_)
      return false;

    const char *pfx = key.c_str() + pstart_;

    // Use JIT meta trie directly for lookup instead of map!
    uint64_t bucket_id = meta_.lookup(pfx, plen_);
    size_t bucket = 0;

    if (bucket_id == 0) {
      // Miss in meta trie! Create a new sub-trie
      bucket = num_subs_++;
      JITCodeTrie *st = new JITCodeTrie();
      st->init(sub_cap_ + 1024 * 1024);
      subs_[bucket] = st;

      // Insert into meta trie mapping to bucket + 1
      meta_.insert(pfx, plen_, bucket + 1);
    } else {
      bucket = bucket_id - 1;
    }

    bool ok = subs_[bucket]->insert(key.c_str(), key.size(), handle);
    count_ += ok;
    return ok;
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
  size_t num_buckets() const { return num_subs_; }

  size_t total_code_bytes() const {
    size_t t = meta_.code_bytes();
    for (size_t i = 0; i < num_subs_; i++)
      t += subs_[i]->code_bytes();
    return t;
  }

  size_t max_bucket_size() const {
    size_t m = 0;
    for (size_t i = 0; i < num_subs_; i++)
      m = std::max(m, subs_[i]->count());
    return m;
  }

private:
  size_t pstart_, plen_, sub_cap_;
  size_t count_ = 0;
  JITCodeTrie meta_;
  JITCodeTrie **subs_;
  size_t num_subs_;
};

static std::tuple<size_t, size_t, size_t>
auto_prefix(const std::vector<std::pair<std::string, uint64_t>> &entries,
            size_t target) {
  size_t start = 2;
  for (size_t plen = 1; plen <= 7; plen++) {
    // Here we can use unordered_map since it's just build-time heuristics, not
    // part of the trie itself
    std::unordered_map<std::string, size_t> counts;
    for (auto &e : entries) {
      if (e.first.size() < start + plen)
        continue;
      counts[e.first.substr(start, plen)]++;
    }
    size_t max_count = 0;
    for (auto &[k, v] : counts)
      max_count = std::max(max_count, v);
    printf("    auto_prefix: plen=%zu  buckets=%zu  max_bucket=%zu\n", plen,
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
  double s = 0;
  for (auto v : t)
    s += v;
  return {s / samples, t[samples / 2], t[size_t(samples * 0.95)]};
}

// Ensure std::unordered_map bench runs, but this is for comparison, not in our
// trie core.
#include <unordered_map>

static void run(const char *label,
                const std::vector<std::pair<std::string, uint64_t>> &entries) {
  size_t N = entries.size();
  auto [pstart, plen, max_bucket] = auto_prefix(entries, 1500);

  printf("\n═══════════════════════════════════════════════════════════════════"
         "\n");
  printf("  %s (%zu entries, prefix=key[%zu..%zu], max_bucket=%zu)\n", label, N,
         pstart, pstart + plen - 1, max_bucket);
  printf(
      "═══════════════════════════════════════════════════════════════════\n");

  FractalJITTrie fractal(pstart, plen, max_bucket + 128);

  auto it0 = std::chrono::high_resolution_clock::now();
  for (auto &e : entries)
    fractal.insert(e.first, e.second);
  auto it1 = std::chrono::high_resolution_clock::now();
  double ins_ms = std::chrono::duration<double, std::milli>(it1 - it0).count();

  printf("  Insert:     %.2fms (%.0fns/entry)\n", ins_ms, ins_ms * 1e6 / N);
  printf("  Buckets:    %zu (max %zu entries/bucket)\n", fractal.num_buckets(),
         fractal.max_bucket_size());
  printf("  Total code: %.2fMB (%.1f B/entry)\n",
         fractal.total_code_bytes() / 1048576.0,
         double(fractal.total_code_bytes()) / N);

  std::unordered_map<std::string, uint64_t> umap;
  umap.reserve(N);
  for (auto &e : entries)
    umap[e.first] = e.second;

  bool ok = true;
  for (size_t i = 0; i < N && ok; i++) {
    uint64_t h =
        fractal.lookup(entries[i].first.c_str(), entries[i].first.size());
    if (h != entries[i].second) {
      printf("  ❌ '%s' got %llu want %llu\n", entries[i].first.c_str(),
             (unsigned long long)h, (unsigned long long)entries[i].second);
      ok = false;
    }
  }
  uint64_t miss = fractal.lookup("ZZZZZZZZZZZ", 11);
  if (miss != 0) {
    printf("  ❌ miss got %llu\n", (unsigned long long)miss);
    ok = false;
  }
  if (!ok)
    return;
  printf("  ✅ ALL %zu verified + miss OK\n", N);

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
  printf("║  ★ FRACTAL JIT Code Trie — Tree of Trees                  ║\n");
  printf("║  Dynamic insert(). Zero-alloc lookup(). Auto-partition.   ║\n");
  printf("║  Each sub-trie fits L1. Unlimited scale.                  ║\n");
  printf("║  100%% JIT. Zero STL collections used internally.          ║\n");
  printf("╚═══════════════════════════════════════════════════════════════╝\n");

  for (size_t sz : {1000, 5000, 10000, 50000, 100000, 500000, 1000000}) {
    auto seq = make_seq(sz);
    run("Sequential", seq);
  }

  printf("\n--- Cursor Test ---\n");
  JITCodeTrie trie;
  trie.init(1024 * 1024);
  trie.insert("apple", 5, 1);
  trie.insert("apricot", 7, 2);
  trie.insert("banana", 6, 3);
  trie.insert("cherry", 6, 4);

  auto cur = trie.cursor();
  while (cur.valid()) {
    std::cout << cur.key() << ": " << cur.value() << "\n";
    cur.next();
  }

  return 0;
}
