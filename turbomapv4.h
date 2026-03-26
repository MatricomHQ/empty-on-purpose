// Copyright 2025 TurboMap Authors. All rights reserved.
//
// TurboMapV4 — Patricia-compressed fractal extendible hash trie.
//
// A high-performance ordered key-value map supporting:
//   - O(1) hash-directed insert/get/erase
//   - Full lexicographic cursor iteration (forward + reverse)
//   - Prefix scans and seek_ge (range queries)
//   - Arbitrary-length binary keys (up to 4GB)
//   - Patricia path compression (zero wasted directory levels)
//   - SIMD-accelerated tag matching (NEON / SSE4.2)
//
// Usage:
//   turbo::TurboMapV4 map;
//   map.insert(key, key_len, value);
//   uint32_t val;
//   if (map.get(key, key_len, val)) { ... }
//
//   auto cur = map.cursor();
//   for (cur.seek_first(); cur.valid(); cur.next()) {
//     const uint8_t* k = cur.key();
//     uint32_t v = cur.val();
//   }

#ifndef TURBOMAP_V4_H_
#define TURBOMAP_V4_H_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_acle.h>
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace turbo {
namespace internal {

// ─────────────────────────────────────────────────────────────────
//  SIMD tag scanning — matches a single byte across 16/32 slots.
// ─────────────────────────────────────────────────────────────────
#if defined(__ARM_NEON)
inline uint16_t neon_movemask(uint8x16_t cmp) {
  const uint8x16_t bits = {1, 2, 4, 8, 16, 32, 64, 128,
                           1, 2, 4, 8, 16, 32, 64, 128};
  uint8x16_t masked = vandq_u8(cmp, bits);
  uint8x8_t lo = vget_low_u8(masked);
  uint8x8_t hi = vget_high_u8(masked);
  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);
  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);
  return (uint16_t)vget_lane_u8(lo, 0) |
         ((uint16_t)vget_lane_u8(hi, 0) << 8);
}

inline uint64_t scan64(const uint8_t* tags, uint8_t needle) {
  uint8x16_t n = vdupq_n_u8(needle);
  return (uint64_t)neon_movemask(vceqq_u8(vld1q_u8(tags), n)) |
         ((uint64_t)neon_movemask(vceqq_u8(vld1q_u8(tags + 16), n)) << 16) |
         ((uint64_t)neon_movemask(vceqq_u8(vld1q_u8(tags + 32), n)) << 32) |
         ((uint64_t)neon_movemask(vceqq_u8(vld1q_u8(tags + 48), n)) << 48);
}
#else
inline uint64_t scan64(const uint8_t* tags, uint8_t needle) {
  __m256i n = _mm256_set1_epi8(needle);
  uint32_t lo = _mm256_movemask_epi8(
      _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)tags), n));
  uint32_t hi = _mm256_movemask_epi8(
      _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i*)(tags+32)), n));
  return (uint64_t)lo | ((uint64_t)hi << 32);
}
#endif

// ─────────────────────────────────────────────────────────────────
//  Arena — Unified bump allocator backed by a single mmap region.
//  All groups, directories, records, and child nodes share one
//  contiguous address space. Records are indexed by 8-byte-aligned
//  slot indices (byte_offset >> 3), supporting up to 32 GB.
// ─────────────────────────────────────────────────────────────────
class Arena {
 public:
  /// Constructs an arena with the given virtual address capacity.
  /// Pages are demand-faulted; only touched pages consume RSS.
  explicit Arena(uint64_t capacity = 4ULL << 30)
      : pos_(0), cap_(capacity) {
    base_ = static_cast<uint8_t*>(
        mmap(nullptr, cap_, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (base_ == MAP_FAILED) {
      fprintf(stderr, "turbo::Arena: mmap failed\n");
      abort();
    }
  }

  ~Arena() { munmap(base_, cap_); }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  /// Resets the arena position to zero (does not unmap).
  void reset() { pos_ = 0; }

  /// Allocates `sz` bytes, 8-byte aligned. Returns raw pointer.
  void* alloc(uint64_t sz) {
    sz = (sz + 7) & ~7ULL;
    void* p = base_ + pos_;
    pos_ += sz;
    return p;
  }

  /// Allocates `sz` bytes, 16-byte aligned (for SIMD structs).
  void* alloc16(uint64_t sz) {
    pos_ = (pos_ + 15) & ~15ULL;
    return alloc(sz);
  }

  /// Allocates a key-value record:
  ///   Layout: [uint32_t key_len][key bytes][uint32_t val]
  /// Returns a slot index (byte_offset >> 3).
  uint32_t alloc_rec(const uint8_t* key, uint32_t key_len, uint32_t val) {
    uint32_t raw = 4 + key_len + 4;
    uint32_t padded = (raw + 7) & ~7;
    uint64_t off = pos_;
    memcpy(base_ + off, &key_len, 4);
    if (key_len) memcpy(base_ + off + 4, key, key_len);
    memcpy(base_ + off + 4 + key_len, &val, 4);
    pos_ += padded;
    return static_cast<uint32_t>(off >> 3);
  }

  /// Resolves a slot index to a raw pointer.
  const uint8_t* rec(uint32_t slot) const {
    return base_ + (static_cast<uint64_t>(slot) << 3);
  }

  /// Returns the stored key length for a given slot.
  uint32_t rec_kl(uint32_t slot) const {
    uint32_t v;
    memcpy(&v, rec(slot), 4);
    return v;
  }

  /// Returns a pointer to the key data for a given slot.
  const uint8_t* rec_key(uint32_t slot) const { return rec(slot) + 4; }

  /// Returns the stored value for a given slot.
  uint32_t rec_val(uint32_t slot) const {
    uint32_t kl = rec_kl(slot);
    uint32_t v;
    memcpy(&v, rec(slot) + 4 + kl, 4);
    return v;
  }

  /// Overwrites the stored value for a given slot.
  void rec_set_val(uint32_t slot, uint32_t v) {
    uint32_t kl = rec_kl(slot);
    memcpy(const_cast<uint8_t*>(rec(slot)) + 4 + kl, &v, 4);
  }

  /// Returns the total bytes consumed so far.
  uint64_t used() const { return pos_; }

 private:
  uint8_t* base_;
  uint64_t pos_;
  uint64_t cap_;
};

// ─────────────────────────────────────────────────────────────────
//  Node — A single level of the Patricia-compressed fractal trie.
//
//  Each node is a complete extendible hash table: a directory of
//  2^gd_ slots pointing to Groups (leaf buckets) or child Nodes
//  (fractal recursion). Groups hold up to CAP entries with SIMD
//  tag scanning. When a group overflows, it either splits within
//  the directory or spawns a child node at a deeper disc_off_.
//
//  Patricia compression: child nodes store a `ref_off_` pointing
//  to a representative key in the arena. The gap between parent
//  and child disc_off_ is verified via memcmp — no intermediate
//  nodes are materialized for shared prefixes.
// ─────────────────────────────────────────────────────────────────
class Node {
 public:
  static constexpr uint32_t kCap = 64;

  /// Group — A leaf bucket holding up to kCap key-value entries.
  struct alignas(16) Group {
    uint8_t tags[64];
    uint64_t entries[64];
    uint32_t first_idx;
    uint32_t last_idx;
    uint8_t count;
    uint8_t local_depth;
    uint8_t sorted;
    uint8_t pad_[5];
  };

  // ── Entry encoding: kf in high 32 bits, arena slot in low 32 ──
  static uint32_t entry_kf(uint64_t e) { return static_cast<uint32_t>(e >> 32); }
  static uint32_t entry_off(uint64_t e) { return static_cast<uint32_t>(e); }
  static uint64_t make_entry(uint32_t kf, uint32_t off) {
    return (static_cast<uint64_t>(kf) << 32) | off;
  }

  // ── Slot tagging: LSB=1 marks a child pointer ──
  static bool is_child(uintptr_t v) { return v & 1; }
  static Node* get_child(uintptr_t v) {
    return reinterpret_cast<Node*>(v & ~1ULL);
  }
  static Group* get_group(uintptr_t v) {
    return reinterpret_cast<Group*>(v);
  }
  static uintptr_t tag_child(Node* c) {
    return reinterpret_cast<uintptr_t>(c) | 1;
  }

  // ── Key fragment + tag extraction ──
  //
  // kf:  First 4 bytes at disc_off_ (big-endian) — level-dependent
  //      directory routing. Changes at each fractal level.
  //
  // tag: CRC of the key's LAST min(16, kl) bytes — level-INDEPENDENT.
  //      Computed once at insert, valid at every fractal depth.
  //      At most 2 CRC64 instructions regardless of key length.
  //      marker bit set.
  //
  // Why tail bytes? The suffix is where keys diverge most (especially
  // structured keys with shared prefixes). And since the tag doesn't
  // depend on disc_off_, it's automatically correct at all child levels
  // — no recomputation during get() descent or spawn_child redistribution.

  /// Compute level-independent tag from key's tail bytes.
  static uint8_t compute_tag(const uint8_t* k, uint32_t kl) {
    if (kl == 0) return 0x80;
    // CRC the last min(16, kl) bytes —.(2 CRC64 ops.(
    uint32_t tail_len = kl > 16 ? 16 : kl;
    const uint8_t* tail = k + kl - tail_len;
#if defined(__ARM_NEON)
    uint32_t crc = 0;
    uint32_t i = 0;
    for (; i + 8 <= tail_len; i += 8) {
      uint64_t v;
      memcpy(&v, tail + i, 8);
      crc = __crc32cd(crc, v);
    }
    for (; i < tail_len; i++) crc = __crc32cb(crc, tail[i]);
#else
    uint32_t crc = 0;
    uint32_t i = 0;
    for (; i + 8 <= tail_len; i += 8) {
      uint64_t v;
      memcpy(&v, tail + i, 8);
      crc = _mm_crc32_u64(crc, v);
    }
    for (; i < tail_len; i++) crc = _mm_crc32_u8(crc, tail[i]);
#endif
    return static_cast<uint8_t>((crc >> 25) | 0x80);
  }

  /// Extract kf (4-byte directory routing key) at disc_off_.
  static void extract_kf(const uint8_t* k, uint32_t kl,
                         uint32_t off, uint32_t& kf) {
    if (kl <= off) { kf = 0; return; }
    const uint8_t* p = k + off;
    uint32_t rem = kl - off;
    kf = 0;
    uint32_t n = rem > 4 ? 4 : rem;
    memcpy(&kf, p, n);
    kf = __builtin_bswap32(kf);
  }

  /// Combined extraction: kf from disc_off_, tag from key tail.
  static void extract(const uint8_t* k, uint32_t kl,
                      uint32_t& kf, uint8_t& tag) {
    kf = 0;
    if (kl > 0) {
      uint32_t n = kl > 4 ? 4 : kl;
      memcpy(&kf, k, n);
    }
    kf = __builtin_bswap32(kf);
    tag = compute_tag(k, kl);
  }

  void extract_at(const uint8_t* k, uint32_t kl,
                  uint32_t& kf, uint8_t& tag) const {
    if (kl <= disc_off_) { kf = 0; tag = compute_tag(k, kl); return; }
    extract_kf(k, kl, disc_off_, kf);
    tag = compute_tag(k, kl);
  }

  /// Branchless directory index from key fragment.
  uint32_t di(uint32_t kf) const {
    return (kf >> (shift_ & 31)) & (ds_sz_ - 1);
  }

  /// Full-key equality check against an arena record.
  bool key_eq(uint32_t slot, const uint8_t* k, uint32_t kl) const {
    uint32_t skl = arena_.rec_kl(slot);
    if (skl != kl) return false;
    return memcmp(arena_.rec_key(slot), k, kl) == 0;
  }

  Group* alloc_group(uint8_t ld, uint32_t fi = 0) {
    auto* g = static_cast<Group*>(arena_.alloc16(sizeof(Group)));
    memset(g->tags, 0, 64);
    g->count = 0;
    g->local_depth = ld;
    g->sorted = 1;
    g->first_idx = fi;
    g->last_idx = fi;
    return g;
  }

  void double_dir() {
    uint32_t ns = ds_sz_ * 2;
    auto* nd = static_cast<uintptr_t*>(arena_.alloc(ns * sizeof(uintptr_t)));
    for (uint32_t i = 0; i < ds_sz_; i++) {
      nd[2 * i] = dir_[i];
      nd[2 * i + 1] = dir_[i];
    }
    for (uint32_t i = 0; i < ds_sz_; i++) {
      if (i == 0 || dir_[i] != dir_[i - 1]) {
        uintptr_t slot = dir_[i];
        if (is_child(slot)) {
          Node* child = get_child(slot);
          child->first_idx_ = child->first_idx_ * 2;
          child->last_idx_ = child->last_idx_ * 2 + 1;
        } else {
          Group* g = get_group(slot);
          g->first_idx = g->first_idx * 2;
          g->last_idx = g->last_idx * 2 + 1;
        }
      }
    }
    dir_ = nd;
    ds_sz_ = ns;
    gd_++;
    shift_--;
  }

  uint32_t compute_lcp(Group* g, uint32_t& min_kl_out) const {
    uint32_t min_kl = UINT32_MAX;
    for (uint32_t i = 0; i < g->count; i++) {
      uint32_t skl = arena_.rec_kl(entry_off(g->entries[i]));
      if (skl < min_kl) min_kl = skl;
    }
    min_kl_out = min_kl;
    if (g->count <= 1 || disc_off_ >= min_kl) return disc_off_;
    const uint8_t* k0 = arena_.rec_key(entry_off(g->entries[0]));
    uint32_t common = min_kl;
    for (uint32_t i = 1; i < g->count; i++) {
      const uint8_t* ki = arena_.rec_key(entry_off(g->entries[i]));
      uint32_t j = disc_off_;
      while (j < common && k0[j] == ki[j]) j++;
      if (j < common) common = j;
      if (common <= disc_off_) break;
    }
    return common;
  }

  bool spawn_child(uint32_t dx) {
    Group* old = get_group(dir_[dx]);
    uint8_t ld = old->local_depth;
    uint32_t min_kl = 0;
    uint32_t child_off = compute_lcp(old, min_kl);
    if (child_off <= disc_off_ || child_off >= min_kl) return false;

    auto* child = static_cast<Node*>(arena_.alloc(sizeof(Node)));
    new (child) Node(arena_, count_, child_off, child_max_gd_, child_max_gd_);

    count_->fetch_sub(old->count, std::memory_order_relaxed);
    for (uint32_t i = 0; i < old->count; i++) {
      uint32_t off = entry_off(old->entries[i]);
      child->insert_internal(arena_.rec_key(off), arena_.rec_kl(off),
                             arena_.rec_val(off), off);
    }

    child->ref_off_ = entry_off(old->entries[0]);
    uint32_t stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(stride - 1);
    child->first_idx_ = st;
    child->last_idx_ = st + stride - 1;
    for (uint32_t i = st; i < st + stride; i++)
      dir_[i] = tag_child(child);
    return true;
  }

  bool split_or_spawn(uint32_t dx) {
    Group* old = get_group(dir_[dx]);
    uint8_t ld = old->local_depth;
    uint32_t kf0 = entry_kf(old->entries[0]);
    uint32_t diff = 0;
    for (uint32_t i = 1; i < old->count; i++)
      diff |= kf0 ^ entry_kf(old->entries[i]);
    if (ld > 0) diff &= ((1u << (32 - ld)) - 1);
    if (diff == 0) return spawn_child(dx);

    uint8_t sld = ld;
    if (sld >= max_gd_) return spawn_child(dx);
    while (sld >= gd_) { double_dir(); dx <<= 1; }

    Group* g0 = alloc_group(sld + 1);
    Group* g1 = alloc_group(sld + 1);
    for (uint32_t i = 0; i < old->count; i++) {
      uint8_t side = (entry_kf(old->entries[i]) >> (31 - sld)) & 1;
      Group* dst = side ? g1 : g0;
      uint32_t p = dst->count;
      dst->tags[p] = old->tags[i];
      dst->entries[p] = old->entries[i];
      dst->count++;
    }
    uint32_t old_stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(old_stride - 1);
    uint32_t half = old_stride / 2;
    for (uint32_t i = 0; i < half; i++) {
      dir_[st + i] = reinterpret_cast<uintptr_t>(g0);
      dir_[st + half + i] = reinterpret_cast<uintptr_t>(g1);
    }
    g0->first_idx = st;
    g0->last_idx = st + half - 1;
    g1->first_idx = st + half;
    g1->last_idx = st + old_stride - 1;
    g0->sorted = 0;
    g1->sorted = 0;
    return true;
  }

  void sort_group(Group* g) const {
    if (g->sorted || g->count <= 1) { g->sorted = 1; return; }
    uint32_t n = g->count;
    for (uint32_t i = 1; i < n; i++) {
      uint64_t ent = g->entries[i];
      uint8_t tag = g->tags[i];
      uint32_t kf = entry_kf(ent);
      uint32_t off = entry_off(ent);
      int j = static_cast<int>(i) - 1;
      while (j >= 0) {
        uint64_t ej = g->entries[j];
        uint32_t kfj = entry_kf(ej);
        bool larger = false;
        if (kfj > kf) {
          larger = true;
        } else if (kfj == kf) {
          uint32_t oj = entry_off(ej);
          uint32_t l1 = arena_.rec_kl(oj);
          uint32_t l2 = arena_.rec_kl(off);
          const uint8_t* s1 = arena_.rec_key(oj);
          const uint8_t* s2 = arena_.rec_key(off);
          uint32_t ml = l1 < l2 ? l1 : l2;
          int cmp = memcmp(s1, s2, ml);
          if (cmp > 0 || (cmp == 0 && l1 > l2)) larger = true;
        }
        if (!larger) break;
        g->entries[j + 1] = ej;
        g->tags[j + 1] = g->tags[j];
        j--;
      }
      g->entries[j + 1] = ent;
      g->tags[j + 1] = tag;
    }
    g->sorted = 1;
  }

 public:
  /// Constructs a root or child node.
  Node(Arena& arena, std::atomic<uint64_t>* cnt,
       uint32_t disc_off = 0, uint32_t max_gd = 22,
       uint32_t child_max_gd = 16)
      : arena_(arena), count_(cnt), disc_off_(disc_off), max_gd_(max_gd),
        child_max_gd_(child_max_gd),
        ref_off_(0xFFFFFFFF), first_idx_(0), last_idx_(0) {
    gd_ = 0;
    ds_sz_ = 1;
    shift_ = 32;
    dir_ = static_cast<uintptr_t*>(arena_.alloc(sizeof(uintptr_t)));
    dir_[0] = reinterpret_cast<uintptr_t>(alloc_group(0));
  }

  /// Inserts a key-value pair. Returns true if new, false if updated.
  bool insert_internal(const uint8_t* k, uint32_t kl, uint32_t v,
                       uint32_t existing_off) {
    Node* node = this;
    uint32_t kf;
    // Tag computed ONCE — level-independent (CRC of key tail).
    const uint8_t tag = compute_tag(k, kl);
    extract_kf(k, kl, node->disc_off_, kf);

    for (int attempt = 0; attempt < 256; attempt++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];

      if (is_child(slot)) {
        Node* child = get_child(slot);
        // Patricia gap verification.
        uint32_t gap_start = node->disc_off_;
        uint32_t gap_end = child->disc_off_;
        if (gap_end > gap_start) {
          const uint8_t* ref = node->arena_.rec_key(child->ref_off_);
          uint32_t ref_kl = node->arena_.rec_kl(child->ref_off_);
          uint32_t check = gap_end;
          if (check > kl) check = kl;
          if (check > ref_kl) check = ref_kl;
          uint32_t div = gap_start;
          while (div < check && k[div] == ref[div]) div++;
          if (div < gap_end) {
            // Gap mismatch — create intermediate branch node.
            auto* branch = static_cast<Node*>(
                node->arena_.alloc(sizeof(Node)));
            new (branch) Node(node->arena_, node->count_, div, node->child_max_gd_, node->child_max_gd_);
            branch->ref_off_ = child->ref_off_;

            uint32_t ref_kf, new_kf;
            extract_kf(ref, ref_kl, branch->disc_off_, ref_kf);
            extract_kf(k, kl, branch->disc_off_, new_kf);

            uint32_t xor_bits = ref_kf ^ new_kf;
            if (xor_bits != 0) {
              uint8_t sb = __builtin_clz(xor_bits);
              while (branch->gd_ <= sb) branch->double_dir();
            }

            // Buddy-split until child and new key are isolated.
            while (true) {
              uint32_t cdx = branch->di(ref_kf);
              uint32_t ndx = branch->di(new_kf);
              if (branch->dir_[cdx] != branch->dir_[ndx]) break;
              Group* old = get_group(branch->dir_[cdx]);
              uint8_t ld = old->local_depth;
              if (ld >= branch->gd_) { branch->double_dir(); continue; }
              Group* g0 = branch->alloc_group(ld + 1);
              Group* g1 = branch->alloc_group(ld + 1);
              uint32_t os = 1u << (branch->gd_ - ld);
              uint32_t st = cdx & ~(os - 1);
              uint32_t h = os / 2;
              for (uint32_t i = 0; i < h; i++) {
                branch->dir_[st + i] = reinterpret_cast<uintptr_t>(g0);
                branch->dir_[st + h + i] = reinterpret_cast<uintptr_t>(g1);
              }
              g0->first_idx = st;
              g0->last_idx = st + h - 1;
              g1->first_idx = st + h;
              g1->last_idx = st + os - 1;
            }

            // Install old child into its isolated slot range.
            uint32_t fc = branch->di(ref_kf);
            Group* cg = get_group(branch->dir_[fc]);
            child->first_idx_ = cg->first_idx;
            child->last_idx_ = cg->last_idx;
            for (uint32_t i = cg->first_idx; i <= cg->last_idx; i++)
              branch->dir_[i] = tag_child(child);

            // Replace child with branch in parent directory.
            uint32_t ps = dx, pe = dx;
            while (ps > 0 && node->dir_[ps - 1] == slot) ps--;
            while (pe + 1 < node->ds_sz_ && node->dir_[pe + 1] == slot) pe++;
            branch->first_idx_ = ps;
            branch->last_idx_ = pe;
            for (uint32_t i = ps; i <= pe; i++)
              node->dir_[i] = tag_child(branch);
            continue;
          }
        }
        node = child;
        extract_kf(k, kl, node->disc_off_, kf);
        continue;
      }

      Group* g = get_group(slot);
      // SIMD tag scan for duplicates.
      uint64_t mm = scan64(g->tags, tag);
      while (mm) {
        int p = __builtin_ctzll(mm);
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          node->arena_.rec_set_val(entry_off(g->entries[p]), v);
          return false;
        }
        mm &= mm - 1;
      }
      if (g->count >= kCap) {
        if (!node->split_or_spawn(dx)) return false;
        continue;
      }

      uint32_t final_off = (existing_off != 0xFFFFFFFF)
                               ? existing_off
                               : node->arena_.alloc_rec(k, kl, v);
      uint32_t s = g->count;
      g->tags[s] = tag;
      g->entries[s] = make_entry(kf, final_off);
      g->count++;
      g->sorted = 0;
      count_->fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

  bool insert(const uint8_t* k, uint32_t kl, uint32_t v) {
    return insert_internal(k, kl, v, 0xFFFFFFFF);
  }

  bool get(const uint8_t* k, uint32_t kl, uint32_t& out) const {
    const Node* node = this;
    uint32_t kf;
    // Tag computed ONCE — level-independent (CRC of key tail).
    const uint8_t tag = compute_tag(k, kl);
    extract_kf(k, kl, node->disc_off_, kf);

    for (int guard = 0; guard < 64; guard++) {
      uintptr_t slot = node->dir_[node->di(kf)];
      if (is_child(slot)) {
        Node* child = get_child(slot);
        uint32_t gs = node->disc_off_, ge = child->disc_off_;
        if (ge > gs) {
          const uint8_t* ref = node->arena_.rec_key(child->ref_off_);
          uint32_t rkl = node->arena_.rec_kl(child->ref_off_);
          uint32_t ce = ge;
          if (ce > kl) ce = kl;
          if (ce > rkl) ce = rkl;
          if (ce < ge ||
              memcmp(k + gs, ref + gs, ge - gs) != 0)
            return false;
        }
        node = child;
        extract_kf(k, kl, node->disc_off_, kf);
        continue;
      }
      Group* g = get_group(slot);
      uint64_t mm = scan64(g->tags, tag);
      while (mm) {
        int p = __builtin_ctzll(mm);
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          out = node->arena_.rec_val(entry_off(g->entries[p]));
          return true;
        }
        mm &= mm - 1;
      }
      return false;
    }
    return false;
  }

  bool erase(const uint8_t* k, uint32_t kl) {
    Node* node = this;
    uint32_t kf;
    // Tag computed ONCE — level-independent (CRC of key tail).
    const uint8_t tag = compute_tag(k, kl);
    extract_kf(k, kl, node->disc_off_, kf);

    for (int guard = 0; guard < 64; guard++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];
      if (is_child(slot)) {
        Node* child = get_child(slot);
        uint32_t gs = node->disc_off_, ge = child->disc_off_;
        if (ge > gs) {
          const uint8_t* ref = node->arena_.rec_key(child->ref_off_);
          uint32_t rkl = node->arena_.rec_kl(child->ref_off_);
          uint32_t ce = ge;
          if (ce > kl) ce = kl;
          if (ce > rkl) ce = rkl;
          if (ce < ge ||
              memcmp(k + gs, ref + gs, ge - gs) != 0)
            return false;
        }
        node = child;
        extract_kf(k, kl, node->disc_off_, kf);
        continue;
      }
      Group* g = get_group(slot);
      uint64_t mm = scan64(g->tags, tag);
      while (mm) {
        int p = __builtin_ctzll(mm);
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          g->count--;
          if (static_cast<uint32_t>(p) < g->count) {
            memmove(&g->tags[p], &g->tags[p + 1], g->count - p);
            memmove(&g->entries[p], &g->entries[p + 1],
                    (g->count - p) * 8);
          }
          g->tags[g->count] = 0;
          g->entries[g->count] = 0;
          count_->fetch_sub(1, std::memory_order_relaxed);
          return true;
        }
        mm &= mm - 1;
      }
      return false;
    }
    return false;
  }

  bool contains(const uint8_t* k, uint32_t kl) const {
    uint32_t dummy;
    return get(k, kl, dummy);
  }

  uint64_t size() const { return count_->load(std::memory_order_relaxed); }
  bool empty() const { return size() == 0; }

  Arena& arena_;
  uintptr_t* dir_;
  uint32_t gd_, ds_sz_, shift_;
  uint32_t disc_off_, max_gd_, child_max_gd_;
  uint32_t ref_off_;
  uint32_t first_idx_, last_idx_;
  std::atomic<uint64_t>* count_;
};

}  // namespace internal

// ═════════════════════════════════════════════════════════════════
//  TurboMapV4 — Public API for the Patricia-compressed fractal
//  extendible hash trie. This is the only user-facing class.
// ═════════════════════════════════════════════════════════════════
class TurboMapV4 {
 public:
  using Node = internal::Node;
  using Group = Node::Group;

  TurboMapV4(uint32_t root_gd = 22, uint32_t child_gd = 16)
      : count_(0), root_gd_(root_gd), child_gd_(child_gd) {
    root_ = new (arena_.alloc(sizeof(Node)))
        Node(arena_, &count_, 0, root_gd_, child_gd_);
  }

  /// Resets the map, releasing all entries. O(1).
  void reset() {
    arena_.reset();
    count_.store(0);
    root_ = new (arena_.alloc(sizeof(Node)))
        Node(arena_, &count_, 0, root_gd_, child_gd_);
  }

  /// Inserts a key-value pair. Updates value if key exists.
  /// Returns true if the key was newly inserted, false if updated.
  bool insert(const uint8_t* key, uint32_t key_len, uint32_t val) {
    return root_->insert(key, key_len, val);
  }

  /// Retrieves the value for a key. Returns true if found.
  bool get(const uint8_t* key, uint32_t key_len, uint32_t& out) const {
    return root_->get(key, key_len, out);
  }

  /// Returns true if the key exists.
  bool contains(const uint8_t* key, uint32_t key_len) const {
    return root_->contains(key, key_len);
  }

  /// Removes a key. Returns true if the key was found and removed.
  bool erase(const uint8_t* key, uint32_t key_len) {
    return root_->erase(key, key_len);
  }

  /// Returns the number of entries in the map.
  uint64_t size() const { return root_->size(); }

  /// Returns true if the map is empty.
  bool empty() const { return root_->empty(); }

  /// Returns the total arena memory consumed in bytes.
  uint64_t arena_used() const { return arena_.used(); }

  // ───────────────────────────────────────────────────────────────
  //  Cursor — Lexicographic ordered traversal over all entries.
  // ───────────────────────────────────────────────────────────────
  class Cursor {
   public:
    Cursor() : root_(nullptr), depth_(-1), cur_group_(nullptr) {}

    /// Positions the cursor at the first (smallest) key.
    bool seek_first() {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0) return false;
      push(root_);
      return advance_fwd();
    }

    /// Positions the cursor at the last (largest) key.
    bool seek_last() {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0) return false;
      return descend_last(root_);
    }

    /// Positions the cursor at the first key >= the given key.
    bool seek_ge(const uint8_t* k, uint32_t kl) {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0) return false;
      return seek_ge_at(root_, k, kl);
    }

    /// Positions the cursor at the first key with the given prefix.
    bool seek_prefix(const uint8_t* prefix, uint32_t pl) {
      return seek_ge(prefix, pl);
    }

    /// Advances the cursor forward. Returns false at end.
    bool next() {
      if (depth_ < 0) return false;
      return advance_fwd();
    }

    /// Moves the cursor backward. Returns false at beginning.
    bool prev() {
      if (depth_ < 0) return false;
      return advance_rev();
    }

    /// Returns true if the cursor points to a valid entry.
    bool valid() const { return depth_ >= 0 && cur_group_ != nullptr; }

    /// Returns a pointer to the current key bytes.
    const uint8_t* key() const { build_key(); return key_buf_.data(); }

    /// Returns the length of the current key.
    uint32_t key_len() const { if (key_dirty_) build_key(); return key_len_; }

    /// Returns the value of the current entry.
    uint32_t val() const {
      if (!cur_group_) return 0;
      auto& f = stack_[depth_];
      return f.node->arena_.rec_val(
          Node::entry_off(cur_group_->entries[f.group_idx]));
    }

    /// Overwrites the value of the current entry.
    void update(uint32_t v) {
      auto& f = stack_[depth_];
      f.node->arena_.rec_set_val(
          Node::entry_off(cur_group_->entries[f.group_idx]), v);
    }

    /// Returns true if the current key starts with the given prefix.
    bool has_prefix(const uint8_t* prefix, uint32_t pl) const {
      build_key();
      return key_len_ >= pl && memcmp(key_buf_.data(), prefix, pl) == 0;
    }

   private:
    friend class TurboMapV4;

    struct Frame {
      const Node* node;
      uint32_t dir_idx;
      int32_t group_idx;
    };

    const Node* root_;
    Frame stack_[128];
    int depth_;
    mutable std::vector<uint8_t> key_buf_ = std::vector<uint8_t>(4096);
    mutable uint32_t key_len_ = 0;
    mutable bool key_dirty_ = true;
    Group* cur_group_;

    void push(const Node* node) {
      stack_[++depth_] = {node, 0, -1};
    }
    void push_at(const Node* node, uint32_t di) {
      stack_[++depth_] = {node, di, -1};
    }

    static uint32_t get_first_idx(uintptr_t slot) {
      return Node::is_child(slot) ? Node::get_child(slot)->first_idx_
                                  : Node::get_group(slot)->first_idx;
    }
    static uint32_t get_last_idx(uintptr_t slot) {
      return Node::is_child(slot) ? Node::get_child(slot)->last_idx_
                                  : Node::get_group(slot)->last_idx;
    }

    void set_leaf() {
      auto& f = stack_[depth_];
      cur_group_ = Node::get_group(f.node->dir_[f.dir_idx]);
      key_dirty_ = true;
    }

    void build_key() const {
      if (!key_dirty_) return;
      auto& f = stack_[depth_];
      Group* g = Node::get_group(f.node->dir_[f.dir_idx]);
      uint32_t off = Node::entry_off(g->entries[f.group_idx]);
      key_len_ = f.node->arena_.rec_kl(off);
      if (key_len_ > 0) {
        if (key_buf_.size() < key_len_) key_buf_.resize(key_len_);
        memcpy(key_buf_.data(), f.node->arena_.rec_key(off), key_len_);
      }
      key_dirty_ = false;
    }

    bool advance_fwd() {
      while (depth_ >= 0) {
        auto& f = stack_[depth_];
        if (f.group_idx >= 0) {
          f.group_idx++;
          if (f.group_idx < static_cast<int32_t>(cur_group_->count)) {
            key_dirty_ = true;
            return true;
          }
          f.group_idx = -1;
          f.dir_idx++;
        }
        while (f.dir_idx < f.node->ds_sz_) {
          uintptr_t slot = f.node->dir_[f.dir_idx];
          if (f.dir_idx != get_first_idx(slot)) { f.dir_idx++; continue; }
          if (Node::is_child(slot)) {
            push(Node::get_child(slot));
            break;
          }
          Group* g = Node::get_group(slot);
          if (g->count > 0) {
            f.node->sort_group(g);
            f.group_idx = 0;
            set_leaf();
            return true;
          }
          f.dir_idx++;
        }
        if (f.dir_idx >= f.node->ds_sz_) {
          depth_--;
          if (depth_ >= 0) stack_[depth_].dir_idx++;
        }
      }
      return false;
    }

    bool advance_rev() {
      while (depth_ >= 0) {
        auto& f = stack_[depth_];
        if (f.group_idx >= 0) {
          f.group_idx--;
          if (f.group_idx >= 0) { key_dirty_ = true; return true; }
          f.group_idx = -1;
        }
        bool pushed = false;
        while (true) {
          if (f.dir_idx == 0 && f.group_idx < 0) break;
          f.dir_idx--;
          uintptr_t slot = f.node->dir_[f.dir_idx];
          if (f.dir_idx != get_last_idx(slot)) {
            if (f.dir_idx == 0) break;
            continue;
          }
          if (Node::is_child(slot)) {
            Node* child = Node::get_child(slot);
            push_at(child, child->ds_sz_);
            pushed = true;
            break;
          }
          Group* g = Node::get_group(slot);
          if (g->count > 0) {
            f.node->sort_group(g);
            f.group_idx = g->count - 1;
            set_leaf();
            return true;
          }
          if (f.dir_idx == 0) break;
        }
        if (!pushed && f.group_idx < 0) {
          depth_--;
          if (depth_ >= 0) stack_[depth_].group_idx = -1;
        }
      }
      return false;
    }

    bool descend_last(const Node* node) {
      push_at(node, node->ds_sz_);
      return advance_rev();
    }

    bool seek_ge_at(const Node* node, const uint8_t* k, uint32_t kl) {
      uint32_t kf;
      uint8_t tag;
      node->extract_at(k, kl, kf, tag);
      uint32_t start = node->di(kf);
      push_at(node, start);
      auto& f = stack_[depth_];

      while (f.dir_idx < node->ds_sz_) {
        uintptr_t slot = node->dir_[f.dir_idx];
        if (f.dir_idx > start && f.dir_idx != get_first_idx(slot)) {
          f.dir_idx++;
          continue;
        }
        if (Node::is_child(slot)) {
          if (seek_ge_at(Node::get_child(slot), k, kl)) return true;
          depth_--;
          f.dir_idx++;
          continue;
        }
        Group* g = Node::get_group(slot);
        if (g->count > 0) {
          node->sort_group(g);
          int pos = find_ge(node, g, kf, k, kl);
          if (pos >= 0) {
            f.group_idx = pos;
            set_leaf();
            return true;
          }
        }
        f.dir_idx++;
      }
      depth_--;
      return false;
    }

    static int find_ge(const Node* node, Group* g, uint32_t search_kf,
                       const uint8_t* key, uint32_t kl) {
      int n = static_cast<int>(g->count);
      int lo = 0, hi = n;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (Node::entry_kf(g->entries[mid]) < search_kf) lo = mid + 1;
        else hi = mid;
      }
      if (lo >= n) return -1;
      if (Node::entry_kf(g->entries[lo]) > search_kf) return lo;
      for (int i = lo; i < n && Node::entry_kf(g->entries[i]) == search_kf;
           i++) {
        uint32_t off = Node::entry_off(g->entries[i]);
        uint32_t skl = node->arena_.rec_kl(off);
        uint32_t ml = skl < kl ? skl : kl;
        int cmp = (ml > 0) ? memcmp(node->arena_.rec_key(off), key, ml) : 0;
        if (cmp > 0 || (cmp == 0 && skl >= kl)) return i;
      }
      int next = lo;
      while (next < n && Node::entry_kf(g->entries[next]) == search_kf)
        next++;
      return (next < n) ? next : -1;
    }
  };

  /// Creates a cursor for ordered traversal.
  Cursor cursor() const {
    Cursor c;
    c.root_ = root_;
    return c;
  }

 private:
  internal::Arena arena_;
  std::atomic<uint64_t> count_;
  Node* root_;
  uint32_t root_gd_, child_gd_;
};

}  // namespace turbo

#endif  // TURBOMAP_V4_H_
