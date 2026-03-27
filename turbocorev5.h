
#ifndef TURBO_MAP_H_
#define TURBO_MAP_H_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_acle.h>
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace turbo {

class TurboMapCore;

namespace internal {

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
  return (uint16_t)vget_lane_u8(lo, 0) | ((uint16_t)vget_lane_u8(hi, 0) << 8);
}

inline uint32_t scan32(const uint8_t *tags, uint8_t needle) {
  uint8x16_t n = vdupq_n_u8(needle);
  return (uint32_t)neon_movemask(vceqq_u8(vld1q_u8(tags), n)) |
         ((uint32_t)neon_movemask(vceqq_u8(vld1q_u8(tags + 16), n)) << 16);
}
#else
inline uint32_t scan32(const uint8_t *tags, uint8_t needle) {
  __m256i n = _mm256_set1_epi8(needle);
  return _mm256_movemask_epi8(
      _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)tags), n));
}
#endif

class Arena {
public:
  explicit Arena(uint64_t capacity = 4ULL << 30) : pos_(0), cap_(capacity) {
    base_ = static_cast<uint8_t *>(mmap(nullptr, cap_, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (base_ == MAP_FAILED) {
      fprintf(stderr, "turbo::Arena: mmap failed\n");
      abort();
    }
  }

  ~Arena() { munmap(base_, cap_); }

  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;

  /// Resets the arena position to zero (does not unmap).
  void reset() { pos_ = 0; }

  /// Allocates `sz` bytes, 8-byte aligned. Returns raw pointer.
  void *alloc(uint64_t sz) {
    sz = (sz + 7) & ~7ULL;
    void *p = base_ + pos_;
    pos_ += sz;
    return p;
  }

  /// Allocates `sz` bytes, 16-byte aligned (for SIMD structs).
  void *alloc16(uint64_t sz) {
    pos_ = (pos_ + 15) & ~15ULL;
    return alloc(sz);
  }

  /// Allocates a key-value record:
  uint32_t alloc_rec(const uint8_t *key, uint32_t key_len, uint32_t val) {
    uint32_t raw = 4 + key_len + 4;
    uint32_t padded = (raw + 7) & ~7;
    uint64_t off = pos_;
    memcpy(base_ + off, &key_len, 4);
    if (key_len)
      memcpy(base_ + off + 4, key, key_len);
    memcpy(base_ + off + 4 + key_len, &val, 4);
    pos_ += padded;
    return static_cast<uint32_t>(off >> 3);
  }

  /// Resolves a slot index to a raw pointer.
  const uint8_t *rec(uint32_t slot) const {
    return base_ + (static_cast<uint64_t>(slot) << 3);
  }

  /// Returns the stored key length for a given slot.
  uint32_t rec_kl(uint32_t slot) const {
    uint32_t v;
    memcpy(&v, rec(slot), 4);
    return v;
  }

  /// Returns a pointer to the key data for a given slot.
  const uint8_t *rec_key(uint32_t slot) const { return rec(slot) + 4; }

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
    memcpy(const_cast<uint8_t *>(rec(slot)) + 4 + kl, &v, 4);
  }

  /// Returns the total bytes consumed so far.
  uint64_t used() const { return pos_; }

private:
  uint8_t *base_;
  uint64_t pos_;
  uint64_t cap_;
};

class Node {
  friend class ::turbo::TurboMapCore;
public:
  static constexpr uint32_t kCap = 32;

  /// Group — A leaf bucket holding up to kCap key-value entries.
  /// Tags use 0=empty, 1-255=valid. Gaps allowed (no packing invariant).
  struct alignas(16) Group {
    uint8_t tags[32];   // 0=empty, 1-255=valid tag
    uint32_t kfs[32];
    uint32_t offs[32];
    uint32_t first_idx;
    uint32_t last_idx;
    uint8_t count;
    uint8_t local_depth;
    uint8_t sorted;
    uint8_t pad_[5];
  };

  // ── Slot type detection (order matters!) ──
  static constexpr uint64_t kInlineBit = 1ULL << 63;
  static bool is_inline(uintptr_t v) { return v & kInlineBit; }
  static bool is_child(uintptr_t v) { return (v & 1) && !(v & kInlineBit); }
  static Node *get_child(uintptr_t v) {
    return reinterpret_cast<Node *>(v & ~1ULL);
  }
  static Group *get_group(uintptr_t v) { return reinterpret_cast<Group *>(v); }
  static uintptr_t tag_child(Node *c) {
    return reinterpret_cast<uintptr_t>(c) | 1;
  }

  // ── Inline entry encoding: [1|kf31:31|off:32] ──
  static uintptr_t pack_inline(uint32_t kf, uint32_t off) {
    return kInlineBit | (uint64_t(kf >> 1) << 32) | uint64_t(off);
  }
  static uint32_t inline_kf31(uintptr_t v) { return (v >> 32) & 0x7FFFFFFF; }
  static uint32_t inline_off(uintptr_t v) { return uint32_t(v); }


  /// Compute level-independent tag from key's tail bytes.
  static uint8_t compute_tag(const uint8_t *k, uint32_t kl, uint32_t off = 0) {
    uint64_t v = 0;
    if (kl > off) {
      uint32_t avail = kl - off;
      uint32_t a8 = avail > 8 ? 8 : avail;
      memcpy(&v, k + off, a8);
    }
#if defined(__ARM_NEON)
    return static_cast<uint8_t>((__crc32cd(0, v) >> 24) | 1);
#else
    return static_cast<uint8_t>((_mm_crc32_u64(0, v) >> 24) | 1);
#endif
  }

  /// Extract kf (4-byte directory routing key) at disc_off_.
  static void extract_kf(const uint8_t *k, uint32_t kl, uint32_t off,
                         uint32_t &kf) {
    if (kl <= off) {
      kf = 0;
      return;
    }
    const uint8_t *p = k + off;
    uint32_t rem = kl - off;
    kf = 0;
    uint32_t n = rem > 4 ? 4 : rem;
    memcpy(&kf, p, n);
    kf = __builtin_bswap32(kf);
  }

  /// Combined extraction: kf from disc_off_, tag from key tail.
  static void extract(const uint8_t *k, uint32_t kl, uint32_t &kf,
                      uint8_t &tag) {
    kf = 0;
    if (kl > 0) {
      uint32_t n = kl > 4 ? 4 : kl;
      memcpy(&kf, k, n);
    }
    kf = __builtin_bswap32(kf);
    tag = compute_tag(k, kl, 0);
  }

  void extract_at(const uint8_t *k, uint32_t kl, uint32_t &kf,
                  uint8_t &tag) const {
    if (__builtin_expect(disc_off_ + 8 <= kl, 1)) {
      // Fast path: 8+ bytes available — single memcpy + CRC
      uint64_t v;
      memcpy(&v, k + disc_off_, 8);
      kf = __builtin_bswap32(static_cast<uint32_t>(v));
#if defined(__ARM_NEON)
      tag = static_cast<uint8_t>((__crc32cd(0, v) >> 25) | 0x80);
#else
      tag = static_cast<uint8_t>((_mm_crc32_u64(0, v) >> 25) | 0x80);
#endif
      return;
    }
    if (__builtin_expect(disc_off_ + 4 <= kl, 1)) {
      // Medium path: 4-7 bytes available — zero-padded single-pass
      uint64_t v = 0;
      uint32_t rem = kl - disc_off_;
      if (rem == 4) {
        uint32_t w;
        memcpy(&w, k + disc_off_, 4);
        v = w;
      } else {
        memcpy(&v, k + disc_off_, rem);
      }
      kf = __builtin_bswap32(static_cast<uint32_t>(v));
#if defined(__ARM_NEON)
      tag = static_cast<uint8_t>((__crc32cd(0, v) >> 25) | 0x80);
#else
      tag = static_cast<uint8_t>((_mm_crc32_u64(0, v) >> 25) | 0x80);
#endif
      return;
    }
    if (kl <= disc_off_) {
      kf = 0;
      tag = compute_tag(k, kl, disc_off_);
      return;
    }
    extract_kf(k, kl, disc_off_, kf);
    tag = compute_tag(k, kl, disc_off_);
  }

  /// Branchless directory index from key fragment.
  uint32_t di(uint32_t kf) const {
    return (kf >> (shift_ & 31)) & (ds_sz_ - 1);
  }

  /// Full-key equality check against an arena record.
  bool key_eq(uint32_t slot, const uint8_t *k, uint32_t kl) const {
    uint32_t skl = arena_.rec_kl(slot);
    if (skl != kl)
      return false;
    uint32_t skip = disc_off_ + 4;
    if (skip >= kl)
      return true;
    return memcmp(arena_.rec_key(slot) + skip, k + skip, kl - skip) == 0;
  }

  Group *alloc_group(uint8_t ld, uint32_t fi = 0) {
    auto *g = static_cast<Group *>(arena_.alloc16(sizeof(Group)));
    memset(g->tags, 0, 32);
    memset(g->offs, 0, sizeof(g->offs));
    g->count = 0;
    g->local_depth = ld;
    g->sorted = 1;
    g->first_idx = fi;
    g->last_idx = fi;
    return g;
  }

  void double_dir() {
    uint32_t ns = ds_sz_ * 2;
    auto *nd = static_cast<uintptr_t *>(arena_.alloc(ns * sizeof(uintptr_t)));
    for (uint32_t i = 0; i < ds_sz_; i++) {
      uintptr_t slot = dir_[i];
      if (slot == 0) {
        nd[2 * i] = 0;
        nd[2 * i + 1] = 0;
      } else if (is_inline(slot)) {
        // Route inline entry to correct half based on kf31
        uint32_t kf31 = inline_kf31(slot);
        uint32_t side = (shift_ >= 2) ? ((kf31 >> (shift_ - 2)) & 1) : 0;
        nd[2 * i + side] = slot;
        nd[2 * i + (1 - side)] = 0;
      } else {
        nd[2 * i] = slot;
        nd[2 * i + 1] = slot;
      }
    }
    for (uint32_t i = 0; i < ds_sz_; i++) {
      uintptr_t slot = dir_[i];
      if (slot == 0 || is_inline(slot)) continue;
      if (i == 0 || dir_[i] != dir_[i - 1]) {
        if (is_child(slot)) {
          Node *child = get_child(slot);
          child->first_idx_ = child->first_idx_ * 2;
          child->last_idx_ = child->last_idx_ * 2 + 1;
        } else {
          Group *gg = get_group(slot);
          gg->first_idx = gg->first_idx * 2;
          gg->last_idx = gg->last_idx * 2 + 1;
        }
      }
    }
    dir_ = nd;
    ds_sz_ = ns;
    gd_++;
    shift_--;
  }

  uint32_t compute_lcp(Group *g, uint32_t &min_kl_out) const {
    
    uint32_t occ = ~scan32(g->tags, 0);  // bitmask of occupied slots
    if (!occ) { min_kl_out = 0; return disc_off_; }
    uint32_t min_kl = UINT32_MAX;
    uint32_t tmp = occ;
    while (tmp) {
      int idx = __builtin_ctz(tmp);
      uint32_t skl = arena_.rec_kl(g->offs[idx]);
      if (skl < min_kl) min_kl = skl;
      tmp &= tmp - 1;
    }
    min_kl_out = min_kl;
    if (g->count <= 1 || disc_off_ >= min_kl)
      return disc_off_;
    int first = __builtin_ctz(occ);
    const uint8_t *k0 = arena_.rec_key(g->offs[first]);
    uint32_t common = min_kl;
    tmp = occ & (occ - 1); // skip first
    while (tmp) {
      int idx = __builtin_ctz(tmp);
      const uint8_t *ki = arena_.rec_key(g->offs[idx]);
      uint32_t j = disc_off_;
      while (j < common && k0[j] == ki[j])
        j++;
      if (j < common) common = j;
      if (common <= disc_off_) break;
      tmp &= tmp - 1;
    }
    return common;
  }

  bool spawn_child(uint32_t dx) {
    Group *old = get_group(dir_[dx]);
    
    uint8_t ld = old->local_depth;
    uint32_t min_kl = 0;
    uint32_t child_off = compute_lcp(old, min_kl);
    if (child_off <= disc_off_)
      return false;  // No advancement possible (all truly identical keys)

    // Handle prefix keys: when LCP extends to the shortest key length,
    // separate the short key(s) from longer ones at the min_kl boundary
    if (child_off >= min_kl) {
      // Short keys (kl == min_kl) are prefixes of longer keys
      // Keep short keys in parent, move longer keys to child at min_kl
      uint32_t occ = ~scan32(old->tags, 0);
      uint32_t short_count = 0, long_count = 0;
      uint32_t tmp = occ;
      while (tmp) {
        int idx = __builtin_ctz(tmp);
        uint32_t kl = arena_.rec_kl(old->offs[idx]);
        if (kl <= min_kl) short_count++;
        else long_count++;
        tmp &= tmp - 1;
      }
      if (long_count == 0) return false;  // All same length = truly identical
      child_off = min_kl;
    }

    uint32_t occ = ~scan32(old->tags, 0);

    // ── Always use child-spawn path — preserves trie gap-checking ──
    auto *child = static_cast<Node *>(arena_.alloc(sizeof(Node)));
    new (child) Node(arena_, count_, child_off, child_max_gd_, child_max_gd_);

    count_->fetch_sub(old->count, std::memory_order_relaxed);
    uint32_t tmp3 = occ;
    uint32_t ref_off = 0;
    while (tmp3) {
      int idx = __builtin_ctz(tmp3);
      uint32_t off = old->offs[idx];
      if (!ref_off) ref_off = off;
      child->insert_internal(arena_.rec_key(off), arena_.rec_kl(off),
                             arena_.rec_val(off), off);
      tmp3 &= tmp3 - 1;
    }

    child->ref_off_ = ref_off;
    uint32_t stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(stride - 1);
    child->first_idx_ = st;
    child->last_idx_ = st + stride - 1;
    for (uint32_t i = st; i < st + stride; i++)
      dir_[i] = tag_child(child);
    return true;
  }

  bool split_or_spawn(uint32_t dx) {
    Group *old = get_group(dir_[dx]);
    
    uint8_t ld = old->local_depth;
    // Find first occupied entry for diff computation (gap-tolerant)
    uint32_t occ = ~scan32(old->tags, 0);  // bitmask of occupied slots
    if (!occ) return false;  // empty group — shouldn't happen
    int first = __builtin_ctz(occ);
    uint32_t kf0 = old->kfs[first];
    uint32_t diff = 0;
    uint32_t tmp = occ & (occ - 1); // skip first
    while (tmp) {
      int idx = __builtin_ctz(tmp);
      diff |= kf0 ^ old->kfs[idx];
      tmp &= tmp - 1;
    }
    if (ld > 0)
      diff &= ((1u << (32 - ld)) - 1);
    if (diff == 0)
      return spawn_child(dx);

    uint8_t sld = ld;
    if (sld >= max_gd_)
      return spawn_child(dx);
    while (sld >= gd_) {
      double_dir();
      dx <<= 1;
    }

    // ── Keep-Majority Split ──
    // Count entries on each side of the split bit
    uint32_t count0 = 0, count1 = 0;
    uint32_t scan = occ;
    while (scan) {
      int idx = __builtin_ctz(scan);
      uint8_t side = (old->kfs[idx] >> (31 - sld)) & 1;
      if (side) count1++; else count0++;
      scan &= scan - 1;
    }

    // Minority side gets moved to a new group; majority stays in-place
    bool minority_is_1 = (count0 >= count1);
    uint32_t minority_count = minority_is_1 ? count1 : count0;
    Group *new_grp = alloc_group(sld + 1);
    

    // Move minority entries to new group, clear in old group
    scan = occ;
    while (scan) {
      int idx = __builtin_ctz(scan);
      uint8_t side = (old->kfs[idx] >> (31 - sld)) & 1;
      bool is_minority = (minority_is_1 ? (side == 1) : (side == 0));
      if (is_minority) {
        uint32_t p = new_grp->count;
        new_grp->tags[p] = old->tags[idx];
        new_grp->kfs[p] = old->kfs[idx];
        new_grp->offs[p] = old->offs[idx];
        new_grp->count++;
        // Clear in old group (gap)
        old->tags[idx] = 0;
        old->kfs[idx] = 0;
        old->offs[idx] = 0;
      }
      scan &= scan - 1;
    }
    old->count -= minority_count;
    old->local_depth = sld + 1;
    old->sorted = 0;
    new_grp->sorted = 0;

    // Update directory: majority side already points to old group
    uint32_t old_stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(old_stride - 1);
    uint32_t half = old_stride / 2;
    if (minority_is_1) {
      // Side 1 (second half) = new minority group
      for (uint32_t i = 0; i < half; i++)
        dir_[st + half + i] = reinterpret_cast<uintptr_t>(new_grp);
      old->first_idx = st;
      old->last_idx = st + half - 1;
      new_grp->first_idx = st + half;
      new_grp->last_idx = st + old_stride - 1;
    } else {
      // Side 0 (first half) = new minority group
      for (uint32_t i = 0; i < half; i++)
        dir_[st + i] = reinterpret_cast<uintptr_t>(new_grp);
      new_grp->first_idx = st;
      new_grp->last_idx = st + half - 1;
      old->first_idx = st + half;
      old->last_idx = st + old_stride - 1;
    }
    return true;
  }

  void sort_group(Group *g) const {
    
    if (g->sorted || g->count <= 1) {
      g->sorted = 1;
      return;
    }
    // Compact gaps: pack occupied entries to front
    uint32_t w = 0;
    for (uint32_t r = 0; r < kCap; r++) {
      if (g->tags[r] != 0) {
        if (w != r) {
          g->tags[w] = g->tags[r]; g->kfs[w] = g->kfs[r]; g->offs[w] = g->offs[r];
          g->tags[r] = 0; g->kfs[r] = 0; g->offs[r] = 0;
        }
        w++;
      }
    }
    uint32_t n = g->count;
    for (uint32_t i = 1; i < n; i++) {
      uint32_t kf = g->kfs[i];
      uint32_t off = g->offs[i];
      uint8_t tag = g->tags[i];
      int j = static_cast<int>(i) - 1;
      while (j >= 0) {
        uint32_t kfj = g->kfs[j];
        bool larger = false;
        if (kfj > kf) {
          larger = true;
        } else if (kfj == kf) {
          uint32_t oj = g->offs[j];
          uint32_t l1 = arena_.rec_kl(oj);
          uint32_t l2 = arena_.rec_kl(off);
          const uint8_t *s1 = arena_.rec_key(oj);
          const uint8_t *s2 = arena_.rec_key(off);
          uint32_t ml = l1 < l2 ? l1 : l2;
          int cmp = memcmp(s1, s2, ml);
          if (cmp > 0 || (cmp == 0 && l1 > l2))
            larger = true;
        }
        if (!larger)
          break;
        g->kfs[j + 1] = kfj;
        g->offs[j + 1] = g->offs[j];
        g->tags[j + 1] = g->tags[j];
        j--;
      }
      g->kfs[j + 1] = kf;
      g->offs[j + 1] = off;
      g->tags[j + 1] = tag;
    }
    g->sorted = 1;
  }

   Node(Arena &arena, std::atomic<uint64_t> *cnt, uint32_t disc_off = 0,
        uint32_t max_gd = 18, uint32_t child_max_gd = 18)
       : arena_(arena), count_(cnt), disc_off_(disc_off), max_gd_(max_gd),
         child_max_gd_(child_max_gd), ref_off_(0xFFFFFFFF), first_idx_(0),
         last_idx_(0) {
     gd_ = 0;
     ds_sz_ = 1;
     shift_ = 32;
     dir_ = static_cast<uintptr_t *>(arena_.alloc(sizeof(uintptr_t)));
     dir_[0] = 0;  // empty slot — inline entries, no initial Group
   }

  /// Inserts a key-value pair. Returns true if new, false if updated.
  bool insert_internal(const uint8_t *k, uint32_t kl, uint32_t v,
                       uint32_t existing_off) {
    Node *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);

    for (int attempt = 0; attempt < 256; attempt++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];

      // ── Empty slot: pack inline entry directly ──
      if (slot == 0) {
        uint32_t final_off = (existing_off != 0xFFFFFFFF)
                                 ? existing_off
                                 : node->arena_.alloc_rec(k, kl, v);
        node->dir_[dx] = pack_inline(kf, final_off);
        count_->fetch_add(1, std::memory_order_relaxed);
        return true;
      }

      // ── Inline slot: check match or promote to Group ──
      if (is_inline(slot)) {
        uint32_t s_off = inline_off(slot);
        // Check if same key (update existing value)
        if (inline_kf31(slot) == (kf >> 1) &&
            node->key_eq(s_off, k, kl)) {
          node->arena_.rec_set_val(s_off, v);
          return false;
        }
        // Different key: promote to Group with 2 entries
        uint32_t final_off = (existing_off != 0xFFFFFFFF)
                                 ? existing_off
                                 : node->arena_.alloc_rec(k, kl, v);
        Group *g = node->alloc_group(node->gd_, dx);
        // Insert existing inline entry
        uint32_t s_kl = node->arena_.rec_kl(s_off);
        const uint8_t *s_key = node->arena_.rec_key(s_off);
        uint32_t s_kf; uint8_t s_tag;
        node->extract_at(s_key, s_kl, s_kf, s_tag);
        g->tags[0] = s_tag;
        g->kfs[0] = s_kf;
        g->offs[0] = s_off;
        // Insert new entry
        g->tags[1] = tag;
        g->kfs[1] = kf;
        g->offs[1] = final_off;
        g->count = 2;
        g->sorted = 0;
        g->first_idx = dx;
        g->last_idx = dx;
        node->dir_[dx] = reinterpret_cast<uintptr_t>(g);
        count_->fetch_add(1, std::memory_order_relaxed);
        return true;
      }

      if (is_child(slot)) {
        Node *child = get_child(slot);
        uint32_t gap_start = node->disc_off_;
        uint32_t gap_end = child->disc_off_;
        if (gap_end > gap_start) {
          const uint8_t *ref = node->arena_.rec_key(child->ref_off_);
          uint32_t ref_kl = node->arena_.rec_kl(child->ref_off_);
          uint32_t check = gap_end;
          if (check > kl)
            check = kl;
          if (check > ref_kl)
            check = ref_kl;
          uint32_t div = gap_start;
          while (div < check && k[div] == ref[div])
            div++;
          if (div < gap_end) {
            auto *branch =
                static_cast<Node *>(node->arena_.alloc(sizeof(Node)));
            new (branch) Node(node->arena_, node->count_, div,
                              node->child_max_gd_, node->child_max_gd_);
            branch->ref_off_ = child->ref_off_;

            uint32_t ref_kf, new_kf;
            extract_kf(ref, ref_kl, branch->disc_off_, ref_kf);
            extract_kf(k, kl, branch->disc_off_, new_kf);

            uint32_t xor_bits = ref_kf ^ new_kf;
            if (xor_bits != 0) {
              uint8_t sb = __builtin_clz(xor_bits);
              while (branch->gd_ <= sb)
                branch->double_dir();
            }

            // Double directory until the two kfs route to different slots
            while (branch->di(ref_kf) == branch->di(new_kf))
              branch->double_dir();

            // Place child at its directory slot
            uint32_t fc = branch->di(ref_kf);
            child->first_idx_ = fc;
            child->last_idx_ = fc;
            branch->dir_[fc] = tag_child(child);

            uint32_t ps = dx, pe = dx;
            while (ps > 0 && node->dir_[ps - 1] == slot)
              ps--;
            while (pe + 1 < node->ds_sz_ && node->dir_[pe + 1] == slot)
              pe++;
            branch->first_idx_ = ps;
            branch->last_idx_ = pe;
            for (uint32_t i = ps; i <= pe; i++)
              node->dir_[i] = tag_child(branch);
            continue;
          }
        }
        node = child;
        node->extract_at(k, kl, kf, tag);
        continue;
      }

      Group *g = get_group(slot);
      uint32_t mm = scan32(g->tags, tag);
      while (mm) {
        int p = __builtin_ctz(mm);
        if (g->kfs[p] == kf && node->key_eq(g->offs[p], k, kl)) {
          node->arena_.rec_set_val(g->offs[p], v);
          return false;
        }
        mm &= mm - 1;
      }
      // Check for overflow: all 32 slots occupied?
      uint32_t empty = scan32(g->tags, 0);
      if (empty == 0) {
        uint32_t old_disc = node->disc_off_;
        if (!node->split_or_spawn(dx))
          return false;
        // Re-extract only if disc_off_ was advanced in-place
        if (node->disc_off_ != old_disc)
          node->extract_at(k, kl, kf, tag);
        continue;
      }

      uint32_t final_off = (existing_off != 0xFFFFFFFF)
                               ? existing_off
                               : node->arena_.alloc_rec(k, kl, v);
      // Find first empty slot via SIMD scan
      uint32_t s = __builtin_ctz(empty);
      g->tags[s] = tag;
      g->kfs[s] = kf;
      g->offs[s] = final_off;
      g->count++;
      g->sorted = 0;
      count_->fetch_add(1, std::memory_order_relaxed);
      return true;
    }
    return false;
  }

  bool insert(const uint8_t *k, uint32_t kl, uint32_t v) {
    return insert_internal(k, kl, v, 0xFFFFFFFF);
  }

  bool get(const uint8_t *k, uint32_t kl, uint32_t &out) const {
    const Node *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);

    for (int guard = 0; guard < 64; guard++) {
      uintptr_t slot = node->dir_[node->di(kf)];

      // Fast path: Group pointer (common case at medium-large scale)
      // Group ptrs are even, bit63=0, non-zero — the most common slot type
      if (__builtin_expect(!(slot & (kInlineBit | 1)) && slot != 0, 1)) {
        Group *g = get_group(slot);
        uint32_t mm = scan32(g->tags, tag);
        while (mm) {
          int p = __builtin_ctz(mm);
          if (g->kfs[p] == kf) {
            uint32_t slot_off = g->offs[p];
            uint32_t skl = node->arena_.rec_kl(slot_off);
            if (skl == kl && memcmp(node->arena_.rec_key(slot_off), k, kl) == 0) {
              out = node->arena_.rec_val(slot_off);
              return true;
            }
          }
          mm &= mm - 1;
        }
        return false;
      }

      // Empty slot
      if (slot == 0) return false;

      // Inline entry
      if (is_inline(slot)) {
        if (inline_kf31(slot) == (kf >> 1)) {
          uint32_t s_off = inline_off(slot);
          uint32_t skl = node->arena_.rec_kl(s_off);
          if (skl == kl && memcmp(node->arena_.rec_key(s_off), k, kl) == 0) {
            out = node->arena_.rec_val(s_off);
            return true;
          }
        }
        return false;
      }

      // Child node
      node = get_child(slot);
      node->extract_at(k, kl, kf, tag);
    }
    return false;
  }

  bool erase(const uint8_t *k, uint32_t kl) {
    Node *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);

    for (int guard = 0; guard < 64; guard++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];

      if (slot == 0) return false;  // empty slot

      // ── Inline entry: check and clear ──
      if (is_inline(slot)) {
        if (inline_kf31(slot) == (kf >> 1)) {
          uint32_t s_off = inline_off(slot);
          uint32_t skl = node->arena_.rec_kl(s_off);
          if (skl == kl && memcmp(node->arena_.rec_key(s_off), k, kl) == 0) {
            node->dir_[dx] = 0;  // clear to empty
            count_->fetch_sub(1, std::memory_order_relaxed);
            return true;
          }
        }
        return false;
      }

      if (is_child(slot)) {
        Node *child = get_child(slot);
        node = child;
        node->extract_at(k, kl, kf, tag);
        continue;
      }
      Group *g = get_group(slot);
      uint32_t mm = scan32(g->tags, tag);
      while (mm) {
        int p = __builtin_ctz(mm);
        if (g->kfs[p] == kf) {
          uint32_t slot_off = g->offs[p];
          uint32_t skl = node->arena_.rec_kl(slot_off);
          if (skl == kl && memcmp(node->arena_.rec_key(slot_off), k, kl) == 0) {
            // Gap-tolerant erase: just clear the slot, no memmove
            g->tags[p] = 0;
            g->kfs[p] = 0;
            g->offs[p] = 0;
            g->count--;
            count_->fetch_sub(1, std::memory_order_relaxed);
            return true;
          }
        }
        mm &= mm - 1;
      }
      return false;
    }
    return false;
  }

  bool contains(const uint8_t *k, uint32_t kl) const {
    uint32_t dummy;
    return get(k, kl, dummy);
  }

  uint64_t size() const { return count_->load(std::memory_order_relaxed); }
  bool empty() const { return size() == 0; }

private:
  Arena &arena_;
  uintptr_t *dir_;
  uint32_t gd_, ds_sz_, shift_;
  uint32_t disc_off_, max_gd_, child_max_gd_;
  uint32_t ref_off_;
  uint32_t first_idx_, last_idx_;
  std::atomic<uint64_t> *count_;
};

} // namespace internal

class TurboMapCore {
public:
  using Node = internal::Node;
  using Group = Node::Group;

  struct KeyRef {
    const uint8_t *data;
    uint32_t len;
    KeyRef(const uint8_t *d, uint32_t l) : data(d), len(l) {}
    KeyRef(const char *s) : data(reinterpret_cast<const uint8_t *>(s)),
                            len(static_cast<uint32_t>(strlen(s))) {}
    KeyRef(const char *s, uint32_t l)
        : data(reinterpret_cast<const uint8_t *>(s)), len(l) {}
    bool operator==(const KeyRef &o) const {
      return len == o.len && (len == 0 || memcmp(data, o.data, len) == 0);
    }
  };

  TurboMapCore(uint32_t root_gd = 18, uint32_t child_gd = 18)
      : count_(0), root_gd_(root_gd), child_gd_(child_gd) {
    root_ = new (arena_.alloc(sizeof(Node)))
        Node(arena_, &count_, 0, root_gd_, child_gd_);
  }

  /// Resets the map, releasing all entries. O(1).
  void clear() {
    arena_.reset();
    count_.store(0);
    root_ = new (arena_.alloc(sizeof(Node)))
        Node(arena_, &count_, 0, root_gd_, child_gd_);
  }


  /// Inserts a key-value pair. Updates value if key exists.
  bool insert(const uint8_t *key, uint32_t key_len, uint32_t val) {
    return root_->insert(key, key_len, val);
  }

  /// Retrieves the value for a key. Returns true if found.
  bool get(const uint8_t *key, uint32_t key_len, uint32_t &out) const {
    return root_->get(key, key_len, out);
  }

  /// Returns true if the key exists.
  bool contains(const uint8_t *key, uint32_t key_len) const {
    return root_->contains(key, key_len);
  }

  /// Removes a key. Returns true if the key was found and removed.
  bool erase(const uint8_t *key, uint32_t key_len) {
    return root_->erase(key, key_len);
  }


  /// umap-style operator[]. Returns reference-like proxy for
  class ValueProxy {
  public:
    operator uint32_t() const {
      uint32_t v = 0;
      map_.get(key_.data, key_.len, v);
      return v;
    }
    ValueProxy &operator=(uint32_t val) {
      map_.insert(key_.data, key_.len, val);
      return *this;
    }
  private:
    friend class TurboMapCore;
    ValueProxy(TurboMapCore &m, KeyRef k) : map_(m), key_(k) {}
    TurboMapCore &map_;
    KeyRef key_;
  };

  ValueProxy operator[](KeyRef key) { return ValueProxy(*this, key); }

  /// umap-style count. Returns 1 if found, 0 otherwise.
  uint64_t count(KeyRef key) const {
    return contains(key.data, key.len) ? 1 : 0;
  }

  /// umap-style find. Returns true and fills `out` if found.
  bool find(KeyRef key, uint32_t &out) const {
    return get(key.data, key.len, out);
  }

  /// umap-style erase by KeyRef.
  bool erase(KeyRef key) { return erase(key.data, key.len); }

  /// umap-style insert by KeyRef.
  bool insert(KeyRef key, uint32_t val) {
    return insert(key.data, key.len, val);
  }


  /// Returns the number of entries in the map.
  uint64_t size() const { return root_->size(); }

  /// Returns true if the map is empty.
  bool empty() const { return root_->empty(); }

  /// Returns the total arena memory consumed in bytes.
  uint64_t arena_used() const { return arena_.used(); }

  class Cursor {
  public:
    Cursor() : root_(nullptr), depth_(-1), cur_group_(nullptr) {}

    /// Positions the cursor at the first (smallest) key.
    bool seek_first() {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0)
        return false;
      push(root_);
      return advance_fwd();
    }

    /// Positions the cursor at the last (largest) key.
    bool seek_last() {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0)
        return false;
      return descend_last(root_);
    }

    /// Positions the cursor at the first key >= the given key.
    bool seek_ge(const uint8_t *k, uint32_t kl) {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root_ || root_->size() == 0)
        return false;
      return seek_ge_at(root_, k, kl);
    }

    /// Positions the cursor at the first key with the given prefix.
    bool seek_prefix(const uint8_t *prefix, uint32_t pl) {
      return seek_ge(prefix, pl);
    }

    /// Advances the cursor forward. Returns false at end.
    bool next() {
      if (depth_ < 0)
        return false;
      return advance_fwd();
    }

    /// Moves the cursor backward. Returns false at beginning.
    bool prev() {
      if (depth_ < 0)
        return false;
      return advance_rev();
    }

    /// Returns true if the cursor points to a valid entry.
    bool valid() const { return depth_ >= 0 && (cur_group_ != nullptr || inline_off_ != 0xFFFFFFFF); }

    /// Returns a pointer to the current key bytes.
    const uint8_t *key() const {
      build_key();
      return key_buf_.data();
    }

    /// Returns the length of the current key.
    uint32_t key_len() const {
      if (key_dirty_)
        build_key();
      return key_len_;
    }

    /// Returns the value of the current entry.
    uint32_t val() const {
      auto &f = stack_[depth_];
      if (inline_off_ != 0xFFFFFFFF)
        return f.node->arena_.rec_val(inline_off_);
      if (!cur_group_) return 0;
      return f.node->arena_.rec_val(cur_group_->offs[f.group_idx]);
    }

    /// Overwrites the value of the current entry.
    void update(uint32_t v) {
      auto &f = stack_[depth_];
      if (inline_off_ != 0xFFFFFFFF) {
        f.node->arena_.rec_set_val(inline_off_, v);
        return;
      }
      f.node->arena_.rec_set_val(cur_group_->offs[f.group_idx], v);
    }

    /// Returns true if the current key starts with the given prefix.
    bool has_prefix(const uint8_t *prefix, uint32_t pl) const {
      build_key();
      return key_len_ >= pl && memcmp(key_buf_.data(), prefix, pl) == 0;
    }

  private:
    friend class TurboMapCore;

    struct Frame {
      const Node *node;
      uint32_t dir_idx;
      int32_t group_idx;
    };

    const Node *root_;
    Frame stack_[128];
    int depth_;
    mutable std::vector<uint8_t> key_buf_ = std::vector<uint8_t>(4096);
    mutable uint32_t key_len_ = 0;
    mutable bool key_dirty_ = true;
    Group *cur_group_;
    uint32_t inline_off_ = 0xFFFFFFFF;  // 0xFFFFFFFF = not on inline

    void push(const Node *node) { stack_[++depth_] = {node, 0, -1}; }
    void push_at(const Node *node, uint32_t di) {
      stack_[++depth_] = {node, di, -1};
    }

    static uint32_t get_first_idx(uintptr_t slot, uint32_t dir_idx) {
      if (Node::is_inline(slot) || slot == 0) return dir_idx;
      return Node::is_child(slot) ? Node::get_child(slot)->first_idx_
                                  : Node::get_group(slot)->first_idx;
    }
    static uint32_t get_last_idx(uintptr_t slot, uint32_t dir_idx) {
      if (Node::is_inline(slot) || slot == 0) return dir_idx;
      return Node::is_child(slot) ? Node::get_child(slot)->last_idx_
                                  : Node::get_group(slot)->last_idx;
    }

    void set_leaf() {
      auto &f = stack_[depth_];
      uintptr_t slot = f.node->dir_[f.dir_idx];
      if (Node::is_inline(slot)) {
        cur_group_ = nullptr;
        inline_off_ = Node::inline_off(slot);
      } else {
        cur_group_ = Node::get_group(slot);
        inline_off_ = 0xFFFFFFFF;
      }
      key_dirty_ = true;
    }

    void build_key() const {
      if (!key_dirty_) return;
      auto &f = stack_[depth_];
      uint32_t off;
      if (inline_off_ != 0xFFFFFFFF) {
        off = inline_off_;
      } else {
        Group *g = Node::get_group(f.node->dir_[f.dir_idx]);
        off = g->offs[f.group_idx];
      }
      key_len_ = f.node->arena_.rec_kl(off);
      if (key_len_ > 0) {
        if (key_buf_.size() < key_len_) key_buf_.resize(key_len_);
        memcpy(key_buf_.data(), f.node->arena_.rec_key(off), key_len_);
      }
      key_dirty_ = false;
    }

    bool advance_fwd() {
      while (depth_ >= 0) {
        auto &f = stack_[depth_];
        // If currently on a group entry, try next within group
        if (f.group_idx >= 0 && cur_group_) {
          f.group_idx++;
          if (f.group_idx < static_cast<int32_t>(cur_group_->count)) {
            key_dirty_ = true;
            inline_off_ = 0xFFFFFFFF;
            return true;
          }
          f.group_idx = -1;
          cur_group_ = nullptr;
          f.dir_idx++;
        } else if (inline_off_ != 0xFFFFFFFF) {
          // Was on an inline entry, move to next dir slot
          inline_off_ = 0xFFFFFFFF;
          f.group_idx = -1;
          f.dir_idx++;
        }
        while (f.dir_idx < f.node->ds_sz_) {
          uintptr_t slot = f.node->dir_[f.dir_idx];
          if (slot == 0) { f.dir_idx++; continue; }
          if (f.dir_idx != get_first_idx(slot, f.dir_idx)) {
            f.dir_idx++;
            continue;
          }
          if (Node::is_inline(slot)) {
            f.group_idx = 0;
            set_leaf();
            return true;
          }
          if (Node::is_child(slot)) {
            push(Node::get_child(slot));
            break;
          }
          Group *g = Node::get_group(slot);
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
          if (depth_ >= 0)
            stack_[depth_].dir_idx++;
        }
      }
      return false;
    }

    bool advance_rev() {
      while (depth_ >= 0) {
        auto &f = stack_[depth_];
        if (f.group_idx >= 0 && cur_group_) {
          f.group_idx--;
          if (f.group_idx >= 0) {
            key_dirty_ = true;
            inline_off_ = 0xFFFFFFFF;
            return true;
          }
          f.group_idx = -1;
          cur_group_ = nullptr;
        } else if (inline_off_ != 0xFFFFFFFF) {
          inline_off_ = 0xFFFFFFFF;
          f.group_idx = -1;
        }
        bool pushed = false;
        while (true) {
          if (f.dir_idx == 0 && f.group_idx < 0)
            break;
          f.dir_idx--;
          uintptr_t slot = f.node->dir_[f.dir_idx];
          if (slot == 0) { if (f.dir_idx == 0) break; continue; }
          if (f.dir_idx != get_last_idx(slot, f.dir_idx)) {
            if (f.dir_idx == 0) break;
            continue;
          }
          if (Node::is_inline(slot)) {
            f.group_idx = 0;
            set_leaf();
            return true;
          }
          if (Node::is_child(slot)) {
            Node *child = Node::get_child(slot);
            push_at(child, child->ds_sz_);
            pushed = true;
            break;
          }
          Group *g = Node::get_group(slot);
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
          if (depth_ >= 0)
            stack_[depth_].group_idx = -1;
        }
      }
      return false;
    }

    bool descend_last(const Node *node) {
      push_at(node, node->ds_sz_);
      return advance_rev();
    }

    bool seek_ge_at(const Node *node, const uint8_t *k, uint32_t kl) {
      uint32_t kf;
      uint8_t tag;
      node->extract_at(k, kl, kf, tag);
      uint32_t start = node->di(kf);
      push_at(node, start);
      auto &f = stack_[depth_];

      while (f.dir_idx < node->ds_sz_) {
        uintptr_t slot = node->dir_[f.dir_idx];
        if (slot == 0) { f.dir_idx++; continue; }
        if (f.dir_idx > start && f.dir_idx != get_first_idx(slot, f.dir_idx)) {
          f.dir_idx++;
          continue;
        }
        if (Node::is_inline(slot)) {
          // Check if this inline entry >= search key
          uint32_t s_off = Node::inline_off(slot);
          uint32_t s_kf31 = Node::inline_kf31(slot);
          if (s_kf31 >= (kf >> 1)) {
            f.group_idx = 0;
            set_leaf();
            return true;
          }
          f.dir_idx++;
          continue;
        }
        if (Node::is_child(slot)) {
          if (seek_ge_at(Node::get_child(slot), k, kl))
            return true;
          depth_--;
          f.dir_idx++;
          continue;
        }
        Group *g = Node::get_group(slot);
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

    static int find_ge(const Node *node, Group *g, uint32_t search_kf,
                       const uint8_t *key, uint32_t kl) {
      int n = static_cast<int>(g->count);
      int lo = 0, hi = n;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (g->kfs[mid] < search_kf)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo >= n)
        return -1;
      if (g->kfs[lo] > search_kf)
        return lo;
      for (int i = lo; i < n && g->kfs[i] == search_kf; i++) {
        uint32_t off = g->offs[i];
        uint32_t skl = node->arena_.rec_kl(off);
        uint32_t ml = skl < kl ? skl : kl;
        int cmp = (ml > 0) ? memcmp(node->arena_.rec_key(off), key, ml) : 0;
        if (cmp > 0 || (cmp == 0 && skl >= kl))
          return i;
      }
      int next = lo;
      while (next < n && g->kfs[next] == search_kf)
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
  Node *root_;
  uint32_t root_gd_, child_gd_;
};

template <class Key>
struct KeySerializer {
  static const uint8_t *data(const Key &k) {
    static_assert(std::is_trivially_copyable<Key>::value,
                  "Key must be trivially copyable or specialize KeySerializer");
    return reinterpret_cast<const uint8_t *>(&k);
  }
  static uint32_t size(const Key &) { return sizeof(Key); }
  static Key deserialize(const uint8_t *p, uint32_t) {
    Key k;
    memcpy(&k, p, sizeof(Key));
    return k;
  }
};

template <>
struct KeySerializer<std::string> {
  static const uint8_t *data(const std::string &k) {
    return reinterpret_cast<const uint8_t *>(k.data());
  }
  static uint32_t size(const std::string &k) {
    return static_cast<uint32_t>(k.size());
  }
  static std::string deserialize(const uint8_t *p, uint32_t len) {
    return std::string(reinterpret_cast<const char *>(p), len);
  }
};

template <class Key, class Val, class KS = KeySerializer<Key>>
class TurboMap {
public:
  using key_type = Key;
  using mapped_type = Val;
  using value_type = std::pair<const Key, Val>;
  using size_type = size_t;

  TurboMap() = default;
  explicit TurboMap(size_t) {}

  void clear() {
    core_.clear();
    val_arena_.reset();
  }

  size_t size() const { return core_.size(); }
  bool empty() const { return core_.empty(); }

  std::pair<bool, uint32_t> internal_insert(const Key &key, const Val &val) {
    auto *kd = KS::data(key);
    auto ks = KS::size(key);
    uint32_t existing;
    if (core_.get(kd, ks, existing)) {
      return {false, existing};
    }
    uint32_t slot = alloc_val_slot();
    *val_at(slot) = val;
    core_.insert(kd, ks, slot);
    return {true, slot};
  }

  std::pair<bool, uint32_t> internal_find(const Key &key) const {
    auto *kd = KS::data(key);
    auto ks = KS::size(key);
    uint32_t idx;
    if (core_.get(kd, ks, idx))
      return {true, idx};
    return {false, 0};
  }

  Val &operator[](const Key &key) {
    auto *kd = KS::data(key);
    auto ks = KS::size(key);
    uint32_t slot;
    if (core_.get(kd, ks, slot))
      return *val_at(slot);
    slot = alloc_val_slot();
    core_.insert(kd, ks, slot);
    return *val_at(slot);
  }

  size_t count(const Key &key) const {
    return core_.contains(KS::data(key), KS::size(key)) ? 1 : 0;
  }

  size_t erase(const Key &key) {
    return core_.erase(KS::data(key), KS::size(key)) ? 1 : 0;
  }

  class iterator {
  public:
    using value_type = std::pair<const Key, Val>;
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;

    struct Reference {
      Key first;
      Val &second;
      Reference(const Key &k, Val &v) : first(k), second(v) {}
      Reference(Key &&k, Val &v) : first(std::move(k)), second(v) {}
    };

    struct ArrowProxy {
      Reference ref;
      const Reference *operator->() const { return &ref; }
    };

    iterator() : map_(nullptr), mode_(END), val_ptr_(nullptr) {}

    Reference operator*() const {
      if (mode_ == POINT)
        return Reference(key_, *val_ptr_);
      return Reference(KS::deserialize(cur_->key(), cur_->key_len()),
                       *map_->val_at(cur_->val()));
    }

    ArrowProxy operator->() const { return {operator*()}; }

    iterator &operator++() {
      if (mode_ == POINT) { mode_ = END; return *this; }
      if (mode_ == SCAN) {
        if (!cur_->next()) mode_ = END;
      }
      return *this;
    }

    iterator operator++(int) { iterator t = *this; ++(*this); return t; }

    bool operator==(const iterator &o) const {
      if (mode_ == END && o.mode_ == END) return true;
      if (mode_ == END || o.mode_ == END) return false;
      return false;
    }
    bool operator!=(const iterator &o) const { return !(*this == o); }

    enum Mode { END, POINT, SCAN };
    TurboMap *map_;
    Mode mode_;
    Key key_;
    Val *val_ptr_;
    std::shared_ptr<typename TurboMapCore::Cursor> cur_;
  };

  using const_iterator = iterator;

  iterator begin() {
    iterator it;
    it.map_ = this;
    it.cur_ = std::make_shared<typename TurboMapCore::Cursor>(core_.cursor());
    if (it.cur_->seek_first()) {
      it.mode_ = iterator::SCAN;
      it.val_ptr_ = val_at(it.cur_->val());
    } else {
      it.mode_ = iterator::END;
      it.val_ptr_ = nullptr;
    }
    return it;
  }
  iterator end() const { return iterator(); }
  const_iterator begin() const { return const_cast<TurboMap *>(this)->begin(); }
  const_iterator end() { return iterator(); }

  iterator find(const Key &key) {
    uint32_t idx;
    if (!core_.get(KS::data(key), KS::size(key), idx))
      return end();
    iterator it;
    it.map_ = this;
    it.mode_ = iterator::POINT;
    it.key_ = key;
    it.val_ptr_ = val_at(idx);
    return it;
  }

  std::pair<iterator, bool> insert(const value_type &kv) {
    auto [is_new, idx] = internal_insert(kv.first, kv.second);
    iterator it;
    it.map_ = this;
    it.mode_ = iterator::POINT;
    it.key_ = kv.first;
    it.val_ptr_ = val_at(idx);
    return {it, is_new};
  }

  template <class... Args>
  std::pair<iterator, bool> emplace(Args &&...args) {
    value_type kv(std::forward<Args>(args)...);
    return insert(kv);
  }

  iterator erase(iterator pos) {
    if (pos.mode_ == iterator::END) return end();
    core_.erase(KS::data(pos.key_), KS::size(pos.key_));
    return end();
  }

private:
  uint32_t alloc_val_slot() {
    uint32_t slot = static_cast<uint32_t>(val_arena_.used() >> 3);
    val_arena_.alloc(sizeof(Val) < 8 ? 8 : ((sizeof(Val) + 7) & ~7));
    return slot;
  }

  Val *val_at(uint32_t slot) {
    return reinterpret_cast<Val *>(
        const_cast<uint8_t *>(val_arena_.rec(slot)));
  }

  const Val *val_at(uint32_t slot) const {
    return reinterpret_cast<const Val *>(val_arena_.rec(slot));
  }

  TurboMapCore core_;
  internal::Arena val_arena_;
};

} // namespace turbo

#endif // TURBO_MAP_H_
