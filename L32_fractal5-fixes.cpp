// FractalTag2: 4-byte kf, CRC-kf tag, CAP=16, val-in-SB
// Group = tags[16] + kfs[16] + offs[16] + header = 152 bytes (72% smaller than
// OG) SB stores: [key_data (kl bytes)][val (4 bytes)] per record Group is a
// PURE INDEX — only routing/matching data in the group Recursion every 4 bytes
// of shared prefix
#include <atomic>
#include <chrono>
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
typedef __m128i uint8x16_t;
typedef __m128i uint8x8_t;
#endif

class Arena {
  uint8_t *base_;
  std::atomic<uint64_t> pos_;
  uint64_t cap_;

public:
  Arena(uint64_t r) {
    cap_ = r;
    base_ = (uint8_t *)mmap(0, cap_, 3, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
      fprintf(stderr, "MMAP FAIL\n");
      abort();
    }
    pos_.store(0);
  }
  ~Arena() { munmap(base_, cap_); }
  void reset() { pos_.store(0); }
  void *alloc(uint64_t b) {
    uint64_t a = (b + 7) & ~7ULL;
    uint64_t o = pos_.fetch_add(a, std::memory_order_relaxed);
    return base_ + o;
  }
  void *alloc16(uint64_t b) {
    uint64_t c = pos_.load(std::memory_order_relaxed);
    uint64_t s = (c + 15) & ~15ULL;
    uint64_t p = (b + 15) & ~15ULL;
    uint64_t t = (s - c) + p;
    uint64_t o = pos_.fetch_add(t, std::memory_order_relaxed);
    return base_ + ((o + 15) & ~15ULL);
  }
  uint64_t used() const { return pos_.load(std::memory_order_relaxed); }
};

struct SB {
  struct Blk {
    uint8_t *d;
    uint32_t c, l;
  };
  Blk *b;
  uint32_t n, mx;
  void init(Arena &a) {
    b = 0;
    n = 0;
    mx = 0;
  }
  // Record format: [uint16_t suf_len] [suffix bytes] [uint32_t val]
  // Total: 2 + suf_len + 4 = 6 + suf_len bytes
  uint32_t al(Arena &a, const uint8_t *data, uint32_t data_len, uint32_t val) {
    uint32_t rl = 2 + data_len + 4;
    if (!n || b[n - 1].l + rl > b[n - 1].c) {
      if (n == mx) {
        uint32_t nm = mx ? mx * 2 : 16;
        auto *nb = (Blk *)a.alloc(nm * sizeof(Blk));
        if (b)
          memcpy(nb, b, n * sizeof(Blk));
        b = nb;
        mx = nm;
      }
      auto &x = b[n];
      x.d = (uint8_t *)a.alloc(4 << 20);
      x.c = 4 << 20;
      x.l = 0;
      n++;
    }
    auto &x = b[n - 1];
    uint32_t o = x.l;
    uint16_t sl16 = (uint16_t)data_len;
    memcpy(x.d + o, &sl16, 2);
    if (data_len)
      memcpy(x.d + o + 2, data, data_len);
    memcpy(x.d + o + 2 + data_len, &val, 4);
    x.l += rl;
    return ((n - 1) << 22) | o;
  }
  const uint8_t *g(uint32_t p) const { return b[p >> 22].d + (p & 0x3FFFFF); }
  // Read stored suf_len from record header
  uint16_t rec_suf_len(uint32_t p) const {
    uint16_t v;
    memcpy(&v, g(p), 2);
    return v;
  }
  // Suffix data starts at offset 2
  const uint8_t *suf(uint32_t p) const { return g(p) + 2; }
  // Val at offset 2 + suf_len
  uint32_t get_val(uint32_t p) const {
    uint16_t sl = rec_suf_len(p);
    uint32_t v;
    memcpy(&v, g(p) + 2 + sl, 4);
    return v;
  }
  void set_val(uint32_t p, uint32_t val) {
    uint16_t sl = rec_suf_len(p);
    memcpy((uint8_t *)g(p) + 2 + sl, &val, 4);
  }
};

#if defined(__ARM_NEON)
static uint16_t nmm(uint8x16_t c) {
  const uint8x16_t bm = {1, 2, 4, 8, 16, 32, 64, 128,
                         1, 2, 4, 8, 16, 32, 64, 128};
  uint8x16_t m = vandq_u8(c, bm);
  uint8x8_t lo = vget_low_u8(m), hi = vget_high_u8(m);
  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);
  lo = vpadd_u8(lo, lo);
  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);
  hi = vpadd_u8(hi, hi);
  return (uint16_t)vget_lane_u8(lo, 0) | ((uint16_t)vget_lane_u8(hi, 0) << 8);
}
// Single-register SIMD scan for 16-entry groups
static uint16_t scan16(const uint8_t *t, uint8_t tag) {
  return nmm(vceqq_u8(vld1q_u8(t), vdupq_n_u8(tag)));
}
// Dual-register scan for OG's 32-entry groups
static uint32_t scan32(const uint8_t *t, uint8_t tag) {
  uint8x16_t n = vdupq_n_u8(tag);
  return (uint32_t)nmm(vceqq_u8(vld1q_u8(t), n)) |
         ((uint32_t)nmm(vceqq_u8(vld1q_u8(t + 16), n)) << 16);
}
#else
static uint16_t scan16(const uint8_t *t, uint8_t tag) {
  __m128i n = _mm_set1_epi8(tag);
  __m128i v = _mm_loadu_si128((const __m128i *)t);
  return _mm_movemask_epi8(_mm_cmpeq_epi8(v, n));
}
static uint32_t scan32(const uint8_t *t, uint8_t tag) {
  __m256i n = _mm256_set1_epi8(tag);
  __m256i v = _mm256_loadu_si256((const __m256i *)t);
  return _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, n));
}
#endif

// ═══════════════════════════════════════════════════════════════
//  UArena — Unified 8-byte-aligned arena (replaces Arena + SB)
//  One mmap: groups, directories, records, children — all in one flat space
//  u32 slot index × 8 = byte offset → 32GB addressable with u32
// ═══════════════════════════════════════════════════════════════
class UArena {
  uint8_t *base_;
  uint64_t pos_, cap_;

public:
  UArena(uint64_t cap = 4ULL << 30) : pos_(0), cap_(cap) {
    base_ = (uint8_t *)mmap(0, cap, 3, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
      fprintf(stderr, "MMAP FAIL\n");
      abort();
    }
  }
  ~UArena() { munmap(base_, cap_); }
  void reset() { pos_ = 0; }
  // Alloc raw bytes (8-byte aligned), returns pointer
  void *alloc(uint64_t sz) {
    sz = (sz + 7) & ~7ULL;
    void *p = base_ + pos_;
    pos_ += sz;
    return p;
  }
  // Alloc 16-byte aligned (for SIMD groups)
  void *alloc16(uint64_t sz) {
    pos_ = (pos_ + 15) & ~15ULL;
    return alloc(sz);
  }
  // Alloc record: [uint16_t full_key_len][full key bytes][uint32_t val], padded
  // to 8 Returns SLOT INDEX (byte_offset >> 3)
  uint32_t alloc_rec(const uint8_t *key, uint32_t key_len, uint32_t val) {
    uint32_t raw = 2 + key_len + 4;
    uint32_t padded = (raw + 7) & ~7;
    uint64_t off = pos_;
    uint16_t sl = (uint16_t)key_len;
    memcpy(base_ + off, &sl, 2);
    if (key_len)
      memcpy(base_ + off + 2, key, key_len);
    memcpy(base_ + off + 2 + key_len, &val, 4);
    pos_ += padded;
    return (uint32_t)(off >> 3);
  }
  // Resolve slot index to pointer
  const uint8_t *rec(uint32_t s) const { return base_ + ((uint64_t)s << 3); }
  // Record access by slot index
  uint16_t rec_kl(uint32_t s) const {
    uint16_t v;
    memcpy(&v, rec(s), 2);
    return v;
  }
  const uint8_t *rec_key(uint32_t s) const { return rec(s) + 2; }
  uint32_t rec_val(uint32_t s) const {
    auto sl = rec_kl(s);
    uint32_t v;
    memcpy(&v, rec(s) + 2 + sl, 4);
    return v;
  }
  void rec_set_val(uint32_t s, uint32_t v) {
    auto sl = rec_kl(s);
    memcpy((uint8_t *)rec(s) + 2 + sl, &v, 4);
  }
  uint64_t used() const { return pos_; }
};

// ═══════════════════════════════════════════════════════════════
//  FractalTag2 — 4-byte kf, zero CRC, key-byte tag, CAP=16
// ═══════════════════════════════════════════════════════════════
static FILE *dbg_log = nullptr;
static void hex_key(FILE *f, const uint8_t *k, uint32_t kl) {
  for (uint32_t i = 0; i < kl; i++)
    fprintf(f, "%02x", k[i]);
}

class FractalTag2 {
  static constexpr uint32_t CAP = 32;
  static constexpr uint32_t CHILD_MAX_GD = 16;

  struct alignas(16) Group {
    uint8_t tags[32];     // 32B
    uint64_t entries[32]; // 256B — (kf<<32)|off
    uint32_t first_idx_;  // 4B — for O(1) fwd cursor dedup
    uint32_t last_idx_;   // 4B — for O(1) rev cursor dedup
    uint8_t count, local_depth, sorted,
        _p[5]; // 8B
  };

  // Entry accessors — kf in high 32 bits (natural uint64_t sort = kf order)
  static uint32_t entry_kf(uint64_t e) { return (uint32_t)(e >> 32); }
  static uint32_t entry_off(uint64_t e) { return (uint32_t)e; }
  static uint64_t make_entry(uint32_t kf, uint32_t off) {
    return ((uint64_t)kf << 32) | off;
  }

  UArena &ua_;
  uintptr_t *dir_;
  uint32_t gd_, ds_sz_;
  uint32_t shift_;     // = 32 - gd_, for branchless di()
  uint32_t disc_off_;  // byte offset into key where this node routes
  uint32_t max_gd_;    // per-node directory limit (root=24, children=8)
  uint32_t ref_off_;   // arena slot of representative entry (for gap prefix verification)
  uint32_t first_idx_; // directory index in parent for cursor dedup
  uint32_t last_idx_;  // last directory index in parent for cursor dedup
  std::atomic<uint64_t>
      *count_; // shared root counter (all levels point to same)

  static bool is_child(uintptr_t v) { return v & 1; }
  static FractalTag2 *get_child(uintptr_t v) {
    return (FractalTag2 *)(v & ~1ULL);
  }
  static Group *get_group(uintptr_t v) { return (Group *)(uintptr_t)v; }
  static uintptr_t tag_child(FractalTag2 *c) { return (uintptr_t)c | 1; }

  // Extract kf (4-byte key fragment) — always reads from byte 0 of passed-in
  // key
  static void extract(const uint8_t *k, uint32_t kl, uint32_t &kf,
                      uint8_t &tag) {
#if defined(__ARM_NEON)
    if (__builtin_expect(8 <= kl, 1)) {
      uint64_t v;
      memcpy(&v, k, 8);
      kf = __builtin_bswap32((uint32_t)v);
      tag = (uint8_t)((__crc32cd(0, v) >> 25) | 0x80);
      return;
    }
#else
    if (__builtin_expect(8 <= kl, 1)) {
      uint64_t v;
      memcpy(&v, k, 8);
      kf = __builtin_bswap32((uint32_t)v);
      tag = (uint8_t)((_mm_crc32_u64(0, v) >> 25) | 0x80);
      return;
    }
#endif
    kf = 0;
    if (kl > 0) {
      uint32_t a4 = kl > 4 ? 4 : kl;
      memcpy(&kf, k, a4);
    }
    kf = __builtin_bswap32(kf);
    uint64_t v = 0;
    if (kl > 0) {
      uint32_t a8 = kl > 8 ? 8 : kl;
      memcpy(&v, k, a8);
    }
#if defined(__ARM_NEON)
    tag = (uint8_t)((__crc32cd(0, v) >> 25) | 0x80);
#else
    tag = (uint8_t)((_mm_crc32_u64(0, v) >> 25) | 0x80);
#endif
  }

  // Extract kf at this node's disc_off_ position
  void extract_at(const uint8_t *k, uint32_t kl, uint32_t &kf,
                  uint8_t &tag) const {
    if (kl <= disc_off_) {
      kf = 0;
      tag = 0x80;
      return;
    }
    extract(k + disc_off_, kl - disc_off_, kf, tag);
  }

  // Branchless: when gd_=0, ds_sz_=1, mask=0 → result always 0. No UB.
  uint32_t di(uint32_t kf) const {
    return (kf >> (shift_ & 31)) & (ds_sz_ - 1);
  }

  // Compare full key stored in arena (the arena record holds the ENTIRE key
  // now)
  bool key_eq(uint32_t off, const uint8_t *k, uint32_t kl) const {
    uint16_t stored_kl = ua_.rec_kl(off);
    if (stored_kl != kl)
      return false;
    if (!stored_kl)
      return true;
    return !memcmp(ua_.rec_key(off), k, kl);
  }

  Group *alloc_group(uint8_t ld, uint32_t first_idx = 0) {
    Group *g = (Group *)ua_.alloc16(sizeof(Group));
    memset(g->tags, 0, 32);
    g->count = 0;
    g->local_depth = ld;
    g->sorted = 1;
    g->first_idx_ = first_idx;
    g->last_idx_ = first_idx;
    return g;
  }

  void double_dir() {
    uint32_t ns = ds_sz_ * 2;
    auto *nd = (uintptr_t *)ua_.alloc(ns * sizeof(uintptr_t));
    for (uint32_t i = 0; i < ds_sz_; i++) {
      nd[2 * i] = dir_[i];
      nd[2 * i + 1] = dir_[i];
    }
    // Update first_idx_/last_idx_ exactly once per unique contiguous block.
    // Process only the start of each block to prevent cascading updates.
    for (uint32_t i = 0; i < ds_sz_; i++) {
      if (i == 0 || dir_[i] != dir_[i - 1]) {
        uintptr_t slot = dir_[i];
        if (is_child(slot)) {
          FractalTag2 *child = get_child(slot);
          child->first_idx_ = child->first_idx_ * 2;
          child->last_idx_ = child->last_idx_ * 2 + 1;
        } else {
          Group *g = get_group(slot);
          g->first_idx_ = g->first_idx_ * 2;
          g->last_idx_ = g->last_idx_ * 2 + 1;
        }
      }
    }
    dir_ = nd;
    ds_sz_ = ns;
    gd_++;
    shift_--;
  }

  // Compute LCP of all keys in group, starting from disc_off_
  // Returns byte position of first differing byte across all keys
  // Uses FULL KEYS from the arena — not limited to 4-byte kf
  uint32_t compute_lcp(Group *g, uint32_t &min_kl_out) const {
    uint32_t min_kl = UINT32_MAX;
    for (uint32_t i = 0; i < g->count; i++) {
      uint16_t skl = ua_.rec_kl(entry_off(g->entries[i]));
      if (skl < min_kl) min_kl = skl;
    }
    min_kl_out = min_kl;
    if (g->count <= 1 || disc_off_ >= min_kl) return disc_off_;

    const uint8_t *k0 = ua_.rec_key(entry_off(g->entries[0]));
    uint32_t common_end = min_kl;
    for (uint32_t i = 1; i < g->count; i++) {
      const uint8_t *ki = ua_.rec_key(entry_off(g->entries[i]));
      uint32_t j = disc_off_;
      while (j < common_end && k0[j] == ki[j]) j++;
      if (j < common_end) common_end = j;
      if (common_end <= disc_off_) break;
    }
    return common_end;
  }

  // Spawn child at LCP-computed disc_off — skips ALL shared prefix in one shot
  // Returns false if spawn is impossible (all keys identical up to min length)
  // Patricia-compressed spawn: jumps directly to LCP (no max_skip cap).
  // Full gap [disc_off_, child_disc_off_) verified via ref_off_ memcmp on insert.
  bool spawn_child(uint32_t dx) {
    Group *old = get_group(dir_[dx]);
    uint8_t ld = old->local_depth;

    uint32_t min_kl = 0;
    uint32_t child_disc_off = compute_lcp(old, min_kl);
    // No max_skip cap — gap is verified by ref_off_ memcmp

    // Guard: child must advance past current disc_off
    if (child_disc_off <= disc_off_)
      return false;
    // Guard: if LCP reaches or exceeds shortest key, keys are indistinguishable
    if (child_disc_off >= min_kl)
      return false;

    auto *child = (FractalTag2 *)ua_.alloc(sizeof(FractalTag2));
    new (child) FractalTag2(ua_, count_, child_disc_off, CHILD_MAX_GD);

    // Re-insert entries via child's insert_internal (reuses arena offsets)
    count_->fetch_sub(old->count, std::memory_order_relaxed);
    for (uint32_t i = 0; i < old->count; i++) {
      uint32_t off = entry_off(old->entries[i]);
      uint32_t val = ua_.rec_val(off);
      uint16_t stored_kl = ua_.rec_kl(off);
      const uint8_t *full_key = ua_.rec_key(off);
      child->insert_internal(full_key, stored_kl, val, off);
    }

    // Store arena offset of first entry as reference for gap verification
    child->ref_off_ = entry_off(old->entries[0]);
    uint32_t stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(stride - 1);
    child->first_idx_ = st;
    child->last_idx_ = st + stride - 1;
    for (uint32_t i = st; i < st + stride; i++)
      dir_[i] = tag_child(child);
    return true;
  }

  // Returns false if bucket can't be split (keys are indistinguishable)
  bool split_or_spawn(uint32_t dx) {
    if (dbg_log)
      fprintf(dbg_log,
              "  SPLIT_OR_SPAWN: node=%p disc_off=%u dx=%u gd=%u "
              "slot_is_child=%d count=%u\n",
              (void *)this, disc_off_, dx, gd_, is_child(dir_[dx]),
              is_child(dir_[dx]) ? 0 : get_group(dir_[dx])->count);
    Group *old = get_group(dir_[dx]);
    uint8_t ld = old->local_depth;

    uint32_t kf0 = entry_kf(old->entries[0]);
    uint32_t diff = 0;
    for (uint32_t i = 1; i < old->count; i++)
      diff |= kf0 ^ entry_kf(old->entries[i]);

    if (ld > 0)
      diff &= ((1u << (32 - ld)) - 1);

    // When diff==0, all entries have identical kf. Spawn child immediately.
    if (diff == 0) {
      return spawn_child(dx);
    }

    uint8_t split_ld = ld;
    if (split_ld >= max_gd_) {
      return spawn_child(dx);
    }
    while (split_ld >= gd_) {
      double_dir();
      dx <<= 1;
    }

    Group *g0 = alloc_group(split_ld + 1), *g1 = alloc_group(split_ld + 1);
    for (uint32_t i = 0; i < old->count; i++) {
      uint32_t ekf = entry_kf(old->entries[i]);
      uint8_t side = (ekf >> (31 - split_ld)) & 1;
      Group *d = side ? g1 : g0;
      uint32_t p = d->count;
      d->tags[p] = old->tags[i];
      d->entries[p] = old->entries[i];
      d->count++;
    }

    uint32_t old_stride = 1u << (gd_ - ld);
    uint32_t st = dx & ~(old_stride - 1);
    uint32_t half_stride = old_stride / 2;
    for (uint32_t i = 0; i < half_stride; i++) {
      dir_[st + i] = (uintptr_t)g0;
      dir_[st + half_stride + i] = (uintptr_t)g1;
    }
    g0->first_idx_ = st;
    g0->last_idx_ = st + half_stride - 1;
    g1->first_idx_ = st + half_stride;
    g1->last_idx_ = st + old_stride - 1;
    g0->sorted = 0;
    g1->sorted = 0;
    return true;
  }

public:
  FractalTag2(UArena &ua, std::atomic<uint64_t> *cnt, uint32_t disc_off = 0,
              uint32_t max_gd = 24)
      : ua_(ua), count_(cnt), disc_off_(disc_off), max_gd_(max_gd),
        ref_off_(0xFFFFFFFF), first_idx_(0), last_idx_(0) {
    gd_ = 0;
    ds_sz_ = 1;
    shift_ = 32;
    dir_ = (uintptr_t *)ua_.alloc(sizeof(uintptr_t));
    dir_[0] = (uintptr_t)alloc_group(0);
  }

  // Insert with optional pre-existing arena offset (for spawn moves)
  bool insert_internal(const uint8_t *k, uint32_t kl, uint32_t v,
                       uint32_t existing_off) {
    FractalTag2 *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);
    if (dbg_log) {
      fprintf(dbg_log, "INSERT key=");
      hex_key(dbg_log, k, kl);
      fprintf(dbg_log, " v=%u disc_off=%u kf=%08x tag=%02x\n", v,
              node->disc_off_, kf, tag);
    }

    for (int a = 0; a < 256; a++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];
      if (is_child(slot)) {
        FractalTag2 *child = get_child(slot);
        // Patricia gap verification: verify ALL bytes in
        // [node->disc_off_, child->disc_off_) match the reference entry.
        uint32_t gap_start = node->disc_off_;
        uint32_t gap_end = child->disc_off_;
        if (gap_end > gap_start) {
          const uint8_t *ref_key = node->ua_.rec_key(child->ref_off_);
          uint16_t ref_kl = node->ua_.rec_kl(child->ref_off_);
          uint32_t check_end = gap_end;
          if (check_end > kl) check_end = kl;
          if (check_end > ref_kl) check_end = ref_kl;
          // Find first diverging byte in the gap
          uint32_t div_byte = gap_start;
          while (div_byte < check_end && k[div_byte] == ref_key[div_byte])
            div_byte++;
          if (div_byte < gap_end) {
            // GAP MISMATCH at div_byte → Patricia edge split.
            // Create intermediate branch node at the divergence point.
            auto *branch = (FractalTag2 *)node->ua_.alloc(sizeof(FractalTag2));
            new (branch) FractalTag2(node->ua_, node->count_, div_byte, CHILD_MAX_GD);
            branch->ref_off_ = child->ref_off_; // same ref, gap [parent, div_byte) is subset

            // Compute kf at div_byte for ref and new key
            uint32_t ref_kf_at_div, new_kf_at_div;
            uint8_t ref_tag_at_div, new_tag_at_div;
            branch->extract_at(ref_key, ref_kl, ref_kf_at_div, ref_tag_at_div);
            branch->extract_at(k, kl, new_kf_at_div, new_tag_at_div);

            // They must differ (we diverged at div_byte)
            uint32_t xor_bits = ref_kf_at_div ^ new_kf_at_div;
            if (xor_bits == 0) {
              // Edge case: keys differ in length but not in kf at div_byte.
              // The short key ends here — just insert at branch level.
            } else {
              uint8_t split_bit = __builtin_clz(xor_bits);
              // Grow branch directory until kfs are separated
              while (branch->gd_ <= split_bit)
                branch->double_dir();
            }

            // Install old child on ref_kf side of branch directory
            uint32_t child_dx_in_branch = branch->di(ref_kf_at_div);
            uint32_t new_dx_in_branch = branch->di(new_kf_at_div);

            // Find the initial group that spans the branch directory
            Group *init_group = get_group(branch->dir_[new_dx_in_branch]);

            // Compute stride for each side: child gets ref_kf slots, group gets new_kf slots.
            // After double_dir, all slots point to init_group. Overwrite child's slots.
            // With gd_ = split_bit+1, the child's stride = 1 (exactly 1 slot matches
            // each distinct kf pattern at the split resolution). But the child needs
            // ALL slots with the same top (split_bit+1) bits as ref_kf.
            // stride = ds_sz_ / (1 << (split_bit + 1)) = 1 (since gd_ = split_bit+1)
            if (child_dx_in_branch != new_dx_in_branch) {
              // Child occupies exactly 1 slot (stride=1 when gd_ = split_bit+1)
              child->first_idx_ = child_dx_in_branch;
              child->last_idx_ = child_dx_in_branch;
              branch->dir_[child_dx_in_branch] = tag_child(child);
              // Empty group keeps all other slots
              init_group->first_idx_ = 0;
              init_group->last_idx_ = branch->ds_sz_ - 1;
              // But fix: init_group shouldn't include child's slot in cursor dedup.
              // Use the next adjacent slot as first_idx if child is at 0.
              if (child_dx_in_branch == 0) {
                init_group->first_idx_ = 1;
              } else if (child_dx_in_branch == branch->ds_sz_ - 1) {
                init_group->last_idx_ = branch->ds_sz_ - 2;
              }
              // Note: init_group spans non-contiguous slots but first/last covers
              // all of them for cursor dedup. The cursor uses canonical first/last checks.
            } else {
              // Same slot — child goes into the single slot, group needs a new slot
              // This happens when xor_bits was 0 (gd stayed 0). Put child at slot 0.
              child->first_idx_ = 0;
              child->last_idx_ = 0;
              branch->dir_[0] = tag_child(child);
              // The new key can't descend into child, insert will retry and
              // split_or_spawn will separate them.
            }

            // Replace old child in parent directory with branch
            uint32_t parent_fi = child->first_idx_;
            uint32_t parent_li = child->last_idx_;
            // Reclaim the original parent slots that pointed to old child
            // We need to re-read them since child's first/last just changed
            // Use the saved values from before we modified them
            // Actually — the parent slots are what we need to overwrite.
            // Find them from dx: the child occupied some contiguous range.
            // Walk left/right from dx to find the extent.
            uint32_t p_start = dx, p_end = dx;
            while (p_start > 0 && node->dir_[p_start - 1] == slot)
              p_start--;
            while (p_end + 1 < node->ds_sz_ && node->dir_[p_end + 1] == slot)
              p_end++;
            branch->first_idx_ = p_start;
            branch->last_idx_ = p_end;
            for (uint32_t i = p_start; i <= p_end; i++)
              node->dir_[i] = tag_child(branch);

            continue; // retry — new key will find branch, then its group
          }
        }
        // Gap verified — descend into child
        node = child;
        node->extract_at(k, kl, kf, tag);
        if (dbg_log)
          fprintf(dbg_log, "  DESCEND child disc_off=%u kf=%08x tag=%02x\n",
                  node->disc_off_, kf, tag);
        continue;
      }
      Group *g = get_group(slot);
      uint32_t mm = scan32(g->tags, tag);
      while (mm) {
        int p = __builtin_ctz(mm);
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          if (dbg_log) {
            fprintf(dbg_log, "  DUP found at p=%d off=%u stored_key=", p,
                    entry_off(g->entries[p]));
            hex_key(dbg_log, node->ua_.rec_key(entry_off(g->entries[p])),
                    node->ua_.rec_kl(entry_off(g->entries[p])));
            fprintf(dbg_log, "\n");
          }
          node->ua_.rec_set_val(entry_off(g->entries[p]), v);
          return false;
        }
        mm &= mm - 1;
      }
      // Overflow scan for positions 32+
      for (uint32_t p = 32; p < g->count; p++) {
        if (g->tags[p] == tag && entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          node->ua_.rec_set_val(entry_off(g->entries[p]), v);
          return false;
        }
      }
      if (g->count >= CAP) {
        if (!node->split_or_spawn(dx)) {
          // Can't split further — this shouldn't happen with proper spawn
          // but as safety net, just retry
          return false;
        }
        continue;
      }
      uint32_t final_off = (existing_off != 0xFFFFFFFF)
                               ? existing_off
                               : node->ua_.alloc_rec(k, kl, v);
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

  bool insert(const uint8_t *k, uint32_t kl, uint32_t v) {
    return insert_internal(k, kl, v, 0xFFFFFFFF);
  }

  bool get(const uint8_t *k, uint32_t kl, uint32_t &out) const {
    const FractalTag2 *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);

    for (int guard = 0; guard < 64; guard++) {
      uintptr_t slot = node->dir_[node->di(kf)];
      if (is_child(slot)) {
        FractalTag2 *child = get_child(slot);
        // Patricia gap verification
        uint32_t gap_start = node->disc_off_;
        uint32_t gap_end = child->disc_off_;
        if (gap_end > gap_start) {
          const uint8_t *ref_key = node->ua_.rec_key(child->ref_off_);
          uint16_t ref_kl = node->ua_.rec_kl(child->ref_off_);
          uint32_t check_end = gap_end;
          if (check_end > kl) check_end = kl;
          if (check_end > ref_kl) check_end = ref_kl;
          if (check_end < gap_end ||
              memcmp(k + gap_start, ref_key + gap_start, gap_end - gap_start) != 0) {
            if (dbg_log) {
              fprintf(dbg_log, "GET GAP MISS: disc_off=%u gap=[%u,%u) check_end=%u kl=%u ref_kl=%u\n",
                      node->disc_off_, gap_start, gap_end, check_end, kl, ref_kl);
              fprintf(dbg_log, "  key bytes: ");
              for (uint32_t b = gap_start; b < gap_end && b < kl; b++) fprintf(dbg_log, "%02x ", k[b]);
              fprintf(dbg_log, "\n  ref bytes: ");
              for (uint32_t b = gap_start; b < gap_end && b < ref_kl; b++) fprintf(dbg_log, "%02x ", ref_key[b]);
              fprintf(dbg_log, "\n");
            }
            return false; // gap mismatch — key not here
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
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          out = node->ua_.rec_val(entry_off(g->entries[p]));
          return true;
        }
        mm &= mm - 1;
      }
      // Overflow scan for positions 32+
      for (uint32_t p = 32; p < g->count; p++) {
        if (g->tags[p] == tag && entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          out = node->ua_.rec_val(entry_off(g->entries[p]));
          return true;
        }
      }
      return false;
    }
    return false;
  }

  bool erase(const uint8_t *k, uint32_t kl) {
    FractalTag2 *node = this;
    uint32_t kf;
    uint8_t tag;
    node->extract_at(k, kl, kf, tag);

    for (int guard = 0; guard < 64; guard++) {
      uint32_t dx = node->di(kf);
      uintptr_t slot = node->dir_[dx];
      if (is_child(slot)) {
        FractalTag2 *child = get_child(slot);
        // Patricia gap verification
        uint32_t gap_start = node->disc_off_;
        uint32_t gap_end = child->disc_off_;
        if (gap_end > gap_start) {
          const uint8_t *ref_key = node->ua_.rec_key(child->ref_off_);
          uint16_t ref_kl = node->ua_.rec_kl(child->ref_off_);
          uint32_t check_end = gap_end;
          if (check_end > kl) check_end = kl;
          if (check_end > ref_kl) check_end = ref_kl;
          if (check_end < gap_end ||
              memcmp(k + gap_start, ref_key + gap_start, gap_end - gap_start) != 0)
            return false; // gap mismatch — key not here
        }
        node = child;
        node->extract_at(k, kl, kf, tag);
        continue;
      }
      Group *g = get_group(slot);
      uint32_t mm = scan32(g->tags, tag);
      while (mm) {
        int p = __builtin_ctz(mm);
        if (entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          g->count--;
          if (p < g->count) {
            memmove(&g->tags[p], &g->tags[p + 1], g->count - p);
            memmove(&g->entries[p], &g->entries[p + 1], (g->count - p) * 8);
          }
          g->tags[g->count] = 0;
          g->entries[g->count] = 0;
          count_->fetch_sub(1, std::memory_order_relaxed);
          return true;
        }
        mm &= mm - 1;
      }
      // Overflow scan for positions 32+
      for (uint32_t p = 32; p < g->count; p++) {
        if (g->tags[p] == tag && entry_kf(g->entries[p]) == kf &&
            node->key_eq(entry_off(g->entries[p]), k, kl)) {
          g->count--;
          if (p < g->count) {
            memmove(&g->tags[p], &g->tags[p + 1], g->count - p);
            memmove(&g->entries[p], &g->entries[p + 1], (g->count - p) * 8);
          }
          g->tags[g->count] = 0;
          g->entries[g->count] = 0;
          count_->fetch_sub(1, std::memory_order_relaxed);
          return true;
        }
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
  // Sort group entries by full key
  void sort_group(Group *g) const {
    if (g->sorted || g->count <= 1) {
      g->sorted = 1;
      return;
    }
    uint32_t n = g->count;
    for (uint32_t i = 1; i < n; i++) {
      uint64_t ent = g->entries[i];
      uint8_t tag = g->tags[i];
      uint32_t kf_ent = entry_kf(ent);
      uint32_t off_ent = entry_off(ent);
      int j = (int)i - 1;
      while (j >= 0) {
        uint64_t ej = g->entries[j];
        uint32_t kfj = entry_kf(ej);
        bool larger = false;
        if (kfj > kf_ent) {
          larger = true;
        } else if (kfj == kf_ent) {
          uint32_t offj = entry_off(ej);
          uint16_t l1 = ua_.rec_kl(offj);
          uint16_t l2 = ua_.rec_kl(off_ent);
          const uint8_t *s1 = ua_.rec_key(offj);
          const uint8_t *s2 = ua_.rec_key(off_ent);
          uint16_t min_l = l1 < l2 ? l1 : l2;
          int cmp = memcmp(s1, s2, min_l);
          if (cmp > 0 || (cmp == 0 && l1 > l2)) {
            larger = true;
          }
        }
        if (!larger)
          break;

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
  // ═══════════════════════════════════════════════════════════════
  //  Ordered Cursor (Lexicographic Traversal)
  // ═══════════════════════════════════════════════════════════════
  class Cursor {
    struct Frame {
      const FractalTag2 *node;
      uint32_t dir_idx;
      int32_t group_idx; // -1 = exploring dir, >=0 = iterating group
    };
    Frame stack_[16];
    int depth_ = -1;

    mutable uint8_t key_buf_[4096];
    mutable uint32_t key_len_ = 0;
    mutable bool key_dirty_ = true;
    Group *cur_group_ = nullptr;

    void push(const FractalTag2 *node) { stack_[++depth_] = {node, 0, -1}; }
    void push_at(const FractalTag2 *node, uint32_t di) {
      stack_[++depth_] = {node, di, -1};
    }

    // O(1) dedup via first_idx_/last_idx_ stored on each group/child
    static uint32_t get_first_idx(uintptr_t slot) {
      return is_child(slot) ? get_child(slot)->first_idx_
                            : get_group(slot)->first_idx_;
    }
    static uint32_t get_last_idx(uintptr_t slot) {
      return is_child(slot) ? get_child(slot)->last_idx_
                            : get_group(slot)->last_idx_;
    }

    void set_leaf() {
      auto &f = stack_[depth_];
      cur_group_ = get_group(f.node->dir_[f.dir_idx]);
      key_dirty_ = true;
    }

    void build_key() const {
      auto &f = stack_[depth_];
      Group *g = get_group(f.node->dir_[f.dir_idx]);
      uint32_t off = entry_off(g->entries[f.group_idx]);

      key_len_ = f.node->ua_.rec_kl(off);
      if (key_len_ > 0) {
        memcpy(key_buf_, f.node->ua_.rec_key(off), key_len_);
      }
      key_dirty_ = false;
    }

    // ── Forward scan ──
    bool advance_fwd() {
      while (depth_ >= 0) {
        auto &f = stack_[depth_];
        if (f.group_idx >= 0) {
          f.group_idx++;
          if (f.group_idx < (int32_t)cur_group_->count) {
            key_dirty_ = true;
            return true;
          }
          f.group_idx = -1;
          f.dir_idx++;
        }
        while (f.dir_idx < f.node->ds_sz_) {
          uintptr_t slot = f.node->dir_[f.dir_idx];
          // O(1) dedup: skip unless this is the canonical first index
          if (f.dir_idx != get_first_idx(slot)) {
            f.dir_idx++;
            continue;
          }
          if (is_child(slot)) {
            push(get_child(slot));
            break;
          }
          Group *g = get_group(slot);
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

    // ── Reverse scan ──
    bool advance_rev() {
      while (depth_ >= 0) {
        auto &f = stack_[depth_];
        if (f.group_idx >= 0) {
          f.group_idx--;
          if (f.group_idx >= 0) {
            key_dirty_ = true;
            return true;
          }
          f.group_idx = -1;
        }
        // Scan directory backwards
        bool pushed_child = false;
        while (true) {
          if (f.dir_idx == 0 && f.group_idx < 0)
            break;
          f.dir_idx--;
          uintptr_t slot = f.node->dir_[f.dir_idx];
          // O(1) dedup: skip unless this is the canonical last index
          if (f.dir_idx != get_last_idx(slot)) {
            if (f.dir_idx == 0)
              break;
            continue;
          }
          if (is_child(slot)) {
            FractalTag2 *child = get_child(slot);
            push_at(child, child->ds_sz_);
            pushed_child = true;
            break;
          }
          Group *g = get_group(slot);
          if (g->count > 0) {
            f.node->sort_group(g);
            f.group_idx = g->count - 1;
            set_leaf();
            return true;
          }
          if (f.dir_idx == 0)
            break;
        }
        if (!pushed_child && f.group_idx < 0) {
          depth_--;
          if (depth_ >= 0) {
            stack_[depth_].group_idx = -1;
          }
        }
      }
      return false;
    }

    // ── seek_first into a subtree (descend to leftmost entry) ──
    bool descend_first(const FractalTag2 *node) {
      push(node);
      return advance_fwd();
    }

    // ── seek_last into a subtree (descend to rightmost entry) ──
    bool descend_last(const FractalTag2 *node) {
      push_at(node, node->ds_sz_);
      return advance_rev();
    }

    // ── seek_ge within a level ──
    // full_key/full_kl is the ORIGINAL search key (all levels use their own
    // offset)
    bool seek_ge_at(const FractalTag2 *node, const uint8_t *full_key,
                    uint32_t full_kl) {
      uint32_t kf;
      uint8_t tag;
      node->extract_at(full_key, full_kl, kf, tag);
      uint32_t start = node->di(kf);
      push_at(node, start);
      auto &f = stack_[depth_];

      while (f.dir_idx < node->ds_sz_) {
        uintptr_t slot = node->dir_[f.dir_idx];
        // O(1) dedup: skip non-first for forward scanning beyond start
        if (f.dir_idx > start && f.dir_idx != get_first_idx(slot)) {
          f.dir_idx++;
          continue;
        }

        if (is_child(slot)) {
          FractalTag2 *child = get_child(slot);
          // Always descend into child and seek_ge on next level
          if (seek_ge_at(child, full_key, full_kl))
            return true;
          depth_--; // pop child if exhausted
          f.dir_idx++;
          continue;
        }

        Group *g = get_group(slot);
        if (g->count > 0) {
          node->sort_group(g);
          int pos = find_ge_in_group(node, g, kf, full_key, full_kl);
          if (pos >= 0) {
            f.group_idx = pos;
            set_leaf();
            return true;
          }
        }
        f.dir_idx++;
      }

      // Level exhausted
      depth_--;
      return false;
    }

    // Binary search + suffix compare within sorted group
    static int find_ge_in_group(const FractalTag2 *node, Group *g,
                                uint32_t search_kf, const uint8_t *full_key,
                                uint32_t full_kl) {
      int n = (int)g->count;
      // Binary search: first entry with entry_kf >= search_kf
      int lo = 0, hi = n;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (entry_kf(g->entries[mid]) < search_kf)
          lo = mid + 1;
        else
          hi = mid;
      }
      if (lo >= n)
        return -1;

      // If kf > search_kf → any entry from here is > search key
      if (entry_kf(g->entries[lo]) > search_kf)
        return lo;

      // kf == search_kf → compare full keys
      for (int i = lo; i < n && entry_kf(g->entries[i]) == search_kf; i++) {
        uint32_t off = entry_off(g->entries[i]);
        uint16_t stored_kl = node->ua_.rec_kl(off);
        uint32_t min_kl = stored_kl < full_kl ? stored_kl : full_kl;
        int cmp =
            (min_kl > 0) ? memcmp(node->ua_.rec_key(off), full_key, min_kl) : 0;
        if (cmp > 0 || (cmp == 0 && stored_kl >= full_kl))
          return i;
      }

      // All entries with this kf < search → next distinct kf
      int next = lo;
      while (next < n && entry_kf(g->entries[next]) == search_kf)
        next++;
      return (next < n) ? next : -1;
    }

  public:
    Cursor() : depth_(-1) {}

    // ── Positioning ──

    bool seek_first(const FractalTag2 *root) {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root || root->size() == 0)
        return false;
      push(root);
      return advance_fwd();
    }

    bool seek_last(const FractalTag2 *root) {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root || root->size() == 0)
        return false;
      return descend_last(root);
    }

    bool seek_ge(const FractalTag2 *root, const uint8_t *k, uint32_t kl) {
      depth_ = -1;
      cur_group_ = nullptr;
      if (!root || root->size() == 0)
        return false;
      return seek_ge_at(root, k, kl);
    }

    bool seek_prefix(const FractalTag2 *root, const uint8_t *prefix,
                     uint32_t pl) {
      // seek_prefix = seek_ge(prefix) — first key with this prefix IS the first
      // key >= prefix
      return seek_ge(root, prefix, pl);
    }

    // ── Movement ──

    bool next() {
      if (depth_ < 0)
        return false;
      return advance_fwd();
    }

    bool prev() {
      if (depth_ < 0)
        return false;
      return advance_rev();
    }

    // ── Accessors ──

    bool valid() const { return depth_ >= 0 && cur_group_ != nullptr; }

    const uint8_t *key() const {
      if (key_dirty_)
        build_key();
      return key_buf_;
    }
    uint32_t key_len() const {
      if (key_dirty_)
        build_key();
      return key_len_;
    }

    uint32_t val() const {
      if (!cur_group_)
        return 0;
      auto &f = stack_[depth_];
      return f.node->ua_.rec_val(entry_off(cur_group_->entries[f.group_idx]));
    }

    // ── Mutation ──

    void update(uint32_t v) {
      auto &f = stack_[depth_];
      f.node->ua_.rec_set_val(entry_off(cur_group_->entries[f.group_idx]), v);
    }

    // ── Prefix check (for prefix scans) ──
    bool has_prefix(const uint8_t *prefix, uint32_t pl) const {
      if (key_dirty_)
        build_key();
      return key_len_ >= pl && memcmp(key_buf_, prefix, pl) == 0;
    }
  };

  Cursor create_cursor() const { return Cursor(); }

public:
  void stats(int indent = 0) const {
    uint32_t n_grp = 0, n_ent = 0;
    for (uint32_t i = 0; i < ds_sz_; i++) {
      if (is_child(dir_[i]))
        continue;
      Group *g = get_group(dir_[i]);
      bool dup = false;
      for (uint32_t j = 0; j < i; j++)
        if (dir_[j] == dir_[i]) {
          dup = true;
          break;
        }
      if (!dup) {
        n_grp++;
        n_ent += g->count;
      }
    }
    uint32_t uc = 0;
    std::vector<FractalTag2 *> seen;
    for (uint32_t i = 0; i < ds_sz_; i++) {
      if (!is_child(dir_[i]))
        continue;
      FractalTag2 *c = get_child(dir_[i]);
      bool found = false;
      for (auto *s : seen)
        if (s == c) {
          found = true;
          break;
        }
      if (!found) {
        seen.push_back(c);
        uc++;
      }
    }
    printf("  D%d: dir=%u grp=%u ent=%u avg=%.1f children=%u arena=%.1fMB\n",
           indent, ds_sz_, n_grp, n_ent, n_grp ? n_ent / (double)n_grp : 0.0,
           uc, ua_.used() / (1024.0 * 1024.0));
    for (auto *c : seen)
      c->stats(indent + 1);
  }
};

// ═══════════════════════════════════════════════════════════════
//  Original TM10L2 (baseline) — CAP=32
// ═══════════════════════════════════════════════════════════════
class TM10L2 {
  static constexpr uint32_t CAP = 32;
  struct Entry {
    uint64_t kh;
    uint32_t val, off;
  };
  struct alignas(16) Group {
    uint8_t tags[32];
    Entry entries[32];
    uint8_t count, local_depth, sorted, _p[5];
  };
  Arena arena_;
  SB sb_;
  uintptr_t *dir_;
  uint32_t gd_, ds_sz_, kl_;
  uint64_t tkh(const uint8_t *k) const {
    uint64_t v = 0;
    memcpy(&v, k, kl_ < 8 ? kl_ : 8);
    return __builtin_bswap64(v);
  }
#if defined(__ARM_NEON)
  static uint8_t mk_tag(uint64_t kh) {
    return (uint8_t)((__crc32cd(0, kh) >> 25) | 0x80);
  }
#else
  static uint8_t mk_tag(uint64_t kh) {
    return (uint8_t)((_mm_crc32_u64(0, kh) >> 25) | 0x80);
  }
#endif
  uint32_t di(uint64_t kh) const {
    return gd_ ? (uint32_t)(kh >> (64 - gd_)) : 0;
  }
  Group *ag(uint8_t ld) {
    auto *g = (Group *)arena_.alloc16(sizeof(Group));
    memset(g, 0, sizeof(Group));
    g->local_depth = ld;
    g->sorted = 1;
    return g;
  }
  void dd() {
    uint32_t ns = ds_sz_ * 2;
    auto *nd = (uintptr_t *)arena_.alloc(ns * sizeof(uintptr_t));
    for (uint32_t i = 0; i < ds_sz_; i++) {
      nd[2 * i] = dir_[i];
      nd[2 * i + 1] = dir_[i];
    }
    dir_ = nd;
    ds_sz_ = ns;
    gd_++;
  }
  void sp(uint32_t dx) {
    Group *o = ((Group *)dir_[dx]);
    uint8_t ld = o->local_depth;
    uint64_t diff = 0;
    for (uint32_t i = 1; i < o->count; i++)
      diff |= o->entries[0].kh ^ o->entries[i].kh;
    uint64_t mask = (ld == 0) ? ~0ULL : ((1ULL << (64 - ld)) - 1);
    diff &= mask;
    if (!diff) {
      return;
    }
    uint8_t sl = (uint8_t)__builtin_clzll(diff);
    while (sl >= gd_) {
      dd();
      dx <<= 1;
    }
    Group *g0 = ag(sl + 1), *g1 = ag(sl + 1);
    for (uint32_t i = 0; i < o->count; i++) {
      uint8_t s = (o->entries[i].kh >> (63 - sl)) & 1;
      Group *d = s ? g1 : g0;
      uint32_t p = d->count;
      d->tags[p] = o->tags[i];
      d->entries[p] = o->entries[i];
      d->count++;
    }
    uint32_t stride = 1u << (gd_ - ld), st = dx & ~(stride - 1);
    for (uint32_t i = st; i < st + stride; i++) {
      uint8_t s = (i >> (gd_ - 1 - sl)) & 1;
      dir_[i] = (uintptr_t)(s ? g1 : g0);
    }
  }

public:
  TM10L2(uint32_t kl) : arena_(2ULL << 30), kl_(kl) {
    sb_.init(arena_);
    gd_ = 0;
    ds_sz_ = 1;
    dir_ = (uintptr_t *)arena_.alloc(sizeof(uintptr_t));
    dir_[0] = (uintptr_t)ag(0);
  }
  void insert(const uint8_t *k, uint32_t v) {
    uint64_t kh = tkh(k);
    uint8_t tag = mk_tag(kh);
    uint32_t so = (kl_ > 8) ? sb_.al(arena_, k, kl_, v) : 0;
    for (int a = 0; a < 64; a++) {
      uint32_t dx = di(kh);
      Group *g = (Group *)dir_[dx];
      uint32_t mm = scan32(g->tags, tag);
      while (mm) {
        int p = __builtin_ctz(mm);
        if (g->entries[p].kh == kh) {
          bool m = (kl_ <= 8) ||
                   !memcmp(sb_.suf(g->entries[p].off) + 8, k + 8, kl_ - 8);
          if (m) {
            g->entries[p].val = v;
            return;
          }
        }
        mm &= mm - 1;
      }
      if (g->count >= CAP) {
        sp(dx);
        continue;
      }
      uint32_t s = g->count;
      g->tags[s] = tag;
      g->entries[s] = {kh, v, so};
      g->count++;
      g->sorted = 0;
      return;
    }
  }
  bool get(const uint8_t *k, uint32_t &out) const {
    uint64_t kh = tkh(k);
    Group *g = (Group *)dir_[di(kh)];
    uint32_t mm = scan32(g->tags, mk_tag(kh));
    while (mm) {
      int p = __builtin_ctz(mm);
      if (g->entries[p].kh == kh) {
        bool m = (kl_ <= 8) ||
                 !memcmp(sb_.suf(g->entries[p].off) + 8, k + 8, kl_ - 8);
        if (m) {
          out = g->entries[p].val;
          return true;
        }
      }
      mm &= mm - 1;
    }
    return false;
  }
  void stats() const {
    uint32_t ng = 0, ne = 0;
    for (uint32_t i = 0; i < ds_sz_; i++) {
      Group *g = (Group *)dir_[i];
      bool d = false;
      for (uint32_t j = 0; j < i; j++)
        if (dir_[j] == dir_[i]) {
          d = true;
          break;
        }
      if (!d) {
        ng++;
        ne += g->count;
      }
    }
    printf("  dir=%u grp=%u ent=%u avg=%.1f arena=%.1fMB\n", ds_sz_, ng, ne,
           ng ? ne / (double)ng : 0.0, arena_.used() / (1024.0 * 1024.0));
  }
};

// ═══════════════════════════════════════════════════════════════
struct FT2W {
  UArena ua;
  FractalTag2 *ft;
  uint32_t kl_;
  std::atomic<uint64_t> count;
  FT2W(uint32_t kl) : kl_(kl), count(0) {
    ft = new (ua.alloc(sizeof(FractalTag2))) FractalTag2(ua, &count);
  }
  void reset() {
    ua.reset();
    count.store(0);
    ft = new (ua.alloc(sizeof(FractalTag2))) FractalTag2(ua, &count);
  }
  void insert(const uint8_t *k, uint32_t v) { ft->insert(k, kl_, v); }
  void insert(const uint8_t *k, uint32_t kl, uint32_t v) {
    ft->insert(k, kl, v);
  }
  bool get(const uint8_t *k, uint32_t &out) const {
    return ft->get(k, kl_, out);
  }
  bool get(const uint8_t *k, uint32_t kl, uint32_t &out) const {
    return ft->get(k, kl, out);
  }
  bool contains(const uint8_t *k) const { return ft->contains(k, kl_); }
  bool contains(const uint8_t *k, uint32_t kl) const {
    return ft->contains(k, kl);
  }
  bool erase(const uint8_t *k) { return ft->erase(k, kl_); }
  bool erase(const uint8_t *k, uint32_t kl) { return ft->erase(k, kl); }
  void clear() { reset(); }

  struct CursorWrapper {
    FractalTag2::Cursor c;
    const FractalTag2 *root;
    bool seek_first() { return c.seek_first(root); }
    bool seek_last() { return c.seek_last(root); }
    bool seek_ge(const uint8_t *k, uint32_t kl) {
      return c.seek_ge(root, k, kl);
    }
    bool seek_prefix(const uint8_t *p, uint32_t pl) {
      return c.seek_prefix(root, p, pl);
    }
    bool next() { return c.next(); }
    bool prev() { return c.prev(); }
    bool valid() const { return c.valid(); }
    const uint8_t *key() const { return c.key(); }
    uint32_t key_len() const { return c.key_len(); }
    uint32_t val() const { return c.val(); }
    void update(uint32_t v) { c.update(v); }
    bool has_prefix(const uint8_t *p, uint32_t pl) const {
      return c.has_prefix(p, pl);
    }
  };
  CursorWrapper create_cursor() const { return {ft->create_cursor(), ft}; }

  uint64_t size() const { return ft->size(); }
  bool empty() const { return ft->empty(); }
  void stats() const { ft->stats(); }
  uint64_t used() const { return ua.used(); }
};

static void gen_keys(std::vector<uint8_t> &out, uint32_t n, uint32_t kl,
                     bool seq) {
  out.resize((size_t)n * kl, 0);
  if (seq) {
    for (uint32_t i = 0; i < n; i++) {
      uint64_t v = i + 1;
      memcpy(&out[(size_t)i * kl], &v, 8 < kl ? 8 : kl);
    }
  } else {
    arc4random_buf(out.data(), out.size());
  }
}

#include <algorithm>

static void bench_mega_cursor() {
  printf("\n  ── MEGA CURSOR BENCHMARK ──\n");
  uint32_t N = 1000000;

  // ─── Test 1: Random 16B keys (baseline, 2 levels max) ───
  {
    FT2W m(0);
    printf("  [1] 1M random 16B keys\n");
    std::vector<uint8_t> keys(N * 16);
    arc4random_buf(keys.data(), keys.size());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < N; i++)
      m.insert(&keys[i * 16], 16, i);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("      Insert: %.1f ns/key\n",
           std::chrono::duration<double, std::nano>(t1 - t0).count() / N);

    auto c = m.create_cursor();
    // Sort+Iter
    auto t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    uint32_t cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("      Sort+Iter: %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt,
           cnt);

    // Pure iterate (pre-sorted)
    t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Pure Iter: %.1f ns/key\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt);

    // Reverse
    t2 = std::chrono::high_resolution_clock::now();
    c.seek_last();
    cnt = 0;
    while (c.valid()) {
      cnt++;
      c.prev();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Reverse:   %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt,
           cnt);

    // Verify sorted order
    c.seek_first();
    uint32_t bad = 0;
    uint8_t prev[16];
    memset(prev, 0, 16);
    while (c.valid()) {
      if (memcmp(c.key(), prev, 16) < 0)
        bad++;
      memcpy(prev, c.key(), 16);
      c.next();
    }
    printf("      Order check: %s\n", bad == 0 ? "✅" : "❌");
  }

  // ─── Test 2: Structured 8B binary keys with shared prefixes ───
  {
    FT2W m(0);
    printf("  [2] 1M structured 8B keys (256 groups × 3906 items)\n");
    std::vector<uint8_t> keys(N * 8);
    for (uint32_t i = 0; i < N; i++) {
      uint32_t grp = __builtin_bswap32(i % 256);
      uint32_t uid = __builtin_bswap32(i);
      memcpy(&keys[i * 8], &grp, 4);
      memcpy(&keys[i * 8 + 4], &uid, 4);
    }
    for (uint32_t i = N - 1; i > 0; i--) {
      uint32_t j = arc4random_uniform(i + 1);
      for (int b = 0; b < 8; b++)
        std::swap(keys[i * 8 + b], keys[j * 8 + b]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < N; i++)
      m.insert(&keys[i * 8], 8, i);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("      Insert: %.1f ns/key, size=%llu\n",
           std::chrono::duration<double, std::nano>(t1 - t0).count() / N,
           m.size());
    // Verify data integrity via get
    uint32_t get_ok = 0;
    for (uint32_t i = 0; i < N; i++) {
      uint32_t v;
      if (m.get(&keys[i * 8], 8, v))
        get_ok++;
    }
    printf("      Get check: %u/%u %s\n", get_ok, N, get_ok == N ? "✅" : "❌");

    auto c = m.create_cursor();
    auto t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    uint32_t cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("      Sort+Iter: %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt,
           cnt);

    // Pure
    t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Pure Iter: %.1f ns/key\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt);

    // Prefix scan: find all items in group 42
    uint32_t pfx_val = __builtin_bswap32(42);
    t2 = std::chrono::high_resolution_clock::now();
    c.seek_prefix((const uint8_t *)&pfx_val, 4);
    cnt = 0;
    while (c.valid() && c.has_prefix((const uint8_t *)&pfx_val, 4)) {
      cnt++;
      c.next();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Prefix scan (grp 42): %.3f ms, %u items\n",
           std::chrono::duration<double, std::milli>(t3 - t2).count(), cnt);

    // seek_ge to middle
    uint32_t seek_grp = __builtin_bswap32(128);
    uint32_t seek_uid = __builtin_bswap32(500000);
    uint8_t seek_key[8];
    memcpy(seek_key, &seek_grp, 4);
    memcpy(seek_key + 4, &seek_uid, 4);
    t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < 10000; r++)
      c.seek_ge(seek_key, 8);
    t3 = std::chrono::high_resolution_clock::now();
    printf("      seek_ge (10K calls): %.1f ns/call\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / 10000);

    // Reverse
    t2 = std::chrono::high_resolution_clock::now();
    c.seek_last();
    cnt = 0;
    while (c.valid()) {
      cnt++;
      c.prev();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Reverse:   %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() / cnt,
           cnt);

    // m.stats(); // DEBUG: disabled — hanging on deep tree
  }

  // ─── Test 3: Deep fractal (same 4B prefix, forces spawn) ───
  {
    FT2W m(0);
    uint32_t ND = 10000;
    printf("  [3] %u deep fractal keys (shared 4B prefix)\n", ND);
    dbg_log = fopen("log.txt", "w");
    for (uint32_t i = 0; i < ND; i++) {
      uint8_t k[12];
      memcpy(k, "DEEP", 4);
      uint32_t be_i = __builtin_bswap32(i);
      memcpy(k + 4, &be_i, 4);
      uint32_t be_i2 = __builtin_bswap32(i * 7);
      memcpy(k + 8, &be_i2, 4);
      m.insert(k, 12, i);
    }
    if (dbg_log) {
      fclose(dbg_log);
      dbg_log = nullptr;
    }
    printf("      Inserted: size=%llu (expect %u) arena=%.1fMB\n", m.size(), ND,
           m.used() / (1024.0 * 1024.0));
    // Verify every key
    dbg_log = fopen("get_log.txt", "w");
    uint32_t ok = 0, bad = 0, miss = 0;
    for (uint32_t i = 0; i < ND; i++) {
      uint8_t k[12];
      memcpy(k, "DEEP", 4);
      uint32_t be_i = __builtin_bswap32(i);
      memcpy(k + 4, &be_i, 4);
      uint32_t be_i2 = __builtin_bswap32(i * 7);
      memcpy(k + 8, &be_i2, 4);
      uint32_t v;
      if (m.get(k, 12, v)) {
        if (v == i)
          ok++;
        else {
          bad++;
          if (bad <= 5)
            printf("      BAD: i=%u got v=%u\n", i, v);
        }
      } else {
        miss++;
        if (miss <= 5)
          printf("      MISS: i=%u\n", i);
      }
    }
    printf("      Verify: ok=%u bad=%u miss=%u %s\n", ok, bad, miss,
           (ok == ND) ? "✅" : "❌");
    if (dbg_log) { fclose(dbg_log); dbg_log = nullptr; }
    auto c = m.create_cursor();
    auto t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    uint32_t cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    printf("      Sort+Iter: %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() /
               (cnt ? cnt : 1),
           cnt);

    t2 = std::chrono::high_resolution_clock::now();
    c.seek_first();
    cnt = 0;
    while (c.valid()) {
      cnt++;
      c.next();
    }
    t3 = std::chrono::high_resolution_clock::now();
    printf("      Pure Iter: %.1f ns/key (%u found)\n",
           std::chrono::duration<double, std::nano>(t3 - t2).count() /
               (cnt ? cnt : 1),
           cnt);

    printf("      Count check: %s (expected %u)\n", cnt == ND ? "✅" : "❌",
           ND);
    // m.stats(); // skipped for now
  }

  printf("  ───────────────────────────────\n\n");
}

int main() {
  printf(
      "\n "
      "═══════════════════════════════════════════════════════════════════\n");
  printf("  FractalTag2 (4B kf, ZERO CRC, CAP=16) vs Original TM10L2\n");
  printf("  F2: 4B kf, key-byte tag, scan16, 216B groups, fractal recursion\n");
  printf("  OG: flat extendible hash, 8B kh, CRC tag, 552B groups\n");
  printf(" ═══════════════════════════════════════════════════════════════════"
         "\n\n");
  printf("                │       Original │    FractalTag2 │       Δ │\n");
  printf("  ──────────────┼────────────────┼────────────────┼──────────┤\n");

  struct R {
    double ins, get;
  };
  struct T {
    uint32_t N;
    int it;
    const char *l;
  };
  T tiers[] = {{100000, 20, "100K"}, {1000000, 5, "1M"}, {5000000, 1, "5M"}};
  uint32_t kls[] = {8, 16, 64};

  for (auto &t : tiers) {
    for (uint32_t kl : kls) {
      for (bool seq : {false, true}) {
        std::vector<uint8_t> keys;
        gen_keys(keys, t.N, kl, seq);
        R bo{1e18, 1e18}, br{1e18, 1e18};
        double f2c = 1e18; // F2 cursor sort+iterate
        double f2p = 1e18; // F2 cursor pure iterate (pre-sorted)

        for (int i = 0; i < t.it; i++) {
          TM10L2 m(kl);
          auto t0 = std::chrono::high_resolution_clock::now();
          for (uint32_t j = 0; j < t.N; j++)
            m.insert(&keys[(size_t)j * kl], j);
          auto t1 = std::chrono::high_resolution_clock::now();
          uint32_t v;
          uint64_t a = 0;
          auto t2 = std::chrono::high_resolution_clock::now();
          for (uint32_t j = 0; j < t.N; j++) {
            if (m.get(&keys[(size_t)j * kl], v))
              a += v;
          }
          auto t3 = std::chrono::high_resolution_clock::now();
          volatile uint64_t s = a;
          (void)s;
          R r = {std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                         .count() /
                     (double)t.N,
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2)
                         .count() /
                     (double)t.N};
          if (r.ins < bo.ins)
            bo.ins = r.ins;
          if (r.get < bo.get)
            bo.get = r.get;
        }
        {
          FT2W m(kl);
          for (int i = 0; i < t.it; i++) {
            if (i > 0)
              m.reset();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (uint32_t j = 0; j < t.N; j++)
              m.insert(&keys[(size_t)j * kl], j);
            auto t1 = std::chrono::high_resolution_clock::now();
            uint32_t v;
            uint64_t a = 0;
            auto t2 = std::chrono::high_resolution_clock::now();
            for (uint32_t j = 0; j < t.N; j++) {
              if (m.get(&keys[(size_t)j * kl], v))
                a += v;
            }
            auto t3 = std::chrono::high_resolution_clock::now();

            // Cursor Benchmark — Pass 1: sort + iterate (includes lazy sort
            // cost)
            auto c0 = std::chrono::high_resolution_clock::now();
            auto c = m.create_cursor();
            uint64_t ca = 0;
            if (c.seek_first()) {
              do {
                ca += c.val();
              } while (c.next());
            }
            auto c1 = std::chrono::high_resolution_clock::now();
            volatile uint64_t cs = ca;
            (void)cs;

            // Cursor Benchmark — Pass 2: pure iterate (all groups already
            // sorted)
            auto c2 = std::chrono::high_resolution_clock::now();
            auto c3w = m.create_cursor();
            uint64_t ca2 = 0;
            if (c3w.seek_first()) {
              do {
                ca2 += c3w.val();
              } while (c3w.next());
            }
            auto c3 = std::chrono::high_resolution_clock::now();
            volatile uint64_t cs2 = ca2;
            (void)cs2;

            volatile uint64_t s = a;
            (void)s;
            R r = {std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                           .count() /
                       (double)t.N,
                   std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2)
                           .count() /
                       (double)t.N};
            if (r.ins < br.ins)
              br.ins = r.ins;
            if (r.get < br.get)
              br.get = r.get;
            double sort_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0)
                    .count() /
                (double)t.N;
            double pure_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(c3 - c2)
                    .count() /
                (double)t.N;
            if (sort_ns < f2c)
              f2c = sort_ns;
            if (pure_ns < f2p)
              f2p = pure_ns;
          }
        }

        double d1 = (bo.ins - br.ins) / bo.ins * 100,
               d2 = (bo.get - br.get) / bo.get * 100;
        printf("  %s %s %3uB │ I:%5.1f G:%5.1f │ I:%5.1f G:%5.1f │ I:%+4.0f%% "
               "G:%+4.0f%% │ Sort+Iter:%5.1f Pure:%5.1fns │\n",
               t.l, seq ? "SEQ" : "RND", kl, bo.ins, bo.get, br.ins, br.get, d1,
               d2, f2c, f2p);
      }
    }
    printf("  ──────────────┼────────────────┼────────────────┼──────────┤\n");
  }

  printf("\n  ── PATHOLOGICAL ──\n");
  auto run_patho = [](uint32_t N, uint32_t kl, uint32_t n_pfx) {
    FT2W m(kl);
    std::vector<uint8_t> keys(N * kl, 0);
    for (uint32_t i = 0; i < N; i++) {
      uint32_t pfx = i % n_pfx;
      memset(&keys[(size_t)i * kl], 'A' + pfx, 8);
      uint64_t suf = i / n_pfx;
      memcpy(&keys[(size_t)i * kl + 8], &suf, 8 < (kl - 8) ? 8 : (kl - 8));
    }
    for (uint32_t i = 0; i < N; i++)
      m.insert(&keys[(size_t)i * kl], i);
    uint32_t found = 0;
    for (uint32_t i = 0; i < N; i++) {
      uint32_t v;
      if (m.get(&keys[(size_t)i * kl], v) && v == i)
        found++;
    }
    printf("  %uK, %uB, %u pfx: Found:%u/%u %s\n", N / 1000, kl, n_pfx, found,
           N, found == N ? "✅" : "❌");
    // m.stats();
  };
  run_patho(10000, 32, 1);
  run_patho(50000, 32, 4);
  run_patho(100000, 32, 1);

  printf("\n  ── CORRECTNESS VERIFICATION (exact per-key) ──\n");
  for (uint32_t kl : {8u, 16u, 64u}) {
    for (bool seq : {false, true}) {
      uint32_t N = 1000000;
      std::vector<uint8_t> keys;
      gen_keys(keys, N, kl, seq);
      FT2W m(kl);
      for (uint32_t i = 0; i < N; i++)
        m.insert(&keys[(size_t)i * kl], i);
      uint32_t ok = 0, bad_val = 0, missing = 0;
      for (uint32_t i = 0; i < N; i++) {
        uint32_t v;
        if (m.get(&keys[(size_t)i * kl], v)) {
          if (v == i)
            ok++;
          else
            bad_val++;
        } else
          missing++;
      }
      printf("  1M %s %2uB: ok=%u bad=%u miss=%u %s\n", seq ? "SEQ" : "RND", kl,
             ok, bad_val, missing,
             (ok == N && bad_val == 0 && missing == 0) ? "✅" : "❌");
    }
  }
  // ═══════════════════════════════════════════════════════════════
  // EDGE CASE TESTS — variable-length key validation
  // ═══════════════════════════════════════════════════════════════
  printf("\n  ── EDGE CASE TESTS ──\n");
  int pass = 0, fail = 0;
  auto CHECK = [&](const char *name, bool cond) {
    if (cond) {
      pass++;
    } else {
      fail++;
      printf("  ❌ FAIL: %s\n", name);
    }
  };

  // T1: Mixed-length keys in same map
  {
    FT2W m(0); // kl_=0, use variable-length overloads
    uint8_t k4[] = {1, 2, 3, 4};
    uint8_t k8[] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t k16[16];
    memset(k16, 0, 16);
    memcpy(k16, k4, 4);
    m.insert(k4, 4, 100);
    m.insert(k8, 8, 200);
    m.insert(k16, 16, 300);
    uint32_t v;
    CHECK("T1a: 4B key found", m.get(k4, 4, v) && v == 100);
    CHECK("T1b: 8B key found", m.get(k8, 8, v) && v == 200);
    CHECK("T1c: 16B key found", m.get(k16, 16, v) && v == 300);
    // Verify they don't interfere
    CHECK("T1d: 4B still correct", m.get(k4, 4, v) && v == 100);
  }

  // T2: Prefix keys — "abcd" vs "abcde" vs "abcdef"
  {
    FT2W m(0);
    uint8_t k4[] = "abcd";   // 4 bytes (not including null)
    uint8_t k5[] = "abcde";  // 5 bytes
    uint8_t k6[] = "abcdef"; // 6 bytes
    m.insert(k4, 4, 10);
    m.insert(k5, 5, 20);
    m.insert(k6, 6, 30);
    uint32_t v;
    CHECK("T2a: prefix 4B found", m.get(k4, 4, v) && v == 10);
    CHECK("T2b: prefix 5B found", m.get(k5, 5, v) && v == 20);
    CHECK("T2c: prefix 6B found", m.get(k6, 6, v) && v == 30);
    // 7B key should NOT be found
    uint8_t k7[] = "abcdefg";
    CHECK("T2d: 7B not found", !m.get(k7, 7, v));
  }

  // T3: Very short keys (1, 2, 3 bytes)
  {
    FT2W m(0);
    uint8_t k1[] = {0x41};
    uint8_t k2[] = {0x41, 0x42};
    uint8_t k3[] = {0x41, 0x42, 0x43};
    m.insert(k1, 1, 1);
    m.insert(k2, 2, 2);
    m.insert(k3, 3, 3);
    uint32_t v;
    CHECK("T3a: 1B key found", m.get(k1, 1, v) && v == 1);
    CHECK("T3b: 2B key found", m.get(k2, 2, v) && v == 2);
    CHECK("T3c: 3B key found", m.get(k3, 3, v) && v == 3);
    // Ensure 1B != 2B prefix
    CHECK("T3d: 1B != 2B prefix", m.get(k1, 1, v) && v == 1);
  }

  // T4: Key update — insert same key twice, verify value changed
  {
    FT2W m(0);
    uint8_t k[] = "testkey!";
    m.insert(k, 8, 42);
    uint32_t v;
    CHECK("T4a: first insert", m.get(k, 8, v) && v == 42);
    m.insert(k, 8, 99);
    CHECK("T4b: update value", m.get(k, 8, v) && v == 99);
  }

  // T5: Keys differing only in length — same content, different kl
  {
    FT2W m(0);
    uint8_t k4[] = {0, 0, 0, 0};
    uint8_t k5[] = {0, 0, 0, 0, 0};
    uint8_t k8[] = {0, 0, 0, 0, 0, 0, 0, 0};
    m.insert(k4, 4, 400);
    m.insert(k5, 5, 500);
    m.insert(k8, 8, 800);
    uint32_t v;
    CHECK("T5a: all-zero 4B", m.get(k4, 4, v) && v == 400);
    CHECK("T5b: all-zero 5B", m.get(k5, 5, v) && v == 500);
    CHECK("T5c: all-zero 8B", m.get(k8, 8, v) && v == 800);
  }

  // T6: Keys differing only in last byte
  {
    FT2W m(0);
    uint8_t ka[16] = {0};
    ka[15] = 0x00;
    uint8_t kb[16] = {0};
    kb[15] = 0x01;
    uint8_t kc[16] = {0};
    kc[15] = 0xFF;
    m.insert(ka, 16, 10);
    m.insert(kb, 16, 20);
    m.insert(kc, 16, 30);
    uint32_t v;
    CHECK("T6a: last byte 0x00", m.get(ka, 16, v) && v == 10);
    CHECK("T6b: last byte 0x01", m.get(kb, 16, v) && v == 20);
    CHECK("T6c: last byte 0xFF", m.get(kc, 16, v) && v == 30);
  }

  // T7: Exactly 4-byte keys (kf covers entire key, suf_len=0)
  {
    FT2W m(0);
    uint8_t ka[] = {1, 2, 3, 4};
    uint8_t kb[] = {1, 2, 3, 5};
    uint8_t kc[] = {5, 4, 3, 2};
    m.insert(ka, 4, 111);
    m.insert(kb, 4, 222);
    m.insert(kc, 4, 333);
    uint32_t v;
    CHECK("T7a: 4B key a", m.get(ka, 4, v) && v == 111);
    CHECK("T7b: 4B key b", m.get(kb, 4, v) && v == 222);
    CHECK("T7c: 4B key c", m.get(kc, 4, v) && v == 333);
  }

  // T8: Many variable-length keys (stress test)
  {
    FT2W m(0);
    uint32_t N = 1000000;
    std::vector<std::vector<uint8_t>> keys(N);
    for (uint32_t i = 0; i < N; i++) {
      uint32_t kl = 4 + (i % 61); // lengths 4..64
      keys[i].resize(kl, 0);
      memcpy(keys[i].data(), &i, 4); // unique prefix from i
      for (uint32_t j = 4; j < kl; j++)
        keys[i][j] = (uint8_t)(i + j);
    }
    for (uint32_t i = 0; i < N; i++)
      m.insert(keys[i].data(), keys[i].size(), i);
    uint32_t ok = 0;
    for (uint32_t i = 0; i < N; i++) {
      uint32_t v;
      if (m.get(keys[i].data(), keys[i].size(), v) && v == i)
        ok++;
    }
    CHECK("T8: 10K variable-length", ok == N);
  }

  // T9: Prefix key that IS another key (4B == prefix of 8B)
  {
    FT2W m(0);
    uint8_t k[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    m.insert(k, 4, 1000); // insert first 4 bytes as a key
    m.insert(k, 8, 2000); // insert all 8 bytes as a different key
    uint32_t v;
    CHECK("T9a: 4B prefix key", m.get(k, 4, v) && v == 1000);
    CHECK("T9b: 8B full key", m.get(k, 8, v) && v == 2000);
    // Update 4B, verify 8B unchanged
    m.insert(k, 4, 1111);
    CHECK("T9c: 4B updated", m.get(k, 4, v) && v == 1111);
    CHECK("T9d: 8B unchanged", m.get(k, 8, v) && v == 2000);
  }

  // T10: Long key (256 bytes)
  {
    FT2W m(0);
    uint8_t klong[256];
    for (int i = 0; i < 256; i++)
      klong[i] = (uint8_t)i;
    uint8_t klong2[256];
    memcpy(klong2, klong, 256);
    klong2[255] ^= 0xFF;
    m.insert(klong, 256, 9999);
    m.insert(klong2, 256, 8888);
    uint32_t v;
    CHECK("T10a: 256B key a", m.get(klong, 256, v) && v == 9999);
    CHECK("T10b: 256B key b", m.get(klong2, 256, v) && v == 8888);
  }

  // T11: Basic erase
  {
    FT2W m(0);
    uint8_t k[] = "erasetest";
    m.insert(k, 9, 42);
    uint32_t v;
    CHECK("T11a: key exists", m.get(k, 9, v) && v == 42);
    CHECK("T11b: erase returns true", m.erase(k, 9));
    CHECK("T11c: key gone", !m.get(k, 9, v));
    CHECK("T11d: double erase false", !m.erase(k, 9));
  }

  // T12: Erase then reinsert
  {
    FT2W m(0);
    uint8_t k[] = {10, 20, 30, 40, 50, 60, 70, 80};
    m.insert(k, 8, 100);
    uint32_t v;
    CHECK("T12a: pre-erase", m.get(k, 8, v) && v == 100);
    m.erase(k, 8);
    CHECK("T12b: erased", !m.get(k, 8, v));
    m.insert(k, 8, 200);
    CHECK("T12c: reinserted", m.get(k, 8, v) && v == 200);
  }

  // T13: Erase doesn't affect other keys
  {
    FT2W m(0);
    uint8_t ka[] = {1, 2, 3, 4};
    uint8_t kb[] = {5, 6, 7, 8};
    uint8_t kc[] = {9, 10, 11, 12};
    m.insert(ka, 4, 10);
    m.insert(kb, 4, 20);
    m.insert(kc, 4, 30);
    m.erase(kb, 4);
    uint32_t v;
    CHECK("T13a: ka survives", m.get(ka, 4, v) && v == 10);
    CHECK("T13b: kb erased", !m.get(kb, 4, v));
    CHECK("T13c: kc survives", m.get(kc, 4, v) && v == 30);
  }

  // T14: Size tracking
  {
    FT2W m(0);
    CHECK("T14a: empty", m.empty() && m.size() == 0);
    uint8_t k1[] = {1, 2, 3, 4};
    uint8_t k2[] = {5, 6, 7, 8};
    uint8_t k3[] = {9, 10, 11, 12};
    m.insert(k1, 4, 1);
    CHECK("T14b: size=1", m.size() == 1);
    m.insert(k2, 4, 2);
    m.insert(k3, 4, 3);
    CHECK("T14c: size=3", m.size() == 3);
    // Update existing key — size should NOT change
    m.insert(k2, 4, 99);
    CHECK("T14d: update no change", m.size() == 3);
    m.erase(k1, 4);
    CHECK("T14e: size=2 after erase", m.size() == 2);
    m.erase(k2, 4);
    m.erase(k3, 4);
    CHECK("T14f: empty again", m.empty());
    // Erase non-existent — size stays 0
    m.erase(k1, 4);
    CHECK("T14g: stays 0", m.size() == 0);
  }

  // T15: Variable-length erase
  {
    FT2W m(0);
    uint8_t k4[] = {0xDE, 0xAD, 0xBE, 0xEF};
    uint8_t k8[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
    m.insert(k4, 4, 100);
    m.insert(k8, 8, 200);
    CHECK("T15a: size=2", m.size() == 2);
    m.erase(k4, 4);
    uint32_t v;
    CHECK("T15b: 4B erased", !m.get(k4, 4, v));
    CHECK("T15c: 8B survives", m.get(k8, 8, v) && v == 200);
    CHECK("T15d: size=1", m.size() == 1);
  }

  // T16: Erase stress — insert 1000, erase all, verify empty
  {
    FT2W m(0);
    uint32_t N = 1000;
    std::vector<std::vector<uint8_t>> keys(N);
    for (uint32_t i = 0; i < N; i++) {
      uint32_t kl = 8 + (i % 25);
      keys[i].resize(kl, 0);
      memcpy(keys[i].data(), &i, 4);
      for (uint32_t j = 4; j < kl; j++)
        keys[i][j] = (uint8_t)(i ^ j);
    }
    for (uint32_t i = 0; i < N; i++)
      m.insert(keys[i].data(), keys[i].size(), i);
    CHECK("T16a: size=1000", m.size() == 1000);
    // Erase every other key
    for (uint32_t i = 0; i < N; i += 2)
      m.erase(keys[i].data(), keys[i].size());
    CHECK("T16b: size=500", m.size() == 500);
    // Verify even keys gone, odd keys survive
    uint32_t ok_gone = 0, ok_live = 0;
    for (uint32_t i = 0; i < N; i++) {
      uint32_t v;
      bool found = m.get(keys[i].data(), keys[i].size(), v);
      if (i % 2 == 0) {
        if (!found)
          ok_gone++;
      } else {
        if (found && v == i)
          ok_live++;
      }
    }
    CHECK("T16c: evens erased", ok_gone == 500);
    CHECK("T16d: odds survive", ok_live == 500);
    // Erase rest
    for (uint32_t i = 1; i < N; i += 2)
      m.erase(keys[i].data(), keys[i].size());
    CHECK("T16e: all erased", m.empty());
  }

  // --- API Edge Case Tests ---
  {
    FT2W m(0);
    bool api_ok = true;
    m.clear();
    if (!m.empty() || m.size() != 0) {
      api_ok = false;
      printf("❌ clear() failed to reset size\n");
    }

    m.insert((const uint8_t *)"test_key", 8, 42);
    if (!m.contains((const uint8_t *)"test_key", 8)) {
      api_ok = false;
      printf("❌ contains() failed for existing key\n");
    }
    if (m.contains((const uint8_t *)"test_kex", 8)) {
      api_ok = false;
      printf("❌ contains() true for missing key\n");
    }

    m.clear();
    if (m.contains((const uint8_t *)"test_key", 8)) {
      api_ok = false;
      printf("❌ contains() true after clear()\n");
    }

    if (!api_ok) {
      fail++;
      printf("TEST FAILED: API Extensions\n");
    } else {
      pass++;
    }
  }

  // T17: Ordered Cursor Iteration
  {
    FT2W m(0);
    bool cursor_ok = true;

    // Empty cursor
    auto c = m.create_cursor();
    if (c.valid()) {
      cursor_ok = false;
      printf("❌ empty cursor initially valid\n");
    }
    if (c.next()) {
      cursor_ok = false;
      printf("❌ empty cursor next() true\n");
    }
    if (c.seek_first()) {
      cursor_ok = false;
      printf("❌ empty cursor seek_first() true\n");
    }

    // Insert some keys in random order
    m.insert((const uint8_t *)"apple", 5, 10);
    m.insert((const uint8_t *)"zebra", 5, 20);
    m.insert((const uint8_t *)"banana", 6, 30);
    m.insert((const uint8_t *)"app", 3, 40); // prefix of apple
    m.insert((const uint8_t *)"apricot", 7, 50);

    // Expected lexicographic order: "app", "apple", "apricot", "banana",
    // "zebra" Values: 40, 10, 50, 30, 20

    if (!c.seek_first()) {
      cursor_ok = false;
      printf("❌ seek_first() failed on non-empty map\n");
    }

    uint32_t expected_vals[] = {40, 10, 50, 30, 20};
    int idx = 0;
    do {
      if (c.val() != expected_vals[idx]) {
        cursor_ok = false;
        printf(
            "❌ cursor returned wrong value at step %d: expected %u, got %u\n",
            idx, expected_vals[idx], c.val());
      }
      idx++;
    } while (c.next());

    if (idx != 5) {
      cursor_ok = false;
      printf("❌ cursor yielded %d items, expected 5\n", idx);
    }
    if (c.valid()) {
      cursor_ok = false;
      printf("❌ cursor valid after exhaustion\n");
    }

    if (!cursor_ok) {
      fail++;
      printf("TEST FAILED: Ordered Cursor\n");
    } else {
      pass++;
    }
  }

  // T18: seek_ge — range scan
  {
    FT2W m(0);
    bool ok = true;
    m.insert((const uint8_t *)"apple", 5, 10);
    m.insert((const uint8_t *)"banana", 6, 20);
    m.insert((const uint8_t *)"cherry", 6, 30);
    m.insert((const uint8_t *)"date", 4, 40);
    m.insert((const uint8_t *)"elderberry", 10, 50);

    // seek_ge("cherry") should land on "cherry"
    auto c = m.create_cursor();
    if (!c.seek_ge((const uint8_t *)"cherry", 6)) {
      ok = false;
      printf("❌ seek_ge(cherry) failed\n");
    } else if (c.val() != 30) {
      ok = false;
      printf("❌ seek_ge(cherry) val=%u expected 30\n", c.val());
    }

    // seek_ge("coconut") should land on "date" (first key > "coconut")
    if (!c.seek_ge((const uint8_t *)"coconut", 7)) {
      ok = false;
      printf("❌ seek_ge(coconut) failed\n");
    } else if (c.val() != 40) {
      ok = false;
      printf("❌ seek_ge(coconut) val=%u expected 40\n", c.val());
    }

    // seek_ge("zzz") should return false (nothing >= "zzz")
    if (c.seek_ge((const uint8_t *)"zzz", 3)) {
      ok = false;
      printf("❌ seek_ge(zzz) should be false\n");
    }

    // seek_ge("a") should land on "apple" (first key)
    if (!c.seek_ge((const uint8_t *)"a", 1)) {
      ok = false;
      printf("❌ seek_ge(a) failed\n");
    } else if (c.val() != 10) {
      ok = false;
      printf("❌ seek_ge(a) val=%u expected 10\n", c.val());
    }

    if (!ok) {
      fail++;
      printf("TEST FAILED: seek_ge\n");
    } else {
      pass++;
    }
  }

  // T19: seek_last + prev — reverse iteration
  {
    FT2W m(0);
    bool ok = true;
    m.insert((const uint8_t *)"apple", 5, 10);
    m.insert((const uint8_t *)"banana", 6, 20);
    m.insert((const uint8_t *)"cherry", 6, 30);

    auto c = m.create_cursor();
    if (!c.seek_last()) {
      ok = false;
      printf("❌ seek_last() failed\n");
    } else if (c.val() != 30) {
      ok = false;
      printf("❌ seek_last val=%u expected 30 (cherry)\n", c.val());
    }

    // prev should give banana
    if (!c.prev()) {
      ok = false;
      printf("❌ prev() from cherry failed\n");
    } else if (c.val() != 20) {
      ok = false;
      printf("❌ prev val=%u expected 20 (banana)\n", c.val());
    }

    // prev should give apple
    if (!c.prev()) {
      ok = false;
      printf("❌ prev() from banana failed\n");
    } else if (c.val() != 10) {
      ok = false;
      printf("❌ prev val=%u expected 10 (apple)\n", c.val());
    }

    // prev should exhaust
    if (c.prev()) {
      ok = false;
      printf("❌ prev() should exhaust\n");
    }

    if (!ok) {
      fail++;
      printf("TEST FAILED: seek_last+prev\n");
    } else {
      pass++;
    }
  }

  // T20: cursor update
  {
    FT2W m(0);
    bool ok = true;
    m.insert((const uint8_t *)"key1", 4, 100);
    m.insert((const uint8_t *)"key2", 4, 200);

    auto c = m.create_cursor();
    c.seek_first();
    c.update(999);

    // Verify via get
    uint32_t v;
    m.get((const uint8_t *)"key1", 4, v);
    if (v != 999) {
      ok = false;
      printf("❌ cursor update: expected 999 got %u\n", v);
    }
    if (!ok) {
      fail++;
      printf("TEST FAILED: cursor update\n");
    } else {
      pass++;
    }
  }

  // T21: seek_prefix + has_prefix
  {
    FT2W m(0);
    bool ok = true;
    m.insert((const uint8_t *)"app", 3, 1);
    m.insert((const uint8_t *)"apple", 5, 2);
    m.insert((const uint8_t *)"application", 11, 3);
    m.insert((const uint8_t *)"banana", 6, 4);
    m.insert((const uint8_t *)"band", 4, 5);

    auto c = m.create_cursor();
    const uint8_t *pfx = (const uint8_t *)"app";
    c.seek_prefix(pfx, 3);
    int count = 0;
    uint32_t sum = 0;
    while (c.valid() && c.has_prefix(pfx, 3)) {
      sum += c.val();
      count++;
      if (!c.next())
        break;
    }
    // Should find: "app"(1), "apple"(2), "application"(3) = sum 6, count 3
    if (count != 3 || sum != 6) {
      ok = false;
      printf("❌ prefix scan: count=%d sum=%u expected 3/6\n", count, sum);
    }

    if (!ok) {
      fail++;
      printf("TEST FAILED: seek_prefix\n");
    } else {
      pass++;
    }
  }

  // T22: Cursor through spawn_child — 20 keys share SAME first 8 bytes
  // Forces L0→child→L1→child→L2 (deep fractal), keys differ at L2 kf
  {
    FT2W m(0);
    bool ok = true;

    const uint32_t N_CHILD = 20;
    uint8_t keys[22][16];
    uint32_t vals[22];
    for (uint32_t i = 0; i < N_CHILD; i++) {
      keys[i][0] = 0x10;
      keys[i][1] = 0x20;
      keys[i][2] = 0x30;
      keys[i][3] = 0x40; // L0 kf
      keys[i][4] = 0x50;
      keys[i][5] = 0x60;
      keys[i][6] = 0x70;
      keys[i][7] = 0x80;             // L1 kf
      keys[i][8] = (uint8_t)(i + 1); // L2 kf byte0 — varies 0x01..0x14
      keys[i][9] = 0xAA;
      keys[i][10] = 0xBB;
      keys[i][11] = 0xCC;
      keys[i][12] = (uint8_t)(i + 1); // suffix
      keys[i][13] = 0;
      keys[i][14] = 0;
      keys[i][15] = 0;
      vals[i] = (i + 1) * 10;
    }
    for (int i = N_CHILD - 1; i >= 0; i--)
      m.insert(keys[i], 16, vals[i]);

    uint8_t kA[] = {0x05, 0x05, 0x05, 0x05, 0, 0, 0, 1, 0xAA};
    uint8_t kZ[] = {0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 1, 0xBB};
    m.insert(kA, 9, 888);
    m.insert(kZ, 9, 999);

    // Forward: verify sorted order + count/sum
    auto c = m.create_cursor();
    c.seek_first();
    printf("  T22 cursor order:\n");

    int fwd_count = 0;
    uint32_t fwd_sum = 0;
    uint8_t prev_key[256] = {0};
    uint32_t prev_kl = 0;
    do {
      uint32_t kl = c.key_len();
      const uint8_t *k = c.key();
      printf("    [%d] kl=%u key=", fwd_count, kl);
      for (uint32_t b = 0; b < kl && b < 16; b++) printf("%02x", k[b]);
      printf(" val=%u\n", c.val());
      if (fwd_count > 0) {
        uint32_t mkl = prev_kl < kl ? prev_kl : kl;
        int cmp = memcmp(k, prev_key, mkl);
        if (cmp < 0 || (cmp == 0 && kl < prev_kl)) {
          ok = false;
          printf("❌ T22 fwd[%d] NOT sorted key=", fwd_count);
          for (uint32_t b = 0; b < kl && b < 16; b++)
            printf("%02x", k[b]);
          printf(" prev=");
          for (uint32_t b = 0; b < prev_kl && b < 16; b++)
            printf("%02x", prev_key[b]);
          printf("\n");
        }
      }
      memcpy(prev_key, k, kl);
      prev_kl = kl;
      fwd_sum += c.val();
      fwd_count++;
    } while (c.next());
    if (fwd_count != 22) {
      ok = false;
      printf("❌ T22 fwd count=%d expected 22\n", fwd_count);
    }
    if (fwd_sum != 3987) {
      ok = false;
      printf("❌ T22 fwd sum=%u expected 3987\n", fwd_sum);
    }

    // Reverse: verify reverse sorted + count/sum
    c.seek_last();
    int rev_count = 0;
    uint32_t rev_sum = 0;
    memset(prev_key, 0xFF, sizeof(prev_key));
    prev_kl = 256;
    do {
      uint32_t kl = c.key_len();
      const uint8_t *k = c.key();
      if (rev_count > 0) {
        uint32_t mkl = prev_kl < kl ? prev_kl : kl;
        int cmp = memcmp(k, prev_key, mkl);
        if (cmp > 0 || (cmp == 0 && kl > prev_kl)) {
          ok = false;
          printf("❌ T22 rev[%d] NOT reverse-sorted\n", rev_count);
        }
      }
      memcpy(prev_key, k, kl);
      prev_kl = kl;
      rev_sum += c.val();
      rev_count++;
    } while (c.prev());
    if (rev_count != 22) {
      ok = false;
      printf("❌ T22 rev count=%d expected 22\n", rev_count);
    }
    if (rev_sum != 3987) {
      ok = false;
      printf("❌ T22 rev sum=%u expected 3987\n", rev_sum);
    }

    // Prefix scan
    uint8_t pfx[] = {0x10, 0x20, 0x30, 0x40};
    c.seek_prefix(pfx, 4);
    int pfx_count = 0;
    uint32_t pfx_sum = 0;
    while (c.valid() && c.has_prefix(pfx, 4)) {
      pfx_sum += c.val();
      pfx_count++;
      if (!c.next())
        break;
    }
    if (pfx_count != 20 || pfx_sum != 2100) {
      ok = false;
      printf("❌ T22 prefix: count=%d sum=%u expected 20/2100\n", pfx_count,
             pfx_sum);
    }

    if (!ok) {
      fail++;
      printf("TEST FAILED: spawn_child cursor\n");
    } else {
      pass++;
    }
  }

  // T23: Deep fractal chain — same 8-byte prefix, 3 levels deep
  {
    FT2W m(0);
    bool ok = true;
    // All share "XXXX" at L0 and "YYYY" at L1 → forces L0→child→L1→child→L2
    m.insert((const uint8_t *)"XXXXYYYYAAAA0001", 16, 100);
    m.insert((const uint8_t *)"XXXXYYYYAAAA0002", 16, 200);
    m.insert((const uint8_t *)"XXXXYYYYBBBB0001", 16, 300);
    m.insert((const uint8_t *)"XXXXYYYYBBBB0002", 16, 400);
    // Different L1 prefix
    m.insert((const uint8_t *)"XXXXZZZZAAAA0001", 16, 500);

    auto c = m.create_cursor();
    c.seek_first();
    // Lex order: XXXXYYYYAAAA0001(100) < XXXXYYYYAAAA0002(200) <
    // XXXXYYYYBBBB0001(300) < XXXXYYYYBBBB0002(400) < XXXXZZZZAAAA0001(500)
    uint32_t deep_exp[] = {100, 200, 300, 400, 500};
    int idx = 0;
    do {
      if (idx < 5 && c.val() != deep_exp[idx]) {
        ok = false;
        printf("❌ T23 deep[%d]: expected %u got %u\n", idx, deep_exp[idx],
               c.val());
      }
      idx++;
    } while (c.next());
    if (idx != 5) {
      ok = false;
      printf("❌ T23 deep count=%d expected 5\n", idx);
    }

    // seek_ge into deep level
    if (!c.seek_ge((const uint8_t *)"XXXXYYYYBBBB0001", 16)) {
      ok = false;
      printf("❌ T23 seek_ge deep failed\n");
    } else if (c.val() != 300) {
      ok = false;
      printf("❌ T23 seek_ge deep val=%u expected 300\n", c.val());
    }

    if (!ok) {
      fail++;
      printf("TEST FAILED: deep fractal cursor\n");
    } else {
      pass++;
    }
  }

  printf("  ────────────────────────────────────\n");
  printf("  EDGE CASES: %d/%d passed %s\n", pass, pass + fail,
         fail == 0 ? "✅" : "❌");

  bench_mega_cursor();

  return 0;
}
