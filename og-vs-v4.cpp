// ═══════════════════════════════════════════════════════════════════════════
//  OG FractalTag2 vs V4 (64-cap) — Unified Head-to-Head Benchmark
//  
//  This file is self-contained. Compile:
//    clang++ -std=c++17 -O3 -march=armv8.1-a+crc -o og_vs_v4 og_vs_v4_bench.cpp
//
//  OG = FractalTag2 from commit 2486d406 ("UNTOUCHABLE")
//  V4 = TurboMapV4 (64-cap, 22/16, CRC tag, 592B AOS groups)
//
//  Both run on identical key sets. Timing is min-of-N to remove noise.
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <atomic>
#include <chrono>
#include <sys/mman.h>
#include <arm_neon.h>
#include <arm_acle.h>

// ═══════════════════════════════════════════════════════════════════════════
//  OG NAMESPACE — verbatim from commit 2486d406
// ═══════════════════════════════════════════════════════════════════════════
namespace og {

class Arena {
    uint8_t* base_; std::atomic<uint64_t> pos_; uint64_t cap_;
public:
    Arena(uint64_t r) { cap_=r; base_=(uint8_t*)mmap(0,cap_,3,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(base_==MAP_FAILED){fprintf(stderr,"MMAP FAIL\n");abort();} pos_.store(0); }
    ~Arena() { munmap(base_,cap_); }
    void reset() { pos_.store(0); }
    void* alloc(uint64_t b) { uint64_t a=(b+7)&~7ULL; uint64_t o=pos_.fetch_add(a,std::memory_order_relaxed); return base_+o; }
    void* alloc16(uint64_t b) { uint64_t c=pos_.load(std::memory_order_relaxed); uint64_t s=(c+15)&~15ULL; uint64_t p=(b+15)&~15ULL; uint64_t t=(s-c)+p; uint64_t o=pos_.fetch_add(t,std::memory_order_relaxed); return base_+((o+15)&~15ULL); }
    uint64_t used() const { return pos_.load(std::memory_order_relaxed); }
};

struct SB {
    struct Blk { uint8_t* d; uint32_t c, l; };
    Blk* b; uint32_t n, mx;
    void init(Arena& a) { b=0; n=0; mx=0; }
    uint32_t al(Arena& a, const uint8_t* data, uint32_t data_len, uint32_t val) {
        uint32_t rl = 2 + data_len + 4;
        if (!n||b[n-1].l+rl>b[n-1].c) { if(n==mx){uint32_t nm=mx?mx*2:16;auto*nb=(Blk*)a.alloc(nm*sizeof(Blk));if(b)memcpy(nb,b,n*sizeof(Blk));b=nb;mx=nm;} auto&x=b[n];x.d=(uint8_t*)a.alloc(4<<20);x.c=4<<20;x.l=0;n++; }
        auto& x=b[n-1]; uint32_t o=x.l;
        uint16_t sl16 = (uint16_t)data_len;
        memcpy(x.d+o, &sl16, 2);
        if (data_len) memcpy(x.d+o+2, data, data_len);
        memcpy(x.d+o+2+data_len, &val, 4);
        x.l += rl;
        return((n-1)<<22)|o;
    }
    const uint8_t* g(uint32_t p) const { return b[p>>22].d+(p&0x3FFFFF); }
    uint16_t rec_suf_len(uint32_t p) const { uint16_t v; memcpy(&v, g(p), 2); return v; }
    const uint8_t* suf(uint32_t p) const { return g(p) + 2; }
    uint32_t get_val(uint32_t p) const { uint16_t sl=rec_suf_len(p); uint32_t v; memcpy(&v, g(p)+2+sl, 4); return v; }
    void set_val(uint32_t p, uint32_t val) { uint16_t sl=rec_suf_len(p); memcpy((uint8_t*)g(p)+2+sl, &val, 4); }
};

static uint16_t nmm(uint8x16_t c) {
    const uint8x16_t bm={1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t m=vandq_u8(c,bm); uint8x8_t lo=vget_low_u8(m),hi=vget_high_u8(m);
    lo=vpadd_u8(lo,lo);lo=vpadd_u8(lo,lo);lo=vpadd_u8(lo,lo);
    hi=vpadd_u8(hi,hi);hi=vpadd_u8(hi,hi);hi=vpadd_u8(hi,hi);
    return(uint16_t)vget_lane_u8(lo,0)|((uint16_t)vget_lane_u8(hi,0)<<8);
}
static uint32_t scan32(const uint8_t* t, uint8_t tag) {
    uint8x16_t n=vdupq_n_u8(tag);
    return(uint32_t)nmm(vceqq_u8(vld1q_u8(t),n))|((uint32_t)nmm(vceqq_u8(vld1q_u8(t+16),n))<<16);
}

class FractalTag2 {
    static constexpr uint32_t CAP = 32;
    static constexpr uint32_t MAX_GD = 24;
    struct alignas(16) Group {
        uint8_t tags[32]; uint32_t kfs[32]; uint32_t offs[32];
        uint8_t count, local_depth, sorted, _p[5];
    };
    Arena& arena_; SB& sb_;
    uintptr_t* dir_;
    uint32_t gd_, ds_sz_, level_, parent_kf_, shift_, vb_;
    static bool is_child(uintptr_t v) { return v & 1; }
    static FractalTag2* get_child(uintptr_t v) { return (FractalTag2*)(v & ~1ULL); }
    static Group* get_group(uintptr_t v) { return (Group*)(uintptr_t)v; }
    static uintptr_t tag_child(FractalTag2* c) { return (uintptr_t)c | 1; }
    void extract(const uint8_t* k, uint32_t kl, uint32_t& kf, uint8_t& tag) const {
        uint32_t o = level_ * 4;
        if (__builtin_expect(o + 8 <= kl, 1)) {
            uint64_t v; memcpy(&v, k + o, 8);
            kf = __builtin_bswap32((uint32_t)v);
            tag = (uint8_t)((__crc32cd(0, v) >> 25) | 0x80);
            return;
        }
        kf = 0;
        uint32_t avail = (o < kl) ? kl - o : 0;
        if (avail > 0) { uint32_t a4 = avail > 4 ? 4 : avail; memcpy(&kf, k + o, a4); }
        kf = __builtin_bswap32(kf);
        uint64_t v = 0;
        if (avail > 0) { uint32_t a8 = avail > 8 ? 8 : avail; memcpy(&v, k + o, a8); }
        tag = (uint8_t)((__crc32cd(0, v) >> 25) | 0x80);
    }
    uint32_t di(uint32_t kf) const { return (kf >> (shift_ & 31)) & (ds_sz_ - 1); }
    bool suffix_eq(uint32_t off, const uint8_t* suf, uint32_t query_suf_len) const {
        uint16_t stored_sl = sb_.rec_suf_len(off);
        if (stored_sl != query_suf_len) return false;
        if (!stored_sl) return true;
        return !memcmp(sb_.suf(off), suf, stored_sl);
    }
    Group* alloc_group(uint8_t ld) {
        auto* g = (Group*)arena_.alloc16(sizeof(Group));
        memset(g->tags, 0, sizeof(g->tags));
        g->count = 0; g->local_depth = ld; g->sorted = 1;
        return g;
    }
    void double_dir() {
        uint32_t ns = ds_sz_ * 2;
        auto* nd = (uintptr_t*)arena_.alloc(ns * sizeof(uintptr_t));
        for (uint32_t i = 0; i < ds_sz_; i++) { nd[2*i] = dir_[i]; nd[2*i+1] = dir_[i]; }
        dir_ = nd; ds_sz_ = ns; gd_++; shift_--;
    }
    void spawn_child(uint32_t dx) {
        Group* old = get_group(dir_[dx]);
        uint32_t shared_kf = old->kfs[0]; uint8_t ld = old->local_depth;
        auto* child = (FractalTag2*)arena_.alloc(sizeof(FractalTag2));
        new (child) FractalTag2(arena_, sb_, level_ + 1, shared_kf);
        for (uint32_t i = 0; i < old->count; i++) {
            uint16_t stored_sl = sb_.rec_suf_len(old->offs[i]);
            const uint8_t* parent_suf = sb_.suf(old->offs[i]);
            uint32_t val = sb_.get_val(old->offs[i]);
            child->insert_from_data(parent_suf, stored_sl, val);
        }
        uint32_t stride = 1u << (gd_ - ld);
        uint32_t st = di(shared_kf) & ~(stride - 1);
        for (uint32_t i = st; i < st + stride; i++) dir_[i] = tag_child(child);
    }
    void separate_from_child(uint32_t dx, uint32_t kf, FractalTag2* child) {
        uint32_t diff = kf ^ child->parent_kf_;
        if (diff == 0) return;
        uint8_t sep_bit = (uint8_t)__builtin_clz(diff);
        if (sep_bit >= MAX_GD) return;
        while (sep_bit >= gd_) { double_dir(); dx <<= 1; }
        uint8_t new_side = (kf >> (31 - sep_bit)) & 1;
        Group* fresh = alloc_group(sep_bit + 1);
        uint32_t new_dx = di(kf);
        uint32_t stride = 1u << (gd_ - sep_bit);
        uint32_t parent_stride = stride * 2;
        uint32_t st = new_dx & ~(parent_stride - 1);
        for (uint32_t i = st; i < st + parent_stride; i++) {
            uint8_t side = (i >> (gd_ - 1 - sep_bit)) & 1;
            if (side == new_side) dir_[i] = (uintptr_t)fresh;
        }
    }
    void split_or_spawn(uint32_t dx) {
        Group* old = get_group(dir_[dx]); uint8_t ld = old->local_depth;
        uint32_t diff = 0;
        for (uint32_t i = 1; i < old->count; i++) diff |= old->kfs[0] ^ old->kfs[i];
        uint32_t mask = (ld == 0) ? ~0u : ((1u << (32 - ld)) - 1);
        if (MAX_GD < 32) mask &= (~0u << (32 - MAX_GD));
        diff &= mask;
        uint8_t split_ld = (diff != 0) ? (uint8_t)__builtin_clz(diff) : 255;
        if (split_ld == 255) { spawn_child(dx); return; }
        while (split_ld >= gd_) { double_dir(); dx <<= 1; }
        Group* g0 = alloc_group(split_ld + 1), *g1 = alloc_group(split_ld + 1);
        for (uint32_t i = 0; i < old->count; i++) {
            uint8_t side = (old->kfs[i] >> (31 - split_ld)) & 1;
            Group* d = side ? g1 : g0; uint32_t p = d->count;
            d->tags[p] = old->tags[i]; d->kfs[p] = old->kfs[i];
            d->offs[p] = old->offs[i]; d->count++;
        }
        uint32_t stride = 1u << (gd_ - ld), st = dx & ~(stride - 1);
        for (uint32_t i = st; i < st + stride; i++) {
            uint8_t side = (i >> (gd_ - 1 - split_ld)) & 1;
            dir_[i] = (uintptr_t)(side ? g1 : g0);
        }
    }
    void insert_from_data(const uint8_t* data, uint32_t data_len, uint32_t v) {
        uint32_t kf; uint8_t tag;
        if (__builtin_expect(data_len >= 8, 1)) {
            uint64_t v8; memcpy(&v8, data, 8);
            kf = __builtin_bswap32((uint32_t)v8);
            tag = (uint8_t)((__crc32cd(0, v8) >> 25) | 0x80);
        } else {
            kf = 0; uint32_t avail = (data_len >= 4) ? 4 : data_len;
            if (avail) memcpy(&kf, data, avail); kf = __builtin_bswap32(kf);
            uint64_t v8 = 0; if (data_len) memcpy(&v8, data, data_len);
            tag = (uint8_t)((__crc32cd(0, v8) >> 25) | 0x80);
        }
        const uint8_t* suf = data + 4;
        uint32_t suf_len = (data_len >= 4) ? data_len - 4 : 0;
        for (int a = 0; a < 64; a++) {
            uint32_t dx = di(kf);
            uintptr_t slot = dir_[dx];
            if (is_child(slot)) {
                FractalTag2* child = get_child(slot);
                if (child->parent_kf_ == kf) { child->insert_from_data(suf, suf_len, v); return; }
                separate_from_child(dx, kf, child); continue;
            }
            Group* g = get_group(slot);
            uint32_t mm = scan32(g->tags, tag);
            while (mm) {
                int p = __builtin_ctz(mm);
                if (g->kfs[p] == kf && suffix_eq(g->offs[p], suf, suf_len)) {
                    sb_.set_val(g->offs[p], v); return;
                }
                mm &= mm - 1;
            }
            if (g->count >= CAP) { split_or_spawn(dx); continue; }
            uint32_t sb_off = sb_.al(arena_, suf, suf_len, v);
            uint32_t s = g->count;
            g->tags[s] = tag; g->kfs[s] = kf; g->offs[s] = sb_off;
            g->count++; g->sorted = 0; return;
        }
    }
public:
    FractalTag2(Arena& a, SB& sb, uint32_t level = 0, uint32_t pkf = 0)
        : arena_(a), sb_(sb), level_(level), parent_kf_(pkf) {
        gd_ = 0; ds_sz_ = 1; shift_ = 32; vb_ = (level + 1) * 4;
        dir_ = (uintptr_t*)arena_.alloc(sizeof(uintptr_t));
        dir_[0] = (uintptr_t)alloc_group(0);
    }
    void insert(const uint8_t* k, uint32_t kl, uint32_t v) {
        uint32_t kf; uint8_t tag; extract(k, kl, kf, tag);
        uint32_t suf_len = (kl > vb_) ? (kl - vb_) : 0;
        const uint8_t* suf = k + vb_;
        for (int a = 0; a < 64; a++) {
            uint32_t dx = di(kf);
            uintptr_t slot = dir_[dx];
            if (is_child(slot)) {
                FractalTag2* child = get_child(slot);
                if (child->parent_kf_ == kf) { child->insert_from_data(suf, suf_len, v); return; }
                separate_from_child(dx, kf, child); continue;
            }
            Group* g = get_group(slot);
            uint32_t mm = scan32(g->tags, tag);
            while (mm) {
                int p = __builtin_ctz(mm);
                if (g->kfs[p] == kf && suffix_eq(g->offs[p], suf, suf_len)) {
                    sb_.set_val(g->offs[p], v); return;
                }
                mm &= mm - 1;
            }
            if (g->count >= CAP) { split_or_spawn(dx); continue; }
            uint32_t sb_off = sb_.al(arena_, suf, suf_len, v);
            uint32_t s = g->count;
            g->tags[s] = tag; g->kfs[s] = kf; g->offs[s] = sb_off;
            g->count++; g->sorted = 0; return;
        }
    }
    bool get(const uint8_t* k, uint32_t kl, uint32_t& out) const {
        uint32_t kf; uint8_t tag; extract(k, kl, kf, tag);
        uint32_t suf_len = (kl > vb_) ? (kl - vb_) : 0;
        return get_inner(k + vb_, suf_len, kf, tag, out);
    }
private:
    bool get_inner(const uint8_t* suf, uint32_t suf_len, uint32_t kf, uint8_t tag, uint32_t& out) const {
        uint32_t dx = di(kf); uintptr_t slot = dir_[dx];
        if (is_child(slot)) {
            FractalTag2* child = get_child(slot);
            if (child->parent_kf_ != kf) return false;
            uint32_t ckf; uint8_t ctag;
            if (__builtin_expect(suf_len >= 8, 1)) {
                uint64_t v8; memcpy(&v8, suf, 8);
                ckf = __builtin_bswap32((uint32_t)v8);
                ctag = (uint8_t)((__crc32cd(0, v8) >> 25) | 0x80);
            } else {
                ckf = 0; uint32_t av = (suf_len >= 4) ? 4 : suf_len;
                if (av) memcpy(&ckf, suf, av); ckf = __builtin_bswap32(ckf);
                uint64_t v8 = 0; if (suf_len) memcpy(&v8, suf, suf_len);
                ctag = (uint8_t)((__crc32cd(0, v8) >> 25) | 0x80);
            }
            uint32_t csuf_len = (suf_len >= 4) ? suf_len - 4 : 0;
            return child->get_inner(suf + 4, csuf_len, ckf, ctag, out);
        }
        Group* g = get_group(slot);
        uint32_t mm = scan32(g->tags, tag);
        while (mm) {
            int p = __builtin_ctz(mm);
            if (g->kfs[p] == kf && suffix_eq(g->offs[p], suf, suf_len)) {
                out = sb_.get_val(g->offs[p]); return true;
            }
            mm &= mm - 1;
        }
        return false;
    }
public:
    uint64_t arena_used() const { return arena_.used(); }
};

// OG Wrapper
struct FT2W {
    Arena a; SB sb; FractalTag2* ft; uint32_t kl_;
    FT2W(uint32_t kl) : a(4ULL<<30), kl_(kl) { sb.init(a); ft = new (a.alloc(sizeof(FractalTag2))) FractalTag2(a, sb); }
    void reset() { a.reset(); sb.init(a); ft = new (a.alloc(sizeof(FractalTag2))) FractalTag2(a, sb); }
    void insert(const uint8_t* k, uint32_t v) { ft->insert(k, kl_, v); }
    bool get(const uint8_t* k, uint32_t& out) const { return ft->get(k, kl_, out); }
    uint64_t used() const { return a.used(); }
};

} // namespace og

// ═══════════════════════════════════════════════════════════════════════════
//  V4 — included from header
// ═══════════════════════════════════════════════════════════════════════════
#include "turbomapv4.h"

// ═══════════════════════════════════════════════════════════════════════════
//  Benchmark Harness
// ═══════════════════════════════════════════════════════════════════════════
static void gen_keys(std::vector<uint8_t>& out, uint32_t n, uint32_t kl, bool seq) {
    out.resize((size_t)n * kl, 0);
    if (seq) { for (uint32_t i = 0; i < n; i++) { uint64_t v=i+1; memcpy(&out[(size_t)i*kl],&v,8<kl?8:kl); } }
    else { arc4random_buf(out.data(), out.size()); }
}

int main() {
    printf("\n ═══════════════════════════════════════════════════════════════════════\n");
    printf("  OG FractalTag2 vs V4 TurboMap — Head-to-Head Benchmark\n");
    printf("  OG: CAP=32, SOA, 4B kf, CRC tag from 8B, suffix-only SB, fractal\n");
    printf("  V4: CAP=64, AOS, 4B kf, CRC tag, full-key Arena, Patricia LCP\n");
    printf(" ═══════════════════════════════════════════════════════════════════════\n\n");
    printf("                │     OG Fractal │     TurboMapV4 │        Δ        │\n");
    printf("  ──────────────┼────────────────┼────────────────┼─────────────────┤\n");

    struct R { double ins, get; uint64_t mem; };
    struct T { uint32_t N; int it; const char* l; };
    T tiers[] = {{100000,20,"100K"}, {1000000,5,"1M"}, {5000000,1,"5M"}};
    uint32_t kls[] = {8, 16, 64};

    for (auto& t : tiers) {
        for (uint32_t kl : kls) {
            for (bool seq : {false, true}) {
                std::vector<uint8_t> keys;
                gen_keys(keys, t.N, kl, seq);
                R og_r{1e18,1e18,0}, v4_r{1e18,1e18,0};

                // ── OG ──
                {
                    og::FT2W m(kl);
                    for (int i = 0; i < t.it; i++) {
                        if (i > 0) m.reset();
                        auto t0=std::chrono::high_resolution_clock::now();
                        for(uint32_t j=0;j<t.N;j++) m.insert(&keys[(size_t)j*kl],j);
                        auto t1=std::chrono::high_resolution_clock::now();
                        uint32_t v;uint64_t a=0;
                        auto t2=std::chrono::high_resolution_clock::now();
                        for(uint32_t j=0;j<t.N;j++){if(m.get(&keys[(size_t)j*kl],v))a+=v;}
                        auto t3=std::chrono::high_resolution_clock::now();
                        volatile uint64_t s=a;(void)s;
                        double ins=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)t.N;
                        double get=std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()/(double)t.N;
                        if(ins<og_r.ins)og_r.ins=ins;
                        if(get<og_r.get)og_r.get=get;
                    }
                    og_r.mem = m.used();
                }

                // ── V4 ──
                {
                    turbo::TurboMapV4 m;
                    for (int i = 0; i < t.it; i++) {
                        if (i > 0) m.reset();
                        auto t0=std::chrono::high_resolution_clock::now();
                        for(uint32_t j=0;j<t.N;j++) m.insert(&keys[(size_t)j*kl],kl,j);
                        auto t1=std::chrono::high_resolution_clock::now();
                        uint32_t v;uint64_t a=0;
                        auto t2=std::chrono::high_resolution_clock::now();
                        for(uint32_t j=0;j<t.N;j++){if(m.get(&keys[(size_t)j*kl],kl,v))a+=v;}
                        auto t3=std::chrono::high_resolution_clock::now();
                        volatile uint64_t s=a;(void)s;
                        double ins=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)t.N;
                        double get=std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()/(double)t.N;
                        if(ins<v4_r.ins)v4_r.ins=ins;
                        if(get<v4_r.get)v4_r.get=get;
                    }
                    v4_r.mem = m.arena_used();
                }

                // Correctness check
                {
                    og::FT2W og_m(kl);
                    turbo::TurboMapV4 v4_m;
                    for(uint32_t j=0;j<t.N;j++) {
                        og_m.insert(&keys[(size_t)j*kl],j);
                        v4_m.insert(&keys[(size_t)j*kl],kl,j);
                    }
                    uint32_t og_ok=0, v4_ok=0;
                    for(uint32_t j=0;j<t.N;j++) {
                        uint32_t v;
                        if(og_m.get(&keys[(size_t)j*kl],v)&&v==j) og_ok++;
                        if(v4_m.get(&keys[(size_t)j*kl],kl,v)&&v==j) v4_ok++;
                    }
                    const char* og_s = og_ok==t.N?"✅":"❌";
                    const char* v4_s = v4_ok==t.N?"✅":"❌";
                    double di=(og_r.ins-v4_r.ins)/og_r.ins*100;
                    double dg=(og_r.get-v4_r.get)/og_r.get*100;
                    printf("  %s %s %3uB │%sI:%5.1f G:%5.1f │%sI:%5.1f G:%5.1f │ I:%+5.0f%% G:%+5.0f%% │ %5.1f/%5.1fMB\n",
                        t.l, seq?"SEQ":"RND", kl,
                        og_s, og_r.ins, og_r.get,
                        v4_s, v4_r.ins, v4_r.get,
                        di, dg,
                        og_r.mem/(1024.0*1024.0), v4_r.mem/(1024.0*1024.0));
                }
            }
        }
        printf("  ──────────────┼────────────────┼────────────────┼─────────────────┤\n");
    }

    // ── Pathological: 28B shared prefix ──
    printf("\n  ── PATHOLOGICAL (28B shared prefix, 32B keys) ──\n");
    for (uint32_t N : {10000u, 100000u}) {
        uint32_t kl = 32;
        std::vector<uint8_t> keys(N * kl, 0);
        for (uint32_t i = 0; i < N; i++) {
            memset(&keys[i*kl], 'A', 28);
            uint32_t be = __builtin_bswap32(i);
            memcpy(&keys[i*kl + 28], &be, 4);
        }
        R og_r{1e18,1e18,0}, v4_r{1e18,1e18,0};
        for (int it = 0; it < 10; it++) {
            { og::FT2W m(kl);
              auto t0=std::chrono::high_resolution_clock::now();
              for(uint32_t j=0;j<N;j++) m.insert(&keys[j*kl],j);
              auto t1=std::chrono::high_resolution_clock::now();
              uint32_t v;uint64_t a=0;auto t2=std::chrono::high_resolution_clock::now();
              for(uint32_t j=0;j<N;j++){if(m.get(&keys[j*kl],v))a+=v;}
              auto t3=std::chrono::high_resolution_clock::now();
              volatile uint64_t s=a;(void)s;
              double ins=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)N;
              double get=std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()/(double)N;
              if(ins<og_r.ins)og_r.ins=ins;if(get<og_r.get)og_r.get=get;
              og_r.mem=m.used();
            }
            { turbo::TurboMapV4 m;
              auto t0=std::chrono::high_resolution_clock::now();
              for(uint32_t j=0;j<N;j++) m.insert(&keys[j*kl],kl,j);
              auto t1=std::chrono::high_resolution_clock::now();
              uint32_t v;uint64_t a=0;auto t2=std::chrono::high_resolution_clock::now();
              for(uint32_t j=0;j<N;j++){if(m.get(&keys[j*kl],kl,v))a+=v;}
              auto t3=std::chrono::high_resolution_clock::now();
              volatile uint64_t s=a;(void)s;
              double ins=std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)N;
              double get=std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t2).count()/(double)N;
              if(ins<v4_r.ins)v4_r.ins=ins;if(get<v4_r.get)v4_r.get=get;
              v4_r.mem=m.arena_used();
            }
        }
        double di=(og_r.ins-v4_r.ins)/og_r.ins*100;
        double dg=(og_r.get-v4_r.get)/og_r.get*100;
        printf("  %uK 32B patho│  I:%5.1f G:%5.1f │  I:%5.1f G:%5.1f │ I:%+5.0f%% G:%+5.0f%% │ %5.1f/%5.1fMB\n",
            N/1000, og_r.ins, og_r.get, v4_r.ins, v4_r.get, di, dg,
            og_r.mem/(1024.0*1024.0), v4_r.mem/(1024.0*1024.0));
    }
    // ── V4 Cursor Performance ──
    printf("\n  ── V4 CURSOR / SORTING PERFORMANCE ──\n");
    printf("                  │  FwdScan │  RevScan │ seek_ge  │  ns/entry\n");
    printf("  ────────────────┼──────────┼──────────┼──────────┤\n");
    for (uint32_t N : {100000u, 1000000u, 5000000u}) {
        for (uint32_t kl : {8u, 16u}) {
            std::vector<uint8_t> keys;
            gen_keys(keys, N, kl, false);
            turbo::TurboMapV4 m;
            for (uint32_t i = 0; i < N; i++) m.insert(&keys[(size_t)i*kl], kl, i);
            uint64_t actual = m.size();

            // Forward scan — seek_first() + next() * N
            double best_fwd = 1e18;
            int iters = N <= 100000 ? 10 : (N <= 1000000 ? 3 : 1);
            for (int r = 0; r < iters; r++) {
                auto cur = m.cursor();
                volatile uint64_t acc = 0;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (cur.seek_first(); cur.valid(); cur.next()) acc += cur.val();
                auto t1 = std::chrono::high_resolution_clock::now();
                double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)actual;
                if (ns < best_fwd) best_fwd = ns;
            }

            // Reverse scan — seek_last() + prev() * N
            double best_rev = 1e18;
            for (int r = 0; r < iters; r++) {
                auto cur = m.cursor();
                volatile uint64_t acc = 0;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (cur.seek_last(); cur.valid(); cur.prev()) acc += cur.val();
                auto t1 = std::chrono::high_resolution_clock::now();
                double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)actual;
                if (ns < best_rev) best_rev = ns;
            }

            // Random seek_ge — 10K random seeks into the dataset
            double best_seek = 1e18;
            uint32_t n_seeks = 10000;
            std::vector<uint8_t> seek_keys(n_seeks * kl);
            arc4random_buf(seek_keys.data(), seek_keys.size());
            for (int r = 0; r < iters; r++) {
                auto cur = m.cursor();
                volatile uint32_t found = 0;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (uint32_t i = 0; i < n_seeks; i++) {
                    if (cur.seek_ge(&seek_keys[i*kl], kl)) found++;
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()/(double)n_seeks;
                if (ns < best_seek) best_seek = ns;
            }

            printf("  %4uK %2uB RND  │ %6.1fns │ %6.1fns │ %6.1fns │\n",
                N/1000, kl, best_fwd, best_rev, best_seek);
        }
        printf("  ────────────────┼──────────┼──────────┼──────────┤\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  V4 CORRECTNESS TEST SUITE — Must pass 100% after any performance mod
    // ═══════════════════════════════════════════════════════════════════════
    printf("\n ═══════════════════════════════════════════════════════════════════════\n");
    printf("  V4 CORRECTNESS TESTS — All must pass!\n");
    printf(" ═══════════════════════════════════════════════════════════════════════\n\n");

    int pass = 0, fail = 0;
    auto CHECK = [&](const char* name, bool cond) {
        if (cond) { pass++; }
        else { fail++; printf("  ❌ FAIL: %s\n", name); }
    };

    // ── Exact per-key verification at 1M scale ──
    printf("  ── 1M EXACT VERIFICATION ──\n");
    for (uint32_t kl : {8u, 16u, 64u}) {
        for (bool seq : {false, true}) {
            uint32_t N = 1000000;
            std::vector<uint8_t> keys;
            gen_keys(keys, N, kl, seq);
            turbo::TurboMapV4 m;
            for (uint32_t i = 0; i < N; i++) m.insert(&keys[(size_t)i*kl], kl, i);
            uint32_t ok = 0, bad_val = 0, missing = 0;
            for (uint32_t i = 0; i < N; i++) {
                uint32_t v;
                if (m.get(&keys[(size_t)i*kl], kl, v)) {
                    if (v == i) ok++; else bad_val++;
                } else missing++;
            }
            char name[64];
            snprintf(name, sizeof(name), "1M %s %2uB: %u/%u", seq?"SEQ":"RND", kl, ok, N);
            CHECK(name, ok==N && bad_val==0 && missing==0);
            printf("  %s\n", ok==N && bad_val==0 && missing==0 ? "  ✅" : "");
        }
    }

    // ── Edge Case Tests ──
    printf("\n  ── EDGE CASE TESTS ──\n");

    // T1: Mixed-length keys
    {
        turbo::TurboMapV4 m;
        uint8_t k4[] = {1,2,3,4};
        uint8_t k8[] = {1,2,3,4,5,6,7,8};
        uint8_t k16[16]; memset(k16, 0, 16); memcpy(k16, k4, 4);
        m.insert(k4, 4, 100); m.insert(k8, 8, 200); m.insert(k16, 16, 300);
        uint32_t v;
        CHECK("T1a: 4B key found",      m.get(k4, 4, v) && v == 100);
        CHECK("T1b: 8B key found",      m.get(k8, 8, v) && v == 200);
        CHECK("T1c: 16B key found",     m.get(k16, 16, v) && v == 300);
        CHECK("T1d: 4B still correct",  m.get(k4, 4, v) && v == 100);
    }

    // T2: Prefix keys
    {
        turbo::TurboMapV4 m;
        uint8_t k4[] = "abcd"; uint8_t k5[] = "abcde"; uint8_t k6[] = "abcdef";
        m.insert(k4, 4, 10); m.insert(k5, 5, 20); m.insert(k6, 6, 30);
        uint32_t v;
        CHECK("T2a: prefix 4B found",   m.get(k4, 4, v) && v == 10);
        CHECK("T2b: prefix 5B found",   m.get(k5, 5, v) && v == 20);
        CHECK("T2c: prefix 6B found",   m.get(k6, 6, v) && v == 30);
        uint8_t k7[] = "abcdefg";
        CHECK("T2d: 7B not found",      !m.get(k7, 7, v));
    }

    // T3: Very short keys
    {
        turbo::TurboMapV4 m;
        uint8_t k1[] = {0x41}; uint8_t k2[] = {0x41,0x42}; uint8_t k3[] = {0x41,0x42,0x43};
        m.insert(k1, 1, 1); m.insert(k2, 2, 2); m.insert(k3, 3, 3);
        uint32_t v;
        CHECK("T3a: 1B key found",      m.get(k1, 1, v) && v == 1);
        CHECK("T3b: 2B key found",      m.get(k2, 2, v) && v == 2);
        CHECK("T3c: 3B key found",      m.get(k3, 3, v) && v == 3);
        CHECK("T3d: 1B != 2B prefix",   m.get(k1, 1, v) && v == 1);
    }

    // T4: Key update
    {
        turbo::TurboMapV4 m;
        uint8_t k[] = "testkey!";
        m.insert(k, 8, 42); uint32_t v;
        CHECK("T4a: first insert",      m.get(k, 8, v) && v == 42);
        m.insert(k, 8, 99);
        CHECK("T4b: update value",      m.get(k, 8, v) && v == 99);
    }

    // T5: Keys differing only in length
    {
        turbo::TurboMapV4 m;
        uint8_t k4[] = {0,0,0,0}; uint8_t k5[] = {0,0,0,0,0}; uint8_t k8[] = {0,0,0,0,0,0,0,0};
        m.insert(k4, 4, 400); m.insert(k5, 5, 500); m.insert(k8, 8, 800);
        uint32_t v;
        CHECK("T5a: all-zero 4B",       m.get(k4, 4, v) && v == 400);
        CHECK("T5b: all-zero 5B",       m.get(k5, 5, v) && v == 500);
        CHECK("T5c: all-zero 8B",       m.get(k8, 8, v) && v == 800);
    }

    // T6: Keys differing only in last byte
    {
        turbo::TurboMapV4 m;
        uint8_t ka[16]={0}; ka[15]=0x00; uint8_t kb[16]={0}; kb[15]=0x01; uint8_t kc[16]={0}; kc[15]=0xFF;
        m.insert(ka, 16, 10); m.insert(kb, 16, 20); m.insert(kc, 16, 30);
        uint32_t v;
        CHECK("T6a: last byte 0x00",    m.get(ka, 16, v) && v == 10);
        CHECK("T6b: last byte 0x01",    m.get(kb, 16, v) && v == 20);
        CHECK("T6c: last byte 0xFF",    m.get(kc, 16, v) && v == 30);
    }

    // T7: Exactly 4-byte keys
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = {1,2,3,4}; uint8_t kb[] = {1,2,3,5}; uint8_t kc[] = {5,4,3,2};
        m.insert(ka, 4, 111); m.insert(kb, 4, 222); m.insert(kc, 4, 333);
        uint32_t v;
        CHECK("T7a: 4B key a",          m.get(ka, 4, v) && v == 111);
        CHECK("T7b: 4B key b",          m.get(kb, 4, v) && v == 222);
        CHECK("T7c: 4B key c",          m.get(kc, 4, v) && v == 333);
    }

    // T8: 10K variable-length stress
    {
        turbo::TurboMapV4 m;
        uint32_t N8 = 10000;
        std::vector<std::vector<uint8_t>> keys(N8);
        for (uint32_t i = 0; i < N8; i++) {
            uint32_t kl = 4 + (i % 61);
            keys[i].resize(kl, 0);
            memcpy(keys[i].data(), &i, 4);
            for (uint32_t j = 4; j < kl; j++) keys[i][j] = (uint8_t)(i + j);
        }
        for (uint32_t i = 0; i < N8; i++) m.insert(keys[i].data(), keys[i].size(), i);
        uint32_t ok = 0;
        for (uint32_t i = 0; i < N8; i++) {
            uint32_t v;
            if (m.get(keys[i].data(), keys[i].size(), v) && v == i) ok++;
        }
        CHECK("T8: 10K variable-length", ok == N8);
    }

    // T9: Prefix key that IS another key
    {
        turbo::TurboMapV4 m;
        uint8_t k[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
        m.insert(k, 4, 1000); m.insert(k, 8, 2000);
        uint32_t v;
        CHECK("T9a: 4B prefix key",     m.get(k, 4, v) && v == 1000);
        CHECK("T9b: 8B full key",       m.get(k, 8, v) && v == 2000);
        m.insert(k, 4, 1111);
        CHECK("T9c: 4B updated",        m.get(k, 4, v) && v == 1111);
        CHECK("T9d: 8B unchanged",      m.get(k, 8, v) && v == 2000);
    }

    // T10: Long key (256 bytes)
    {
        turbo::TurboMapV4 m;
        uint8_t klong[256]; for (int i = 0; i < 256; i++) klong[i] = (uint8_t)i;
        uint8_t klong2[256]; memcpy(klong2, klong, 256); klong2[255] ^= 0xFF;
        m.insert(klong, 256, 9999); m.insert(klong2, 256, 8888);
        uint32_t v;
        CHECK("T10a: 256B key a",       m.get(klong, 256, v) && v == 9999);
        CHECK("T10b: 256B key b",       m.get(klong2, 256, v) && v == 8888);
    }

    // T11: Basic erase
    {
        turbo::TurboMapV4 m;
        uint8_t k[] = "erasetest";
        m.insert(k, 9, 42); uint32_t v;
        CHECK("T11a: key exists",        m.get(k, 9, v) && v == 42);
        CHECK("T11b: erase returns true", m.erase(k, 9));
        CHECK("T11c: key gone",          !m.get(k, 9, v));
        CHECK("T11d: double erase false", !m.erase(k, 9));
    }

    // T12: Erase then reinsert
    {
        turbo::TurboMapV4 m;
        uint8_t k[] = {10,20,30,40,50,60,70,80};
        m.insert(k, 8, 100); uint32_t v;
        CHECK("T12a: pre-erase",        m.get(k, 8, v) && v == 100);
        m.erase(k, 8);
        CHECK("T12b: erased",           !m.get(k, 8, v));
        m.insert(k, 8, 200);
        CHECK("T12c: reinserted",       m.get(k, 8, v) && v == 200);
    }

    // T13: Erase doesn't affect other keys
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = {1,2,3,4}; uint8_t kb[] = {5,6,7,8}; uint8_t kc[] = {9,10,11,12};
        m.insert(ka, 4, 10); m.insert(kb, 4, 20); m.insert(kc, 4, 30);
        m.erase(kb, 4); uint32_t v;
        CHECK("T13a: ka survives",      m.get(ka, 4, v) && v == 10);
        CHECK("T13b: kb erased",        !m.get(kb, 4, v));
        CHECK("T13c: kc survives",      m.get(kc, 4, v) && v == 30);
    }

    // T14: Size tracking
    {
        turbo::TurboMapV4 m;
        CHECK("T14a: empty",            m.empty() && m.size() == 0);
        uint8_t k1[] = {1,2,3,4}; uint8_t k2[] = {5,6,7,8}; uint8_t k3[] = {9,10,11,12};
        m.insert(k1, 4, 1);
        CHECK("T14b: size=1",           m.size() == 1);
        m.insert(k2, 4, 2); m.insert(k3, 4, 3);
        CHECK("T14c: size=3",           m.size() == 3);
        m.insert(k2, 4, 99);
        CHECK("T14d: update no change", m.size() == 3);
        m.erase(k1, 4);
        CHECK("T14e: size=2",           m.size() == 2);
        m.erase(k2, 4); m.erase(k3, 4);
        CHECK("T14f: empty again",      m.empty());
        m.erase(k1, 4);
        CHECK("T14g: stays 0",          m.size() == 0);
    }

    // T15: Variable-length erase
    {
        turbo::TurboMapV4 m;
        uint8_t k4[] = {0xDE,0xAD,0xBE,0xEF};
        uint8_t k8[] = {0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE};
        m.insert(k4, 4, 100); m.insert(k8, 8, 200);
        CHECK("T15a: size=2",           m.size() == 2);
        m.erase(k4, 4); uint32_t v;
        CHECK("T15b: 4B erased",        !m.get(k4, 4, v));
        CHECK("T15c: 8B survives",      m.get(k8, 8, v) && v == 200);
        CHECK("T15d: size=1",           m.size() == 1);
    }

    // T16: Erase stress — insert 1000, erase all
    {
        turbo::TurboMapV4 m;
        uint32_t N16 = 1000;
        std::vector<std::vector<uint8_t>> keys(N16);
        for (uint32_t i = 0; i < N16; i++) {
            uint32_t kl = 8 + (i % 25);
            keys[i].resize(kl, 0);
            memcpy(keys[i].data(), &i, 4);
            for (uint32_t j = 4; j < kl; j++) keys[i][j] = (uint8_t)(i ^ j);
        }
        for (uint32_t i = 0; i < N16; i++) m.insert(keys[i].data(), keys[i].size(), i);
        CHECK("T16a: size=1000",        m.size() == 1000);
        for (uint32_t i = 0; i < N16; i += 2) m.erase(keys[i].data(), keys[i].size());
        CHECK("T16b: size=500",         m.size() == 500);
        uint32_t ok_gone = 0, ok_live = 0;
        for (uint32_t i = 0; i < N16; i++) {
            uint32_t v;
            bool found = m.get(keys[i].data(), keys[i].size(), v);
            if (i % 2 == 0) { if (!found) ok_gone++; }
            else { if (found && v == i) ok_live++; }
        }
        CHECK("T16c: evens erased",     ok_gone == 500);
        CHECK("T16d: odds survive",     ok_live == 500);
        for (uint32_t i = 1; i < N16; i += 2) m.erase(keys[i].data(), keys[i].size());
        CHECK("T16e: all erased",       m.empty());
    }

    // ── Cursor & Sorting Tests ──
    printf("\n  ── CURSOR & SORTING TESTS ──\n");

    // T17: Forward traversal — lexicographic order
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "cherry"; uint8_t kb[] = "apple"; uint8_t kc[] = "banana";
        m.insert(ka, 6, 1); m.insert(kb, 5, 2); m.insert(kc, 6, 3);
        auto cur = m.cursor();
        CHECK("T17a: seek_first",       cur.seek_first());
        CHECK("T17b: first=apple",      cur.valid() && cur.key_len()==5 && !memcmp(cur.key(),"apple",5) && cur.val()==2);
        CHECK("T17c: next=banana",      cur.next() && cur.key_len()==6 && !memcmp(cur.key(),"banana",6) && cur.val()==3);
        CHECK("T17d: next=cherry",      cur.next() && cur.key_len()==6 && !memcmp(cur.key(),"cherry",6) && cur.val()==1);
        CHECK("T17e: end",              !cur.next());
    }

    // T18: Reverse traversal
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "aaa"; uint8_t kb[] = "bbb"; uint8_t kc[] = "ccc";
        m.insert(ka, 3, 10); m.insert(kb, 3, 20); m.insert(kc, 3, 30);
        auto cur = m.cursor();
        CHECK("T18a: seek_last",        cur.seek_last());
        CHECK("T18b: last=ccc",         cur.valid() && cur.key_len()==3 && !memcmp(cur.key(),"ccc",3) && cur.val()==30);
        CHECK("T18c: prev=bbb",         cur.prev() && cur.key_len()==3 && !memcmp(cur.key(),"bbb",3) && cur.val()==20);
        CHECK("T18d: prev=aaa",         cur.prev() && cur.key_len()==3 && !memcmp(cur.key(),"aaa",3) && cur.val()==10);
        CHECK("T18e: begin",            !cur.prev());
    }

    // T19: seek_ge exact match
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "dog"; uint8_t kb[] = "cat"; uint8_t kc[] = "fox";
        m.insert(ka, 3, 1); m.insert(kb, 3, 2); m.insert(kc, 3, 3);
        auto cur = m.cursor();
        CHECK("T19a: seek_ge(cat)",     cur.seek_ge(kb, 3));
        CHECK("T19b: found cat",        cur.valid() && cur.key_len()==3 && !memcmp(cur.key(),"cat",3) && cur.val()==2);
        CHECK("T19c: next=dog",         cur.next() && cur.key_len()==3 && !memcmp(cur.key(),"dog",3));
    }

    // T20: seek_ge between keys
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "aaa"; uint8_t kb[] = "ccc"; uint8_t kc[] = "eee";
        m.insert(ka, 3, 10); m.insert(kb, 3, 20); m.insert(kc, 3, 30);
        auto cur = m.cursor();
        uint8_t bbb[] = "bbb";
        CHECK("T20a: seek_ge(bbb)",     cur.seek_ge(bbb, 3));
        CHECK("T20b: lands on ccc",     cur.valid() && cur.key_len()==3 && !memcmp(cur.key(),"ccc",3) && cur.val()==20);
    }

    // T21: seek_ge past end
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "aaa"; uint8_t kb[] = "bbb";
        m.insert(ka, 3, 1); m.insert(kb, 3, 2);
        auto cur = m.cursor();
        uint8_t zzz[] = "zzz";
        CHECK("T21: seek_ge past end",  !cur.seek_ge(zzz, 3));
    }

    // T22: 10K sorted order verification — binary keys
    {
        turbo::TurboMapV4 m;
        uint32_t N22 = 10000;
        std::vector<uint8_t> keys(N22 * 8);
        for (uint32_t i = 0; i < N22; i++) {
            uint64_t r = (uint64_t)i * 6364136223846793005ULL + 1;
            memcpy(&keys[i*8], &r, 8);
            m.insert(&keys[i*8], 8, i);
        }
        auto cur = m.cursor();
        bool ordered = true;
        uint32_t count = 0;
        std::vector<uint8_t> prev_key;
        for (cur.seek_first(); cur.valid(); cur.next()) {
            uint32_t kl = cur.key_len();
            const uint8_t* k = cur.key();
            if (!prev_key.empty()) {
                int cmp = memcmp(prev_key.data(), k, prev_key.size() < kl ? prev_key.size() : kl);
                if (cmp > 0 || (cmp == 0 && prev_key.size() > kl)) { ordered = false; break; }
            }
            prev_key.assign(k, k + kl);
            count++;
        }
        CHECK("T22a: 10K sorted order", ordered);
        CHECK("T22b: 10K count",        count == N22);
    }

    // T23: Mixed-length keys cursor ordering
    {
        turbo::TurboMapV4 m;
        // "a" < "aa" < "aaa" < "aab" < "ab" < "b"
        uint8_t k1[] = "b"; uint8_t k2[] = "a"; uint8_t k3[] = "ab";
        uint8_t k4[] = "aa"; uint8_t k5[] = "aaa"; uint8_t k6[] = "aab";
        m.insert(k1, 1, 1); m.insert(k2, 1, 2); m.insert(k3, 2, 3);
        m.insert(k4, 2, 4); m.insert(k5, 3, 5); m.insert(k6, 3, 6);
        auto cur = m.cursor();
        cur.seek_first();
        CHECK("T23a: first=a",          cur.valid() && cur.key_len()==1 && !memcmp(cur.key(),"a",1));
        cur.next();
        CHECK("T23b: next=aa",          cur.valid() && cur.key_len()==2 && !memcmp(cur.key(),"aa",2));
        cur.next();
        CHECK("T23c: next=aaa",         cur.valid() && cur.key_len()==3 && !memcmp(cur.key(),"aaa",3));
        cur.next();
        CHECK("T23d: next=aab",         cur.valid() && cur.key_len()==3 && !memcmp(cur.key(),"aab",3));
        cur.next();
        CHECK("T23e: next=ab",          cur.valid() && cur.key_len()==2 && !memcmp(cur.key(),"ab",2));
        cur.next();
        CHECK("T23f: next=b",           cur.valid() && cur.key_len()==1 && !memcmp(cur.key(),"b",1));
        CHECK("T23g: end",              !cur.next());
    }

    // T24: Cursor after erase
    {
        turbo::TurboMapV4 m;
        uint8_t ka[] = "aaa"; uint8_t kb[] = "bbb"; uint8_t kc[] = "ccc";
        m.insert(ka, 3, 1); m.insert(kb, 3, 2); m.insert(kc, 3, 3);
        m.erase(kb, 3);
        auto cur = m.cursor();
        cur.seek_first();
        CHECK("T24a: first=aaa",        cur.valid() && !memcmp(cur.key(),"aaa",3));
        cur.next();
        CHECK("T24b: next=ccc",         cur.valid() && !memcmp(cur.key(),"ccc",3));
        CHECK("T24c: end",              !cur.next());
    }

    // T25: Prefix scan
    {
        turbo::TurboMapV4 m;
        uint8_t k1[] = "user:alice"; uint8_t k2[] = "user:bob"; uint8_t k3[] = "user:charlie";
        uint8_t k4[] = "post:1"; uint8_t k5[] = "post:2";
        m.insert(k1, 10, 1); m.insert(k2, 8, 2); m.insert(k3, 12, 3);
        m.insert(k4, 6, 4); m.insert(k5, 6, 5);
        auto cur = m.cursor();
        uint8_t prefix[] = "user:";
        cur.seek_ge(prefix, 5);
        uint32_t user_count = 0;
        while (cur.valid() && cur.key_len() >= 5 && !memcmp(cur.key(), "user:", 5)) {
            user_count++;
            cur.next();
        }
        CHECK("T25: prefix scan found 3 users", user_count == 3);
    }

    // T26: 100K cursor stress — forward count == reverse count == size
    {
        turbo::TurboMapV4 m;
        uint32_t N26 = 100000;
        std::vector<uint8_t> keys(N26 * 16);
        arc4random_buf(keys.data(), keys.size());
        for (uint32_t i = 0; i < N26; i++) m.insert(&keys[i*16], 16, i);
        uint64_t actual_size = m.size();
        auto cur = m.cursor();
        uint64_t fwd_count = 0;
        for (cur.seek_first(); cur.valid(); cur.next()) fwd_count++;
        uint64_t rev_count = 0;
        for (cur.seek_last(); cur.valid(); cur.prev()) rev_count++;
        CHECK("T26a: fwd count == size", fwd_count == actual_size);
        CHECK("T26b: rev count == size", rev_count == actual_size);
        CHECK("T26c: fwd == rev",        fwd_count == rev_count);
    }
    printf("  V4 TESTS: %d/%d passed %s\n", pass, pass+fail, fail==0 ? "✅" : "❌");
    if (fail > 0) printf("  🚨 %d FAILURES — DO NOT SHIP!\n", fail);
    printf("  ────────────────────────────────────\n\n");

    return fail > 0 ? 1 : 0;
}
