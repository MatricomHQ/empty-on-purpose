#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <random>
#include <string_view>
#include <atomic>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// --- Common Code ---
using ArenaOffset = uint32_t;
constexpr uint32_t FTU_MAGIC_NUMBER = 0xDEADBEEF;
enum ChildType : uint8_t { CT_UNUSED = 0, CT_INTERNAL = 1, CT_JUMP = 2, CT_LEAF = 3 };
enum NodeTypeBit : uint8_t { NTB_INTERNAL = 0, NTB_OFFSET = 1 };
constexpr uint8_t FTU_NODE_CAPACITY = 127;
constexpr uint8_t MAX_BIT_INDEX = 127;

struct DataPayload {
    uint32_t key_len;
    uint32_t val_len;
    const char* get_data() const { return reinterpret_cast<const char*>(this + 1); }
    char* get_data() { return reinterpret_cast<char*>(this + 1); }
};

class MmapArena {
private:
    uint8_t* memory;
    size_t capacity;
    std::atomic<size_t> used;
    int fd;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMap;
#else
    std::string filename;
#endif
public:
    MmapArena(const std::string& file, size_t size_in_mb) : memory(nullptr), capacity(0), used(1), fd(-1) {
        capacity = size_in_mb * 1024 * 1024;
        if (capacity > std::numeric_limits<uint32_t>::max()) { throw std::invalid_argument("Arena size cannot exceed 4GB (32-bit offset limit)"); }
#ifdef _WIN32
        hFile = CreateFile(file.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("Failed to open or create mmap file.");
        hMap = CreateFileMapping(hFile, NULL, PAGE_READWRITE, 0, capacity, NULL);
        if (hMap == NULL) { CloseHandle(hFile); throw std::runtime_error("Failed to create file mapping."); }
        memory = (uint8_t*)MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, capacity);
        if (memory == NULL) { CloseHandle(hMap); CloseHandle(hFile); throw std::runtime_error("Failed to map view of file."); }
#else
        filename = file;
        fd = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd == -1) throw std::runtime_error("Failed to open or create mmap file: " + filename);
        if (ftruncate(fd, capacity) == -1) { close(fd); throw std::runtime_error("Failed to set size of mmap file (ftruncate)"); }
        memory = static_cast<uint8_t*>(mmap(NULL, capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        if (memory == MAP_FAILED) { close(fd); throw std::runtime_error("Failed to map file to memory (mmap)"); }
#endif
    }
    ~MmapArena() {
#ifdef _WIN32
        if (memory) { FlushViewOfFile(memory, used.load()); UnmapViewOfFile(memory); }
        if (hMap) CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (memory != nullptr && memory != MAP_FAILED) {
            msync(memory, used.load(), MS_SYNC);
            munmap(memory, capacity);
        }
        if (fd != -1) close(fd);
        std::remove(filename.c_str());
#endif
    }
    MmapArena(const MmapArena&) = delete;
    MmapArena& operator=(const MmapArena&) = delete;
    void advise(int advice) {
#ifndef _WIN32
        if (madvise(memory, capacity, advice) != 0) perror("madvise failed");
#endif
    }
    ArenaOffset allocate(size_t size) {
        size_t aligned_size = (size + 7) & ~7;
        size_t offset = used.fetch_add(aligned_size, std::memory_order_relaxed);
        if (offset + aligned_size > capacity) return 0;
        return static_cast<ArenaOffset>(offset);
    }
    template<typename T> T* get(ArenaOffset offset) {
        if (offset == 0 || offset >= used.load(std::memory_order_relaxed)) return nullptr;
        return reinterpret_cast<T*>(memory + offset);
    }
    template<typename T> const T* get(ArenaOffset offset) const {
        if (offset == 0 || offset >= used.load(std::memory_order_relaxed)) return nullptr;
        return reinterpret_cast<const T*>(memory + offset);
    }
};

inline int get_bit_at(std::string_view key, int bit_idx) {
    int byte_idx = bit_idx >> 3;
    int bit_in_byte = 7 - (bit_idx & 7);
    if (byte_idx >= key.length()) return 0;
    return (key[byte_idx] >> bit_in_byte) & 1;
}

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)
#endif

inline int find_first_differing_bit(std::string_view k1, std::string_view k2) {
    size_t len1 = k1.length();
    size_t len2 = k2.length();
    size_t min_len = std::min(len1, len2);
    size_t i = 0;
#if defined(__GNUC__) || defined(__clang__)
    for (; i + 7 < min_len; i += 8) {
        uint64_t v1 = *reinterpret_cast<const uint64_t*>(k1.data() + i);
        uint64_t v2 = *reinterpret_cast<const uint64_t*>(k2.data() + i);
        if (v1 != v2) {
            uint64_t xor_val = v1 ^ v2;
            return (i * 8) + __builtin_clzll(__builtin_bswap64(xor_val));
        }
    }
#endif
    for (; i < min_len; ++i) {
        if (k1[i] != k2[i]) {
            uint8_t xor_val = k1[i] ^ k2[i];
#if defined(__GNUC__) || defined(__clang__)
            return (i * 8) + __builtin_clz(xor_val) - ((sizeof(unsigned int) - 1) * 8);
#elif _MSC_VER
            unsigned long index; _BitScanReverse(&index, xor_val); return (i * 8) + (7-index);
#else
            for(int j=0; j<8; ++j) { if (((k1[i] >> (7-j)) & 1) != ((k2[i] >> (7-j)) & 1)) { return (i*8) + j; } }
#endif
        }
    }
    if (len1 != len2) { return min_len * 8; }
    return -1;
}

// --- Implementation 1: Array of Structures (AOS) ---
namespace aos {
struct Node {
    uint16_t data;
    uint8_t get_test_bit_index() const { return static_cast<uint8_t>((data >> 9) & 0x7F); }
    uint8_t get_child_base_idx() const { return static_cast<uint8_t>((data >> 2) & 0x7F); }
    NodeTypeBit get_left_type_bit() const { return static_cast<NodeTypeBit>((data >> 1) & 0x01); }
    NodeTypeBit get_right_type_bit() const { return static_cast<NodeTypeBit>(data & 0x01); }
    void set_instruction_test(uint8_t bit_index, uint8_t child_base_idx, NodeTypeBit left_type, NodeTypeBit right_type) {
        if (bit_index > MAX_BIT_INDEX || child_base_idx >= FTU_NODE_CAPACITY) throw std::out_of_range("Node instruction out of range");
        data = (static_cast<uint16_t>(bit_index) << 9) | (static_cast<uint16_t>(child_base_idx) << 2) | (static_cast<uint16_t>(left_type) << 1) | static_cast<uint16_t>(right_type);
    }
};
inline ArenaOffset get_offset_from_nodes(const Node* nodes, uint8_t index) { return (static_cast<uint32_t>(nodes[index].data) << 16) | nodes[index + 1].data; }
inline void set_offset_in_nodes(Node* nodes, uint8_t index, ArenaOffset offset) {
    nodes[index].data = static_cast<uint16_t>(offset >> 16);
    nodes[index + 1].data = static_cast<uint16_t>(offset & 0xFFFF);
}
struct FTU { uint32_t magic; uint8_t nodes_used; ChildType root_node_type; uint8_t padding[2]; Node nodes[FTU_NODE_CAPACITY]; };
class FractalTree {
    MmapArena arena; ArenaOffset root_ftu_offset;
    ArenaOffset allocate_new_ftu() {
        ArenaOffset ftu_offset = arena.allocate(sizeof(FTU)); if (ftu_offset == 0) throw std::runtime_error("Arena out of memory for new FTU");
        FTU* ftu = arena.get<FTU>(ftu_offset);
        ftu->magic = FTU_MAGIC_NUMBER; ftu->nodes_used = 0; ftu->root_node_type = CT_UNUSED;
        memset(ftu->nodes, 0, sizeof(Node) * FTU_NODE_CAPACITY); return ftu_offset;
    }
    ArenaOffset allocate_data_payload(std::string_view key, std::string_view value) {
        size_t total_size = sizeof(DataPayload) + key.length() + value.length();
        ArenaOffset offset = arena.allocate(total_size); if (offset == 0) throw std::runtime_error("Arena out of memory for data payload");
        DataPayload* payload = arena.get<DataPayload>(offset);
        payload->key_len = key.length(); payload->val_len = value.length();
        char* data_ptr = payload->get_data();
        memcpy(data_ptr, key.data(), key.length()); memcpy(data_ptr + key.length(), value.data(), value.length());
        return offset;
    }
public:
    FractalTree(const std::string& filename, size_t arena_size_mb) : arena(filename, arena_size_mb) { root_ftu_offset = allocate_new_ftu(); }
    void advise_memory_usage(int advice) { arena.advise(advice); }
    std::string get_name() const { return "FractalTree (AOS)"; }
    std::string find(std::string_view key) const {
        const FTU* ftu = arena.get<FTU>(root_ftu_offset);
        if (ftu->root_node_type == CT_UNUSED) return "";
        ChildType current_type = ftu->root_node_type; uint8_t node_idx = 0;
        while (true) {
            switch (current_type) {
                case CT_INTERNAL: {
                    const Node* node = &ftu->nodes[node_idx];
                    int bit = get_bit_at(key, node->get_test_bit_index());
                    NodeTypeBit child_type_bit; uint8_t child_base_idx = node->get_child_base_idx();
                    if (bit == 0) { node_idx = child_base_idx; child_type_bit = node->get_left_type_bit(); }
                    else { node_idx = child_base_idx + 2; child_type_bit = node->get_right_type_bit(); }
                    if (child_type_bit == NTB_INTERNAL) { current_type = CT_INTERNAL; }
                    else {
                        ArenaOffset offset = get_offset_from_nodes(ftu->nodes, node_idx);
                        const FTU* target_ftu_header = arena.get<const FTU>(offset);
                        if (target_ftu_header && target_ftu_header->magic == FTU_MAGIC_NUMBER) { current_type = CT_JUMP; }
                        else { current_type = CT_LEAF; }
                    }
                    break;
                }
                case CT_JUMP: {
                    ArenaOffset offset = get_offset_from_nodes(ftu->nodes, node_idx);
                    ftu = arena.get<FTU>(offset); if (!ftu) return "";
                    current_type = ftu->root_node_type; node_idx = 0;
                    break;
                }
                case CT_LEAF: {
                    ArenaOffset offset = get_offset_from_nodes(ftu->nodes, node_idx);
                    const DataPayload* payload = arena.get<DataPayload>(offset); if (!payload) return "";
                    std::string_view stored_key(payload->get_data(), payload->key_len);
                    if (key == stored_key) { return std::string(payload->get_data() + payload->key_len, payload->val_len); }
                    return "";
                }
                case CT_UNUSED: default: return "";
            }
        }
    }
    bool insert(std::string_view key, std::string_view value) {
        FTU* ftu = arena.get<FTU>(root_ftu_offset);
        if (ftu->root_node_type == CT_UNUSED) {
            ArenaOffset data_offset = allocate_data_payload(key, value);
            set_offset_in_nodes(ftu->nodes, 0, data_offset);
            ftu->root_node_type = CT_LEAF; ftu->nodes_used = 2; return true;
        }
        FTU* parent_ftu = nullptr; uint8_t parent_node_idx = FTU_NODE_CAPACITY; bool is_left_child = false;
    restart_insertion:
        ftu = arena.get<FTU>(root_ftu_offset);
        ChildType current_type = ftu->root_node_type; uint8_t node_idx = 0;
        parent_ftu = nullptr; parent_node_idx = FTU_NODE_CAPACITY; is_left_child = false;
        while (true) {
            if (current_type == CT_INTERNAL) {
                parent_ftu = ftu; parent_node_idx = node_idx;
                Node* node = &ftu->nodes[node_idx];
                int bit = get_bit_at(key, node->get_test_bit_index());
                uint8_t child_base_idx = node->get_child_base_idx(); NodeTypeBit child_type_bit;
                if (bit == 0) { node_idx = child_base_idx; child_type_bit = node->get_left_type_bit(); is_left_child = true; }
                else { node_idx = child_base_idx + 2; child_type_bit = node->get_right_type_bit(); is_left_child = false; }
                if (child_type_bit == NTB_INTERNAL) { current_type = CT_INTERNAL; }
                else {
                    ArenaOffset offset = get_offset_from_nodes(ftu->nodes, node_idx);
                    const FTU* target_ftu = arena.get<const FTU>(offset);
                    if (target_ftu && target_ftu->magic == FTU_MAGIC_NUMBER) { current_type = CT_JUMP; }
                    else { current_type = CT_LEAF; }
                }
                continue;
            }
            if (current_type == CT_JUMP) {
                ArenaOffset offset = get_offset_from_nodes(ftu->nodes, node_idx);
                ftu = arena.get<FTU>(offset); if (!ftu) return false;
                current_type = ftu->root_node_type; node_idx = 0;
                parent_ftu = nullptr; parent_node_idx = FTU_NODE_CAPACITY;
                continue;
            }
            if (current_type == CT_LEAF) {
                ArenaOffset leaf_offset = get_offset_from_nodes(ftu->nodes, node_idx);
                const DataPayload* payload = arena.get<DataPayload>(leaf_offset); if (!payload) return false;
                std::string_view existing_key(payload->get_data(), payload->key_len);
                if (existing_key == key) return false;
                int crit_bit = find_first_differing_bit(key, existing_key);
                if (crit_bit == -1 || crit_bit / 8 >= 16) return false;
                if (ftu->nodes_used + 4 > FTU_NODE_CAPACITY) {
                    ArenaOffset new_ftu_offset = allocate_new_ftu();
                    FTU* new_ftu = arena.get<FTU>(new_ftu_offset);
                    set_offset_in_nodes(new_ftu->nodes, 0, leaf_offset);
                    new_ftu->root_node_type = CT_LEAF; new_ftu->nodes_used = 2;
                    set_offset_in_nodes(ftu->nodes, node_idx, new_ftu_offset);
                    if (!parent_ftu) { ftu->root_node_type = CT_JUMP; }
                    goto restart_insertion;
                }
                uint8_t internal_node_idx = node_idx; uint8_t child_base_idx = ftu->nodes_used;
                ftu->nodes_used += 4;
                ArenaOffset new_data_offset = allocate_data_payload(key, value);
                int bit = get_bit_at(key, crit_bit);
                uint8_t new_key_leaf_node_idx = (bit == 0) ? child_base_idx : child_base_idx + 2;
                uint8_t old_key_leaf_node_idx = (bit == 0) ? child_base_idx + 2 : child_base_idx;
                set_offset_in_nodes(ftu->nodes, new_key_leaf_node_idx, new_data_offset);
                set_offset_in_nodes(ftu->nodes, old_key_leaf_node_idx, leaf_offset);
                ftu->nodes[internal_node_idx].set_instruction_test(crit_bit, child_base_idx, NTB_OFFSET, NTB_OFFSET);
                if (parent_ftu) {
                    Node* parent_node = &parent_ftu->nodes[parent_node_idx];
                    uint8_t p_bit = parent_node->get_test_bit_index(); uint8_t p_child_base = parent_node->get_child_base_idx();
                    NodeTypeBit p_left_type = parent_node->get_left_type_bit(); NodeTypeBit p_right_type = parent_node->get_right_type_bit();
                    if (is_left_child) { parent_node->set_instruction_test(p_bit, p_child_base, NTB_INTERNAL, p_right_type); }
                    else { parent_node->set_instruction_test(p_bit, p_child_base, p_left_type, NTB_INTERNAL); }
                } else { ftu->root_node_type = CT_INTERNAL; }
                return true;
            }
            return false;
        }
    }
};
} // namespace aos

// --- Implementation 2: Structure of Arrays (SOA) ---
namespace soa {
struct FTU {
    uint32_t magic;
    uint8_t nodes_used;
    ChildType root_node_type;
    uint8_t padding[2];
    uint16_t node_data[FTU_NODE_CAPACITY];
};
inline uint8_t get_test_bit_index(const FTU* ftu, uint8_t node_idx) { return static_cast<uint8_t>((ftu->node_data[node_idx] >> 9) & 0x7F); }
inline uint8_t get_child_base_idx(const FTU* ftu, uint8_t node_idx) { return static_cast<uint8_t>((ftu->node_data[node_idx] >> 2) & 0x7F); }
inline NodeTypeBit get_left_type_bit(const FTU* ftu, uint8_t node_idx) { return static_cast<NodeTypeBit>((ftu->node_data[node_idx] >> 1) & 0x01); }
inline NodeTypeBit get_right_type_bit(const FTU* ftu, uint8_t node_idx) { return static_cast<NodeTypeBit>(ftu->node_data[node_idx] & 0x01); }
inline void set_instruction_test(FTU* ftu, uint8_t node_idx, uint8_t bit_index, uint8_t child_base_idx, NodeTypeBit left_type, NodeTypeBit right_type) {
    if (bit_index > MAX_BIT_INDEX || child_base_idx >= FTU_NODE_CAPACITY) throw std::out_of_range("Node instruction out of range");
    ftu->node_data[node_idx] = (static_cast<uint16_t>(bit_index) << 9) | (static_cast<uint16_t>(child_base_idx) << 2) | (static_cast<uint16_t>(left_type) << 1) | static_cast<uint16_t>(right_type);
}
inline ArenaOffset get_offset_from_nodes(const FTU* ftu, uint8_t index) { return (static_cast<uint32_t>(ftu->node_data[index]) << 16) | ftu->node_data[index + 1]; }
inline void set_offset_in_nodes(FTU* ftu, uint8_t index, ArenaOffset offset) {
    ftu->node_data[index] = static_cast<uint16_t>(offset >> 16);
    ftu->node_data[index + 1] = static_cast<uint16_t>(offset & 0xFFFF);
}
class FractalTree {
    MmapArena arena; ArenaOffset root_ftu_offset;
    ArenaOffset allocate_new_ftu() {
        ArenaOffset ftu_offset = arena.allocate(sizeof(FTU)); if (ftu_offset == 0) throw std::runtime_error("Arena out of memory for new FTU");
        FTU* ftu = arena.get<FTU>(ftu_offset);
        ftu->magic = FTU_MAGIC_NUMBER; ftu->nodes_used = 0; ftu->root_node_type = CT_UNUSED;
        memset(ftu->node_data, 0, sizeof(uint16_t) * FTU_NODE_CAPACITY); return ftu_offset;
    }
    ArenaOffset allocate_data_payload(std::string_view key, std::string_view value) {
        size_t total_size = sizeof(DataPayload) + key.length() + value.length();
        ArenaOffset offset = arena.allocate(total_size); if (offset == 0) throw std::runtime_error("Arena out of memory for data payload");
        DataPayload* payload = arena.get<DataPayload>(offset);
        payload->key_len = key.length(); payload->val_len = value.length();
        char* data_ptr = payload->get_data();
        memcpy(data_ptr, key.data(), key.length()); memcpy(data_ptr + key.length(), value.data(), value.length());
        return offset;
    }
public:
    FractalTree(const std::string& filename, size_t arena_size_mb) : arena(filename, arena_size_mb) { root_ftu_offset = allocate_new_ftu(); }
    void advise_memory_usage(int advice) { arena.advise(advice); }
    std::string get_name() const { return "FractalTree (SOA)"; }
    std::string find(std::string_view key) const {
        const FTU* ftu = arena.get<FTU>(root_ftu_offset);
        if (ftu->root_node_type == CT_UNUSED) return "";
        ChildType current_type = ftu->root_node_type; uint8_t node_idx = 0;
        while (true) {
            switch (current_type) {
                case CT_INTERNAL: {
                    int bit = get_bit_at(key, get_test_bit_index(ftu, node_idx));
                    uint8_t child_base_idx = get_child_base_idx(ftu, node_idx);
                    NodeTypeBit left_type = get_left_type_bit(ftu, node_idx);
                    NodeTypeBit right_type = get_right_type_bit(ftu, node_idx);
                    NodeTypeBit child_type_bit;
                    if (bit == 0) { node_idx = child_base_idx; child_type_bit = left_type; }
                    else { node_idx = child_base_idx + 2; child_type_bit = right_type; }
                    if (child_type_bit == NTB_INTERNAL) { current_type = CT_INTERNAL; }
                    else {
                        ArenaOffset offset = get_offset_from_nodes(ftu, node_idx);
                        const FTU* target_ftu_header = arena.get<const FTU>(offset);
                        if (target_ftu_header && target_ftu_header->magic == FTU_MAGIC_NUMBER) { current_type = CT_JUMP; }
                        else { current_type = CT_LEAF; }
                    }
                    break;
                }
                case CT_JUMP: {
                    ArenaOffset offset = get_offset_from_nodes(ftu, node_idx);
                    ftu = arena.get<FTU>(offset); if (!ftu) return "";
                    current_type = ftu->root_node_type; node_idx = 0;
                    break;
                }
                case CT_LEAF: {
                    ArenaOffset offset = get_offset_from_nodes(ftu, node_idx);
                    const DataPayload* payload = arena.get<DataPayload>(offset); if (!payload) return "";
                    std::string_view stored_key(payload->get_data(), payload->key_len);
                    if (key == stored_key) { return std::string(payload->get_data() + payload->key_len, payload->val_len); }
                    return "";
                }
                case CT_UNUSED: default: return "";
            }
        }
    }
    bool insert(std::string_view key, std::string_view value) {
        FTU* ftu = arena.get<FTU>(root_ftu_offset);
        if (ftu->root_node_type == CT_UNUSED) {
            ArenaOffset data_offset = allocate_data_payload(key, value);
            set_offset_in_nodes(ftu, 0, data_offset);
            ftu->root_node_type = CT_LEAF; ftu->nodes_used = 2; return true;
        }
        FTU* parent_ftu = nullptr; uint8_t parent_node_idx = FTU_NODE_CAPACITY; bool is_left_child = false;
    restart_insertion:
        ftu = arena.get<FTU>(root_ftu_offset);
        ChildType current_type = ftu->root_node_type; uint8_t node_idx = 0;
        parent_ftu = nullptr; parent_node_idx = FTU_NODE_CAPACITY; is_left_child = false;
        while (true) {
            if (current_type == CT_INTERNAL) {
                parent_ftu = ftu; parent_node_idx = node_idx;
                int bit = get_bit_at(key, get_test_bit_index(ftu, node_idx));
                uint8_t child_base_idx = get_child_base_idx(ftu, node_idx);
                NodeTypeBit left_type = get_left_type_bit(ftu, node_idx);
                NodeTypeBit right_type = get_right_type_bit(ftu, node_idx);
                NodeTypeBit child_type_bit;
                if (bit == 0) { node_idx = child_base_idx; child_type_bit = left_type; is_left_child = true; }
                else { node_idx = child_base_idx + 2; child_type_bit = right_type; is_left_child = false; }
                if (child_type_bit == NTB_INTERNAL) { current_type = CT_INTERNAL; }
                else {
                    ArenaOffset offset = get_offset_from_nodes(ftu, node_idx);
                    const FTU* target_ftu = arena.get<const FTU>(offset);
                    if (target_ftu && target_ftu->magic == FTU_MAGIC_NUMBER) { current_type = CT_JUMP; }
                    else { current_type = CT_LEAF; }
                }
                continue;
            }
            if (current_type == CT_JUMP) {
                ArenaOffset offset = get_offset_from_nodes(ftu, node_idx);
                ftu = arena.get<FTU>(offset); if (!ftu) return false;
                current_type = ftu->root_node_type; node_idx = 0;
                parent_ftu = nullptr; parent_node_idx = FTU_NODE_CAPACITY;
                continue;
            }
            if (current_type == CT_LEAF) {
                ArenaOffset leaf_offset = get_offset_from_nodes(ftu, node_idx);
                const DataPayload* payload = arena.get<DataPayload>(leaf_offset); if (!payload) return false;
                std::string_view existing_key(payload->get_data(), payload->key_len);
                if (existing_key == key) return false;
                int crit_bit = find_first_differing_bit(key, existing_key);
                if (crit_bit == -1 || crit_bit / 8 >= 16) return false;
                if (ftu->nodes_used + 4 > FTU_NODE_CAPACITY) {
                    ArenaOffset new_ftu_offset = allocate_new_ftu();
                    FTU* new_ftu = arena.get<FTU>(new_ftu_offset);
                    set_offset_in_nodes(new_ftu, 0, leaf_offset);
                    new_ftu->root_node_type = CT_LEAF; new_ftu->nodes_used = 2;
                    set_offset_in_nodes(ftu, node_idx, new_ftu_offset);
                    if (!parent_ftu) { ftu->root_node_type = CT_JUMP; }
                    goto restart_insertion;
                }
                uint8_t internal_node_idx = node_idx; uint8_t child_base_idx = ftu->nodes_used;
                ftu->nodes_used += 4;
                ArenaOffset new_data_offset = allocate_data_payload(key, value);
                int bit = get_bit_at(key, crit_bit);
                uint8_t new_key_leaf_node_idx = (bit == 0) ? child_base_idx : child_base_idx + 2;
                uint8_t old_key_leaf_node_idx = (bit == 0) ? child_base_idx + 2 : child_base_idx;
                set_offset_in_nodes(ftu, new_key_leaf_node_idx, new_data_offset);
                set_offset_in_nodes(ftu, old_key_leaf_node_idx, leaf_offset);
                set_instruction_test(ftu, internal_node_idx, crit_bit, child_base_idx, NTB_OFFSET, NTB_OFFSET);
                if (parent_ftu) {
                    uint8_t p_bit = get_test_bit_index(parent_ftu, parent_node_idx); uint8_t p_child_base = get_child_base_idx(parent_ftu, parent_node_idx);
                    NodeTypeBit p_left_type = get_left_type_bit(parent_ftu, parent_node_idx); NodeTypeBit p_right_type = get_right_type_bit(parent_ftu, parent_node_idx);
                    if (is_left_child) { set_instruction_test(parent_ftu, parent_node_idx, p_bit, p_child_base, NTB_INTERNAL, p_right_type); }
                    else { set_instruction_test(parent_ftu, parent_node_idx, p_bit, p_child_base, p_left_type, NTB_INTERNAL); }
                } else { ftu->root_node_type = CT_INTERNAL; }
                return true;
            }
            return false;
        }
    }
};
} // namespace soa

// --- Implementation 3: Hybrid (AOS/SOA) ---
namespace hybrid {

constexpr uint32_t NODE_IS_INSTRUCTION_BIT = 1u << 31;
constexpr uint32_t NODE_OFFSET_MASK = 0x7FFFFFFF;

struct FTU {
    uint32_t magic;
    uint8_t nodes_used;
    uint8_t root_node_idx;
    uint8_t padding[2];
    uint32_t nodes[FTU_NODE_CAPACITY];
};

inline bool is_instruction(uint32_t node_data) { return (node_data & NODE_IS_INSTRUCTION_BIT) != 0; }
inline ArenaOffset get_offset(uint32_t node_data) { return node_data & NODE_OFFSET_MASK; }
inline uint32_t make_offset_node(ArenaOffset offset) {
    if (offset > NODE_OFFSET_MASK) throw std::out_of_range("Arena offset too large for hybrid node");
    return offset;
}

inline uint8_t get_test_bit_index(uint32_t node_data) { return (node_data >> 23) & 0x7F; }
inline uint8_t get_left_child_idx(uint32_t node_data) { return (node_data >> 15) & 0x7F; }
inline uint8_t get_right_child_idx(uint32_t node_data) { return (node_data >> 7) & 0x7F; }

inline uint32_t make_instruction_node(uint8_t bit_index, uint8_t left_idx, uint8_t right_idx) {
    if (bit_index > MAX_BIT_INDEX || left_idx >= FTU_NODE_CAPACITY || right_idx >= FTU_NODE_CAPACITY) {
        throw std::out_of_range("Node instruction out of range");
    }
    return NODE_IS_INSTRUCTION_BIT | (static_cast<uint32_t>(bit_index) << 23) | (static_cast<uint32_t>(left_idx) << 15) | (static_cast<uint32_t>(right_idx) << 7);
}

class FractalTree {
    MmapArena arena;
    ArenaOffset root_ftu_offset;

    ArenaOffset allocate_new_ftu() {
        ArenaOffset ftu_offset = arena.allocate(sizeof(FTU));
        if (ftu_offset == 0) throw std::runtime_error("Arena out of memory for new FTU");
        FTU* ftu = arena.get<FTU>(ftu_offset);
        ftu->magic = FTU_MAGIC_NUMBER;
        ftu->nodes_used = 0;
        ftu->root_node_idx = 0;
        memset(ftu->nodes, 0, sizeof(uint32_t) * FTU_NODE_CAPACITY);
        return ftu_offset;
    }

    ArenaOffset allocate_data_payload(std::string_view key, std::string_view value) {
        size_t total_size = sizeof(DataPayload) + key.length() + value.length();
        ArenaOffset offset = arena.allocate(total_size);
        if (offset == 0) throw std::runtime_error("Arena out of memory for data payload");
        DataPayload* payload = arena.get<DataPayload>(offset);
        payload->key_len = key.length();
        payload->val_len = value.length();
        char* data_ptr = payload->get_data();
        memcpy(data_ptr, key.data(), key.length());
        memcpy(data_ptr + key.length(), value.data(), value.length());
        return offset;
    }

public:
    FractalTree(const std::string& filename, size_t arena_size_mb) : arena(filename, arena_size_mb) {
        root_ftu_offset = allocate_new_ftu();
    }
    void advise_memory_usage(int advice) { arena.advise(advice); }
    std::string get_name() const { return "FractalTree (Hybrid)"; }

    std::string find(std::string_view key) const {
        const FTU* ftu = arena.get<FTU>(root_ftu_offset);
        uint8_t node_idx = ftu->root_node_idx;

        while (true) {
            if (ftu->nodes_used == 0) return "";
            uint32_t node_data = ftu->nodes[node_idx];

            if (is_instruction(node_data)) {
                int bit = get_bit_at(key, get_test_bit_index(node_data));
                node_idx = (bit == 0) ? get_left_child_idx(node_data) : get_right_child_idx(node_data);
                continue;
            }

            ArenaOffset offset = get_offset(node_data);
            const auto* header_check = arena.get<FTU>(offset);
            if (header_check && header_check->magic == FTU_MAGIC_NUMBER) { // Jump to another FTU
                ftu = header_check;
                node_idx = ftu->root_node_idx;
                continue;
            }

            // Leaf node
            const DataPayload* payload = arena.get<DataPayload>(offset);
            if (!payload) return "";
            std::string_view stored_key(payload->get_data(), payload->key_len);
            if (key == stored_key) {
                return std::string(payload->get_data() + payload->key_len, payload->val_len);
            }
            return "";
        }
    }

    bool insert(std::string_view key, std::string_view value) {
        FTU* ftu = arena.get<FTU>(root_ftu_offset);

        if (ftu->nodes_used == 0) {
            ArenaOffset data_offset = allocate_data_payload(key, value);
            ftu->nodes[0] = make_offset_node(data_offset);
            ftu->nodes_used = 1;
            ftu->root_node_idx = 0;
            return true;
        }

        FTU* parent_ftu = nullptr;
        uint8_t parent_node_idx = FTU_NODE_CAPACITY;

    restart_insertion:
        ftu = arena.get<FTU>(root_ftu_offset);
        uint8_t node_idx = ftu->root_node_idx;
        parent_ftu = nullptr;
        parent_node_idx = FTU_NODE_CAPACITY;

        while (true) {
            uint32_t node_data = ftu->nodes[node_idx];

            if (is_instruction(node_data)) {
                parent_ftu = ftu;
                parent_node_idx = node_idx;
                int bit = get_bit_at(key, get_test_bit_index(node_data));
                node_idx = (bit == 0) ? get_left_child_idx(node_data) : get_right_child_idx(node_data);
                continue;
            }

            ArenaOffset offset = get_offset(node_data);
            const auto* header_check = arena.get<const FTU>(offset);

            if (header_check && header_check->magic == FTU_MAGIC_NUMBER) { // Jump
                ftu = arena.get<FTU>(offset);
                node_idx = ftu->root_node_idx;
                parent_ftu = nullptr;
                parent_node_idx = FTU_NODE_CAPACITY;
                continue;
            }

            // Leaf node
            const DataPayload* payload = arena.get<DataPayload>(offset);
            if (!payload) return false;
            std::string_view existing_key(payload->get_data(), payload->key_len);
            if (existing_key == key) return false; // Key already exists

            int crit_bit = find_first_differing_bit(key, existing_key);
            if (crit_bit == -1 || crit_bit / 8 >= 16) return false;

            if (ftu->nodes_used + 2 > FTU_NODE_CAPACITY) { // Need 2 slots for children
                ArenaOffset new_ftu_offset = allocate_new_ftu();
                FTU* new_ftu = arena.get<FTU>(new_ftu_offset);
                new_ftu->nodes[0] = make_offset_node(offset); // Move old leaf to new FTU
                new_ftu->nodes_used = 1;
                new_ftu->root_node_idx = 0;
                ftu->nodes[node_idx] = make_offset_node(new_ftu_offset); // Point old slot to new FTU
                goto restart_insertion;
            }

            ArenaOffset new_data_offset = allocate_data_payload(key, value);

            uint8_t old_leaf_child_idx = ftu->nodes_used++;
            uint8_t new_leaf_child_idx = ftu->nodes_used++;

            int bit = get_bit_at(key, crit_bit);
            uint8_t left_child_idx = (bit == 0) ? new_leaf_child_idx : old_leaf_child_idx;
            uint8_t right_child_idx = (bit == 0) ? old_leaf_child_idx : new_leaf_child_idx;

            ftu->nodes[old_leaf_child_idx] = make_offset_node(offset);
            ftu->nodes[new_leaf_child_idx] = make_offset_node(new_data_offset);

            ftu->nodes[node_idx] = make_instruction_node(crit_bit, left_child_idx, right_child_idx);

            return true;
        }
    }
};
} // namespace hybrid

const size_t BENCH_ARENA_SIZE_MB = 256;
const int BENCH_ITEM_COUNT = 1000000;
template<typename TreeType>
void run_fractal_tree_benchmark(const std::string& mmap_file_prefix, size_t arena_size_mb) {
    std::string tree_name;
    const std::string seq_mmap_file = mmap_file_prefix + "_sequential.mmap";
    {
        TreeType seq_tree(seq_mmap_file, arena_size_mb);
        tree_name = seq_tree.get_name();
        std::cout << "--- " << tree_name << " Benchmark ---" << std::endl;
        std::cout << "Items: " << BENCH_ITEM_COUNT << ", Arena: " << arena_size_mb << " MB" << std::endl;
        std::vector<std::string> seq_keys;
        seq_keys.reserve(BENCH_ITEM_COUNT);
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { char key_buf[17]; snprintf(key_buf, sizeof(key_buf), "key-%08d", i); seq_keys.push_back(key_buf); }
        seq_tree.advise_memory_usage(MADV_SEQUENTIAL);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { seq_tree.insert(seq_keys[i], "v"); }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long avg_insert_ns = BENCH_ITEM_COUNT > 0 ? duration.count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Sequential] Insertions: " << std::setw(5) << avg_insert_ns << " ns/op." << std::endl;
        seq_tree.advise_memory_usage(MADV_RANDOM);
        long long successful_finds = 0;
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { if (seq_tree.find(seq_keys[i]) == "v") { successful_finds++; } }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long avg_get_ns = BENCH_ITEM_COUNT > 0 ? duration.count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Sequential] Gets:         " << std::setw(5) << avg_get_ns << " ns/op." << std::endl;
        if (successful_finds != BENCH_ITEM_COUNT) { throw std::runtime_error(tree_name + " sequential verification failed! Found " + std::to_string(successful_finds) + " keys, expected " + std::to_string(BENCH_ITEM_COUNT)); }
    }
    const std::string rand_mmap_file = mmap_file_prefix + "_random.mmap";
    {
        TreeType rand_tree(rand_mmap_file, arena_size_mb);
        std::vector<std::string> rand_keys;
        rand_keys.reserve(BENCH_ITEM_COUNT);
        std::mt19937_64 rng(std::random_device{}());
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) {
            uint64_t r1 = rng(); uint64_t r2 = rng(); std::string rkey(16, ' ');
            memcpy(&rkey[0], &r1, 8); memcpy(&rkey[8], &r2, 8); rand_keys.push_back(rkey);
        }
        rand_tree.advise_memory_usage(MADV_SEQUENTIAL);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { rand_tree.insert(rand_keys[i], "v"); }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long avg_insert_ns = BENCH_ITEM_COUNT > 0 ? duration.count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Random]     Insertions: " << std::setw(5) << avg_insert_ns << " ns/op." << std::endl;
        rand_tree.advise_memory_usage(MADV_RANDOM);
        long long successful_finds = 0;
        start = std::chrono::high_resolution_clock::now();
        std::shuffle(rand_keys.begin(), rand_keys.end(), rng);
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { if (rand_tree.find(rand_keys[i]) == "v") { successful_finds++; } }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long avg_get_ns = BENCH_ITEM_COUNT > 0 ? duration.count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Random]     Gets:         " << std::setw(5) << avg_get_ns << " ns/op." << std::endl;
        if (successful_finds != BENCH_ITEM_COUNT) { throw std::runtime_error(tree_name + " random verification failed! Found " + std::to_string(successful_finds) + " keys, expected " + std::to_string(BENCH_ITEM_COUNT)); }
    }
}
template<typename ContainerType>
void run_std_container_benchmark(const std::string& container_name) {
    std::cout << "--- " << container_name << " Benchmark ---" << std::endl;
    std::cout << "Items: " << BENCH_ITEM_COUNT << std::endl;
    {
        ContainerType c; std::vector<std::string> keys; keys.reserve(BENCH_ITEM_COUNT);
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { char k[17]; snprintf(k, sizeof(k), "key-%08d", i); keys.push_back(k); }
        auto s = std::chrono::high_resolution_clock::now();
        for (const auto& k : keys) { c.insert({k, "v"}); }
        auto e = std::chrono::high_resolution_clock::now();
        long long avg_i = BENCH_ITEM_COUNT > 0 ? std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Sequential] Insertions: " << std::setw(5) << avg_i << " ns/op." << std::endl;
        long long finds = 0;
        s = std::chrono::high_resolution_clock::now();
        for (const auto& k : keys) { if (c.find(k) != c.end()) { finds++; } }
        e = std::chrono::high_resolution_clock::now();
        long long avg_g = BENCH_ITEM_COUNT > 0 ? std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Sequential] Gets:         " << std::setw(5) << avg_g << " ns/op." << std::endl;
        if (finds != BENCH_ITEM_COUNT) { throw std::runtime_error(container_name + " sequential verification failed!"); }
    }
    {
        ContainerType c; std::vector<std::string> keys; keys.reserve(BENCH_ITEM_COUNT);
        std::mt19937_64 rng(std::random_device{}());
        for (int i = 0; i < BENCH_ITEM_COUNT; ++i) { uint64_t r1 = rng(), r2 = rng(); std::string k(16, ' '); memcpy(&k[0], &r1, 8); memcpy(&k[8], &r2, 8); keys.push_back(k); }
        auto s = std::chrono::high_resolution_clock::now();
        for (const auto& k : keys) { c.insert({k, "v"}); }
        auto e = std::chrono::high_resolution_clock::now();
        long long avg_i = BENCH_ITEM_COUNT > 0 ? std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Random]     Insertions: " << std::setw(5) << avg_i << " ns/op." << std::endl;
        long long finds = 0;
        std::shuffle(keys.begin(), keys.end(), rng);
        s = std::chrono::high_resolution_clock::now();
        for (const auto& k : keys) { if (c.find(k) != c.end()) { finds++; } }
        e = std::chrono::high_resolution_clock::now();
        long long avg_g = BENCH_ITEM_COUNT > 0 ? std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count() / BENCH_ITEM_COUNT : 0;
        std::cout << "[Random]     Gets:         " << std::setw(5) << avg_g << " ns/op." << std::endl;
        if (finds != BENCH_ITEM_COUNT) { throw std::runtime_error(container_name + " random verification failed!"); }
    }
}

int main() {
    try {
        run_fractal_tree_benchmark<aos::FractalTree>("aos_tree", BENCH_ARENA_SIZE_MB);
        run_fractal_tree_benchmark<soa::FractalTree>("soa_tree", BENCH_ARENA_SIZE_MB);
        run_fractal_tree_benchmark<hybrid::FractalTree>("hybrid_tree", BENCH_ARENA_SIZE_MB);
        std::cout << "\n" << std::string(80, '=') << "\n" << std::endl;
        run_std_container_benchmark<std::unordered_map<std::string, std::string>>("std::unordered_map");
        std::cout << "\n" << std::string(80, '=') << "\n" << std::endl;
        run_std_container_benchmark<std::map<std::string, std::string>>("std::map");
    } catch (const std::exception& e) {
        std::cerr << "\n\n*** An error occurred: " << e.what() << " ***" << std::endl;
        return 1;
    }
    return 0;
}
