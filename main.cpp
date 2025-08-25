#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <limits>

// --- Handle Types & Node Structs ---
using NodeHandle16 = uint16_t;
using NodeHandle32 = uint32_t;

// The primary 4-byte node
struct Node { NodeHandle16 left; NodeHandle16 right; };

// A larger node for the "leap space" for nodes that have been upgraded via COW
struct LeapCowNode { uint32_t bit_index; NodeHandle32 left; NodeHandle32 right; };

class CritBitTree {
public:
    CritBitTree();
    void insert(const std::string& key_str);
    bool get(const std::string& key_str) const;

private:
    NodeHandle16 root_handle;
    std::vector<Node> nodes;
    std::vector<uint32_t> bit_indices;
    std::vector<std::vector<unsigned char>> leaves;
    std::vector<LeapCowNode> leap_space;

    // --- Handle System Constants ---
    static constexpr uint16_t HANDLE16_LEAF_FLAG = 0x8000;
    static constexpr uint16_t HANDLE16_INDEX_MASK = 0x7FFF;
    static constexpr uint32_t HANDLE32_LEAF_FLAG = 0x80000000;
    static constexpr uint32_t HANDLE32_LEAP_FLAG = 0x40000000; // Points to leap_space
    static constexpr uint32_t HANDLE32_INDEX_MASK = 0x3FFFFFFF;
    static constexpr uint32_t REDIRECT_FLAG = std::numeric_limits<uint32_t>::max();

    // --- Handle Helper Methods ---
    bool is_leaf16(NodeHandle16 h) const;
    uint16_t get_index16(NodeHandle16 h) const;
    NodeHandle16 make_leaf_handle16(uint16_t i) const;
    NodeHandle16 make_internal_handle16(uint16_t i) const;
    bool is_leaf32(NodeHandle32 h) const;
    bool is_leap32(NodeHandle32 h) const;
    uint32_t get_index32(NodeHandle32 h) const;
    NodeHandle32 make_leaf_handle32(uint32_t i) const;
    NodeHandle32 make_internal_handle32(uint32_t i) const;
    NodeHandle32 make_leap_handle32(uint32_t i) const;
    NodeHandle32 convert_handle16_to_32(NodeHandle16 h16) const;
    NodeHandle16 convert_handle32_to_16(NodeHandle32 h32) const;

    int get_bit(const std::vector<unsigned char>& key, uint32_t bit_index) const;
    uint32_t perform_cow_leap(uint16_t node_idx_16);
    NodeHandle32 find_closest_leaf(const std::vector<unsigned char>& key) const;
};

// --- Constructor ---
CritBitTree::CritBitTree() : root_handle(0) {
    nodes.reserve(HANDLE16_INDEX_MASK + 1);
    bit_indices.reserve(HANDLE16_INDEX_MASK + 1);
    leaves.reserve(HANDLE16_INDEX_MASK + 1);
    // Reserve index 0 in all pools as a sentinel for null handles.
    nodes.push_back({0, 0});
    bit_indices.push_back(0);
    leaves.emplace_back();
    leap_space.emplace_back();
}

// --- Handle Helper Implementations ---
bool CritBitTree::is_leaf16(NodeHandle16 h) const { return (h & HANDLE16_LEAF_FLAG) != 0; }
uint16_t CritBitTree::get_index16(NodeHandle16 h) const { return h & HANDLE16_INDEX_MASK; }
NodeHandle16 CritBitTree::make_leaf_handle16(uint16_t i) const { return i | HANDLE16_LEAF_FLAG; }
NodeHandle16 CritBitTree::make_internal_handle16(uint16_t i) const { return i; }
bool CritBitTree::is_leaf32(NodeHandle32 h) const { return (h & HANDLE32_LEAF_FLAG) != 0; }
bool CritBitTree::is_leap32(NodeHandle32 h) const { return (h & HANDLE32_LEAP_FLAG) != 0; }
uint32_t CritBitTree::get_index32(NodeHandle32 h) const { return h & HANDLE32_INDEX_MASK; }
NodeHandle32 CritBitTree::make_leaf_handle32(uint32_t i) const { return i | HANDLE32_LEAF_FLAG; }
NodeHandle32 CritBitTree::make_internal_handle32(uint32_t i) const { return i; }
NodeHandle32 CritBitTree::make_leap_handle32(uint32_t i) const { return i | HANDLE32_LEAP_FLAG; }

NodeHandle32 CritBitTree::convert_handle16_to_32(NodeHandle16 h16) const {
    if (is_leaf16(h16)) return make_leaf_handle32(get_index16(h16));
    return make_internal_handle32(get_index16(h16));
}

NodeHandle16 CritBitTree::convert_handle32_to_16(NodeHandle32 h32) const {
    if (is_leaf32(h32)) return make_leaf_handle16(get_index32(h32));
    if (is_leap32(h32)) throw std::logic_error("Cannot downcast a leap handle to 16-bit");
    return make_internal_handle16(get_index32(h32));
}

int CritBitTree::get_bit(const std::vector<unsigned char>& key, uint32_t bit_index) const {
    uint32_t byte_pos = bit_index / 8;
    if (byte_pos >= key.size()) return 0;
    return (key[byte_pos] >> (7 - (bit_index % 8))) & 1;
}

// --- Core Method Implementations ---

uint32_t CritBitTree::perform_cow_leap(uint16_t node_idx_16) {
    if (bit_indices[node_idx_16] == REDIRECT_FLAG) {
        return *reinterpret_cast<const uint32_t*>(&nodes[node_idx_16]);
    }
    uint32_t leap_idx = leap_space.size();
    leap_space.push_back({
        bit_indices[node_idx_16],
        convert_handle16_to_32(nodes[node_idx_16].left),
        convert_handle16_to_32(nodes[node_idx_16].right)
    });
    bit_indices[node_idx_16] = REDIRECT_FLAG;
    *reinterpret_cast<uint32_t*>(&nodes[node_idx_16]) = leap_idx;
    return leap_idx;
}

NodeHandle32 CritBitTree::find_closest_leaf(const std::vector<unsigned char>& key) const {
    if (root_handle == 0) return 0;
    NodeHandle32 current_h = convert_handle16_to_32(root_handle);
    while (!is_leaf32(current_h)) {
        if (is_leap32(current_h)) {
            uint32_t leap_idx = get_index32(current_h);
            const auto& leap_node = leap_space[leap_idx];
            int bit = get_bit(key, leap_node.bit_index);
            current_h = (bit == 0) ? leap_node.left : leap_node.right;
        } else {
            uint32_t node_idx = get_index32(current_h);
            if (bit_indices[node_idx] == REDIRECT_FLAG) {
                current_h = make_leap_handle32(*reinterpret_cast<const uint32_t*>(&nodes[node_idx]));
                continue;
            }
            const auto& node = nodes[node_idx];
            int bit = get_bit(key, bit_indices[node_idx]);
            current_h = convert_handle16_to_32((bit == 0) ? node.left : node.right);
        }
    }
    return current_h;
}

bool CritBitTree::get(const std::string& key_str) const {
    std::vector<unsigned char> key(key_str.begin(), key_str.end());
    if (root_handle == 0) return false;
    NodeHandle32 leaf_h = find_closest_leaf(key);
    if (leaf_h == 0) return false;
    return key == leaves[get_index32(leaf_h)];
}

void CritBitTree::insert(const std::string& key_str) {
    std::vector<unsigned char> key(key_str.begin(), key_str.end());
    if (root_handle == 0) {
        root_handle = make_leaf_handle16(leaves.size());
        leaves.push_back(key);
        return;
    }

    NodeHandle32 closest_h = find_closest_leaf(key);
    const auto& existing_key = leaves[get_index32(closest_h)];
    if (existing_key == key) return;

    uint32_t crit_bit_index = 0;
    uint32_t max_len = std::max(key.size(), existing_key.size());
    bool found_diff = false;
    for (uint32_t i = 0; i < max_len * 8; ++i) {
        if (get_bit(key, i) != get_bit(existing_key, i)) { crit_bit_index = i; found_diff = true; break; }
    }
    if (!found_diff) crit_bit_index = max_len * 8;

    NodeHandle32 parent_h = 0;
    NodeHandle32 current_h = convert_handle16_to_32(root_handle);
    while (true) {
        if (is_leaf32(current_h)) break;

        uint32_t node_bit_index;
        bool current_is_leap_handle = is_leap32(current_h);
        uint32_t node_idx = get_index32(current_h);

        if (current_is_leap_handle) {
            node_bit_index = leap_space[node_idx].bit_index;
        } else {
             if (bit_indices[node_idx] == REDIRECT_FLAG) {
                current_h = make_leap_handle32(*reinterpret_cast<const uint32_t*>(&nodes[node_idx]));
                continue;
            }
            node_bit_index = bit_indices[node_idx];
        }

        if (node_bit_index > crit_bit_index) break;
        parent_h = current_h;

        if (is_leap32(current_h)) {
            const auto& leap_node = leap_space[get_index32(current_h)];
            int bit = get_bit(key, leap_node.bit_index);
            current_h = (bit == 0) ? leap_node.left : leap_node.right;
        } else {
            const auto& node = nodes[get_index32(current_h)];
            int bit = get_bit(key, bit_indices[get_index32(current_h)]);
            current_h = convert_handle16_to_32((bit == 0) ? node.left : node.right);
        }
    }

    bool must_use_leap_node = (nodes.size() >= HANDLE16_INDEX_MASK || leaves.size() >= HANDLE16_INDEX_MASK);
    if (parent_h != 0 && is_leap32(parent_h)) {
        must_use_leap_node = true;
    }
    if (must_use_leap_node && parent_h != 0 && !is_leap32(parent_h)) {
        uint32_t leap_idx = perform_cow_leap(get_index32(parent_h));
        parent_h = make_leap_handle32(leap_idx);
    }

    NodeHandle32 new_leaf_h = make_leaf_handle32(leaves.size());
    leaves.push_back(key);
    int new_key_bit = get_bit(key, crit_bit_index);
    NodeHandle32 child_a = (new_key_bit == 0) ? new_leaf_h : current_h;
    NodeHandle32 child_b = (new_key_bit == 0) ? current_h : new_leaf_h;
    NodeHandle32 new_internal_h;

    if (must_use_leap_node) {
        uint32_t new_leap_idx = leap_space.size();
        leap_space.push_back({crit_bit_index, child_a, child_b});
        new_internal_h = make_leap_handle32(new_leap_idx);
    } else {
        uint16_t new_node_idx = nodes.size();
        bit_indices.push_back(crit_bit_index);
        nodes.push_back({convert_handle32_to_16(child_a), convert_handle32_to_16(child_b)});
        new_internal_h = make_internal_handle32(new_node_idx);
    }

    if (parent_h == 0) {
        root_handle = convert_handle32_to_16(new_internal_h);
    } else {
        if (is_leap32(parent_h)) {
            const auto& parent_data = leap_space[get_index32(parent_h)];
            int bit = get_bit(key, parent_data.bit_index);
            if (bit == 0) leap_space[get_index32(parent_h)].left = new_internal_h;
            else leap_space[get_index32(parent_h)].right = new_internal_h;
        } else {
            const auto& parent_data = bit_indices[get_index32(parent_h)];
            int bit = get_bit(key, parent_data);
            if (bit == 0) nodes[get_index32(parent_h)].left = convert_handle32_to_16(new_internal_h);
            else nodes[get_index32(parent_h)].right = convert_handle32_to_16(new_internal_h);
        }
    }
}

// --- Test Functions ---
void run_correctness_tests() {
    std::cout << "--- Running Correctness Tests ---" << std::endl;
    CritBitTree tree;
    std::vector<std::string> keys = {"test", "team", "toast", "tesla", "a", "b", "c", ""};
    for (const auto& key : keys) tree.insert(key);

    bool all_passed = true;
    for (const auto& key : keys) {
        if (!tree.get(key)) { std::cerr << "TEST FAILED: Could not find key '" << key << "'" << std::endl; all_passed = false; }
    }
    std::vector<std::string> not_found_keys = {"tes", "te", "toastz", "d", " "};
    for (const auto& key : not_found_keys) {
        if (tree.get(key)) { std::cerr << "TEST FAILED: Found key '" << key << "' that should not exist" << std::endl; all_passed = false; }
    }
    std::cout << (all_passed ? "All correctness tests passed!" : "Some correctness tests failed.") << std::endl;
}

void run_stress_test() {
    std::cout << "\n--- Running Stress Test (1M keys) ---" << std::endl;
    CritBitTree tree;
    size_t num_keys = 1000000;
    std::vector<std::string> keys;
    keys.reserve(num_keys);

    std::mt19937_64 rng(0);
    for (size_t i = 0; i < num_keys; ++i) {
        uint64_t val = rng();
        keys.push_back(std::string(reinterpret_cast<char*>(&val), 8));
    }

    auto start_insert = std::chrono::high_resolution_clock::now();
    try {
        for (const auto& key : keys) tree.insert(key);
    } catch (const std::exception& e) {
        std::cerr << "Stress test failed during insertion: " << e.what() << std::endl;
        return;
    }
    auto end_insert = std::chrono::high_resolution_clock::now();
    auto insert_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_insert - start_insert);
    std::cout << "Insert " << num_keys << " keys: " << insert_duration.count() << " ms ("
              << (double)insert_duration.count() * 1000000 / num_keys << " ns/key)" << std::endl;

    std::shuffle(keys.begin(), keys.end(), std::mt19937(0));
    auto start_get = std::chrono::high_resolution_clock::now();
    bool all_found = true;
    for (const auto& key : keys) {
        if (!tree.get(key)) { all_found = false; break; }
    }
    auto end_get = std::chrono::high_resolution_clock::now();

    if (!all_found) {
        std::cerr << "STRESS TEST FAILED: Some keys could not be found." << std::endl;
    } else {
        std::cout << "STRESS TEST PASSED: All keys found successfully." << std::endl;
        auto get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_get - start_get);
        std::cout << "Get " << num_keys << " keys: " << get_duration.count() << " ms ("
                  << (double)get_duration.count() * 1000000 / num_keys << " ns/key)" << std::endl;
    }
}

int main() {
    run_correctness_tests();
    run_stress_test();
    return 0;
}
