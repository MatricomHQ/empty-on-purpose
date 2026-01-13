#ifndef MEMORY_ARENA_HPP
#define MEMORY_ARENA_HPP

#include <vector>
#include <stdexcept>

template <typename T>
class MemoryArena {
public:
    explicit MemoryArena(size_t capacity) {
        pool.reserve(capacity);
    }

    // Allocate a new object in the pool and return its index
    size_t allocate() {
        if (pool.size() >= pool.capacity()) {
            throw std::runtime_error("MemoryArena: Out of memory");
        }
        pool.emplace_back();
        return pool.size() - 1;
    }

    // Get reference by index
    T& get(size_t index) {
        if (index >= pool.size()) {
            throw std::out_of_range("MemoryArena: Index out of range");
        }
        return pool[index];
    }

    // Get pointer by index (unsafe if reallocation happens, but we reserved)
    T* get_ptr(size_t index) {
         if (index >= pool.size()) {
            throw std::out_of_range("MemoryArena: Index out of range");
        }
        return &pool[index];
    }

    size_t size() const {
        return pool.size();
    }

    size_t capacity() const {
        return pool.capacity();
    }

    void clear() {
        pool.clear();
    }

private:
    std::vector<T> pool;
};

#endif // MEMORY_ARENA_HPP
