#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "memory_arena.hpp"
#include "neuron.hpp"
#include "scheduler.hpp"
#include <map>
#include <string>
#include <vector>
#include <iostream>

class Network {
public:
    Network(size_t neuron_capacity = 10000) : arena(neuron_capacity), scheduler(arena) {}

    // Add a neuron and return its index
    size_t add_neuron() {
        return arena.allocate();
    }

    // Connect two neurons
    void connect(size_t source, size_t target, int32_t weight, int32_t delay) {
        Neuron& src = arena.get(source);
        src.synapses.emplace_back(target, weight, delay);

        Neuron& tgt = arena.get(target);
        tgt.incoming_sources.push_back(source);
    }

    // Run simulation for duration
    void run(int64_t duration) {
        int64_t end_time = scheduler.get_current_time() + duration;
        scheduler.run_until(end_time);
    }

    // Stimulate a neuron
    void stimulate(size_t neuron_idx, int32_t weight) {
        // Schedule immediate event
        scheduler.schedule_event(scheduler.get_current_time() + 1, neuron_idx, weight);
    }

    // For debugging/output: Check if a neuron fired recently
    bool did_fire_recently(size_t neuron_idx, int64_t window) {
        Neuron& n = arena.get(neuron_idx);
        return (scheduler.get_current_time() - n.last_fire_time) <= window;
    }

    MemoryArena<Neuron> arena;
    Scheduler scheduler;
};

#endif // NETWORK_HPP
