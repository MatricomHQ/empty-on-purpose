#ifndef NEURON_HPP
#define NEURON_HPP

#include <cstdint>
#include <functional>
#include <vector>
#include "synapse.hpp"

struct Neuron {
    std::vector<Synapse> synapses; // Adjacency list (Outgoing)
    std::vector<size_t> incoming_sources; // Incoming connections (for STDP)
    int32_t membrane_potential;
    int32_t threshold;
    int32_t leak_rate;
    int32_t refractory_period;
    int64_t last_fire_time;

    // Callback for when neuron fires
    std::function<void(Neuron*)> on_fire;

    Neuron()
        : membrane_potential(0),
          threshold(100),
          leak_rate(5),
          refractory_period(10),
          last_fire_time(-1000), // Start available
          on_fire(nullptr) {}

    void receive_impulse(int32_t input) {
        membrane_potential += input;
    }

    // Process time step: leak and check threshold
    // Returns true if fired
    bool tick(int64_t current_time) {
        // Refractory period check
        if (current_time - last_fire_time < refractory_period) {
            // Cannot fire, but potential might still decay or stay reset
            membrane_potential = 0; // Or keep it low? Spec doesn't strictly say, but usually reset.
            return false;
        }

        // Leak
        if (membrane_potential > 0) {
            membrane_potential -= leak_rate;
            if (membrane_potential < 0) membrane_potential = 0;
        } else if (membrane_potential < 0) {
            membrane_potential += leak_rate;
            if (membrane_potential > 0) membrane_potential = 0;
        }

        // Fire check
        if (membrane_potential >= threshold) {
            fire(current_time);
            return true;
        }
        return false;
    }

    void fire(int64_t current_time) {
        membrane_potential = 0; // Reset
        last_fire_time = current_time;
        if (on_fire) {
            on_fire(this);
        }
    }
};

#endif // NEURON_HPP
