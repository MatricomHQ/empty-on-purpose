#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

#include <cstdint>

struct Synapse {
    size_t target_neuron_index;
    int32_t weight;
    int32_t delay;
    int32_t permanence;

    Synapse(size_t target, int32_t w, int32_t d)
        : target_neuron_index(target), weight(w), delay(d), permanence(50) {}
};

#endif // SYNAPSE_HPP
