#include "test_harness.hpp"
#include "src/memory_arena.hpp"
#include "src/neuron.hpp"

// Simple simulation helper to manually propagate events
struct Simulation {
    MemoryArena<Neuron> arena;

    Simulation(size_t size) : arena(size) {}

    // Process one step: tick all neurons
    // Returns indices of neurons that fired
    std::vector<size_t> step(int64_t time) {
        std::vector<size_t> fired_indices;
        for (size_t i = 0; i < arena.size(); ++i) {
            Neuron& n = arena.get(i);

            // Capture fire event
            bool fired_this_tick = false;
            auto old_cb = n.on_fire;
            n.on_fire = [&](Neuron* self) {
                fired_this_tick = true;
                if (old_cb) old_cb(self);
            };

            n.tick(time);

            if (fired_this_tick) {
                fired_indices.push_back(i);
            }
            n.on_fire = old_cb; // Restore
        }
        return fired_indices;
    }

    // Propagate spikes from fired neurons to their targets (naively, instant or delayed manually)
    void propagate(const std::vector<size_t>& fired_indices, int64_t current_time, std::vector<std::pair<int64_t, std::pair<size_t, int32_t>>>& event_queue) {
         for (size_t src_idx : fired_indices) {
             Neuron& src = arena.get(src_idx);
             for (auto& syn : src.synapses) {
                 // Schedule event: time + delay -> target receives weight
                 event_queue.push_back({current_time + syn.delay, {syn.target_neuron_index, syn.weight}});
             }
         }
    }
};

// Test 1: Connect A -> B.
TEST(test_synapse_connect) {
    MemoryArena<Neuron> arena(10);
    size_t idxA = arena.allocate();
    size_t idxB = arena.allocate();

    Neuron& A = arena.get(idxA);
    // Connect A to B with weight 10, delay 1
    A.synapses.emplace_back(idxB, 10, 1);

    ASSERT_EQ(A.synapses.size(), 1);
    ASSERT_EQ(A.synapses[0].target_neuron_index, idxB);
    ASSERT_EQ(A.synapses[0].weight, 10);
}

// Test 2: Force A to fire. Assert B receives input after `delay` ticks.
TEST(test_synapse_propagation) {
    Simulation sim(10);
    size_t idxA = sim.arena.allocate();
    size_t idxB = sim.arena.allocate();

    Neuron& A = sim.arena.get(idxA);
    Neuron& B = sim.arena.get(idxB);

    // Connect A -> B, delay 2, weight 50
    A.synapses.emplace_back(idxB, 50, 2);

    // Events: (time, (target_idx, weight))
    std::vector<std::pair<int64_t, std::pair<size_t, int32_t>>> event_queue;

    // Force A to fire at time 0
    A.receive_impulse(110); // > threshold
    std::vector<size_t> fired = sim.step(0);

    ASSERT_EQ(fired.size(), 1);
    ASSERT_EQ(fired[0], idxA);

    sim.propagate(fired, 0, event_queue);

    ASSERT_EQ(event_queue.size(), 1);
    ASSERT_EQ(event_queue[0].first, 2); // Time 0 + Delay 2 = 2

    // Simulate time steps
    // Time 1: Nothing happens to B
    sim.step(1);
    ASSERT_EQ(B.membrane_potential, 0);

    // Time 2: B receives impulse (manual processing of queue)
    for (auto it = event_queue.begin(); it != event_queue.end(); ) {
        if (it->first == 2) {
            sim.arena.get(it->second.first).receive_impulse(it->second.second);
            it = event_queue.erase(it);
        } else {
            ++it;
        }
    }

    ASSERT_EQ(B.membrane_potential, 50);
}

// Test 3: Hebbian (STDP) - Potentiation
TEST(test_hebbian_potentiation) {
    MemoryArena<Neuron> arena(10);
    size_t idxA = arena.allocate();
    size_t idxB = arena.allocate();

    Neuron& A = arena.get(idxA);
    Neuron& B = arena.get(idxB);

    // A -> B
    A.synapses.emplace_back(idxB, 50, 1);
    Synapse& syn = A.synapses[0];
    int32_t initial_weight = syn.weight;

    // Fire A at t=10
    A.last_fire_time = 10;

    // Fire B at t=12 (B fires AFTER A input arrives approx)
    // Actually STDP rule: if Pre fires slightly before Post, strengthen.
    B.last_fire_time = 12;

    // Apply naive STDP
    int64_t dt = B.last_fire_time - A.last_fire_time; // 2
    if (dt > 0 && dt < 10) {
        syn.weight += 10; // Strengthen
    }

    ASSERT_EQ(syn.weight, initial_weight + 10);
}

// Test 4: Hebbian (STDP) - Depression
TEST(test_hebbian_depression) {
    MemoryArena<Neuron> arena(10);
    size_t idxA = arena.allocate();
    size_t idxB = arena.allocate();

    Neuron& A = arena.get(idxA);
    Neuron& B = arena.get(idxB);

    // A -> B
    A.synapses.emplace_back(idxB, 50, 1);
    Synapse& syn = A.synapses[0];
    int32_t initial_weight = syn.weight;

    // Fire A at t=10
    A.last_fire_time = 10;

    // B fired BEFORE A (or didn't fire). Let's say B fired at t=5.
    // This implies A didn't contribute to B's firing.
    B.last_fire_time = 5;

    // Apply naive STDP
    int64_t dt = B.last_fire_time - A.last_fire_time; // -5
    if (dt < 0 && dt > -10) {
        syn.weight -= 10; // Weaken
    }

    ASSERT_EQ(syn.weight, initial_weight - 10);
}

int main() {
    return TestHarness::instance().run_all();
}
