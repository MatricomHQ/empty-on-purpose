#include "test_harness.hpp"
#include "src/memory_arena.hpp"
#include "src/neuron.hpp"

// Test 1: Assert potential increases when receive_impulse() is called.
TEST(test_neuron_impulse) {
    MemoryArena<Neuron> arena(10);
    size_t idx = arena.allocate();
    Neuron& n = arena.get(idx);

    n.membrane_potential = 0;
    n.receive_impulse(10);
    ASSERT_EQ(n.membrane_potential, 10);
    n.receive_impulse(5);
    ASSERT_EQ(n.membrane_potential, 15);
}

// Test 2: Assert potential decreases over simulated time (Leak).
TEST(test_neuron_leak) {
    MemoryArena<Neuron> arena(10);
    size_t idx = arena.allocate();
    Neuron& n = arena.get(idx);

    n.membrane_potential = 50;
    n.leak_rate = 5;
    n.tick(0); // Tick at time 0
    // After 1 tick, it should leak 5
    ASSERT_EQ(n.membrane_potential, 45);

    n.tick(1);
    ASSERT_EQ(n.membrane_potential, 40);
}

// Test 3: Assert fire() callback is triggered ONLY when threshold is crossed.
TEST(test_neuron_fire) {
    MemoryArena<Neuron> arena(10);
    size_t idx = arena.allocate();
    Neuron& n = arena.get(idx);

    n.threshold = 100;
    n.membrane_potential = 90;

    bool fired = false;
    n.on_fire = [&](Neuron*) { fired = true; };

    n.tick(0);
    ASSERT(fired == false); // Should not fire yet
    ASSERT_EQ(n.membrane_potential, 85); // Leaked

    n.receive_impulse(20); // 85 + 20 = 105
    n.tick(1);
    ASSERT(fired == true);
    ASSERT_EQ(n.membrane_potential, 0); // Reset after fire
}

// Test 4: Assert refractory_period prevents firing immediately after a previous fire.
TEST(test_neuron_refractory) {
    MemoryArena<Neuron> arena(10);
    size_t idx = arena.allocate();
    Neuron& n = arena.get(idx);

    n.threshold = 100;
    n.refractory_period = 10;

    // Force fire at time 0
    int fire_count = 0;
    n.on_fire = [&](Neuron*) { fire_count++; };

    n.receive_impulse(110);
    n.tick(0);
    ASSERT_EQ(fire_count, 1);
    ASSERT_EQ(n.last_fire_time, 0);

    // Try to fire again at time 5 (inside refractory period)
    n.receive_impulse(200);
    n.tick(5);
    ASSERT_EQ(fire_count, 1); // Should NOT have fired again
    ASSERT_EQ(n.membrane_potential, 0); // Should remain reset/low
}

int main() {
    return TestHarness::instance().run_all();
}
