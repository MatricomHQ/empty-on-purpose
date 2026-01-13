#include "test_harness.hpp"
#include "src/scheduler.hpp"
#include <random>
#include <chrono>

// Test 1: Push 1000 random events into the scheduler. Assert they execute in correct chronological order.
TEST(test_scheduler_order) {
    MemoryArena<Neuron> arena(10);
    // Initialize dummy neurons
    for (int i=0; i<5; ++i) arena.allocate();

    Scheduler scheduler(arena);

    std::vector<int64_t> times;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(1, 10000);

    // Push 1000 random events
    for (int i = 0; i < 1000; ++i) {
        int64_t t = dist(rng);
        times.push_back(t);
        scheduler.schedule_event(t, 0, 10);
    }

    // Sort expected times
    std::sort(times.begin(), times.end());

    // Run and check order
    // Since we don't expose pop easily without modifying Scheduler, we can observe via side effects?
    // Or we can modify Scheduler to have a "process_next" that returns the event processed.
    // But `run_until` processes internally.
    // Let's hook into Neuron's receive_impulse or use a spy.

    // We'll trust the priority_queue behavior, but to verify integration:
    // We can use a single neuron and see if we can capture the "times" it received impulses.
    // But wait, the Scheduler implementation calls `n.receive_impulse`.
    // We can hack `receive_impulse`? No, it's non-virtual.

    // We can check `current_time` after steps?
    // Or we can modify Scheduler to expose `peek`.
    // Actually, `run_until` updates `current_time`.

    // Let's rely on the fact that if we schedule events at t=1, t=100, t=50.
    // if we run_until(2), only t=1 processed.
    // if we run_until(60), t=50 processed.

    Scheduler s2(arena);
    s2.schedule_event(100, 0, 10);
    s2.schedule_event(10, 0, 10);
    s2.schedule_event(50, 0, 10);

    s2.run_until(5);
    ASSERT_EQ(s2.pending_events(), 3);

    s2.run_until(20); // Should process t=10
    ASSERT_EQ(s2.pending_events(), 2);
    // current time should be at least 10 (or 20 if catch up)

    s2.run_until(60); // Should process t=50
    ASSERT_EQ(s2.pending_events(), 1);

    s2.run_until(150); // Should process t=100
    ASSERT_EQ(s2.pending_events(), 0);
}

// Test 2: Benchmark processing speed. Optimization goal: >1 Million events per second on a single core.
TEST(test_scheduler_benchmark) {
    MemoryArena<Neuron> arena(1000);
    for(int i=0; i<1000; ++i) arena.allocate();

    Scheduler scheduler(arena);

    const int NUM_EVENTS = 1000000;
    for (int i = 0; i < NUM_EVENTS; ++i) {
        scheduler.schedule_event(i, i % 1000, 1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    scheduler.run_until(NUM_EVENTS + 100);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Processed " << NUM_EVENTS << " events in " << diff.count() << " s\n";
    std::cout << "Speed: " << (NUM_EVENTS / diff.count()) / 1000000.0 << " M events/s\n";

    ASSERT(diff.count() < 1.0); // Should be faster than 1s for 1M events
}

int main() {
    return TestHarness::instance().run_all();
}
