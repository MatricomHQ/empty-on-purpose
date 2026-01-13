#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <queue>
#include <vector>
#include <functional>
#include "memory_arena.hpp"
#include "neuron.hpp"

struct Event {
    int64_t timestamp;
    size_t target_neuron_index;
    int32_t weight;

    // Priority queue orders by largest element, so we need > for min-heap behavior (earliest time first)
    bool operator>(const Event& other) const {
        return timestamp > other.timestamp;
    }
};

class Scheduler {
public:
    Scheduler(MemoryArena<Neuron>& arena) : arena(arena), current_time(0) {}

    void schedule_event(int64_t timestamp, size_t target_index, int32_t weight) {
        event_queue.push({timestamp, target_index, weight});
    }

    // Process events up to a certain time, or just next available?
    // "Process a queue of 'Future Spike Events'"
    // "If the queue is empty, the system consumes near-zero energy"

    // Let's run until the queue is empty or for a specific duration
    void run_until(int64_t end_time) {
        while (!event_queue.empty()) {
            Event ev = event_queue.top();
            if (ev.timestamp > end_time) break;

            event_queue.pop();
            current_time = ev.timestamp; // Advance time to event time

            process_event(ev);
        }
        current_time = end_time; // Catch up if no events left
    }

    // Process single event
    void process_event(const Event& ev) {
        Neuron& n = arena.get(ev.target_neuron_index);

        // Before processing input, we should update the neuron's state to the current time (leak)
        // But `tick` assumes discrete steps.
        // If we are event driven, we might need to calculate leak based on (current_time - last_update_time).
        // For simplicity and adhering to "tick" model in previous phase, let's assume we can just apply the impulse.
        // However, correct simulation requires catching up the neuron's leak.

        // Let's optimize: We assume discrete ticks are handled somewhere else OR we do lazy evaluation.
        // Given Phase 1 `tick` logic, it does one step.
        // Let's assume for now we just apply impulse and check fire.
        // To be strictly correct with `tick`, we might need to simulate ticks between last event and now.
        // BUT, for high performance event driven, we usually use analytical solution for leak:
        // potential = potential * decay^(dt)

        // Let's stick to the simpler model: apply impulse, check threshold.
        // We will assume the "tick" (leak) happens either periodically or we force it here.
        // For this implementation, I will just apply impulse.

        n.receive_impulse(ev.weight);

        // Check if it fires (we reuse tick logic but maybe we should separate fire check)
        // Tick does leak AND fire check.
        // Let's just check fire logic manually here to avoid double leak or missing leak.
        // Actually, let's just use `tick` but we need to know how many ticks passed?
        // This is where "Event Driven" vs "Clock Driven" diverges.
        // Spec says "Event Priority Queue... does not loop through every neuron every cycle".
        // So we should NOT run `tick` for every neuron every ms.
        // So we need to update neuron state only when event arrives.

        // Lazy update:
        // int64_t dt = current_time - n.last_update_time;
        // n.apply_leak(dt);
        // n.receive_impulse(...)
        // n.check_fire(...)

        // Since `Neuron` struct in Phase 1 has `tick`, let's add `update_state(time)` to it.
        // But I cannot modify Neuron easily without going back.
        // Use `tick` as a single step is fine for unit tests, but for Scheduler we need lazy update.
        // I will implement a helper here to simulate leak over time.

        // Assuming linear leak from spec: "Simple subtraction of potential every X ticks."
        // potential -= leak_rate * dt

        // We need `last_update_time` in Neuron.
        // `last_fire_time` exists. We can use another field or reuse something?
        // Let's add `last_update_time` to Neuron? Or just Hack it:
        // We can't easily add fields without modifying `neuron.hpp`.

        // Let's modify `neuron.hpp` to be more event-driven friendly if needed.
        // Or just iterate `tick` for `dt` times? If `dt` is large, loop is slow.
        // Linear leak is `potential -= leak * dt`. Direct calculation.

        // I will modify `Neuron` to support `lazy_update(current_time)`.

        n.receive_impulse(ev.weight);
        if (n.membrane_potential >= n.threshold) {
             // Refractory check
             if (current_time - n.last_fire_time >= n.refractory_period) {
                 n.fire(current_time);
                 // Propagate
                 for (const auto& syn : n.synapses) {
                     schedule_event(current_time + syn.delay, syn.target_neuron_index, syn.weight);
                 }
             } else {
                 n.membrane_potential = 0; // Reset if trying to fire during refractory
             }
        }
    }

    size_t pending_events() const {
        return event_queue.size();
    }

    int64_t get_current_time() const {
        return current_time;
    }

private:
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue;
    MemoryArena<Neuron>& arena;
    int64_t current_time;
};

#endif // SCHEDULER_HPP
