#include "../src/omega.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    try {
        std::cout << "Testing Phase 4: The Loop (Self-Awareness)..." << std::endl;
        unlink("brain_loop.map");
        Omega::Brain brain("brain_loop.map", 100);

        // We need to simulate "Hello" -> "Hi".
        // Let's manually create neurons for "Hello" and "Hi".
        // And link them: Hello -> Hi.

        uint32_t n_hello = brain.create_neuron(100);
        uint32_t n_hi = brain.create_neuron(100);

        // Link Hello -> Hi (Strong)
        brain.connect(n_hello, n_hi, 150);

        // Link Hi -> Hi (Loop? Or Hi -> ... -> Hi?)
        // Prompt says: "System inputs 'Hi' (Self-hearing). System avoids looping 'Hi Hi Hi' because of Refractory Periods."
        // So we need to feed Hi back to Input.
        // And if Hi fires, it triggers Input(Hi).
        // If Input(Hi) triggers Hi (Cortex), and Hi fires again...
        // We need Refractory Period to stop it.

        // Does our Neuron have Refractory Period?
        // `run_cycles`: "if (n.potential >= n.threshold) { n.potential = 0; ... }"
        // It resets to 0.
        // If we stimulate it immediately again, it starts from 0.
        // If we have fatigue, threshold increases? Or potential drops below 0?
        // Prompt implies we need to handle "Refractory Periods (Fatigue)".

        // Let's implement Fatigue in `omega.hpp`? Or is resetting to 0 enough if input is constant?
        // If Input is "Hi" (Voltage 100).
        // Hi fires. Reset to 0.
        // Next tick: Input still "Hi" (Voltage 100).
        // Hi += 100. Fires.
        // It loops forever.

        // We need a Refractory Period where it CANNOT fire or Threshold is higher.
        // Or "Hyperpolarization": Potential = -100 after fire.

        // Action: Modify `omega.hpp` to set Potential = -Threshold after fire (Hyperpolarization).
        // Or add `last_fired_time` check in `run_cycles` to ignore inputs for K ticks?

        // Let's update `omega.hpp` to implement Refractory Period.
        // If (current_tick - last_fired_time < REFRACTORY_PERIOD) -> Ignore input or Don't fire.

        // Setup Test:
        // 1. Stimulate Hello.
        // 2. Run cycles.
        // 3. Check if Hi fired.
        // 4. If Hi fired, Feed Hi (Stimulate Hi).
        // 5. Run cycles.
        // 6. Check if Hi fires AGAIN immediately. It should NOT.

        std::cout << "Step 1: Feed Hello" << std::endl;
        brain.stimulate(n_hello, 150);
        brain.run_cycles(10);

        // Check if Hi fired.
        bool hi_fired = false;
        for(uint32_t id : brain.fired_neurons_in_last_run) {
            if (id == n_hi) hi_fired = true;
        }

        // Actually, `fired_neurons_in_last_run` is cleared at start of `run_cycles`.
        // So we need to check if it fired during the run.
        // We can check `brain.last_fired_time[n_hi]`.

        if (brain.last_fired_time[n_hi] > 0) {
            std::cout << "Hi fired! (Response)" << std::endl;
        } else {
            std::cerr << "Hi did NOT fire." << std::endl;
            return 1;
        }

        // Step 2: Feedback Loop (Self-hearing)
        std::cout << "Step 2: Feedback Hi (Self-hearing)" << std::endl;
        // Stimulate Hi immediately
        brain.stimulate(n_hi, 150);

        // Run cycles again
        brain.run_cycles(10);

        // Check if Hi fired again
        // We need to know if it fired *in this run*.
        // `fired_neurons_in_last_run` contains IDs fired in THIS call.

        bool hi_looped = false;
        for(uint32_t id : brain.fired_neurons_in_last_run) {
            if (id == n_hi) hi_looped = true;
        }

        if (hi_looped) {
            std::cerr << "Hi looped! (Refractory failure)" << std::endl;
            std::cout << "Potential: " << brain.get_neuron(n_hi).potential << std::endl;
            // It might fail because I haven't implemented Refractory yet.
        } else {
            std::cout << "Hi did NOT loop. (Refractory success)" << std::endl;
        }

        unlink("brain_loop.map");

        if (hi_looped) return 1;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
