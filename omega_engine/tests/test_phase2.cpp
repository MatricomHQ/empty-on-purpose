#include "../src/omega.hpp"
#include <cassert>
#include <iostream>

int main() {
    try {
        std::cout << "Testing Logic Gate (Inhibition)..." << std::endl;
        // Clean slate
        unlink("brain_logic.map");

        Omega::Brain brain("brain_logic.map", 100);

        // Setup Logic Gate
        // A -> C (+100)
        // B -> C (-100)
        // C -> D (+1)  (To verify C fired)

        uint32_t nA = brain.create_neuron(100);
        uint32_t nB = brain.create_neuron(100);
        uint32_t nC = brain.create_neuron(100); // Threshold 100
        uint32_t nD = brain.create_neuron(100); // Detector

        brain.connect(nA, nC, 100);
        brain.connect(nB, nC, -100);
        brain.connect(nC, nD, 1);

        std::cout << "Test 1: Fire A (Expect C fires -> D+=1)" << std::endl;
        brain.stimulate(nA, 100);
        brain.run_cycles();

        Omega::Neuron& d1 = brain.get_neuron(nD);
        std::cout << "D Potential: " << d1.potential << std::endl;
        assert(d1.potential == 1);

        // Reset D for next test?
        d1.potential = 0;
        // Also ensure A, B, C are reset.
        brain.get_neuron(nA).potential = 0;
        brain.get_neuron(nB).potential = 0;
        brain.get_neuron(nC).potential = 0;

        std::cout << "Test 2: Fire A and B (Expect C inhibited -> D+=0)" << std::endl;
        brain.stimulate(nA, 100);
        brain.stimulate(nB, 100);
        brain.run_cycles();

        Omega::Neuron& d2 = brain.get_neuron(nD);
        std::cout << "D Potential: " << d2.potential << std::endl;
        assert(d2.potential == 0);

        std::cout << "PASS: Logic Gate" << std::endl;

        unlink("brain_logic.map");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
