#include "../src/omega.hpp"
#include <cassert>
#include <iostream>

void test_allocation() {
    std::cout << "Testing Allocation..." << std::endl;
    Omega::Brain brain("brain.map", 100); // 100MB
    assert(brain.memory_ != nullptr);
    assert(brain.header_->magic == 0xDEADBEEF);
    std::cout << "PASS: Allocation" << std::endl;
}

void test_linkage() {
    std::cout << "Testing Linkage..." << std::endl;
    Omega::Brain brain("brain.map", 100);

    uint32_t n1 = brain.create_neuron(100);
    uint32_t n2 = brain.create_neuron(100);

    brain.connect(n1, n2, 50);

    // Verify
    Omega::Neuron& neuron1 = brain.get_neuron(n1);
    uint32_t syn_idx = neuron1.synapse_head_index;
    assert(syn_idx != 0);

    Omega::Synapse& s = brain.synapses_[syn_idx];
    assert(s.target_neuron == n2);
    assert(s.weight == 50);
    std::cout << "PASS: Linkage" << std::endl;
}

void test_stimulation() {
    std::cout << "Testing Stimulation..." << std::endl;
    Omega::Brain brain("brain.map", 100);

    // Create new neurons to avoid previous state issues (or reset)
    // Assuming persistent file, we just add new ones.
    uint32_t n1 = brain.create_neuron(100);
    uint32_t n2 = brain.create_neuron(100);

    brain.connect(n1, n2, 100); // weight 100 triggers n2 (threshold 100)

    brain.stimulate(n1, 100); // Fire N1
    brain.run_cycles();

    // Test N3->N4
    uint32_t n3 = brain.create_neuron(100);
    uint32_t n4 = brain.create_neuron(200);
    brain.connect(n3, n4, 100);

    brain.stimulate(n3, 110); // Fire N3
    brain.run_cycles();

    Omega::Neuron& neuron4 = brain.get_neuron(n4);
    std::cout << "Neuron 4 Potential: " << neuron4.potential << std::endl;
    assert(neuron4.potential == 100); // 0 + 100 = 100. Threshold 200 not reached.
    std::cout << "PASS: Stimulation" << std::endl;
}

void test_persistence() {
    std::cout << "Testing Persistence..." << std::endl;
    // Already using same file "brain.map" across tests?
    // The OS might keep it cached, but let's close and reopen explicitly in new scope.

    uint32_t saved_n1, saved_n2;
    {
        Omega::Brain brain("brain_persist.map", 100);
        saved_n1 = brain.create_neuron();
        saved_n2 = brain.create_neuron();
        brain.connect(saved_n1, saved_n2, 77);
    } // closed

    {
        Omega::Brain brain("brain_persist.map", 100);
        Omega::Neuron& n1 = brain.get_neuron(saved_n1);
        uint32_t syn_idx = n1.synapse_head_index;
        assert(syn_idx != 0);
        Omega::Synapse& s = brain.synapses_[syn_idx];
        assert(s.target_neuron == saved_n2);
        assert(s.weight == 77);
    }
    std::cout << "PASS: Persistence" << std::endl;
}

int main() {
    try {
        // Clean up
        unlink("brain.map");
        unlink("brain_persist.map");

        test_allocation();
        test_linkage();
        test_stimulation();
        test_persistence();

        std::cout << "Phase 1 Complete." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
