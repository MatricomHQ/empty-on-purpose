#include "src/network.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>

// Simple tokenizer: split by space, remove punctuation
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : text) {
        if (std::isalpha(c)) {
            current += std::tolower(c);
        } else if (!current.empty()) {
            tokens.push_back(current);
            current = "";
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

int main() {
    std::cout << "Initializing ENMA (Elastic Neuro-Mesh Architecture)..." << std::endl;

    // 1. Load Data
    std::ifstream file("training_data.txt");
    if (!file.is_open()) {
        std::cerr << "Could not open training_data.txt" << std::endl;
        return 1;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::vector<std::string> tokens = tokenize(buffer.str());

    if (tokens.empty()) {
        std::cerr << "No tokens found." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << tokens.size() << " tokens." << std::endl;

    // 2. Setup Network
    Network net(2000);
    std::unordered_map<std::string, size_t> word_to_neuron;
    std::vector<std::string> neuron_to_word;

    auto get_or_create_neuron = [&](const std::string& word) {
        if (word_to_neuron.find(word) == word_to_neuron.end()) {
            size_t idx = net.add_neuron();
            word_to_neuron[word] = idx;
            neuron_to_word.push_back(word);

            Neuron& n = net.arena.get(idx);
            n.threshold = 500;
            n.leak_rate = 20;

            n.on_fire = [&net, idx](Neuron* self) {
                int64_t now = net.scheduler.get_current_time();

                // 1. LTP (Long-Term Potentiation): Strengthen connections from neurons that fired recently
                for (size_t src_idx : self->incoming_sources) {
                    Neuron& src = net.arena.get(src_idx);
                    int64_t dt = now - src.last_fire_time;

                    // Window of 15 ticks (words are spaced by 10 ticks)
                    if (dt > 0 && dt <= 15) {
                        for (auto& syn : src.synapses) {
                            if (syn.target_neuron_index == idx) {
                                syn.weight += 50;
                                if (syn.weight > 500) syn.weight = 500;
                                break;
                            }
                        }
                    }
                }

                // 2. LTD (Long-Term Depression): Weaken connections to neurons that fired BEFORE we fired (anti-causal)
                // If I fire now, and target fired recently, it means target fired BEFORE me.
                // So My firing did NOT cause Target. My connection to Target should be weak.
                for (auto& syn : self->synapses) {
                    Neuron& target = net.arena.get(syn.target_neuron_index);
                    int64_t dt = now - target.last_fire_time;

                     if (dt > 0 && dt < 20) {
                        syn.weight -= 20;
                        if (syn.weight < 0) syn.weight = 0;
                     }
                }
            };

            return idx;
        }
        return word_to_neuron[word];
    };

    // Initialize neurons
    for (const auto& t : tokens) {
        get_or_create_neuron(t);
    }

    std::cout << "Vocabulary size: " << neuron_to_word.size() << std::endl;

    // Connect all to all with 0 weight
    for (size_t i = 0; i < neuron_to_word.size(); ++i) {
        for (size_t j = 0; j < neuron_to_word.size(); ++j) {
            if (i == j) continue;
            net.connect(i, j, 0, 5);
        }
    }

    // 3. Training
    std::cout << "Training..." << std::endl;
    int epochs = 20;

    for (int e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            size_t idx = word_to_neuron[tokens[i]];
            net.stimulate(idx, 600); // Force fire
            net.run(10); // Run simulation
        }
        net.run(100); // Cool down between epochs
    }

    std::cout << "Training complete." << std::endl;

    // 4. Testing / Prediction
    auto predict = [&](const std::string& word) {
        if (word_to_neuron.find(word) == word_to_neuron.end()) {
            std::cout << "Word '" << word << "' not in vocabulary." << std::endl;
            return;
        }
        size_t idx = word_to_neuron[word];
        Neuron& n = net.arena.get(idx);

        std::vector<Synapse> sorted_synapses = n.synapses;
        std::sort(sorted_synapses.begin(), sorted_synapses.end(), [](const Synapse& a, const Synapse& b){
            return a.weight > b.weight;
        });

        std::cout << "Prediction for '" << word << "':" << std::endl;
        bool found = false;
        for (int i=0; i < (int)sorted_synapses.size(); ++i) {
            if (sorted_synapses[i].weight > 250) { // Threshold for strong association
                std::string target = neuron_to_word[sorted_synapses[i].target_neuron_index];
                std::cout << "  -> " << target << " (strength=" << sorted_synapses[i].weight << ")" << std::endl;
                found = true;
            }
        }
        if (!found) std::cout << "  (No strong prediction)" << std::endl;
    };

    // Predict next words for the rhyme
    predict("mary");
    predict("had");
    predict("a");
    predict("little");

    return 0;
}
