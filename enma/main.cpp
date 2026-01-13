#include "src/network.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cctype>
#include <set>

// Robust tokenizer
std::vector<std::vector<std::string>> load_phrases(const std::string& filename) {
    std::vector<std::vector<std::string>> phrases;
    std::ifstream file(filename);
    if (!file.is_open()) return phrases;

    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> tokens;
        std::string current;
        for (char c : line) {
            if (std::isalpha(c)) {
                current += std::tolower(c);
            } else if (!current.empty()) {
                tokens.push_back(current);
                current = "";
            }
        }
        if (!current.empty()) tokens.push_back(current);
        if (!tokens.empty()) phrases.push_back(tokens);
    }
    return phrases;
}

int main() {
    std::cout << "Initializing ENMA (Elastic Neuro-Mesh Architecture)..." << std::endl;

    // 1. Load Data
    auto phrases = load_phrases("training_data.txt");

    if (phrases.empty()) {
        std::cerr << "No phrases found." << std::endl;
        return 1;
    }

    size_t total_tokens = 0;
    for (const auto& p : phrases) total_tokens += p.size();
    std::cout << "Loaded " << phrases.size() << " phrases with " << total_tokens << " tokens." << std::endl;

    // 2. Setup Network
    std::unordered_map<std::string, size_t> word_to_neuron;
    std::vector<std::string> neuron_to_word;

    // Build vocab
    for (const auto& phrase : phrases) {
        for (const auto& word : phrase) {
            if (word_to_neuron.find(word) == word_to_neuron.end()) {
                word_to_neuron[word] = 0;
            }
        }
    }
    size_t vocab_size = word_to_neuron.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    Network net(vocab_size + 100);
    word_to_neuron.clear();

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

                // 1. LTP
                for (size_t src_idx : self->incoming_sources) {
                    Neuron& src = net.arena.get(src_idx);
                    int64_t dt = now - src.last_fire_time;

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

                // 2. LTD
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
    for (const auto& phrase : phrases) {
        for (const auto& word : phrase) {
            get_or_create_neuron(word);
        }
    }

    // SPARSE INITIALIZATION: Only connect words that appear nearby in the text.
    std::cout << "Initializing sparse connections..." << std::endl;
    std::set<std::pair<size_t, size_t>> connections;

    for (const auto& phrase : phrases) {
        for (size_t i = 0; i < phrase.size(); ++i) {
            size_t src_idx = word_to_neuron[phrase[i]];
            // Connect to next 3 words?
            for (size_t j = i + 1; j < phrase.size() && j < i + 4; ++j) {
                size_t tgt_idx = word_to_neuron[phrase[j]];
                if (src_idx == tgt_idx) continue;

                if (connections.find({src_idx, tgt_idx}) == connections.end()) {
                    net.connect(src_idx, tgt_idx, 0, 5);
                    connections.insert({src_idx, tgt_idx});
                }
            }
        }
    }
    std::cout << "Created " << connections.size() << " connections." << std::endl;

    // 3. Training
    std::cout << "Training..." << std::endl;
    int epochs = 10; // 10 epochs should be enough

    for (int e = 0; e < epochs; ++e) {
        if (e % 2 == 0) std::cout << "  Epoch " << e << std::endl;

        for (const auto& phrase : phrases) {
            for (const auto& word : phrase) {
                size_t idx = word_to_neuron[word];
                net.stimulate(idx, 600);
                net.run(10);
            }
            net.run(50);
        }
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
        int count = 0;
        for (int i=0; i < (int)sorted_synapses.size(); ++i) {
            if (sorted_synapses[i].weight > 200) {
                std::string target = neuron_to_word[sorted_synapses[i].target_neuron_index];
                std::cout << "  -> " << target << " (strength=" << sorted_synapses[i].weight << ")" << std::endl;
                found = true;
                count++;
                if (count >= 5) break;
            }
        }
        if (!found) std::cout << "  (No strong prediction)" << std::endl;
    };

    std::vector<std::string> test_words = {
        "absence", "actions", "history", "knowledge", "time", "early", "don't", "money", "rome", "practice", "too", "two", "blood"
    };

    for (const auto& w : test_words) {
        predict(w);
    }

    return 0;
}
