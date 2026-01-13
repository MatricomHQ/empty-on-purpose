#include "omega.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace Omega;

class Trainer {
public:
    Brain brain;
    std::string data_path;

    Trainer(const std::string& db_path, const std::string& txt_path)
        : brain(db_path, 200), data_path(txt_path) {
        // 200MB Brain
    }

    void train(int seconds, const std::string& label) {
        std::cout << "\n=== Starting Training Session: " << label << " (" << seconds << "s) ===" << std::endl;

        std::ifstream file(data_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_path << std::endl;
            return;
        }

        auto start_time = std::chrono::steady_clock::now();
        uint32_t window = 0;
        char c;
        uint64_t bytes_processed = 0;

        // Metrics
        uint64_t predictions_total = 0;
        uint64_t predictions_correct = 0;

        // Rolling window of last active neurons (from previous step)
        // To check if they PREDICTED the current input.
        // But wait, prediction is: "When seeing A, does it predict B?"
        // So at Step N (Input A), we see what fires.
        // If B fires (due to A), we check if B corresponds to the Next Token.
        // So we need to know the Next Token ID *before* we process it?
        // Or store the prediction and check it in the *next* step?

        // Let's store `predicted_ids` from Step N.
        // In Step N+1, we determine `current_id` from input.
        // If `predicted_ids` contains `current_id`, then Score++!

        std::vector<uint32_t> predicted_ids;
        uint32_t prev_id = 0;
        bool ignore_line = false;

        while (true) {
             if (!file.get(c)) {
                 file.clear();
                 file.seekg(0);
                 ignore_line = false;
                 continue;
             }

             // Ingestion Rule: Ignore headers (lines starting with =)
             if (c == '\n') {
                 if (ignore_line) {
                     ignore_line = false;
                     continue;
                 }
                 // End of Thought: Reset Potentials
                 brain.reset_state();
                 window = 0; // Reset sensory window
                 prev_id = 0; // Break sequence link
                 continue;
             }

             if (c == '=' && (window == 0 || prev_id == 0)) {
                 // Heuristic: If = appears at start of line (after reset)
                 // But wait, window is 3 bytes. We don't track "start of line" explicitly unless we flag it.
                 // Let's assume after \n we are at start of line.
                 // But we just processed \n and `continue`. So next char is start of line.
                 // So if c == '=', ignore rest of line.
                 ignore_line = true;
                 continue;
             }

             if (ignore_line) continue;

            // Check time
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() >= seconds) {
                break;
            }

            // Update Window (3 bytes)
            window = ((window << 8) | (uint8_t)c) & 0xFFFFFF;

            // Map to Neuron
            // input_field_ is array of uint32
            // window is index.
            uint32_t& mapped_id = brain.input_field_[window];
            if (mapped_id == 0) {
                // Create new cortex neuron for this trigram
                try {
                    mapped_id = brain.create_neuron(100); // Threshold 100
                } catch (const std::exception& e) {
                    std::cerr << "OOM at byte " << bytes_processed << std::endl;
                    break;
                }
            }

            uint32_t current_id = mapped_id;

            // Hebbian Link: Sequence learning
            if (prev_id != 0 && prev_id != current_id) {
                brain.connect_or_update(prev_id, current_id, 2, 20);
            }
            prev_id = current_id;

            // Check Prediction from PREVIOUS step
            if (!predicted_ids.empty()) {
                predictions_total++;
                bool hit = false;
                for (uint32_t pid : predicted_ids) {
                    if (pid == current_id) {
                        hit = true;
                        break;
                    }
                }
                if (hit) predictions_correct++;
            }

            // Stimulate
            brain.stimulate(current_id, 150); // Force fire (Sensory input strong)

            // Run Physics
            brain.run_cycles(50); // Run enough cycles for propagation

            // Identify Predicted Neurons
            predicted_ids = brain.fired_neurons_in_last_run;
            // Remove the current_id if present (since it's the sensory input)
            // Actually, if current_id is in list, it fired. We want *others* that fired.
            // Or if current_id fired due to internal dynamics BEFORE we forced it?
            // But we forced it.

            bytes_processed++;
        }

        // Report
        double accuracy = predictions_total > 0 ? (100.0 * predictions_correct / predictions_total) : 0.0;
        std::cout << "Bytes Processed: " << bytes_processed << std::endl;
        std::cout << "Predictions: " << predictions_correct << "/" << predictions_total << std::endl;
        std::cout << "Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Neurons: " << brain.header_->neuron_count << std::endl;
        std::cout << "Synapses: " << brain.header_->synapse_count << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./trainer <data_path>" << std::endl;
        return 1;
    }

    // Clean room
    unlink("brain.map");

    Trainer trainer("brain.map", argv[1]);

    trainer.train(5, "Test A (5s)");
    trainer.train(5, "Test B (10s)"); // Accumulate 5 more seconds? No, prompt says "Train for 10s". Total time?
    // "Test A (5 Seconds): Train for 5s."
    // "Test B (10 Seconds): Train for 10s."
    // Does it mean *additional* 10s or *total* 10s?
    // "Test C (20 Seconds): Train for 20s."
    // Usually means cumulative or fresh run.
    // "Success Condition: The accuracy must be statistically higher than the 10s test."
    // Implies we can continue training.

    // I will run 5s, report. Then continue for 5s (Total 10s). Report. Continue for 10s (Total 20s).

    trainer.train(10, "Test C (20s)");

    // Research Request: 30s test
    trainer.train(10, "Test D (30s)");

    // Research Request: 40s test
    trainer.train(10, "Test E (40s)");

    // Research Request: 60s test (Scaling check)
    trainer.train(10, "Test F (50s)");
    trainer.train(10, "Test G (60s)");

    return 0;
}
