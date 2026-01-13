#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace Omega {

    struct Neuron {
        uint32_t id;
        int32_t potential; // Voltage
        int32_t threshold;
        uint32_t synapse_head_index;
    };

    struct Synapse {
        uint32_t target_neuron;
        int16_t weight;
        uint16_t padding; // Align to 4 bytes if needed, or just 2 bytes.
        // 4+2+2+4 = 12 bytes.
        uint32_t next_synapse_index;
    };

    struct Header {
        uint64_t magic;
        uint32_t neuron_count;
        uint32_t synapse_count;
        size_t input_field_offset;
        size_t neurons_offset;
        size_t synapses_offset;
        size_t max_neurons;
        size_t max_synapses;
    };

    struct Event {
        uint32_t neuron_index;
        int32_t voltage_delta;
        uint32_t source_neuron; // 0 if sensory/manual
        uint32_t synapse_index; // 0 if sensory/manual
    };

    class Brain {
    public:
        int fd_ = -1;
        uint8_t* memory_ = nullptr;
        size_t size_ = 0;

        Header* header_ = nullptr;
        uint32_t* input_field_ = nullptr;
        Neuron* neurons_ = nullptr;
        Synapse* synapses_ = nullptr;

        std::queue<Event> event_queue;
        std::vector<uint64_t> last_fired_time;
        std::vector<uint32_t> fired_neurons_in_last_run;
        uint64_t current_tick = 0;

        // STDP Parameters
        int16_t LTP_REWARD = 2;
        int16_t LTD_PENALTY = 2;
        uint64_t LTD_WINDOW = 5;

        // Refractory Period
        uint64_t REFRACTORY_PERIOD = 5; // Ticks

        // Track modified neurons for efficient reset
        std::vector<uint32_t> modified_neurons;

        Brain(const std::string& filepath, size_t size_mb) {
            size_ = size_mb * 1024 * 1024;
            bool new_file = false;

            fd_ = open(filepath.c_str(), O_RDWR | O_CREAT, 0644);
            if (fd_ == -1) throw std::runtime_error("Failed to open file");

            struct stat st;
            fstat(fd_, &st);
            if (st.st_size == 0) {
                new_file = true;
                if (ftruncate(fd_, size_) == -1) throw std::runtime_error("Failed to resize file");
            } else {
                size_ = st.st_size;
            }

            memory_ = (uint8_t*)mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (memory_ == MAP_FAILED) {
                close(fd_);
                throw std::runtime_error("mmap failed");
            }

            header_ = (Header*)memory_;

            if (new_file) {
                header_->magic = 0xDEADBEEF;
                header_->neuron_count = 1; // 1-based indexing for null safety
                header_->synapse_count = 1;

                // 64MB Input Field
                size_t input_size = (1 << 24) * sizeof(uint32_t);
                header_->input_field_offset = sizeof(Header);
                // Align to 4096 page boundary if possible, but packing is key here.
                // Just keep it simple.

                header_->neurons_offset = header_->input_field_offset + input_size;

                size_t remaining = size_ - header_->neurons_offset;

                // 30% Neurons, 70% Synapses
                header_->max_neurons = (remaining * 0.3) / sizeof(Neuron);
                header_->max_synapses = (remaining * 0.7) / sizeof(Synapse);

                header_->synapses_offset = header_->neurons_offset + (header_->max_neurons * sizeof(Neuron));

                // Initialize Input Field to 0
                // mmap zeros it, but to be safe/explicit? No, trust mmap for new files.
            }

            input_field_ = (uint32_t*)(memory_ + header_->input_field_offset);
            neurons_ = (Neuron*)(memory_ + header_->neurons_offset);
            synapses_ = (Synapse*)(memory_ + header_->synapses_offset);

            // Initialize STDP tracking
            last_fired_time.resize(header_->max_neurons + 1, 0);
        }

        ~Brain() {
            if (memory_) munmap(memory_, size_);
            if (fd_ != -1) close(fd_);
        }

        uint32_t create_neuron(int32_t threshold = 100) {
            if (header_->neuron_count >= header_->max_neurons) throw std::runtime_error("Neuron OOM");
            uint32_t id = header_->neuron_count++;
            Neuron& n = neurons_[id];
            n.id = id;
            n.potential = 0;
            n.threshold = threshold;
            n.synapse_head_index = 0;
            return id;
        }

        void connect(uint32_t source_id, uint32_t target_id, int16_t weight) {
            if (header_->synapse_count >= header_->max_synapses) throw std::runtime_error("Synapse OOM");
            uint32_t syn_id = header_->synapse_count++;
            Synapse& s = synapses_[syn_id];
            s.target_neuron = target_id;
            s.weight = weight;

            Neuron& source = neurons_[source_id];
            s.next_synapse_index = source.synapse_head_index;
            source.synapse_head_index = syn_id;
        }

        void connect_or_update(uint32_t source_id, uint32_t target_id, int16_t weight_inc, int16_t initial_weight) {
             Neuron& source = neurons_[source_id];
             uint32_t syn_idx = source.synapse_head_index;
             while(syn_idx != 0) {
                 Synapse& s = synapses_[syn_idx];
                 if (s.target_neuron == target_id) {
                     // Update
                     int32_t new_w = s.weight + weight_inc;
                     if (new_w > 32000) new_w = 32000; // Clamp
                     if (new_w < -32000) new_w = -32000;
                     s.weight = (int16_t)new_w;
                     return;
                 }
                 syn_idx = s.next_synapse_index;
             }
             // Not found, create
             connect(source_id, target_id, initial_weight);
        }

        void stimulate(uint32_t neuron_id, int32_t voltage, uint32_t source_id = 0, uint32_t synapse_idx = 0) {
            event_queue.push({neuron_id, voltage, source_id, synapse_idx});
        }

        void run_cycles(int limit = 1000) {
            fired_neurons_in_last_run.clear();
            int cycles = 0;
            while(!event_queue.empty() && cycles < limit) {
                current_tick++;
                size_t batch_size = event_queue.size();

                // Track causations for LTP: Map<TargetID, Vector<SynapseIndex>>
                // But simplified: Store list of contributing events for each neuron
                // Since we iterate active_neurons later, we can re-check events or store temp structure.
                // Using a temp map is cleaner.
                std::vector<Event> current_events;
                current_events.reserve(batch_size);

                for(size_t i=0; i<batch_size; ++i) {
                    Event e = event_queue.front();
                    event_queue.pop();
                    current_events.push_back(e);

                    // STDP LTD: Check if Target (e.neuron_index) fired recently?
                    // "If A fires after B (useless noise), weaken A->B."
                    // A (source) just fired (causing this event).
                    // This event arrives at B (target) NOW.
                    // If B fired recently (e.g. within LTD_WINDOW), but A is late.
                    // Then weaken.

                    if (e.source_neuron != 0 && e.synapse_index != 0) {
                        uint64_t last = last_fired_time[e.neuron_index];
                        if (last > 0 && current_tick > last && (current_tick - last) <= LTD_WINDOW) {
                             // Weaken
                             synapses_[e.synapse_index].weight -= LTD_PENALTY;
                        }
                    }

                    Neuron& n = neurons_[e.neuron_index];
                    n.potential += e.voltage_delta;
                    modified_neurons.push_back(e.neuron_index);
                }

                // Identify active neurons
                std::vector<uint32_t> active_indices;
                active_indices.reserve(batch_size);
                for(const auto& e : current_events) {
                    active_indices.push_back(e.neuron_index);
                }
                std::sort(active_indices.begin(), active_indices.end());
                auto unique_end = std::unique(active_indices.begin(), active_indices.end());

                // Check thresholds
                for(auto it = active_indices.begin(); it != unique_end; ++it) {
                    uint32_t nid = *it;
                    Neuron& n = neurons_[nid];

                    // Refractory Check
                    // Ensure last_fired_time is not 0 (never fired)
                    if (last_fired_time[nid] != 0 && (current_tick - last_fired_time[nid] < REFRACTORY_PERIOD)) {
                        // Refractory: Cannot fire.
                        n.potential = 0;
                        continue;
                    }

                    if (n.potential >= n.threshold) {
                        // Fire
                        n.potential = -50; // Hyperpolarization (Reset to below 0)
                        last_fired_time[nid] = current_tick;
                        fired_neurons_in_last_run.push_back(nid);

                        // LTP: Strengthen synapses that contributed to this fire
                        // We need to find events targeting nid in current_events
                        for(const auto& e : current_events) {
                            if (e.neuron_index == nid && e.source_neuron != 0 && e.synapse_index != 0) {
                                // Strengthen
                                synapses_[e.synapse_index].weight += LTP_REWARD;
                            }
                        }

                        // Propagate
                        uint32_t syn_idx = n.synapse_head_index;
                        while(syn_idx != 0) {
                            Synapse& s = synapses_[syn_idx];

                            // Usage-Dependent Decay (Pruning)
                            // Every time a synapse fires, it loses a tiny bit of strength.
                            // Only if the target fires (LTP) does it gain back more.
                            if (s.weight > 0) {
                                s.weight -= 1;
                            }

                            stimulate(s.target_neuron, s.weight, nid, syn_idx);
                            syn_idx = s.next_synapse_index;
                        }
                    }
                }
                cycles++;
            }
        }

        Neuron& get_neuron(uint32_t id) {
            return neurons_[id];
        }

        void reset_state() {
            // Clear Queue
            std::queue<Event> empty;
            std::swap(event_queue, empty);

            // Reset Potentials of modified neurons
            for(uint32_t id : modified_neurons) {
                neurons_[id].potential = 0;
            }
            modified_neurons.clear();

            // Reset STDP tracking?
            // "Reset Neuron potentials to 0". It doesn't say reset weights or history.
            // Keeping last_fired_time is probably fine (Long Term state),
            // but if "End of Thought" means "Start fresh sequence", maybe last_fired should be considered "old"?
            // If we keep last_fired, STDP might bridge the gap across thoughts (e.g. End of one sentence -> Start of next).
            // Usually we want that.
        }
    };
}
