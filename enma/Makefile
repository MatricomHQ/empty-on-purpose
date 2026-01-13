CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I. -O2

# Directories
SRC_DIR = src
TEST_DIR = tests

# Targets
all: test_neuron test_synapse test_scheduler enma_main

test_neuron: $(TEST_DIR)/test_neuron.cpp
	$(CXX) $(CXXFLAGS) -o test_neuron $(TEST_DIR)/test_neuron.cpp

test_synapse: $(TEST_DIR)/test_synapse.cpp
	$(CXX) $(CXXFLAGS) -o test_synapse $(TEST_DIR)/test_synapse.cpp

test_scheduler: $(TEST_DIR)/test_scheduler.cpp
	$(CXX) $(CXXFLAGS) -o test_scheduler $(TEST_DIR)/test_scheduler.cpp

enma_main: main.cpp
	$(CXX) $(CXXFLAGS) -o enma_main main.cpp

clean:
	rm -f test_neuron test_synapse test_scheduler enma_main
	rm -f *.o *.d
