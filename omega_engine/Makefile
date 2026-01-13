CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread
SRC_DIR = src
TEST_DIR = tests
DATA_DIR = data
BUILD_DIR = build
BIN_DIR = bin

# Targets
all: directories test_phase1 test_phase2 trainer

directories:
	mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Phase 1 Test
test_phase1: directories
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_phase1.cpp -o $(BIN_DIR)/test_phase1

# Phase 2 Test
test_phase2: directories
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_phase2.cpp -o $(BIN_DIR)/test_phase2

# Phase 3 Trainer
trainer: directories
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/trainer_main.cpp -o $(BIN_DIR)/trainer

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) brain.map

.PHONY: all clean directories test_phase1 test_phase2 trainer
# Phase 4 Test
test_phase4: directories
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_phase4.cpp -o $(BIN_DIR)/test_phase4
