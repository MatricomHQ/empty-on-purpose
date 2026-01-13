# Project OMEGA Documentation

## Architecture

### Substrate
- **Memory Model:** `mmap` backed by `brain.map`.
- **Addressing:** 32-bit indices allowing for 4 billion neurons.

### Structs
- **Neuron:** 16 bytes.
  - `ID` (u32)
  - `Potential` (u32) - interpreted as voltage.
  - `Threshold` (u32)
  - `Synapse_Head_Index` (u32)
- **Synapse:** 12 bytes (aligned to 16? or packed? Prompt says u32, u32, Int16... wait. u32=4, u32=4, Int16=2. 10 bytes. Padding might be needed for alignment).
  - `Target_Neuron` (u32)
  - `Weight` (Int16)
  - `Next_Synapse_Index` (u32)

## Roadmap
1. Phase 1: Virtual Machine & Persistence.
2. Phase 2: Logic Gates (Inhibition).
3. Phase 3: Scaling (WikiText-2 Trainer).
4. Phase 4: Self-Loop.
