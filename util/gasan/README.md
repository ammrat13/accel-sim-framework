# GASan Simulation

This project looks to evaluate the performance and memory overheads of a
potential port of ASan to GPUs. It uses AccelSim to fulfil that goal. The
proposal, paper, and presentation are available in the `submission/` directory.

The `gasan-trace-process/` folder contains scripts to instrument processed
AccelSim traces with GASan. It goes over the trace and inserts assembly code
before every `LDG` and `STG` instruction.

To simulate memory footprint, this project has a custom simulator. The
`gasan-alloc-record/` folder has a shared library which can be used to record
allocation calls via `LD_PRELOAD`. The generated trace can then be fed into the
simulator in `gasan-alloc-sim/`.
