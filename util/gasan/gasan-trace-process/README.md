# GASan Trace Processing

While the usages for the other two components are mostly self-explanatory, the
trace processor requires some explanation. The main file is
`gasan-trace-process.py`. It takes in exactly two command-line arguments. The
first one is the trace file to process, and the second is where to output the
processed file. Note that the input trace should have been fed through
AccelSim's own `/util/tracer_nvbit/tracer_tool/traces-processing/`.

Additionally, the trace processor has a configuration for what kind of GASan
should be added. That is defined in `gasan_trace_process_config.py`. It's
imported by the main file. To change the configuration, change that file and run
the program.

Finally, there is a utility script, `gen-gasan-trace-process-all.sh`, which
generates the commands to instrument all the traces in a directory. Its first
argument is the source directory to find traces inside, and the second is the
target directory in which to copy all the traces. The directory structure is
expected to be the same between the source and the target. The output is
`gasan-trace-process-all.sh`.
