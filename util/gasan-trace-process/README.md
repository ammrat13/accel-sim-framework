# Gpu Address SANitizer (GASAN)

This project aims to simulate the effects of a hypothetical implementation of
Address Sanitizer for Graphics Processing Units (GPUs). Effectively, it looks
over the provided traces and inserts extra instructions into it for every load
or store. For more theoretical information, look in the `submission/` directory.

## Usage

The script runs on processed traces - ones that have already had
`post-traces-processing` run on them.
