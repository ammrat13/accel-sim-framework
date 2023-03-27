#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/")

import os
import gasan_pkg.instruction
from gasan_pkg.context import Context
from gasan_pkg.instruction import Instruction


# Get the input and output files, and open them
# We don't close them, but the OS will take care of that for us
INP_FNAME = sys.argv[1]
OUT_FNAME = sys.argv[2]
INP = open(INP_FNAME)
OUT = open(OUT_FNAME, "w")


ctx = Context()

while True:
    # Termination logic
    inp_line = INP.readline()
    if inp_line == "":
        break
    # Get rid of newlines
    inp_line = inp_line.strip();

    # Handle instruction blocks
    if inp_line.startswith("insts = "):

        # Context should have all the information
        assert ctx.is_finalized(), "Context doesn't have all its information"

        # Get the number of instructions
        # Pool all of them into an array
        num_insts = int(inp_line.split("=")[1].strip())
        insts = []
        for _ in range(num_insts):
            # Read the text
            il = INP.readline()
            assert il != "", "Expected an instruction"
            # Parse it
            insts.append(Instruction.from_str(il))

        # Do processing
        insts = gasan_pkg.instruction.process_block_insts(insts)

        # Write out
        OUT.write(f"insts = {len(insts)}\n")
        for i in insts:
            OUT.write(str(i) + "\n")

    # All other lines
    # Take what information we need, and write or passtrhough
    else:
        cl = ctx.process_line(inp_line)
        OUT.write(cl or inp_line + "\n")
