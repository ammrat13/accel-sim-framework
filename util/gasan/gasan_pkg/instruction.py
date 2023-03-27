from typing import List

class Instruction:
    """Represents a single instruction from a processed trace. The format is
    slightly different than that of a raw trace. See `tracer_tool.cu` and
    `post-traces-processing.cpp` for more information.

    Note that we keep addresses around for all threads, even if they're unused.
    """

    def __init__(
        self,
        pc: int, mask: List[bool], opcode: str,
        dsts: List[str], srcs: List[str],
        mem_width: int, mem_addrs: List[int]
    ):
        # Check execution mask
        assert len(mask) == 32, "Incorrect number of masks"
        # Check memory addresses
        assert mem_width in [0, 1, 2, 4, 8], "Weird memory granularity"
        assert len(mem_addrs) == 32, "Incorrect number of addresses"

        # Set the variables
        self.pc = pc
        self.mask = mask
        self.opcode = opcode
        self.dsts = dsts
        self.srcs = srcs
        self.mem_width = mem_width
        self.mem_addrs = mem_addrs

    def __str__(self):
        """Prints a string representation of this object, compatible with the
        trace format.
        """
        ret = ""

        # PC
        ret += f"{self.pc:04x} "
        # Mask
        mask_num = 0
        for b in self.mask:
            mask_num *= 2
            mask_num += 1 if b else 0
        ret += f"{mask_num:08x} "
        # Destination registers
        ret += f"{len(self.dsts)} "
        if len(self.dsts) != 0:
            ret += f"{' '.join(self.dsts)} "
        # Opcode
        ret += self.opcode + " "
        # Source registers
        ret += f"{len(self.srcs)} "
        if len(self.srcs) != 0:
            ret += f"{' '.join(self.srcs)} "
        # Memory width
        ret += f"{self.mem_width} "

        # Addresses
        # Only if we have them
        # TODO: Try to compress
        if self.mem_width != 0:
            ret += "0 "
            for i, a in enumerate(self.mem_addrs):
                if self.mask[i]:
                    ret += f"0x{a:016x} "

        return ret

    @staticmethod
    def from_str(s: str):
        """Parses an instruction from the trace. Takes in the string
        representation and returns the object.
        """

        # Tokenize
        ts = s.strip().split(" ")
        assert len(ts) >= 6, "Bad tokenization"

        # PC
        pc = int(ts[0], 16)
        ts.pop(0)
        # Mask
        mask_bin = int(ts[0], 16)
        mask = [mask_bin & (1 << i) != 0 for i in range(32)]
        ts.pop(0)
        # Destination registers
        num_dsts = int(ts[0])
        ts.pop(0)
        dsts = []
        for _ in range(num_dsts):
            dsts.append(ts[0])
            ts.pop(0)
        # Opcode
        opcode = ts[0]
        ts.pop(0)
        # Source registers
        num_srcs = int(ts[0])
        ts.pop(0)
        srcs = []
        for _ in range(num_srcs):
            srcs.append(ts[0])
            ts.pop(0)
        # Memory width
        mem_width = int(ts[0])
        ts.pop(0)

        # Addresses
        # Only do the decoding if we have memory
        mem_addrs = [0x0] * 32
        if mem_width != 0:
            # Get the encoded mode
            mode = int(ts[0])
            assert mode in [0, 1, 2], "Bad mode"
            ts.pop(0)

            # TODO: Actually decode the memory operands
            pass

        return Instruction(pc, mask, opcode, dsts, srcs, mem_width, mem_addrs)


def process_block(block: List[Instruction]) -> List[Instruction]:
    return block
