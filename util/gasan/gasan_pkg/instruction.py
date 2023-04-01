from typing import List
from gasan_config import VARIANT
from gasan_pkg.context import Context

class Instruction:
    """Represents a single instruction from a processed trace. The format is
    slightly different than that of a raw trace. See `tracer_tool.cu` and
    `post-traces-processing.cpp` for more information.

    Note that we keep addresses around for all threads, even if they're unused.
    """

    WARP_SIZE = 32

    def __init__(
        self,
        pc: int, mask: List[bool], opcode: str,
        dsts: List[str], srcs: List[str],
        mem_width: int, mem_addrs: List[int] = [0x0]*32
    ):
        # Check execution mask
        assert len(mask) == Instruction.WARP_SIZE, "Incorrect number of masks"
        # Check memory addresses
        assert mem_width in [0, 1, 2, 4, 8, 16], "Weird memory granularity"
        assert len(mem_addrs) == Instruction.WARP_SIZE, "Incorrect number of addresses"

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
        for t,a in enumerate(self.mask):
            mask_num |= (1 << t) if a else 0
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
                    ret += f"0x{a:x} "

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
        mask = [
            int(ts[0], 16) & (1 << t) != 0
            for t in range(Instruction.WARP_SIZE)
        ]
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
        mem_addrs = [0x0] * Instruction.WARP_SIZE
        if mem_width != 0:
            # Get the encoded mode
            mode = int(ts[0])
            assert mode in [0, 1, 2], "Bad mode"
            ts.pop(0)

            # address_format::list_all
            if mode == 0:
                for t in range(Instruction.WARP_SIZE):
                    if mask[t]:
                        mem_addrs[t] = int(ts[0], 16)
                        ts.pop(0)

            # address_format::base_stride
            if mode == 1:
                # Get information
                cur_addr = int(ts[0], 16)
                ts.pop(0)
                stride = int(ts[0])
                ts.pop(0)
                # Parse
                first_bit1_found = False
                last_bit1_found = False
                for t in range(Instruction.WARP_SIZE):
                    if not first_bit1_found:
                        if mask[t]:
                            first_bit1_found = True
                            mem_addrs[t] = cur_addr
                    elif not last_bit1_found:
                        if mask[t]:
                            cur_addr += stride
                            mem_addrs[t] = cur_addr
                        else:
                            last_bit1_found = True
                    else:
                        assert not mask[t], "Bad base-stride encoding"

            # address_format::base_delta
            if mode == 2:
                first_bit1_found = False
                for t in range(Instruction.WARP_SIZE):
                    if mask[t]:
                        if not first_bit1_found:
                            first_bit1_found = True
                            mem_addrs[t] = int(ts[0], 16)
                            ts.pop(0)
                            cur_addr = mem_addrs[t]
                        else:
                            mem_addrs[t] = cur_addr + int(ts[0])
                            ts.pop(0)
                            cur_addr = mem_addrs[t]

        assert len(ts) == 0, "Not all operands parsed"
        return Instruction(pc, mask, opcode, dsts, srcs, mem_width, mem_addrs)

def process_block_insts(insts: List[Instruction], ctx: Context) -> List[Instruction]:
    """Inserts instructions before every load or store to simulate the effects
    of ASAN.
    """

    ret = []
    for i in insts:

        # Check if load or store
        # Based on what the opcode is
        is_ld = i.opcode.startswith("LDG")
        is_st = i.opcode.startswith("STG")
        if is_ld or is_st:
            assert not is_ld or not is_st, "Instruction can't both load and store"
            assert i.mem_width != 0, "Memory operation must have width"

            # Get the register with the address
            # Also check the operands have the right length
            r_addr = None
            if is_ld:
                assert len(i.srcs) == 1, "Wrong number of operands for LDG"
                r_addr = i.srcs[0]
            if is_st:
                assert len(i.srcs) == 2, "Wrong number of operands for STG"
                r_addr = i.srcs[1]

            # Generate different code depending on the variant
            assert VARIANT in [128, 256], "Bad variant"

            if VARIANT == 128:
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.SHR",
                    [ctx.r_block_val], [r_addr], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "LOP32I.AND",
                    [ctx.r_block_val], [ctx.r_block_val, ctx.r_shadow_mask], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.IADD",
                    [ctx.r_block_val], [ctx.r_block_val, ctx.r_shadow_base], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "LDG.E.U8.SYS",
                    [ctx.r_block_val], [ctx.r_block_val], 1,
                    list(map(lambda a: (a // 128) % 2**28, i.mem_addrs))))
                ret.append(Instruction(
                    i.pc, i.mask, "LOP.AND",
                    [ctx.r_block_off], [r_addr], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.IADD",
                    [ctx.r_block_off], [ctx.r_block_off], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "ISETP.GT.AND",
                    [], [ctx.r_block_val, ctx.r_block_off], 0))
                ret.append(Instruction(
                    i.pc, [False]*32, "BRA",
                    [], [], 0))

            if VARIANT == 256:
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.SHR",
                    [ctx.r_block_val], [r_addr], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "LOP32I.AND",
                    [ctx.r_block_val], [ctx.r_block_val, ctx.r_shadow_mask], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.IADD",
                    [ctx.r_block_val], [ctx.r_block_val, ctx.r_shadow_base], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "LDG.E.U8.SYS",
                    [ctx.r_block_val], [ctx.r_block_val], 1,
                    list(map(lambda a: (a // 256) % 2**27, i.mem_addrs))))
                ret.append(Instruction(
                    i.pc, i.mask, "LOP.AND",
                    [ctx.r_block_off], [r_addr], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "IMAD.IADD",
                    [ctx.r_block_off], [ctx.r_block_off], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "ISETP.NE.AND",
                    [], [ctx.r_block_val], 0))
                ret.append(Instruction(
                    i.pc, i.mask, "ISETP.GE.AND",
                    [], [ctx.r_block_val, ctx.r_block_off], 0))
                ret.append(Instruction(
                    i.pc, [False]*32, "BRA",
                    [], [], 0))

        # Always append the current instruction
        ret.append(i)

    return ret
