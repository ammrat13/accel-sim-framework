from typing import Optional
from sys import stderr

class Context:
    """Represents global information about the file currently being processed.
    Most importantly, it contains the registers to use when generating GASAN
    instructions.

    :ivar Optional[str] r_block_val: Register to compute the address and hold
        the value of the "block" of shadow memory
    :ivar Optional[str] r_block_off: Register to hold the offset of the address
        into the "block" represented by shadow memory
    :ivar Optional[str] r_shadow_base: Register to store the base address of
        shadow memory
    """

    def __init__(self):
        # Set the variables to default values
        self.r_block_val = None
        self.r_block_off = None
        self.r_shadow_base = None

    def is_finalized(self) -> bool:
        """Whether all of the context information has a non-None value. This
        should always return true during instruction processing.
        """
        return self.r_block_val is not None and \
            self.r_block_off is not None and \
            self.r_shadow_base is not None

    def process_line(self, line: str) -> Optional[str]:
        """Update the context information based on a non-instruction line. May
        optionally supply the value to write to the file instead of just passing
        the line through.

        :return: The line to output, or None if passthrough
        """

        # Use register number information
        if line.startswith("-nregs = "):
            old_nregs = int(line.split("=")[1])

            # Compute the new number of registers
            # Warn if there's not enough space
            if old_nregs > 255 - 3:
                print(f"WARN: Not enough registers with -nregs = {old_nregs}", file=stderr)
            new_nregs = min(255, old_nregs + 3)

            # Compute the register numbers to use
            self.r_block_val = f"R{new_nregs - 3}"
            self.r_block_off = f"R{new_nregs - 2}"
            self.r_shadow_base = f"R{new_nregs - 1}"

            # Write the new nregs
            return f"-nregs = {new_nregs}\n"

        # Do nothing by default
        return None
