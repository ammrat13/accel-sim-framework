from typing import Optional

class Context:
    """Represents global information about the file currently being processed.
    Most importantly, it contains the registers to use when generating GASAN
    instructions.

    :ivar Optional[str] r_addr: The extra register to use for address
        calculation
    :ivar Optional[str] r_off: The extra register to use to store and compare
        the offset
    """

    def __init__(self):
        # Set the variables to default values
        self.r_addr = None
        self.r_off = None

    def is_finalized(self) -> bool:
        """Whether all of the context information has a non-None value. This
        should always return true during instruction processing.
        """
        return self.r_addr != None and \
            self.r_off != None

    def process_line(self, line: str) -> Optional[str]:
        """Update the context information based on a non-instruction line. May
        optionally supply the value to write to the file instead of just passing
        the line through.

        :return: The line to output, or None if passthrough
        """

        # Use register number information
        if line.startswith("-nregs = "):
            nregs = int(line.split("=")[1])

            # Compute the register numbers to use
            r_addr_num = min(253, nregs + 0)
            r_off_num = min(254, nregs + 1)
            # Write into variables
            self.r_addr = f"R{r_addr_num}"
            self.r_off = f"R{r_off_num}"

            # Write the new nregs
            new_nregs = min(255, nregs + 2)
            return f"-nregs = {new_nregs}\n"

        # Do nothing by default
        return None
