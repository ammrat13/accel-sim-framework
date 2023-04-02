VARIANT: int = 128
"""int: What mode to run in. Must be either 8, 128, or 256."""

EMIT_LOADS: bool = True
"""bool: Whether to emit loads from shadow memory. If this is unset, only the
arithmetic operations are emitted, which can be useful to see how much overhead
they cause.
"""
