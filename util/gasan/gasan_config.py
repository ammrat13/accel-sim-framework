VARIANT: int = 128
"""int: What mode to run in. Must be either 8, 128, or 256."""

EMIT_ARITH: bool = True
"""bool: Whether to emit loads from shadow memory. If this is unset, only the
memory operations are emitted, given EMIT_LOADS is True. If both this and
EMIT_LOADS are False, no extra code is generated.
"""

EMIT_LOADS: bool = True
"""bool: Whether to emit loads from shadow memory. If this is unset, only the
arithmetic operations are emitted, given EMIT_ARITH is True. If both this and
EMIT_ARITH are False, no extra code is generated.
"""
