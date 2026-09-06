"""Single place to declare which serial port each instrument is plugged into.

Moving to a different machine? Change the numbers/paths in this file only --
every script imports from here instead of hard-coding a COM port.

Usage:
    from coms import port
    SERIAL_PORT = port('keithley')     # -> 'COM6' on Windows, '/dev/ttyUSB0' on Linux

Scripts in subfolders need the repo root on sys.path first:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from coms import port
"""

import platform

IS_LINUX = platform.system() == 'Linux'

# Windows: bare COM numbers (or a full 'COM12' string if you prefer).
WINDOWS_PORTS = {
    'arduino':  5,
    'keithley': 6,
}

# Linux: device paths for the same instruments.
LINUX_PORTS = {
    'arduino':  '/dev/ttyACM0',
    'keithley': '/dev/ttyUSB0',
}

# Kept for older code that did `from coms import ports`.
ports = WINDOWS_PORTS


def port(name):
    """Return the OS-appropriate serial port string for a named instrument."""
    key = name.lower()
    table = LINUX_PORTS if IS_LINUX else WINDOWS_PORTS
    try:
        value = table[key]
    except KeyError:
        raise KeyError(
            f"Unknown instrument {name!r}. Known names: "
            f"{', '.join(sorted(table))}. Add it to coms.py."
        ) from None
    if isinstance(value, int):
        return f'COM{value}'
    return value
