"""Channel names, colors and plot order, in one place.

Data columns are by ADC channel number: raw0..raw3 / v0..v3 = MAX1032 CH0..CH3
= Electro1..4. capture.py writes these names into each session's meta.txt.
"""

# ============ CONFIG ============
CHANNEL_NAMES = [
    "Top ring (E1, CH0)",
    "Floating (E2, CH1)",     # Electro2 isn't connected to anything
    "Middle ring (E3, CH2)",
    "Bottom ring (E4, CH3)",
]
COLORS = ["tab:green", "tab:gray", "tab:orange", "tab:blue"]
PLOT_ORDER = [0, 2, 3, 1]  # panels top-to-bottom like the tube, floating channel last
# ================================

# Sessions recorded with the old 3x ADS1115 board (before 2026-10) have 3 channels.
LEGACY_ADS = {
    "names": ["Bottom ring (0x48)", "Middle ring (0x49)", "Top ring (0x4A)"],
    "colors": ["tab:blue", "tab:orange", "tab:green"],
    "order": [2, 1, 0],
}


def channel_info(n_ch, meta=None):
    """(names, colors, order) for a session with n_ch channels.
    Prefers the names saved in meta.txt, then this file, then the legacy layout."""
    meta = meta or {}
    if "channel_names" in meta:
        names = meta["channel_names"].split("|")
        if len(names) == n_ch:
            order = [int(x) for x in meta.get("plot_order", "").split(",") if x != ""]
            if sorted(order) != list(range(n_ch)):
                order = list(range(n_ch))
            colors = (COLORS * 2)[:n_ch]
            return names, colors, order
    if n_ch == len(CHANNEL_NAMES):
        return CHANNEL_NAMES, COLORS, PLOT_ORDER
    if n_ch == 3:
        return LEGACY_ADS["names"], LEGACY_ADS["colors"], LEGACY_ADS["order"]
    return [f"CH{i}" for i in range(n_ch)], (COLORS * 2)[:n_ch], list(range(n_ch))


def read_meta(session):
    meta = {}
    p = session / "meta.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            k, _, v = line.partition("\t")
            meta[k] = v
    return meta


def n_channels(names):
    """Number of v<i> columns in a structured array's field names."""
    n = 0
    while f"v{n}" in names:
        n += 1
    return n
