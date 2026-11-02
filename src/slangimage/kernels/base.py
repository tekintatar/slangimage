# kernels/base.py

from __future__ import annotations


class ImageOperator:
    """Minimal base for operator classes: holds the device. Not abstract —
    grouped operators expose several methods instead of a single apply()."""

    def __init__(self, device):
        self.device = device
