"""
MuJoCo doesn't simulate RF propagation, so the control-link between the main drone's
sensors and its controller is modeled here as a software-layer channel: a delay buffer
whose latency/packet-loss can be driven up by Attack 4 (comm jamming), and whose baseline
"reach" is parameterized by the true inter-drone distance computed from the GNSS sensors
on both drones (Section 11.2).
"""
from collections import deque
import numpy as np


class CommLink:
    def __init__(self, base_delay_steps=1, max_buffer=64):
        self.base_delay_steps = base_delay_steps
        self.buffer = deque(maxlen=max_buffer)

    def reset(self):
        self.buffer.clear()

    def push_and_pop(self, packet, extra_latency_steps=0, drop_prob=0.0, rng=None):
        """Push the newest control-loop packet, return the packet that should arrive
        this tick (or None if nothing has arrived yet / it was dropped)."""
        rng = rng or np.random
        if rng.random() < drop_prob:
            packet = None  # jammed: this sample never arrives
        self.buffer.append(packet)
        delay = self.base_delay_steps + extra_latency_steps
        if len(self.buffer) <= delay:
            return None
        return self.buffer[-1 - delay]


def inter_drone_range(main_pos, att_pos):
    return float(np.linalg.norm(np.asarray(main_pos) - np.asarray(att_pos)))
