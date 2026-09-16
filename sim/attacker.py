"""
The attacker drone does NOT run a full ArduCopter/SITL instance (Section 11.4) -- it's
driven by this lightweight scripted policy, using the same ArduCopterStyleController for
its own flight. Rather than dynamically chasing the main drone (a moving-target-tracking
problem that needs its own well-tuned guidance law), it uses the fact that the main
drone's mission is a known straight-line waypoint flight: it stations itself, ahead of
time, at the point along that line where it wants to be in range when the attack is
scheduled to fire. This is simple station-keeping (a stable, well-posed control problem)
rather than pursuit, and is a realistic threat model too -- a real attacker positioning
itself along a known/observed flight corridor rather than trying to out-maneuver the target.
"""
import numpy as np
from .controller import ArduCopterStyleController
from .attacks import AttackState

EFFECTIVE_RANGE = 16.0  # meters; attacks only "reach" the main drone within this range.
                          # Increased for the battlefield-scale map (Section 11 update) --
                          # still a physically plausible reach for GPS spoofing/jamming
                          # equipment (real jammers can reach well beyond this).


class AttackerPolicy:
    def __init__(self, dt, rng=None):
        self.controller = ArduCopterStyleController(dt=dt, max_speed=2.6)
        self.attack_state = AttackState()
        self.rng = rng or np.random.default_rng()
        self.scheduled_attack_id = 0
        self.scheduled_onset_frac = 0.5
        self._intercept_target = None

    def schedule(self, attack_id, onset_frac=0.5, start_xy=None, goal_xy=None, altitude=1.2,
                 standoff_distance=12.0, side=1.0):
        """start_xy/goal_xy: the main drone's known mission endpoints, used to precompute
        a station-keeping point along its path near where the attack should trigger.
        standoff_distance/side: how far, and to which side (+1/-1) of the flight line,
        the attacker positions itself -- e.g. across a border into "enemy territory"."""
        self.scheduled_attack_id = attack_id
        self.scheduled_onset_frac = onset_frac
        if start_xy is not None and goal_xy is not None:
            start_xy, goal_xy = np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float)
            waypoint_xy = start_xy + onset_frac * (goal_xy - start_xy)
            path_dir = goal_xy - start_xy
            path_dir = path_dir / (np.linalg.norm(path_dir) + 1e-6)
            perp = side * np.array([path_dir[1], -path_dir[0]])
            standoff_xy = waypoint_xy + standoff_distance * perp
            self._intercept_target = np.array([standoff_xy[0], standoff_xy[1], altitude])
        else:
            self._intercept_target = None

    def reset(self):
        self.controller.reset()
        self.attack_state.clear()

    def step(self, pos, vel, quat, gyro, main_pos, t, episode_frac, main_vel=None):
        """Hold station at the precomputed intercept point; trigger the scheduled attack
        once the main drone is actually close enough and the scheduled onset has passed."""
        rng_dist = np.linalg.norm(main_pos - pos)
        target = self._intercept_target if self._intercept_target is not None else main_pos
        ctrl = self.controller.update(pos, vel, quat, gyro, pos_target=target)

        if (self.scheduled_attack_id != 0
                and self.attack_state.active_id == 0
                and episode_frac >= self.scheduled_onset_frac
                and rng_dist <= EFFECTIVE_RANGE):
            self.attack_state.trigger(self.scheduled_attack_id, t)

        return ctrl, rng_dist
