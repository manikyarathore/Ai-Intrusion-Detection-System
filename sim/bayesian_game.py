"""
Bayesian Game-based IDS/IES decision module, implementing Sedjelmaci, Senouci & Ansari,
"Intrusion Detection and Ejection Framework Against Lethal Attacks in UAV-Aided Networks:
A Bayesian Game-Theoretic Methodology" (IEEE T-ITS, 2017).

The paper's core idea, adapted here: a suspected drone shouldn't be ejected the moment it
shows *any* sign of misbehavior -- that sign could be transitory (noise, unreliable
comms, or the drone genuinely recovering) rather than a permanent, lethal attack. Two
Bayesian games decide this:

  Game 1 (IDS activation, Section IV-A): should the monitoring drone (the "friend") even
  bother activating its monitoring process against a target? Decided by comparing the
  target's Misbehavior Rate (MR) against a Bayesian-equilibrium threshold B_attacker.

  Game 2 (IES ejection, Section IV-B): once monitoring, should the friend drone eject the
  suspected drone? Decided the same way, against a threshold B_node, using the *history*
  of MR (a drone that persists in misbehaving crosses the "permanent misbehavior" bar;
  one that oscillates back to normal does not -- this is the paper's key distinction
  between transitory and permanent misbehavior, Section III-B).

This module intentionally keeps the paper's economic/profit formulation (Eqs. 2-9,
16-21) rather than replacing it with something simpler, because the profit trade-off
(cost of monitoring/ejecting vs. cost of a missed or false detection) is the actual
mechanism that makes the Bayesian equilibrium meaningful -- collapsing it to a bare
threshold on the detector's raw confidence would lose the "low overhead + few false
positives" property the paper is built around.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from collections import deque


# ----------------------------------------------------------------------------------
# Misbehavior Rate (MR) -- Eq. (1) in the paper. The paper defines MR from counts of
# False-Information and DoS incidents observed by the monitor; here, the "incidents"
# are the trained PRHN-lite detector's per-tick attack classifications of the target
# drone (fed to the friend drone over a trusted inter-drone link -- see module docstring
# in env_multi.py for how this is wired to the physics simulation).
# ----------------------------------------------------------------------------------

def misbehavior_rate(n_flagged_incidents: int, n_window_ticks: int,
                      alpha=0.35, beta=0.05) -> float:
    """MR in [0,1], Eq. (1): a smooth (exponential-based) function of how many ticks in
    the monitoring window were flagged as an attack by the friend drone's detector, not
    a raw fraction -- so a couple of isolated misclassifications don't already read as
    near-certain misbehavior (matches the paper's exponential MR1 for the more dangerous
    incident type; this module treats all 4 attack types as equally "lethal" incidents,
    since Section 11 of the framework doesn't rank them)."""
    if n_window_ticks <= 0:
        return 0.0
    incident_frac = n_flagged_incidents / n_window_ticks
    mr1 = alpha * (np.exp(incident_frac) - 1) + beta  # Eq. (1)'s MR1 form, normalized to start at beta
    return float(np.clip(mr1, 0.0, 1.0))


# ----------------------------------------------------------------------------------
# Cost/profit parameters -- Eqs. (2)-(9) and (16)-(21). Kept as a dataclass so they're
# explicit, tunable, and visible rather than buried as magic numbers.
# ----------------------------------------------------------------------------------

@dataclass
class GameParams:
    # First game (IDS activation) params
    fne_mb: float = 0.15     # expected false-negative rate of the *lightweight* PRHN-lite
                              # detector (i.e. how often it misses a real attack) -- taken
                              # from this project's own measured recall gaps (README.md)
    g_mb: float = 0.85       # damage incurred by a missed malicious act, in [0,1]
    cost_mb_attacker: float = 0.25   # attacker's cost to carry out an attack
    cost_ids: float = 0.10           # monitoring drone's cost to run its IDS (compute/comms)
    edr_mb: float = 0.85     # IDS's expected detection rate (matches this project's
                              # measured recall, see README.md's results table)
    fpr_mb: float = 0.02     # expected false-positive rate (measured, see README.md)

    # Second game (IES ejection) params
    fne_attacker: float = 0.15
    eff_ies: float = 0.85    # IES's effectiveness at correctly ejecting a real attacker
    cost_attacker: float = 0.25
    cost_ies: float = 0.10   # cost of running the ejection process (blacklisting, BC write)
    edr_attacker: float = 0.85
    fpe_attacker: float = 0.02  # false-positive ejection rate (ejecting a normal drone)


def _ids_profits(p: GameParams):
    """Eqs. (2)-(9): profits (Q'_ji, Q_ji) for (JIDS, Jattacker) under each strategy pair."""
    # (i) IDS idle, attacker attacks -- Eqs. (2),(3)
    q_idle_attack_ids = -p.fne_mb * p.g_mb
    q_idle_attack_atk = p.fne_mb * p.g_mb - p.cost_mb_attacker
    # (ii) IDS monitors, attacker attacks -- Eqs. (4),(5)
    q_mon_attack_ids = p.edr_mb - p.cost_ids
    q_mon_attack_atk = -(p.edr_mb + p.cost_mb_attacker)
    # (iii) attacker normal, IDS monitors -- Eqs. (6),(7)
    q_mon_normal_atk = p.fpr_mb
    q_mon_normal_ids = -(p.fpr_mb + p.cost_ids)
    # (iv) attacker normal, IDS idle -- Eqs. (8),(9)
    q_idle_normal_atk = p.fpr_mb
    q_idle_normal_ids = -p.fpr_mb
    return {
        ("idle", "attack"): (q_idle_attack_ids, q_idle_attack_atk),
        ("monitor", "attack"): (q_mon_attack_ids, q_mon_attack_atk),
        ("monitor", "normal"): (q_mon_normal_ids, q_mon_normal_atk),
        ("idle", "normal"): (q_idle_normal_ids, q_idle_normal_atk),
    }


def _ies_profits(p: GameParams):
    """Eqs. (16)-(21): profits (S'_ji, S_ji) for (JIES, Jnode)."""
    ine_ies = 1 - p.eff_ies
    # (i) attacker, not ejected -- Eqs. (16),(17)
    s_noeject_attacker_node = p.fne_attacker * ine_ies * p.cost_attacker
    s_noeject_attacker_ies = -p.fne_attacker * ine_ies
    # (ii) attacker, ejected -- Eqs. (18),(19)
    s_eject_attacker_node = -(p.eff_ies * p.edr_attacker + p.cost_attacker)
    s_eject_attacker_ies = p.eff_ies * p.edr_attacker - p.cost_ies
    # (iii) normal node, categorized as attacker -- Eqs. (20),(21)
    s_eject_normal_node = p.fpe_attacker * ine_ies
    s_eject_normal_ies = -(p.fpe_attacker * ine_ies + p.cost_ies)
    return {
        ("noeject", "attacker"): (s_noeject_attacker_ies, s_noeject_attacker_node),
        ("eject", "attacker"): (s_eject_attacker_ies, s_eject_attacker_node),
        ("eject", "normal"): (s_eject_normal_ies, s_eject_normal_node),
    }


def bne_threshold_attacker(p: GameParams) -> float:
    """B_attacker (Eq. 13's lower bound, specialized via Theorem 1): the MR value above
    which the attacker's own best response is to actually attack, i.e. the point past
    which the IDS should have activated monitoring. Derived from the profit table by
    finding the MR at which the IDS is indifferent between idle and monitor, given the
    attacker plays its equilibrium mixed strategy -- implemented here as the closed-form
    break-even point of Eqs. (2)-(9), which is algebraically what Theorem 1's BNE
    condition reduces to for this 2x2 game."""
    profits = _ids_profits(p)
    # IDS's payoff difference between monitoring and staying idle, as a function of the
    # attacker's mixed-strategy probability of attacking (= MR, per Section IV-A's own
    # identification of MR with the attacker's attack probability y_j)
    ids_monitor_attack, _ = profits[("monitor", "attack")]
    ids_idle_attack, _ = profits[("idle", "attack")]
    ids_monitor_normal, _ = profits[("monitor", "normal")]
    ids_idle_normal, _ = profits[("idle", "normal")]
    # IDS prefers monitor when: MR*ids_monitor_attack + (1-MR)*ids_monitor_normal
    #                          > MR*ids_idle_attack   + (1-MR)*ids_idle_normal
    numerator = ids_idle_normal - ids_monitor_normal
    denominator = (ids_monitor_attack - ids_idle_attack) + (ids_idle_normal - ids_monitor_normal)
    if abs(denominator) < 1e-9:
        return 0.5
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def bne_threshold_node(p: GameParams) -> float:
    """B_node (Eq. 25's lower bound, via Theorem 2): the MR value above which the IES's
    best response is to eject, following the same break-even construction as
    bne_threshold_attacker but over the ejection game's profit table."""
    profits = _ies_profits(p)
    ies_eject_attacker, _ = profits[("eject", "attacker")]
    ies_noeject_attacker, _ = profits[("noeject", "attacker")]
    ies_eject_normal, _ = profits[("eject", "normal")]
    ies_noeject_normal = 0.0  # paper doesn't define a (noeject, normal) cell -- zero-profit baseline
    numerator = ies_noeject_normal - ies_eject_normal
    denominator = (ies_eject_attacker - ies_noeject_attacker) + (ies_noeject_normal - ies_eject_normal)
    if abs(denominator) < 1e-9:
        return 0.5
    return float(np.clip(numerator / denominator, 0.0, 1.0))


# ----------------------------------------------------------------------------------
# Stateful monitor: one instance per (friend drone, watched drone) pair. This is what
# a "friend" drone runs continuously against its teammate.
# ----------------------------------------------------------------------------------

@dataclass
class FriendMonitor:
    """Tracks a sliding window of the friend drone's detector flags against its
    teammate, computes MR, and applies the two-game Bayesian decision. Distinguishes
    *transitory* misbehavior (Section II-B: MR crosses the threshold briefly, then
    drops back) from *permanent* misbehavior (MR stays over threshold for
    `persistence_ticks` in a row) -- only the latter triggers ejection, exactly the
    paper's central point (a single bad reading should not eject a teammate)."""
    params: GameParams = field(default_factory=GameParams)
    window_ticks: int = 100          # 1.0s at dt=0.01, matches detector.py's feature window
    persistence_ticks: int = 300     # 3.0s of sustained over-threshold MR before ejecting
    _flags: deque = field(default_factory=lambda: deque(maxlen=100))
    _over_threshold_streak: int = 0
    monitoring_active: bool = False
    ejected: bool = False

    def __post_init__(self):
        self._flags = deque(maxlen=self.window_ticks)
        self.b_attacker = bne_threshold_attacker(self.params)
        self.b_node = bne_threshold_node(self.params)

    def update(self, detector_flagged_attack: bool):
        """Call once per tick with whether the friend's detector flagged an attack on
        the watched drone this tick. Returns a dict with the current decision state."""
        if self.ejected:
            return self._status(mr=1.0)

        self._flags.append(1 if detector_flagged_attack else 0)
        mr = misbehavior_rate(sum(self._flags), len(self._flags))

        # ---- Game 1: IDS activation ----
        self.monitoring_active = mr >= self.b_attacker

        # ---- Game 2: IES ejection (only meaningful once monitoring) ----
        if self.monitoring_active and mr >= self.b_node:
            self._over_threshold_streak += 1
        else:
            self._over_threshold_streak = 0  # transitory misbehavior: streak resets

        if self._over_threshold_streak >= self.persistence_ticks:
            self.ejected = True

        return self._status(mr=mr)

    def _status(self, mr: float):
        return {
            "mr": mr,
            "b_attacker": self.b_attacker,
            "b_node": self.b_node,
            "monitoring_active": self.monitoring_active,
            "over_threshold_streak": self._over_threshold_streak,
            "persistence_required": self.persistence_ticks,
            "ejected": self.ejected,
        }
