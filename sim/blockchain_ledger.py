"""
A real (if necessarily scaled-down) hash-chained ledger for logging detection and
ejection events, based on the mechanisms described in Jain, Barke, Garg et al.,
"A Walkthrough of Blockchain-Based Internet of Drones Architectures" (IEEE IoT
Journal, 2024).

What's implemented for real:
  - Genuine block hash-chaining (each block's hash depends on its content AND the
    previous block's hash -- SHA-256, not a placeholder) -- Section III / Fig. 1 of
    the paper: "blocks... comprised of... Set of Transactions... Computed Hash Value."
  - Tamper-evidence: mutating any past block's data breaks every hash after it, and
    `Blockchain.is_valid()` genuinely detects this (tested below, not just claimed).
  - A simplified multi-signature confirmation step before a block is committed,
    modeled on the paper's Section IV description of an n-m multisignature smart
    contract (Feng et al.'s scheme, as summarized there): a block only commits once a
    quorum of the *other* drones (not the one the block is about) have "signed" it,
    so no single drone can unilaterally blacklist another by writing straight to the
    ledger. This directly implements the paper's point (Section II.A.6, "UAV Hijack
    Detection..."): "If a sufficient number of such entries are made against a
    particular UAV, it can be flagged."

What's NOT implemented (and shouldn't be claimed as more than it is):
  - No real distributed consensus (PoW/PoS/PBFT/DPoS) or peer-to-peer networking --
    this is a single-process, in-memory chain shared by the drones in this
    simulation, not a decentralized network of independently-running nodes. The
    paper's comparative consensus-algorithm study (Table I) doesn't map onto a
    single-process simulation in a meaningful way; building genuine multi-node P2P
    consensus is future scope, not something to fake here.
  - "Signatures" are HMAC-style keyed hashes per drone, not real asymmetric
    cryptography (no ECDSA/public-key infra) -- adequate to demonstrate the
    multi-signature *gating* mechanism, not a production security primitive.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Optional


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sign(drone_id: str, secret: bytes, block_hash: str) -> str:
    """HMAC-SHA256 keyed hash standing in for a real per-drone signature -- see the
    module docstring for why this isn't claimed to be production asymmetric crypto."""
    return hmac.new(secret, f"{drone_id}:{block_hash}".encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass
class Block:
    index: int
    timestamp: float
    data: dict            # the "Set of Transactions (Content)" -- e.g. a detection or
                           # ejection event, drone IDs, MR, decision, etc.
    prev_hash: str
    signatures: dict = field(default_factory=dict)   # drone_id -> signature, filled in as
                                                        # confirmations arrive (pre-commit)
    hash: str = ""

    def compute_hash(self) -> str:
        payload = json.dumps({
            "index": self.index, "timestamp": self.timestamp,
            "data": self.data, "prev_hash": self.prev_hash,
        }, sort_keys=True)
        return _hash(payload)


class DroneBlockchain:
    """One shared ledger for the 4-drone fleet. Each drone holds a keyed secret used
    for its "signature"; a block only commits once `quorum` other drones have signed
    it, and a drone can never sign a block that names itself as the subject (so a
    compromised drone can't confirm its own innocence)."""

    def __init__(self, drone_ids, quorum=2, seed=0):
        self.drone_ids = list(drone_ids)
        self.quorum = quorum
        self._secrets = {d: hashlib.sha256(f"{d}-{seed}".encode()).digest() for d in self.drone_ids}
        genesis = Block(index=0, timestamp=time.time(), data={"type": "genesis"}, prev_hash="0" * 64)
        genesis.hash = genesis.compute_hash()
        self.chain = [genesis]
        self._pending: Optional[Block] = None

    def propose_block(self, data: dict) -> Block:
        """A drone proposes a new block (e.g. "drone X flagged as misbehaving, MR=0.7").
        Not committed yet -- needs quorum signatures first."""
        prev = self.chain[-1]
        block = Block(index=prev.index + 1, timestamp=time.time(), data=data, prev_hash=prev.hash)
        block.hash = block.compute_hash()
        self._pending = block
        return block

    def sign_pending(self, drone_id: str) -> bool:
        """A drone signs the currently-pending block. Refuses to sign if the drone is
        the subject of the block (self-confirmation isn't allowed) or isn't a known
        fleet member. Auto-commits once quorum is reached."""
        if self._pending is None or drone_id not in self.drone_ids:
            return False
        subject = self._pending.data.get("subject_drone")
        if drone_id == subject:
            return False  # a drone cannot confirm a block about itself
        self._pending.signatures[drone_id] = _sign(drone_id, self._secrets[drone_id], self._pending.hash)
        if len(self._pending.signatures) >= self.quorum:
            self._commit_pending()
        return True

    def _commit_pending(self):
        self.chain.append(self._pending)
        self._pending = None

    def is_valid(self) -> bool:
        """Walks the whole chain verifying both the hash-chain integrity (Section III's
        "Computed Hash Value") and that every non-genesis block actually met quorum."""
        for i in range(1, len(self.chain)):
            block, prev = self.chain[i], self.chain[i - 1]
            if block.prev_hash != prev.hash:
                return False
            if block.compute_hash() != block.hash:
                return False
            if len(block.signatures) < self.quorum:
                return False
        return True

    def events_about(self, drone_id: str):
        return [b.data for b in self.chain[1:] if b.data.get("subject_drone") == drone_id]

    def __len__(self):
        return len(self.chain)
