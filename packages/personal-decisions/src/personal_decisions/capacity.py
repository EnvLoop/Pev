"""`capacity`: how many questions of each family a shard (or KB) can supply, as counts only.

No records are rendered or written. For every user and every real later choice rated >= 4 (a possible decision time
and pick_option target) each family's `build` is tried `trials` times with the clean variant and RNGs seeded by
("capacity", user id, target index, family, trial), so the result does not depend on the shard's name.

- pick_option: a target is *feasible* when at least one trial yields candidates. Distinct feasible (user, target)
  pairs bound the non-repeating pick_option questions; the other families are synthesized per state and are
  bounded by the number of distinct decision times instead.
- `balanced_questions`: questions per family when pick_option may not repeat a (user, target) pair and every
  family gets the same number of questions (pick_option oversampled to 1/7 of all questions).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .families import FAMILIES, FAMILY_NAMES
from .families.base import Context
from .families.shard import ShardData
from .jsonl import seeded_rng
from .kb.rules import parse_time
from .kb.schema import UserProfile

PICK = "pick_option"


@dataclass
class UserCapacity:
    domain: str
    targets: int  # later choices rated >= 4 (distinct decision times)
    pick_feasible: int  # of those, targets pick_option can be built around
    family_success: dict[str, float]  # family -> share of (target, trial) attempts that built a fragment


def try_family(shard: ShardData, user: UserProfile, family: str, index: int, trial: int) -> bool:
    build, _ = FAMILIES[family]
    rng = seeded_rng("capacity", user.user_id, index, family, trial)
    ctx = Context(user=user, shard=shard, rng=rng, now=parse_time(user.later_interactions[index].timestamp),
                  target_index=index, letter="A", variant="clean")
    return build(ctx) is not None


def user_capacity(shard: ShardData, user: UserProfile, trials: int = 6, families=FAMILY_NAMES) -> UserCapacity:
    liked = [index for index, row in enumerate(user.later_interactions) if row.rating >= 4]
    domains = Counter(shard.items[row.item_id].domain for row in user.later_interactions if row.item_id in shard.items)
    feasible, success = 0, {}
    for family in families:
        ok = 0
        for index in liked:
            hits = sum(try_family(shard, user, family, index, trial) for trial in range(trials))
            ok += hits
            if family == PICK and hits:
                feasible += 1
        success[family] = ok / max(1, len(liked) * trials)
    return UserCapacity(domain=domains.most_common(1)[0][0] if domains else "?", targets=len(liked),
                        pick_feasible=feasible, family_success=success)


def summarize(capacities: list[UserCapacity], families=FAMILY_NAMES) -> dict:
    targets = sum(entry.targets for entry in capacities)
    pick = sum(entry.pick_feasible for entry in capacities)
    per_domain = {}
    for domain in sorted({entry.domain for entry in capacities}):
        group = [entry for entry in capacities if entry.domain == domain]
        per_domain[domain] = {"users": len(group), "targets": sum(entry.targets for entry in group),
                              "pick_feasible_targets": sum(entry.pick_feasible for entry in group),
                              "users_with_pick": sum(1 for entry in group if entry.pick_feasible)}
    success = {family: round(sum(entry.family_success[family] * entry.targets for entry in capacities)
                             / max(1, targets), 4) for family in families}
    users_with = {family: sum(1 for entry in capacities if entry.family_success[family] > 0) for family in families}
    return {"users": len(capacities), "decision_times": targets, "pick_feasible_targets": pick,
            "users_with_pick": sum(1 for entry in capacities if entry.pick_feasible),
            "users_with_ge2_pick": sum(1 for entry in capacities if entry.pick_feasible >= 2),
            "per_domain": per_domain, "attempt_success_rate": success, "users_able": users_with,
            "balanced_questions": {"per_family": pick, "total": pick * len(families),
                                   "states_at_3_questions": round(pick * len(families) / 3)}}


def shard_capacity(shard: ShardData, trials: int = 6, families=FAMILY_NAMES) -> dict:
    return summarize([user_capacity(shard, user, trials, families) for user in shard.users], families)
