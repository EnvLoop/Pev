"""The seven question families of docs/PREREGISTRATION.md: name -> (build function, supported variants)."""
from __future__ import annotations

from collections.abc import Callable

from . import approval, memory_use, notify, pick_option, routing, sharing
from .base import Context, Fragment

FAMILIES: dict[str, tuple[Callable[[Context], Fragment | None], tuple[str, ...]]] = {
    pick_option.FAMILY: (pick_option.build, pick_option.VARIANTS),
    approval.FAMILY: (approval.build, approval.VARIANTS),
    memory_use.APPLY: (memory_use.build_apply, memory_use.APPLY_VARIANTS),
    memory_use.FORGOTTEN: (memory_use.build_forgotten, memory_use.FORGOTTEN_VARIANTS),
    sharing.FAMILY: (sharing.build, sharing.VARIANTS),
    notify.FAMILY: (notify.build, notify.VARIANTS),
    routing.FAMILY: (routing.build, routing.VARIANTS),
}
FAMILY_NAMES = tuple(FAMILIES)
