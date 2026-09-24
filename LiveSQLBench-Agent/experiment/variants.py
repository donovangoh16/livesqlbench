"""Single source of truth for the SQL-agent experiment variants.

Agent prompt improvements are active for variants 1 and 3. Observation-only
harness instrumentation is active for variants 2 and 3.

Variants 4 and 5 use the forward-only four-agent coordinator. Variant 4 keeps
the Variant 1 agent/harness profiles; Variant 5 keeps the Variant 3 profiles.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class VariantConfig:
    number: int
    requested_agent_profile: str
    requested_harness_profile: str
    requested_orchestration_profile: str = "single"
    active_agent_profile: str = "baseline"
    active_harness_profile: str = "baseline"
    active_orchestration_profile: str = "single"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def behavior_signature(self) -> tuple[str, str]:
        """Backward-compatible agent/harness behavior signature."""
        return self.active_agent_profile, self.active_harness_profile

    @property
    def runtime_signature(self) -> tuple[str, str, str]:
        """All profiles that currently affect runtime behavior."""
        return (
            self.active_agent_profile,
            self.active_harness_profile,
            self.active_orchestration_profile,
        )


VARIANTS = {
    0: VariantConfig(0, "baseline", "baseline"),
    1: VariantConfig(1, "improved", "baseline", active_agent_profile="improved"),
    2: VariantConfig(2, "baseline", "improved", active_harness_profile="improved"),
    3: VariantConfig(
        3, "improved", "improved",
        active_agent_profile="improved", active_harness_profile="improved",
    ),
    4: VariantConfig(
        4, "improved", "baseline", requested_orchestration_profile="multi",
        active_agent_profile="improved", active_harness_profile="baseline",
        active_orchestration_profile="multi",
    ),
    5: VariantConfig(
        5, "improved", "improved", requested_orchestration_profile="multi",
        active_agent_profile="improved", active_harness_profile="improved",
        active_orchestration_profile="multi",
    ),
}


def get_variant(number: int) -> VariantConfig:
    try:
        return VARIANTS[number]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment variant {number}; expected one of {sorted(VARIANTS)}"
        ) from exc
