"""Single source of truth for the four SQL-agent experiment variants.

Agent prompt improvements are active for variants 1 and 3. Harness behavior
remains baseline until its intervention is implemented.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class VariantConfig:
    number: int
    requested_agent_profile: str
    requested_harness_profile: str
    active_agent_profile: str = "baseline"
    active_harness_profile: str = "baseline"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def behavior_signature(self) -> tuple[str, str]:
        """Profiles that currently affect runtime behavior."""
        return self.active_agent_profile, self.active_harness_profile


VARIANTS = {
    0: VariantConfig(0, "baseline", "baseline"),
    1: VariantConfig(1, "improved", "baseline", active_agent_profile="improved"),
    2: VariantConfig(2, "baseline", "improved"),
    3: VariantConfig(3, "improved", "improved", active_agent_profile="improved"),
}


def get_variant(number: int) -> VariantConfig:
    try:
        return VARIANTS[number]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment variant {number}; expected one of {sorted(VARIANTS)}"
        ) from exc
