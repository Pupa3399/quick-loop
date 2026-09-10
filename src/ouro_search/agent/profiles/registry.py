from __future__ import annotations

from ouro_search.agent.profiles.base import PromptProfile
from ouro_search.agent.profiles.hermes import HermesProfile
from ouro_search.agent.profiles.search_r1 import SearchR1Profile

_PROFILE_TYPES: dict[str, type[PromptProfile]] = {
    SearchR1Profile.name: SearchR1Profile,
    HermesProfile.name: HermesProfile,
}


def get_prompt_profile(profile: str | PromptProfile) -> PromptProfile:
    if isinstance(profile, PromptProfile):
        return profile
    normalized = profile.strip().lower()
    try:
        return _PROFILE_TYPES[normalized]()
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILE_TYPES))
        raise ValueError(f"Unknown prompt profile {profile!r}; choose one of: {choices}") from exc


def available_prompt_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILE_TYPES))
