from dataclasses import dataclass

@dataclass
class PersonalityProfile:
    """Simple Big Five personality model for response style control."""

    extraversion: float = 0.5
    agreeableness: float = 0.5
    conscientiousness: float = 0.5
    neuroticism: float = 0.5
    openness: float = 0.5

    def to_prompt(self) -> str:
        """Return a system prompt snippet describing the personality profile."""
        return (
            "Adopt a personality with these Big Five trait levels (0-1 scale): "
            f"extraversion {self.extraversion:.2f}, "
            f"agreeableness {self.agreeableness:.2f}, "
            f"conscientiousness {self.conscientiousness:.2f}, "
            f"neuroticism {self.neuroticism:.2f}, "
            f"openness {self.openness:.2f}. "
            "Adjust tone and phrasing accordingly."
        )
