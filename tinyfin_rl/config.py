from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    seed: int = 0
    log_level: str = "INFO"
    device: str = "cpu"
    extras: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> "Config":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extras[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "seed": self.seed,
            "log_level": self.log_level,
            "device": self.device,
        }
        data.update(self.extras)
        return data
