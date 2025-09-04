from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    data_dir: Path
    out_dir: Path
    windows: tuple[int, ...] = (5, 10)
    xi: float = 0.001
    max_goals: int = 10
    min_train_matches: int = 9100  # Only ~20 iterations for quick testing


#Typed config centralizes rolling windows, time‑decay ξ, max goals grid, and minimum rolling training size for walk‑forward evaluation