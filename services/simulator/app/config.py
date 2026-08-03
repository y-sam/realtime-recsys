import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:19092")
    events_topic: str = os.getenv("EVENTS_TOPIC", "user_events")

    # traffic volume
    events_per_second: float = _float("EVENTS_PER_SECOND", 8.0)
    n_users: int = _int("N_USERS", 5_000)
    n_items: int = _int("N_ITEMS", 2_000)

    # behavior
    new_user_rate: float = _float("NEW_USER_RATE", 0.05)   # share of sessions from cold-start users
    base_ctr: float = _float("BASE_CTR", 0.08)
    fatigue_decay: float = _float("FATIGUE_DECAY", 0.55)   # CTR multiplier per re-impression
    purchase_given_click: float = _float("PURCHASE_GIVEN_CLICK", 0.09)
    reward_delay_s: float = _float("REWARD_DELAY_S", 45.0) # mean impression -> conversion lag

    seed: int = _int("SEED", 42)


settings = Settings()
