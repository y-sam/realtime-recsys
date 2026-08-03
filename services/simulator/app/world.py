"""Synthetic world: catalog, users, and behavioral rules.

The point of the simulator is not detailed realism. It is to produce a stream with
the properties the recommender actually has to cope with:

- long-tail (Zipf) popularity   -> retrieval must go beyond top-popular
- latent user x category affinity -> the two-tower has something to learn
- new users arriving continuously -> cold-start
- fatigue: re-impressing an item decays its CTR
- delayed reward: purchases land minutes after the click
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

CATEGORIES = [
    "action", "comedy", "drama", "docu", "sports",
    "music", "kids", "horror", "reality", "news",
]

DEVICES = ["ios", "android", "web", "ctv"]
SURFACES = ["home_feed", "search", "detail_page", "midroll"]


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    popularity: float  # Zipf weight, used when sampling impressions
    price_tier: int


@dataclass
class User:
    user_id: str
    affinity: dict[str, float]        # category -> weight
    device: str
    is_new: bool
    seen: dict[str, int] = field(default_factory=dict)  # item_id -> impression count (fatigue)


class World:
    def __init__(self, n_users: int, n_items: int, seed: int = 42):
        self.rng = random.Random(seed)
        self.items = self._build_catalog(n_items)
        self.item_ids = [i.item_id for i in self.items]
        self.item_weights = [i.popularity for i in self.items]
        self.item_by_id = {i.item_id: i for i in self.items}
        self.users = {u.user_id: u for u in (self._make_user(f"u_{n}") for n in range(n_users))}
        self._new_user_seq = n_users

    # ---------- construction ----------
    def _build_catalog(self, n_items: int) -> list[Item]:
        items = []
        for rank in range(1, n_items + 1):
            items.append(
                Item(
                    item_id=f"i_{rank:05d}",
                    category=self.rng.choice(CATEGORIES),
                    popularity=1.0 / (rank ** 0.9),  # Zipf-ish
                    price_tier=self.rng.randint(1, 4),
                )
            )
        self.rng.shuffle(items)
        return items

    def _make_user(self, user_id: str, is_new: bool = False) -> User:
        favs = self.rng.sample(CATEGORIES, k=self.rng.randint(2, 4))
        affinity = {c: (self.rng.uniform(1.5, 4.0) if c in favs else self.rng.uniform(0.1, 0.8))
                    for c in CATEGORIES}
        return User(
            user_id=user_id,
            affinity=affinity,
            device=self.rng.choice(DEVICES),
            is_new=is_new,
        )

    def spawn_new_user(self) -> User:
        self._new_user_seq += 1
        user = self._make_user(f"u_{self._new_user_seq}", is_new=True)
        self.users[user.user_id] = user
        return user

    # ---------- sampling ----------
    def pick_user(self, new_user_rate: float) -> User:
        if self.rng.random() < new_user_rate:
            return self.spawn_new_user()
        return self.users[self.rng.choice(list(self.users.keys()))]

    def pick_items(self, k: int) -> list[Item]:
        return self.rng.choices(self.items, weights=self.item_weights, k=k)

    # ---------- behavior ----------
    def click_probability(self, user: User, item: Item, base_ctr: float, fatigue_decay: float) -> float:
        p = base_ctr * user.affinity[item.category]
        # new users carry less signal and click less (noisier cold-start CTR)
        if user.is_new:
            p *= 0.6
        # fatigue: every re-impression pushes the probability down
        times_seen = user.seen.get(item.item_id, 0)
        p *= fatigue_decay ** times_seen
        return min(p, 0.95)

    def register_impression(self, user: User, item: Item) -> None:
        user.seen[item.item_id] = user.seen.get(item.item_id, 0) + 1
