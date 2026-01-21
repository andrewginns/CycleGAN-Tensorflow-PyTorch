from __future__ import annotations

import random
from collections import deque

import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self._items: deque[torch.Tensor] = deque(maxlen=capacity)

    @torch.no_grad()
    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        """Mimics the TF ItemPool behavior: sometimes return historical fakes."""

        out = []
        for item in batch:
            item = item.detach()
            if len(self._items) < self.capacity:
                self._items.append(item)
                out.append(item)
            else:
                if random.random() > 0.5:
                    idx = random.randrange(0, self.capacity)
                    old = self._items[idx]
                    self._items[idx] = item
                    out.append(old)
                else:
                    out.append(item)
        return torch.stack(out, dim=0)
