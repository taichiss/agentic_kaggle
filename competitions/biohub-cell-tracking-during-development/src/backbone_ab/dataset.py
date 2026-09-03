"""Deterministic dataset adapters for controlled corrected-v2 experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch.utils.data import Dataset


class DeterministicAugmentationDataset(Dataset):
    """Apply organizer augmentations with an item/epoch-derived RNG.

    The wrapped dataset must return unaugmented items. Recreating non-persistent
    DataLoader workers each epoch propagates ``set_epoch`` without shared state.
    """

    def __init__(
        self,
        dataset: Dataset,
        augmentations: list[Callable],
        *,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.augmentations = list(augmentations)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        images = item["imgs"]
        coords = item["coords"]
        masks = item["masks"]
        seed_sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        rng = np.random.default_rng(seed_sequence)
        for augmentation in self.augmentations:
            images, coords, masks = augmentation(images, coords, masks, rng=rng)
        return {**item, "imgs": images, "coords": coords, "masks": masks}
