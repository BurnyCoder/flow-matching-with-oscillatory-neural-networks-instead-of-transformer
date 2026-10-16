"""Local: audited image splits and resumable batches. Global: paired runs share data without test leakage.

Sources: https://docs.pytorch.org/vision/stable/datasets.html;
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html.
"""

import hashlib  # Audits equality of decoded pixels across split boundaries.
import json  # Loads the cached data provenance manifest.
from dataclasses import dataclass  # Groups prepared data without a framework-specific loader.
from pathlib import Path  # Restricts caches and downloads to the configured workspace directory.

import numpy as np  # Hashing and sklearn operate on CPU arrays.
import torch  # Holds compact uint8 data and reproducible sampling state.
from sklearn.model_selection import train_test_split  # Reuses established stratified splitting.
from torchvision import datasets  # Reuses official dataset downloads and archive integrity checks.

from ofm.artifacts import event, file_hash, save_checkpoint, write_json  # Shared persistence and logging.


def split_indices(labels: torch.Tensor, validation_size: int, seed: int):
    """Local: reserve a stratified holdout. Global: validation examples never enter optimizer batches."""
    train, validation = train_test_split(np.arange(len(labels)), test_size=validation_size, stratify=labels.numpy(),
                                         random_state=seed)  # Uses a fixed split RNG independent of training seeds.
    return torch.from_numpy(train), torch.from_numpy(validation)  # Keeps indices compatible with tensor batch selection.


def image_hashes(images: torch.Tensor) -> list[str]:
    """Local: hash each decoded image. Global: identify exact train/validation/test contamination before training."""
    return [hashlib.sha256(image.numpy().tobytes()).hexdigest() for image in images]  # Hashes all channels and pixels.


@dataclass
class PreparedData:
    """Local: compact CPU tensors and audited indices. Global: all architectures receive the same prepared population."""

    images: torch.Tensor  # Original official training pixels in NCHW uint8 format.
    test_images: torch.Tensor  # Official test pixels, isolated from optimizer indexing.
    train_indices: torch.Tensor  # Rows remaining after held-out duplicate exclusion.
    validation_indices: torch.Tensor  # Validation rows remaining after exclusion against the test set.
    manifest: dict  # Dataset/split hashes and counts, copied into each run.

    def batch(self, indices: torch.Tensor, device: str, test: bool = False) -> torch.Tensor:
        """Local: select and normalize pixels. Global: keep identical preprocessing for every training arm."""
        source = self.test_images if test else self.images  # Prevents accidental test/training array confusion.
        return source[indices].to(device=device, dtype=torch.float32).div_(127.5).sub_(1)  # Maps uint8 to [-1,1].


def prepare_data(name: str, root: str, validation_size: int = 5000) -> PreparedData:
    """Local: download, audit, and cache data. Global: frozen manifests make every comparison reproducible."""
    directory = Path(root) / "prepared"  # Keeps processed artifacts separate from torchvision's source archives.
    directory.mkdir(parents=True, exist_ok=True)  # Allows a fresh clone to run without manual directory setup.
    stem = directory / f"{name}-v1-val{validation_size}"  # Cache identity includes the split size and preprocessing version.
    tensor_path, manifest_path = stem.with_suffix(".pt"), stem.with_suffix(".json")  # Separates tensors from readable provenance.
    if tensor_path.exists() and manifest_path.exists():  # Reuses a previously audited preparation.
        manifest = json.loads(manifest_path.read_text())  # Loads the checksum recorded when the cache was produced.
        if file_hash(tensor_path) != manifest["cache_sha256"]:  # Detects corrupted or manually replaced caches.
            raise ValueError(f"Prepared dataset checksum mismatch: {tensor_path}")  # Refuses silent data changes.
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)  # Avoids arbitrary pickle execution.
        return PreparedData(**tensors, manifest=manifest)  # Restores the exact frozen index sets.
    dataset_type = datasets.MNIST if name == "mnist" else datasets.CIFAR10  # Both sources expose uint8 pixels and labels.
    original = dataset_type(root=root, train=True, download=True)  # torchvision verifies supported archive checksums.
    heldout = dataset_type(root=root, train=False, download=True)  # Downloads test data without using it for model selection.
    images = torch.as_tensor(np.asarray(original.data))  # Materializes the official training pixels on CPU.
    test_images = torch.as_tensor(np.asarray(heldout.data))  # Keeps test pixels in a separate tensor.
    images = images[:, None] if name == "mnist" else images.permute(0, 3, 1, 2).contiguous()  # NCHW for both modalities.
    test_images = test_images[:, None] if name == "mnist" else test_images.permute(0, 3, 1, 2).contiguous()  # Same convention.
    train, validation = split_indices(torch.as_tensor(original.targets), validation_size, 20261016)  # Fixed stratified split.
    hashes, test_hashes = image_hashes(images), image_hashes(test_images)  # Audits equality before normalization.
    test_set = set(test_hashes)  # Exclusion priority gives the official test set precedence.
    clean_validation = torch.tensor([i for i in validation.tolist() if hashes[i] not in test_set], dtype=torch.long)  # No overlap.
    excluded = test_set | {hashes[i] for i in validation.tolist()}  # Even excluded validation rows stay unavailable for training.
    clean_train = torch.tensor([i for i in train.tolist() if hashes[i] not in excluded], dtype=torch.long)  # Removes leakage.
    if len(clean_train) == 0 or len(clean_validation) == 0:  # A pathological audit must fail rather than train on empty data.
        raise ValueError("Duplicate exclusion left an empty training or validation split")  # Explicitly identifies the problem.
    tensors = {"images": images, "test_images": test_images, "train_indices": clean_train,
               "validation_indices": clean_validation}  # Save only tensors, avoiding custom pickle classes.
    save_checkpoint(tensor_path, tensors)  # Atomically stores the exact prepared data and indices.
    manifest = {"dataset": name, "split_seed": 20261016, "normalization": "uint8 / 127.5 - 1", "augmentation": "none",
                "train_count": len(clean_train), "validation_count": len(clean_validation), "test_count": len(test_images),
                "removed_train_overlap": len(train) - len(clean_train), "removed_validation_overlap": len(validation) - len(clean_validation),
                "train_indices": clean_train.tolist(), "validation_indices": clean_validation.tolist(),
                "train_pixel_sha256": hashlib.sha256("".join(hashes).encode()).hexdigest(),
                "test_pixel_sha256": hashlib.sha256("".join(test_hashes).encode()).hexdigest(),
                "cache_sha256": file_hash(tensor_path)}  # Contains enough evidence to reproduce exclusion exactly.
    write_json(manifest_path, manifest)  # Commit metadata after the tensor cache is safely written.
    event("data_prepared", dataset=name, train=len(clean_train), validation=len(clean_validation), test=len(test_images),
          removed_train_overlap=manifest["removed_train_overlap"])  # Exposes changes to effective dataset size.
    return PreparedData(**tensors, manifest=manifest)  # Training receives only audited indices.


class BatchStream:
    """Local: explicit shuffle state. Global: prefetch workers cannot make resumed data streams diverge."""

    def __init__(self, indices: torch.Tensor, seed: int):
        """Local: initialize one data RNG. Global: model construction cannot consume data-order randomness."""
        self.indices = indices.clone()  # Isolates the stream from external modifications to its population.
        self.generator = torch.Generator().manual_seed(seed)  # CPU permutation RNG is independent of flow noise.
        self.order = self.indices[torch.randperm(len(indices), generator=self.generator)]  # Shuffles without replacement.
        self.position, self.epoch = 0, 0  # Cursor and epoch complete the resumable iteration state.

    def next_indices(self, count: int) -> torch.Tensor:
        """Local: assemble a full batch across boundaries. Global: every optimizer update sees the requested batch size."""
        parts = []  # Accumulates slices without discarding a short final batch.
        while count > 0:  # Handles any number of epoch boundaries for small test datasets.
            if self.position == len(self.order):  # Reshuffle only after every row was consumed.
                self.order = self.indices[torch.randperm(len(self.indices), generator=self.generator)]  # Next deterministic order.
                self.position, self.epoch = 0, self.epoch + 1  # Advance the auditable epoch counter.
            take = min(count, len(self.order) - self.position)  # Consume only the available suffix.
            parts.append(self.order[self.position:self.position + take])  # Retains each selected dataset index.
            self.position, count = self.position + take, count - take  # Update cursor and remaining batch demand.
        return torch.cat(parts)  # Produces one flat index vector for tensor-based loading.

    def state_dict(self) -> dict:
        """Local: snapshot iteration state. Global: restarting cannot change future minibatch prefixes."""
        return {"order": self.order.clone(), "position": self.position, "epoch": self.epoch,
                "generator": self.generator.get_state().clone()}  # Copies prevent subsequent iteration mutating a snapshot.

    def load_state_dict(self, state: dict) -> None:
        """Local: validate and restore a snapshot. Global: reject resuming against a different training population."""
        if not torch.equal(state["order"].sort().values, self.indices.sort().values):  # Compare complete population identities.
            raise ValueError("Batch stream checkpoint does not match the prepared training split")  # No silent data replacement.
        self.order, self.position, self.epoch = state["order"].clone(), state["position"], state["epoch"]  # Restore the next row.
        self.generator.set_state(state["generator"])  # Restore future epoch permutations as well.
