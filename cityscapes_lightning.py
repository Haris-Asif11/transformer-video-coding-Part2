from typing import List, Optional, Union, Sequence

import torchvision
from torchvision import transforms
from datamodules.video_data_api import CityscapesDataset
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomChoice,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
)
from pytorch_lightning import LightningDataModule

from cityscapes import CityscapesVCTLoader  # Ensure this import matches your implementation


class CityscapesDataModule(LightningDataModule):
    """
    PyTorch Lightning data module for the Cityscapes dataset tailored for video compression.

    Args:
        data_dir: Root directory of the Cityscapes dataset.
        train_batch_size: Batch size for training.
        val_batch_size: Batch size for validation.
        crop_size: The size of the crop to take from the original images.
        num_workers: Number of parallel workers for data loading.
        pin_memory: If True, uses pinned GPU memory to help faster transfers to GPU.
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        crop_size: Union[int, Sequence[int]] = (512, 1024),
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pil_transform1 = transforms.Compose([transforms.Resize((512, 1024)), transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])


    def _video_transform(self, mode: str) -> torchvision.transforms.Compose:
        scaling = []
        if mode == "train":
            augmentations = [
                RandomChoice([
                    RandomCrop(size=self.crop_size, pad_if_needed=True, padding_mode="edge"),
                    RandomResizedCrop(size=self.crop_size, scale=(0.6, 1.0)),
                ]),
                RandomHorizontalFlip(p=0.5),
            ]
        else:
            augmentations = [CenterCrop(size=self.crop_size)]

        return Compose(scaling + augmentations)


    def _custom_collate(self, batch) -> CityscapesDataset:
        batch = default_collate(batch)
        return CityscapesDataset(video_tensor=batch)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CityscapesVCTLoader(
            root=self.data_dir,
            as_video=True,
            frames_per_group=5,  # Assuming 30 frames per video sequence
            split="train",
            pil_transform=self.pil_transform1,
            tensor_transform=self._video_transform(mode="train")
        )

        self.val_dataset = CityscapesVCTLoader(
            root=self.data_dir,
            as_video=True,
            frames_per_group=5,  # Consistent with training
            split="test",
            pil_transform=self.pil_transform1,
            tensor_transform=self._video_transform(mode="test")
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=self._custom_collate,
        )