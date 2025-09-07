from pathlib import Path
from typing import Any, Callable, Optional, Union
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

class CityscapesVCTLoader(Dataset):
    def __init__(
            self,
            root: Union[str, os.PathLike],
            as_video: bool = False,
            frames_per_group: int = 5,  # Number of frames in each group
            split: str = 'train',
            pil_transform: Optional[Callable[[Any], torch.Tensor]] = None,
            tensor_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.root = Path(root)
        self.as_video = as_video
        self.frames_per_group = frames_per_group
        self.pil_transform = pil_transform or ToTensor()
        self.tensor_transform = tensor_transform
        self.split = split

        # Validate the split
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be either 'train', 'val', or 'test'.")

        # Adjusted folder path to fit the directory structure
        self.folder_path = self.root / 'leftImg8bit_sequence' / split
        print(f"Looking in: {self.folder_path}")

        # Generate the list of files and group them by unique city-clip identifier
        self.clips = self.group_files_by_clip()

        if not self.clips:
            raise RuntimeError("No images found. Check the dataset path and structure.")

    def group_files_by_clip(self):
        # Dictionary to hold city and clip number as key and list of frames as values
        clips = {}
        for file_path in sorted(self.folder_path.glob('*/*.png')):
            # Parse the city, clip number, and frame number from the filename
            parts = file_path.stem.split('_')
            city = parts[0]  # Extract the city name
            clip_number = parts[1]  # Extract the clip number
            frame_number = int(parts[2])  # Extract the frame number

            # Create a unique key for each city and clip combination
            unique_clip_key = f"{city}_{clip_number}"

            if unique_clip_key not in clips:
                clips[unique_clip_key] = []
            clips[unique_clip_key].append((frame_number, file_path))

        # Sort frames within each clip by frame number
        for clip in clips.keys():
            clips[clip].sort()

        return clips

    def __len__(self) -> int:
        # Count total number of items depending on the 'as_video' flag
        if self.as_video:
            total_groups = 0
            for frames in self.clips.values():
                total_groups += len(frames) // self.frames_per_group
            return total_groups
        else:
            total_frames = sum(len(frames) for frames in self.clips.values())
            return total_frames

    def load_image(self, img_path: Path) -> torch.Tensor:
        img = default_loader(str(img_path))
        img = self.pil_transform(img)
        if not isinstance(img, torch.Tensor):
            raise RuntimeError("Expected transformation to return a tensor.")
        return img

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.as_video:
            # Compute which clip and frame group this index corresponds to
            cumulative_groups = 0
            for clip_frames in self.clips.values():
                num_groups_in_clip = len(clip_frames) // self.frames_per_group
                if idx < cumulative_groups + num_groups_in_clip:
                    # Compute local index within this clip
                    local_idx = idx - cumulative_groups
                    start_index = local_idx * self.frames_per_group
                    end_index = start_index + self.frames_per_group
                    images = [self.load_image(frame_path) for _, frame_path in clip_frames[start_index:end_index]]
                    item = torch.stack(images)
                    if self.tensor_transform:
                        item = self.tensor_transform(item)
                    return item
                cumulative_groups += num_groups_in_clip
        else:
            # Compute which frame this index corresponds to in a flattened list of all frames
            cumulative_frames = 0
            for clip_frames in self.clips.values():
                if idx < cumulative_frames + len(clip_frames):
                    # Compute local index within this clip
                    local_idx = idx - cumulative_frames
                    img_path = clip_frames[local_idx][1]
                    item = self.load_image(img_path)
                    if self.tensor_transform:
                        item = self.tensor_transform(item)
                    return item
                cumulative_frames += len(clip_frames)

        raise IndexError("Index out of range")