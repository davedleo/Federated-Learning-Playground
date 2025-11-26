from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Tuple, Union
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from src.data.transforms import Compose


class BaseDataset(torch.utils.data.Dataset, ABC):
    """
    Abstract dataset with optional caching.
    Subclasses must implement load_data(path) -> torch.Tensor
    """

    def __init__(
        self,
        data: List[Tuple[Union[str, Path, torch.Tensor], int]],
        transforms: List[Callable] = [],
        use_cache: bool = True
    ):
        super().__init__()
        self.data = data
        self.transform = Compose(transforms)
        self.use_cache = use_cache
        self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Preload if caching is enabled
        if self.use_cache:
            self.load()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cache:
            x, y = self.cache[i]
        else:
            x, y = self.data[i]
            x = x if isinstance(x, torch.Tensor) else self.load_data(x)
            y = torch.tensor(y, dtype=torch.long)

        if self.transform:
            x = self.transform(x)
        return x, y

    def load(self):
        """
        Populate cache if caching is enabled or cache is empty.
        """
        if (not self.use_cache) or (not self.cache):
            append = self.cache.append
            for x, y in self.data:
                tensor = x if isinstance(x, torch.Tensor) else self.load_data(x)
                append((tensor, torch.tensor(y, dtype=torch.long)))

    def clear(self):
        if self.use_cache:
            self.cache.clear()

    @abstractmethod
    def load_data(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Subclasses implement this.
        Should read the file at path and return a torch.Tensor
        """
        pass

class TorchDataset(BaseDataset):
    """Load .pt/.pth tensors"""

    def load_data(self, path: Union[str, Path]) -> torch.Tensor:
        if isinstance(path, str):
            path = Path(path)
        ext = path.suffix.lower()
        if ext not in {".pt", ".pth"}:
            raise ValueError(f"TorchDataset only supports .pt/.pth files: {path}")
        return torch.load(path)


class RGBDataset(BaseDataset):
    """Load RGB images"""

    def load_data(self, path: Union[str, Path]) -> torch.Tensor:
        if isinstance(path, str):
            path = Path(path)
        img = Image.open(path).convert("RGB")
        tensor = pil_to_tensor(img).float() / 255.0  # C,H,W normalized
        return tensor


class BWDataset(BaseDataset):
    """Load grayscale images"""

    def load_data(self, path: Union[str, Path]) -> torch.Tensor:
        if isinstance(path, str):
            path = Path(path)
        img = Image.open(path).convert("L")
        tensor = pil_to_tensor(img).float() / 255.0  # 1,H,W
        return tensor