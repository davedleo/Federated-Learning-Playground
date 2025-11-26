from typing import Callable, Iterable, Any

class Compose:
    """
    A unified Compose that works for:
    - torchvision-style transforms
    - torchaudio-style transforms
    - custom callables
    - lambdas
    - nn.Module transforms (optional)
    """

    def __init__(self, transforms: Iterable[Callable]): 
        self.transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        name = self.__class__.__name__
        ops = ",\n  ".join(repr(t) for t in self.transforms)
        return f"{name}([\n  {ops}\n])"