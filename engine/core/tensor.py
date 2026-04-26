from dataclasses import dataclass
from typing import Tuple, Any, Optional

@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    offset: int = 0
    size_bytes: int = 0

class Tensor:
    """
    Dummy Tensor representation for the inference lab.
    Does not allocate real memory unless requested.
    """
    def __init__(self, info: TensorInfo, data: Optional[Any] = None):
        self.info = info
        self.data = data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.info.shape

    @property
    def dtype(self) -> str:
        return self.info.dtype
