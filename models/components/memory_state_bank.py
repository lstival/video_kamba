import torch
import torch.nn as nn

class MemoryStateBank(nn.Module):
    """Pure state container. No learnable parameters."""
    
    def __init__(self):
        super().__init__()
        self._state = None
        
    def reset(self) -> None:
        self._state = None
        
    def set_state(self, states: list[torch.Tensor]) -> None:
        # detach() all tensors — breaks graph across frames
        self._state = [s.detach() for s in states]
        
    def get_state(self) -> list[torch.Tensor] | None:
        return self._state
        
    def update_state(self, states: list[torch.Tensor]) -> None:
        self._state = [s.detach() for s in states]
