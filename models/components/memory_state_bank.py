import torch
import torch.nn as nn

class MemoryStateBank(nn.Module):
    """Pure state container. No learnable parameters.
    
    Supports both unidirectional (list of tensors) and bidirectional
    (tuple of lists of tensors) SSM states.
    """
    
    def __init__(self):
        super().__init__()
        self._state = None
        
    def reset(self) -> None:
        self._state = None
        
    def _detach_states(self, states):
        if states is None:
            return None
        if isinstance(states, torch.Tensor):
            return states.detach()
        if isinstance(states, (list, tuple)):
            return type(states)(self._detach_states(s) for s in states)
        return states

    def set_state(self, states) -> None:
        self._state = self._detach_states(states)
        
    def get_state(self):
        return self._state
        
    def update_state(self, states) -> None:
        self._state = self._detach_states(states)
