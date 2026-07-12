from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Compute:
    device: torch.device
    use_flash: bool  # flash/sdpa attention on CUDA, manual path on CPU

    @classmethod
    def resolve(cls, spec: str = "auto") -> "Compute":
        if spec == "auto":
            spec = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(spec)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {spec!r} but CUDA is not available")
        return cls(device=device, use_flash=device.type == "cuda")

    @property
    def disable_flash_attention(self) -> bool:
        return not self.use_flash

    def __str__(self) -> str:
        
        return f"Compute(device={self.device}, flash={self.use_flash})"
