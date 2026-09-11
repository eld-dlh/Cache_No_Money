"""env package — importable by Persons 2, 3, 4."""
from env.radar_env     import RadarEnv
from env.memmap_loader import PDWMemmap

__all__ = ["RadarEnv", "PDWMemmap"]

try:
    from env.pdw_dataset import PDWDataset
    __all__.append("PDWDataset")
except ImportError:
    pass  # PyTorch not installed; skip DRL components
