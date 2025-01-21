
__version__ = "1.0.0"

from .model import MachineLearningModel
from .schemas import PredictionInput

MODEL_CONFIG = {
    "features": ["Temperature", "Run_Time"],
    "target": "Downtime_Flag",
    "test_size": 0.2,
    "random_state": 42
}

__all__ = ["MachineLearningModel", "PredictionInput", "MODEL_CONFIG"]