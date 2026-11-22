"""Public inference entry point for the single YAMNet model."""
from .labels import KAZAKH
from experiments.accuracy_v2.inference import YAMNetPredictor as Predictor
