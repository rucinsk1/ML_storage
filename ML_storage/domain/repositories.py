from abc import ABC, abstractmethod
from pathlib import Path

from ML_storage.domain.interfaces import Predictor, Trainer

class ModelRepository(ABC):
    
    @abstractmethod
    def get_predictor_for_image(self, name : str) -> Predictor:
        pass
    
    @abstractmethod
    def get_trainer_for_image(self, name : str, annotations_path : Path, device : str = "cpu", max_iterations : int = 1000) -> Trainer:
        pass
    
    @abstractmethod
    def new_trainer(self, annotations_path : Path, iterations : int, name : str) -> Trainer:
        pass
    
    @abstractmethod
    def save_predictor(self, predictor : Predictor) -> None:
        pass
    