from abc import ABC, abstractmethod
from pathlib import Path
from typing import ParamSpecArgs
from urllib.parse import ParseResultBytes

from ML_storage.domain.types import CocoDict, Image, PredictionResult


class Predictor(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def predict(self, image : Image) -> PredictionResult:
        pass
    
    @abstractmethod
    def preprocess(self, img : Image) -> Image:
        pass
    
    @abstractmethod
    def set_score_threshold(self, score_threshold : float) -> None:
        pass
    
    @abstractmethod
    def set_name(self, new_name : str) -> None:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, o: object) -> bool:
        pass
    
class Trainer(ABC):
    @abstractmethod
    def fit(self) -> None:
        pass
    
    @abstractmethod
    def get_coco_categories(self) -> CocoDict:
        pass
    
    @abstractmethod
    def to_predictor(self) -> Predictor:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
       pass 
   
    @abstractmethod
    def set_iterations(self, iterations : int) -> None:
       pass