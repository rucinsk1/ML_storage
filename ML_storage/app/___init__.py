
from asyncio.log import logger
from pathlib import Path
from tkinter import Image
from typing import Optional

from ML_storage import get_logger
from ML_storage.domain.types import PredictionResult


class Application:
    def __init__(self, debug_directory : Optional[Path] = None) -> None:
        self.logger = get_logger()
        self.debug_directory = debug_directory
        
    def train_model(self, model_name : str,  iterations : int, annotations_dir : Path, save_path : Path) -> None:
        pass
    
    def get_predictions_for_image(self, img : Image, model_name : str, score_threshold : Optional[float] = None) -> PredictionResult:
        pass