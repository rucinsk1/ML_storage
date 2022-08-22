from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt

CocoDict = Dict[str, int]
TrainerTuple = Tuple[Path, CocoDict]


@dataclass(frozen=True)
class Image:
    """
    class that we use to store the image data and its name (without extension)
    """

    data_bgr: npt.NDArray[np.uint8]
    name: str

    @cached_property
    def black_and_white(self) -> npt.NDArray[np.uint8]:
        """
        white background and black foreground
        """
        # into gray
        im_gray = cv2.cvtColor(self.data_bgr, cv2.COLOR_BGR2GRAY)
        # into black and white
        im_bw = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY)[1]
        # if majority of the image is black,
        # we invert it so that it has white background
        if np.mean(im_bw) < 128:
            im_bw = 255 - im_bw
        return im_bw

    @cached_property
    def height(self) -> int:
        """
        y's range
        """
        return int(self.data_bgr.shape[0])

    @cached_property
    def width(self) -> int:
        """
        x's range
        """
        return int(self.data_bgr.shape[1])

    @cached_property
    def depth(self) -> int:
        """
        color channels
        """
        if len(self.data_bgr.shape) == 2:
            return 1
        return int(self.data_bgr.shape[2])

    def clone(self) -> Image:
        return Image(
            self.data_bgr.copy(),
            self.name,
        )

    def __hash__(self) -> int:
        """
        hashing for caching purposes (lru_cache)
        """
        return hash_image_array(self.data_bgr) ^ hash(self.name)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Image):
            return False
        return (
            np.array_equal(self.data_bgr, o.data_bgr) and self.name == o.name
        )


@dataclass(frozen=True, eq=True)
class Box:
    """
    a box type that is used to represent a bounding box - rectangle
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @cached_property
    def rectangle_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Two opposite pairs of points that represent the rectangle
        """
        return (
            (
                self.xmin,
                self.ymin,
            ),
            (
                self.xmax,
                self.ymax,
            ),
        )

    @cached_property
    def width(self) -> int:
        return abs(self.xmax - self.xmin)

    @cached_property
    def height(self) -> int:
        return abs(self.ymax - self.ymin)

    @cached_property
    def area(self) -> int:
        return self.height * self.width

    @cached_property
    def xmid(self) -> float:
        return (self.xmax + self.xmin) / 2

    @cached_property
    def ymid(self) -> float:
        return (self.ymax + self.ymin) / 2

    def __lt__(self, other: Box) -> bool:
        """
        the only method that is needed to sort boxes
        boxes are sorted per location (left to right, top to bottom), not by area
        """
        return (self.ymid, self.xmid) < (other.ymid, other.xmid)

    def clone(self) -> Box:
        return Box(self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass(frozen=True, eq=True)
class Prediction:
    label: str
    box: Box
    score: float
    predicted_class: int
    class_name: str

    def are_duplicates(
        self,
        other: Prediction,
        overlap_threshold: float = 0.3,
        one_in_another_threshold: float = 2,
    ) -> bool:
        """
        checks if two matches are duplicates. We check for the label first.
        If the class_names are the same, we check for the box overlap.
        """
        # if not same class_name, not a duplicate
        if self.class_name != other.class_name:
            return False
        relative_intersetion = intersection_area(self.box, other.box) / max(
            self.box.area, other.box.area
        )
        # cases where we have an equipment in another equipment etc.
        # if area of one is much different than the other, not a duplicate
        if (
            max(self.box.area, other.box.area)
            / min(self.box.area, other.box.area)
            > one_in_another_threshold
        ):
            return False
        if relative_intersetion > overlap_threshold:
            return True
        return False

    def __lt__(self, other: Prediction) -> bool:
        """
        the only needed method to be able to sort predictions
        we sort by box location, not by score
        """
        return self.box.__lt__(other.box)


@dataclass
class ImageMetadata:
    """
    metadata of an image that we matched the templates against
    """

    image_name: str
    width: int
    height: int
    depth: int


@dataclass(frozen=True)
class PredictionResult:
    """
    container for resulting predictions with context
    """

    predictions: List[Prediction]
    image_metadata: ImageMetadata


def intersection_area(a: Box, b: Box) -> float:
    """
    calculates the intersection area of two boxes
    """
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def hash_image_array(arr: npt.NDArray[np.uint8]) -> int:
    view: npt.NDArray[np.uint8] = np.ascontiguousarray(arr, dtype=np.uint8)
    return int(
        hashlib.sha1(view).hexdigest(),  # type: ignore # nosec
        16,
    )
