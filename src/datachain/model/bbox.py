from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from pydantic import Field

from datachain.lib.data_model import DataModel

from .utils import convert_bbox, validate_bbox

if TYPE_CHECKING:
    from .pose import Pose, Pose3D


class BBox(DataModel):
    """
    A data model representing a bounding box.

    Attributes:
        title (str): The title or label associated with the bounding box.
        coords (list[int]): A list of four bounding box coordinates.

    The bounding box follows the PASCAL VOC format, where:
        - (x1, y1) represents the pixel coordinates of the top-left corner.
        - (x2, y2) represents the pixel coordinates of the bottom-right corner.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_albumentations(
        coords: Sequence[float],
        img_size: Sequence[int],
        title: str = "",
    ) -> "BBox":
        """
        Create a bounding box from Albumentations format.

        Albumentations represents bounding boxes as `[x_min, y_min, x_max, y_max]`
        with normalized coordinates (values between 0 and 1) relative to the image size.

        Args:
            coords (Sequence[float]): The bounding box coordinates in
                Albumentations format.
            img_size (Sequence[int]): The reference image size as `[width, height]`.
            title (str, optional): The title or label of the bounding box.
                Defaults to an empty string.

        Returns:
            BBox: The bounding box data model.
        """
        validate_bbox(coords, float)
        bbox_coords = convert_bbox(coords, img_size, "albumentations", "voc")
        return BBox(title=title, coords=list(map(round, bbox_coords)))

    def to_albumentations(self, img_size: Sequence[int]) -> list[float]:
        """
        Convert the bounding box coordinates to Albumentations format.

        Albumentations represents bounding boxes as `[x_min, y_min, x_max, y_max]`
        with normalized coordinates (values between 0 and 1) relative to the image size.

        Args:
            img_size (Sequence[int]): The reference image size as `[width, height]`.

        Returns:
            list[float]: The bounding box coordinates in Albumentations format.
        """
        return convert_bbox(self.coords, img_size, "voc", "albumentations")

    @staticmethod
    def from_coco(
        coords: Sequence[float],
        title: str = "",
    ) -> "BBox":
        """
        Create a bounding box from COCO format.

        COCO format represents bounding boxes as [x_min, y_min, width, height], where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - width and height define the size of the bounding box in pixels.

        Args:
            coords (Sequence[float]): The bounding box coordinates in COCO format.
            title (str): The title of the bounding box.

        Returns:
            BBox: The bounding box data model.
        """
        validate_bbox(coords, float, int)
        bbox_coords = convert_bbox(coords, [], "coco", "voc")
        return BBox(title=title, coords=list(map(round, bbox_coords)))

    def to_coco(self) -> list[int]:
        """
        Return the bounding box coordinates in COCO format.

        COCO format represents bounding boxes as [x_min, y_min, width, height], where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - width and height define the size of the bounding box in pixels.

        Returns:
            list[int]: The bounding box coordinates in COCO format.
        """
        res = convert_bbox(self.coords, [], "voc", "coco")
        return list(map(round, res))

    @staticmethod
    def from_voc(
        coords: Sequence[float],
        title: str = "",
    ) -> "BBox":
        """
        Create a bounding box from PASCAL VOC format.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - (x_max, y_max) are the pixel coordinates of the bottom-right corner.

        Args:
            coords (Sequence[float]): The bounding box coordinates in VOC format.
            title (str): The title of the bounding box.

        Returns:
            BBox: The bounding box data model.
        """
        validate_bbox(coords, float, int)
        return BBox(title=title, coords=list(map(round, coords)))

    def to_voc(self) -> list[int]:
        """
        Return the bounding box coordinates in PASCAL VOC format.

        PASCAL VOC format represents bounding boxes as [x_min, y_min, x_max, y_max],
        where:
        - (x_min, y_min) are the pixel coordinates of the top-left corner.
        - (x_max, y_max) are the pixel coordinates of the bottom-right corner.

        Returns:
            list[int]: The bounding box coordinates in VOC format.
        """
        return self.coords

    @staticmethod
    def from_yolo(
        coords: Sequence[float],
        img_size: Sequence[int],
        title: str = "",
    ) -> "BBox":
        """
        Create a bounding box from YOLO format.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
        - (x_center, y_center) are the normalized coordinates of the box center.
        - width and height normalized values define the size of the bounding box.

        Args:
            coords (Sequence[float]): The bounding box coordinates in YOLO format.
            img_size (Sequence[int]): The reference image size as `[width, height]`.
            title (str): The title of the bounding box.

        Returns:
            BBox: The bounding box data model.
        """
        validate_bbox(coords, float)
        bbox_coords = convert_bbox(coords, img_size, "yolo", "voc")
        return BBox(title=title, coords=list(map(round, bbox_coords)))

    def to_yolo(self, img_size: Sequence[int]) -> list[float]:
        """
        Return the bounding box coordinates in YOLO format.

        YOLO format represents bounding boxes as [x_center, y_center, width, height],
        where:
        - (x_center, y_center) are the normalized coordinates of the box center.
        - width and height normalized values define the size of the bounding box.

        Args:
            img_size (Sequence[int]): The reference image size as `[width, height]`.

        Returns:
            list[float]: The bounding box coordinates in YOLO format.
        """
        return convert_bbox(self.coords, img_size, "voc", "yolo")

    def point_inside(self, x: int, y: int) -> bool:
        """
        Return True if the point is inside the bounding box.

        Assumes that if the point is on the edge of the bounding box,
        it is considered inside.
        """
        x1, y1, x2, y2 = self.coords
        return x1 <= x <= x2 and y1 <= y <= y2

    def pose_inside(self, pose: Union["Pose", "Pose3D"]) -> bool:
        """Return True if the pose is inside the bounding box."""
        return all(
            self.point_inside(x, y)
            for x, y in zip(pose.x, pose.y, strict=False)
            if x > 0 or y > 0
        )

    @staticmethod
    def from_list(coords: Sequence[float], title: str = "") -> "BBox":
        return BBox.from_voc(coords, title=title)

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "BBox":
        keys = ("x1", "y1", "x2", "y2")
        if not isinstance(coords, dict) or set(coords) != set(keys):
            raise ValueError("Bounding box must be a dictionary with coordinates.")
        return BBox.from_voc([coords[k] for k in keys], title=title)


class OBBox(DataModel):
    """
    A data model for representing oriented bounding boxes.

    Attributes:
        title (str): The title of the oriented bounding box.
        coords (list[int]): The coordinates of the oriented bounding box.

    The oriented bounding box is defined by four points:
        - (x1, y1): The first corner of the box.
        - (x2, y2): The second corner of the box.
        - (x3, y3): The third corner of the box.
        - (x4, y4): The fourth corner of the box.
    """

    title: str = Field(default="")
    coords: list[int] = Field(default=[])

    @staticmethod
    def from_list(coords: Sequence[float], title: str = "") -> "OBBox":
        if not isinstance(coords, (list, tuple)):
            raise TypeError("Oriented bounding box must be a list of coordinates.")
        if len(coords) != 8:
            raise ValueError("Oriented bounding box must have 8 coordinates.")
        if not all(isinstance(value, (int, float)) for value in coords):
            raise ValueError(
                "Oriented bounding box coordinates must be floats or integers."
            )
        return OBBox(title=title, coords=list(map(round, coords)))

    @staticmethod
    def from_dict(coords: dict[str, float], title: str = "") -> "OBBox":
        keys = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
        if not isinstance(coords, dict) or set(coords) != set(keys):
            raise ValueError(
                "Oriented bounding box must be a dictionary with coordinates."
            )
        return OBBox.from_list([coords[k] for k in keys], title=title)
