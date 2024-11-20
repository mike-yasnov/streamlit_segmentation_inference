from ultralytics import YOLO
from src.utils.paths import get_segmentation_model_path


class SegmentationInference:
    def __init__(self):
        """
        Initialize the SegmentationInference class.

        This class is used to perform object detection segmentation inference on frames.

        :param:
            None

        :return:
            None
        """
        self.model = YOLO(get_segmentation_model_path())

    def predict(self, frame):
        """
        Perform object detection segmentation inference on a given frame.

        :param frame: The frame on which to perform the inference
        :type frame: numpy.ndarray

        :return: The object detection results
        :rtype: ultralytics.YOLO
        """
        return self.model.predict(source=frame, conf=0.6, iou=0.6, show=False, verbose=False)[0]