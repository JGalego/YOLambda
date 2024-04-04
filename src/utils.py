"""
YOLONNX utilities
"""

import numpy as np

from PIL import Image
from yaml import safe_load

def scale_boxes(orig_shape: tuple, boxes: np.ndarray, scaled_shape: tuple) -> np.ndarray:
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were
    originally specified in (orig_shape) to the shape of a different image (scaled_shape).

    Adapted from `ultralytics.utils.ops.scale_boxes`
    https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.scale_boxes

    Args:
        orig_shape (tuple): The shape of the image that the bounding boxes are for,
                            in the format of (height, width).
        boxes (np.ndarray): the bounding boxes of the objects in the image,
                            in the format of (x1, y1, x2, y2)
        scaled_shape (tuple): the shape of the target image, in the format of (height, width).

    Returns:
        boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    gain = min(orig_shape[0] / scaled_shape[0], orig_shape[1] / scaled_shape[1])
    boxes *= gain
    return boxes


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Converts a tensor of bounding box coordinates in the (x, y, width, height)
    format to the (x1, y1, x2, y2) format where (x1, y1) is the top-left corner
    and (x2, y2) is the bottom-right corner.

    Adapted from `ultralytics.utils.ops.xyxy2xywh`
    https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.xyxy2xywh

    Args:
        x (np.ndarray): The input bounding box coordinates in (x, y, width, height) format

    Returns:
        y (np.ndarray): The bounding box coordinates in (x1, y1, x2, y2) format
    """
    y = np.empty_like(x)

    dw = x[..., 2] / 2    # half-width
    dh = x[..., 3] / 2    # half-height

    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y

    return y


def nms(
        boxes: np.ndarray,
        confs: np.ndarray,
        iou_thres: float = 0.45
    ) -> np.ndarray:
    # pylint: disable=too-many-locals
    """
    Perform non-maximum suppression (NMS) to avoid detecting too many overlapping boxes
    for a given object.

    Adapted from LearnOpenCV
    https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    and `ultralytics.utils.ops.non_max_suppression`
    https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.non_max_suppression

    Args:
        boxes: (np.ndarray) The location predictions for the image, Shape: [num_boxes, 4]
        confs: (np.ndarray) The class prediction scores, Shape: [num_boxes, 1]
        iou_thres: (float) The overlap threshold for suppressing unnecessary boxes

    Returns:
        A list of filtered boxes, Shape: [ , 4]
    """

    # Checks
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    # Extract the coordinates of every prediction box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate the areas of all prediction box
    areas = (x2 - x1) * (y2 - y1)

    # Sort the prediction boxes according to their confidence score
    order = confs.argsort()

    # Initialize an empty list for filtered prediction boxes
    keep = []

    while len(order) > 0:
        # >>>>> Step 1 <<<<<
        # Select the prediction with highest confidence score (S),
        # remove it from the initial list of predictions (P) and
        # add it to the filtered prediction list 'keep'.

        # Extract the index of the prediction with the highest confidence score (S)
        idx = order[-1]

        # Push S to the filtered prediction list
        keep.append(idx)

        # Remove S from the initial list of predictions (P)
        order = order[:-1]

        # >>>>> Step 2 <<<<<
        # Compare prediction S with all the predictions present in P.
        # Calculate the IoU of this prediction S with every other predictions in P.
        # If the IoU is greater than the given threshold for any prediction T present in P,
        #   remove prediction T from P.

        # Find the coordinates of the intersection boxes
        xx1 = np.maximum(x1[idx], x1[order])
        yy1 = np.maximum(y1[idx], y1[order])
        xx2 = np.minimum(x2[idx], x2[order])
        yy2 = np.minimum(y2[idx], y2[order])

        # Find the dimensions of the intersection boxes
        # - take max to avoid negative dimensions due to non-overlapping boxes
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # Find the intersection area
        inter = w * h

        # Find the union of every prediction with S
        union = areas[idx] + areas[order] - inter

        # Find the IoU of every prediction with S
        iou = inter / union

        # Keep prediction boxes with IoU less than the given threshold
        mask = iou <= iou_thres
        order = order[mask]

    return keep


def class_nms(
        boxes: np.ndarray,
        confs: np.ndarray,
        cls_ids: np.ndarray,
        iou_thres: float = 0.45
    ) -> np.ndarray:
    """
    Performs class-based non-maximum suppression (NMS).

    Args:
        boxes: (np.ndarray) The location predictions for the image, Shape: [num_boxes, 4]
        confs: (np.ndarray) The class prediction scores, Shape: [num_boxes, 1]
        cls_ids: (np.ndarray) The class values of the boxes, Shape: [num_boxes, 1]
        iou_thres: (float) The overlap threshold for suppressing unnecessary boxes
    Returns:
        A list of filtered boxes, Shape: [ , 4]
    """
    
	# Checks
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are between 0.0 and 1.0"

    # Get a list of unique classes
    unique_cls_ids = np.unique(cls_ids)

    # Initialize an empty list for filtered prediction boxes
    keep = []

    for cls_id in unique_cls_ids:
        # Extract prediction boxes and confidence scores for a given class
        cls_idxs = np.where(cls_ids == cls_id)[0]
        cls_boxes = boxes[cls_ids == cls_id, :]
        cls_confs = confs[cls_idxs]

        # Perform NMS separately for each class and keep the results
        cls_keep = nms(cls_boxes, cls_confs, iou_thres=iou_thres)
        keep.extend(cls_idxs[cls_keep])

    return keep


def prepare_input(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple, tuple]:
    """
    Processes an image to be used as input to YOLO.

    Args:
        img: (np.ndarray) The input image
        imgsz: (int) size of input images as integer
    Returns:
        A tuple containing the prepared input, the original and the scaled shape of the image
    """

    # Scale image
    orig_shape = orig_w, orig_h = img.size
    ratio = min(imgsz / orig_w, imgsz / orig_h)
    scaled_shape = scaled_w, scaled_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))
    img = img.resize(
        (scaled_w, scaled_h),
        resample=Image.Resampling.BILINEAR
    )

    # Add padding
    dw = 0 if (scaled_w % 32) == 0 else 32 - (scaled_w % 32)
    dh = 0 if (scaled_h % 32) == 0 else 32 - (scaled_h % 32)
    inp = np.full(
        (scaled_h + dh, scaled_w + dw, 3),
        114,  # mean ImageNet intensity (UINT8 0-255)
        dtype=np.float32
    )
    inp[:scaled_h, :scaled_w, :] = np.array(img)

    # Scale input pixel values and format input
    inp = inp / 255.0
    inp = inp.transpose(2, 0, 1)
    inp = inp[None, :, :, :]

    return inp, orig_shape, scaled_shape


def process_output(
        out: np.ndarray,
        orig_shape: tuple[int],
        scaled_shape: tuple[int],
        conf_thres: float = 0.7,
        iou_thres: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes YOLO's output.

    Args:
        out: (np.ndarray) The output from YOLO
        conf_thres: (float) The confidence score threshold
        iou_thres: (float) The IoU threshold
    Returns:
        A tuple containing arrays of bounding boxes, confidence scores and class IDs
    """
    # Get predictions from output
    preds = np.squeeze(out[0]).T

    # Filter out low-confidence predictions
    confs = np.max(preds[:, 4:], axis=1)
    keep = confs > conf_thres
    preds = preds[keep, :]

    # Get prediction boxes
    boxes = preds[:, :4]
    boxes = xywh2xyxy(boxes)

	# Scale boxes
    boxes = scale_boxes(orig_shape, boxes, scaled_shape)

    # Get confidence scores
    confs = np.max(preds[:, 4:], axis=1)

    # Get class IDs
    cls_ids = np.argmax(preds[:, 4:], axis=1)

    # Perform NMS
    idxs = class_nms(boxes, confs, cls_ids, iou_thres=iou_thres)

    return boxes[idxs], confs[idxs], cls_ids[idxs]


def parse_detections(
        boxes: np.ndarray,
        confs: np.ndarray,
        cls_ids: np.ndarray,
        meta: dict = None
    ) -> dict:
    """
    Generates YOLO's predictions in a human-readable format.

    Args:
        boxes: (np.ndarray) The location predictions for the image, Shape: [num_boxes, 4]
        confs: (np.ndarray) The class prediction scores, Shape: [num_boxes, 1]
        cls_ids: (np.ndarray) The class values of the boxes, Shape: [num_boxes, 1]
        meta: (dict) The model's metadata
    Returns:
        A list of filtered boxes, Shape: [ , 4]
    """
    detections = []
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        detections.append({
            'box': box.astype(np.int32).tolist(),
            'conf': float(conf),
            'cls': safe_load(
                meta['names']
            )[cls_id] if isinstance(meta, dict) and 'names' in meta.keys() else cls_id
        })
    return detections
