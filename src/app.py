"""
YOLambda - YOLONNX on Lambda
"""

import base64
import json
import logging

from io import BytesIO
from PIL import Image

from yolonnx import YOLONNX

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load YOLO model
logger.info("Loading model")
model = YOLONNX("/opt/yolov8n.onnx")

def handler(
        event,
        context,  # pylint: disable=unused-argument
    ):
    """
    Lambda handler
    """

    # Process payload
    logger.info("Processing payload")
    body = json.loads(event['body'])
    img = body['image']
    imgsz = body.get('imgsz', 640)
    conf_thres = body.get('conf_thres', 0.3)
    iou_thres = body.get('iou_thres', 0.3)

    # Open image
    logger.info("Opening image")
    img = Image.open(
        BytesIO(
            base64.b64decode(
                img.encode('ascii')
            )
        )
    )

    # Get detections
    logger.info("Running inference")
    detections = model(
        img,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres
    )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'detections': detections
        }),
    }
