"""
YOLONNX
"""

from onnxruntime import InferenceSession

from utils import (
    prepare_input,
    process_output,
    parse_detections
)

class YOLODetector:
    """
    YOLO Detector for ONNX models.

    Supports YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11 and future versions
    """
    def __init__(self, model_path):
        self.sess = InferenceSession(model_path)
        self.meta = self.sess.get_modelmeta().custom_metadata_map

        # Get input and output names dynamically
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def __repr__(self):
        meta = "\n".join([f"\t{prop}: {val}" for prop, val in self.meta.items()])
        return f"<YOLODetector\n" \
               f"\tInput: {self.input_name}\n" \
               f"\tOutput: {self.output_name}\n" \
               f"{meta}\n>"

    def __call__(self, img, imgsz=640, conf_thres=0.3, iou_thres=0.5):
        # Prepare input
        inp, orig_shape, scaled_shape = prepare_input(img, imgsz)

        # Run inference
        out = self.inference(inp)

        # Process output
        boxes, confs, cls_ids = process_output(
            out,
            orig_shape,
            scaled_shape,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )

        # Parse detections
        return parse_detections(boxes, confs, cls_ids, self.meta)

    def inference(self, inp):
        """
        Runs YOLO inference on a prepared image.
        
        Dynamically handles different YOLO versions by using the actual
        input/output names from the model rather than hardcoded values.
        
        Args:
            inp: (np.ndarray) The input tensor
        Returns:
            The inference results
        """
        return self.sess.run([self.output_name], {self.input_name: inp})
