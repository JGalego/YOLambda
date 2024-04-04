"""
YOLONNX
"""

from onnxruntime import InferenceSession

from utils import (
    prepare_input,
    process_output,
    parse_detections
)

class YOLONNX:
    """
    YOLONN Class
    """
    def __init__(self, model):
        self.sess = InferenceSession(model)
        self.meta = self.sess.get_modelmeta().custom_metadata_map

    def __repr__(self):
        meta = "\n".join([f"\t{prop}: {val}" for prop, val in self.meta.items()])
        return f"<YOLONNX \n{meta}\n>"

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
        
        Args:
            inp: (np.ndarray) The input tensor
        Returns:
            The inference results
        """
        return self.sess.run(['output0'], {'images': inp})
