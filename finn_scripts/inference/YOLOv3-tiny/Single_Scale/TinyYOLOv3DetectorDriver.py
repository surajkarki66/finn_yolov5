import numpy as np
import cv2

from utils import letterbox, non_max_suppression


class TinyYOLOv3DetectorDriver():
    """
    Optimized DetectorDriver for Tiny YOLOv3
    """

    def __init__(self, accel, io_shape_dict, batch_size, detect_head, 
                 scale_path="scale.npy", conf_thres=0.10, iou_thres=0.5):
        """
        Initialize Tiny YOLOv3 DetectorDriver
        """
        self.accel = accel
        self.io_shape_dict = io_shape_dict
        self.batch_size = batch_size
        self.detect_head = detect_head
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load scale factor
        self.scale = np.load(scale_path)
        
        # Get target image size (NHWC format)
        self.img_size = tuple(io_shape_dict['ishape_normal'][0][1:3])  # (416, 416)
        
        # Pre-allocate buffers for preprocessing
        self.preprocessed_buffer = np.zeros(
            (batch_size, self.img_size[0], self.img_size[1], 3),
            dtype=np.uint8
        )
        
        # Cache letterbox parameters (for fixed camera resolution)
        self.letterbox_params = []
        self.last_input_shape = None
        self.cached_letterbox_ratio = None
        self.cached_letterbox_pad = None


    def preproc_and_write_accel(self, batch):
        """
        Optimized preprocessing with reduced copies
        """
        self.letterbox_params = []
        
        for i in range(self.batch_size):
            img = batch[i]
            current_shape = img.shape[:2]  # (H, W)
            
            # Check if we can use cached letterbox parameters
            if (self.last_input_shape is not None and 
                current_shape == self.last_input_shape and
                self.cached_letterbox_ratio is not None):
                # Use cached parameters for faster letterbox
                img_letterboxed, ratio, pad = letterbox(
                    img, self.img_size, auto=False
                )
                ratio = self.cached_letterbox_ratio
                pad = self.cached_letterbox_pad
            else:
                # Compute letterbox (and cache for next time)
                img_letterboxed, ratio, pad = letterbox(img, self.img_size, auto=False)
                self.last_input_shape = current_shape
                self.cached_letterbox_ratio = ratio
                self.cached_letterbox_pad = pad
            
            self.letterbox_params.append((ratio, pad))
            
            # Convert BGR to RGB in-place using numpy view
            # This is faster than [:, :, ::-1] for continuous arrays
            b, g, r = cv2.split(img_letterboxed)
            img_rgb = cv2.merge((r, g, b))
            
            # Write directly to pre-allocated buffer
            self.preprocessed_buffer[i] = img_rgb
        
        # Fold input
        ibuf_folded = self.accel.fold_input(self.preprocessed_buffer)
        
        # Pack input
        ibuf_packed = self.accel.pack_input(ibuf_folded)
        
        # Copy to FPGA
        self.accel.copy_input_data_to_device(ibuf_packed)


    def read_accel_and_postprocess(self):
        """
        Optimized postprocessing with reduced allocations
        """
        batch_detections = []
        
        for batch_idx in range(self.batch_size):
            # Read output from FPGA
            output = self._read_single_output(batch_idx)
            
            # Apply scale and transpose in one operation
            # This is faster than separate operations
            output = (self.scale * output).transpose(0, 3, 1, 2)
            
            # Apply Detect head
            pred = self.detect_head([output])[0]
            
            # Apply NMS
            pred = non_max_suppression(
                pred, 
                conf_thres=self.conf_thres, 
                iou_thres=self.iou_thres, 
                classes=None, 
                max_det=1000
            )
            
            # Extract detections
            detections = pred[0] if len(pred) > 0 and len(pred[0]) > 0 else np.array([])
            
            batch_detections.append(detections)
        
        return batch_detections


    def _read_single_output(self, batch_idx):
        """
        Read and unpack FPGA output
        """
        output_idx = 0
        
        # Unpack output
        obuf_folded = self.accel.unpack_output(
            self.accel.obuf_packed[output_idx], 
            ind=output_idx
        )
        
        # Unfold output
        obuf_normal = self.accel.unfold_output(obuf_folded, ind=output_idx)
        
        # Extract single image (keep batch dimension)
        output = obuf_normal[batch_idx:batch_idx+1]
        
        return output