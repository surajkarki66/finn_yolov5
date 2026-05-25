#!/usr/bin/env python3
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import argparse
import numpy as np
from time import time
from typing import List, Tuple

import cpp_detect

from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

from TinyYOLOv3DetectorDriver import TinyYOLOv3DetectorDriver
from models import Detect


def find_working_camera(max_index: int = 10) -> Tuple[cv2.VideoCapture, int, int, int]:
    """Try camera indices and return first working camera"""
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✓ Using camera index {idx} with resolution {w}x{h}")
            return cap, idx, int(w), int(h)

        cap.release()

    raise RuntimeError(f"❌ No working camera found in indices 0..{max_index-1}")


class ResultWindow:
    """Detection window - using __slots__ for memory efficiency"""
    __slots__ = ['x', 'y', 'y_frombottom', 'width', 'height', 'value', 'active']
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 value: int, image_height: int):
        self.x = x
        self.y = y
        self.y_frombottom = image_height - y - height
        self.width = width
        self.height = height
        self.value = value
        self.active = True


class HPD2PeopleCounter:
    """
    HPD2 People Counter with vectorized operations
    """
    
    # Constants
    HUNDRED_TO_DENSITY_MUL = 5000
    
    # People averaging parameters
    PEOPLE_AVER_SECOND = 4
    PEOPLE_AVER_SECOND_HI_LIMIT = 6
    PEOPLE_AVER_SECOND_LOW_LIMIT = 3
    NOOF_PEOPLE_AVER_LENGTH_LOW_LIMIT = 3
    NOOF_PEOPLE_AVER_LENGTH_HI_LIMIT = 20
    PEOPLE_AVER_SECOND_SLOWDOWN_COEF = 0.8
    PEOPLE_AVER_SECOND_SPEEDUP_REGULAR_COEF = 7.0
    
    # Non-maxima suppression
    WALLM_DENSITY_MAX_WNDS = 500
    WALLM_DENSITY_WEIGHT = 1000000
    
    def __init__(self, image_width: int = 640, image_height: int = 480,
                 fps: float = 25.0, threshold: float = 0.5, peak_hold: bool = True):
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps
        self.threshold = threshold
        # peak_hold=True: display never goes down until scene is empty (original HPD2 behavior).
        # peak_hold=False: display tracks smoothed count so it can decrease when people leave.
        self.peak_hold = peak_hold
        
        # State variables
        self.max_noof_people = 0
        self.f_noof_people = 0.0
        self.f_one_person = 0.0
        self.person_frame_count = 0
        self.switch_relay_on_frame_fixed = True
        
        # Pre-compute constants for vectorization
        self.int_threshold = int(threshold * (100 * self.HUNDRED_TO_DENSITY_MUL))
        self.height_scale = (1 << 10) / self.image_height
        
    def apply_threshold_by_size_and_distance_vectorized(self, windows: List[ResultWindow],
                                                        threshold: float) -> List[ResultWindow]:
        """Vectorized adaptive thresholding"""
        if not windows:
            return windows
            
        n = len(windows)
        
        # Extract window properties into arrays
        widths = np.array([w.width for w in windows], dtype=np.int32)
        heights = np.array([w.height for w in windows], dtype=np.int32)
        y_frombottom = np.array([w.y_frombottom for w in windows], dtype=np.int32)
        values = np.array([w.value for w in windows], dtype=np.int32)
        
        # Vectorized calculations
        relative_size = (((widths + heights) // 2) * (1 << 10)) // self.image_height
        relative_pos = ((y_frombottom + heights // 2) * (1 << 10)) // self.image_height
        
        # Threshold ratios
        thresh_ratio1 = np.minimum(relative_size * 2 + 700, 1024)
        thresh_ratio2 = np.maximum(((800 - 500 + 800) << 10) // (relative_pos - 500 + 800), 1024)
        thresh_ratio = np.minimum((thresh_ratio1 * thresh_ratio2) >> 10, 1024)
        
        # Apply threshold
        adapted_threshold = (self.int_threshold * thresh_ratio) >> 10
        
        # Filter windows (vectorized comparison)
        mask = values >= adapted_threshold
        
        # Update values that passed but are below standard threshold
        low_conf_mask = (values >= adapted_threshold) & (values < self.int_threshold)
        values[low_conf_mask] = self.int_threshold
        
        # Build filtered list
        filtered_windows = []
        for i in range(n):
            if mask[i]:
                windows[i].value = values[i]
                filtered_windows.append(windows[i])
                
        return filtered_windows
    
    def suppress_non_maxima_optimized(self, windows: List[ResultWindow]) -> List[ResultWindow]:
        """Optimized non-maxima suppression with cv2.dnn.NMSBoxes fallback"""
        if not windows:
            return windows
        
        # For small number of windows, use OpenCV's optimized NMS
        if len(windows) <= 50:
            return self._suppress_with_cv2_nms(windows)
            
        # For larger numbers, use the original algorithm but with optimizations
        return self._suppress_custom_algorithm(windows)
    
    def _suppress_with_cv2_nms(self, windows: List[ResultWindow]) -> List[ResultWindow]:
        """Fast NMS using OpenCV's C++ implementation"""
        n = len(windows)
        
        # Build boxes and scores arrays
        boxes = np.zeros((n, 4), dtype=np.float32)
        scores = np.zeros(n, dtype=np.float32)
        
        for i, w in enumerate(windows):
            boxes[i] = [w.x, w.y, w.width, w.height]  # xywh format
            scores[i] = w.value / (100.0 * self.HUNDRED_TO_DENSITY_MUL)  # normalize
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=0.34  # 34% overlap threshold from original
        )
        
        # Handle different OpenCV versions
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        elif len(indices):
            indices = np.array(indices).flatten()
        else:
            return []
        
        return [windows[i] for i in indices]
    
    def _suppress_custom_algorithm(self, windows: List[ResultWindow]) -> List[ResultWindow]:
        """Optimized version of original algorithm"""
        # Sort by value (descending)
        windows.sort(key=lambda w: w.value, reverse=True)
        
        if len(windows) > self.WALLM_DENSITY_MAX_WNDS:
            windows = windows[:self.WALLM_DENSITY_MAX_WNDS]
            
        win_count = len(windows)
        
        # Pre-allocate arrays
        boxes = np.zeros((win_count, 4), dtype=np.int32)
        areas = np.zeros(win_count, dtype=np.int32)
        
        # Vectorized box extraction
        for i, w in enumerate(windows):
            boxes[i] = [w.x, w.y, w.x + w.width, w.y + w.height]
            areas[i] = w.width * w.height
        
        # Build overlap matrix (vectorized intersection)
        W = np.zeros((win_count, win_count), dtype=np.uint8)
        
        for i in range(win_count):
            # Vectorized intersection with all remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
            
            w_intersect = np.maximum(0, xx2 - xx1)
            h_intersect = np.maximum(0, yy2 - yy1)
            intersection = w_intersect * h_intersect
            
            # Get minimum area for each pair
            min_area = np.minimum(areas[i], areas[i+1:])
            
            # Check overlap threshold (>34%)
            overlap_mask = intersection > (min_area * 22 // 64)
            
            # Update matrix
            indices = np.where(overlap_mask)[0] + i + 1
            W[indices, i] = 1
            W[i, indices] = 1
        
        # Select best non-overlapping set (same algorithm)
        win_value = np.array([w.value for w in windows], dtype=np.int64)
        x_activewnd = np.zeros(win_count, dtype=np.uint8)
        xt_m = W.astype(np.int64)
        
        for _ in range(1000):
            x_idx = -1
            cost = 0x7fffffffffffffff
            
            for i in range(win_count):
                if x_activewnd[i]:
                    continue
                    
                x_activewnd[i] = 1
                x0 = -(np.dot(win_value, x_activewnd))
                x1 = int(np.dot(xt_m[i], x_activewnd)) * self.WALLM_DENSITY_WEIGHT
                x_activewnd[i] = 0
                
                if x0 + x1 < cost:
                    cost = x0 + x1
                    x_idx = i
                    
            if x_idx >= 0:
                x_activewnd[x_idx] = 1
                xt_m += W[x_idx][:, np.newaxis]  # Vectorized update
            else:
                break
        
        # Return only active windows
        return [windows[i] for i in range(win_count) if x_activewnd[i] > 0]
    
    def calculate_people_count(self, windows: List[ResultWindow]) -> Tuple[int, int]:
        """Calculate smoothed people count"""
        noof_detections = len(windows)
        confirmed_detection = 1 if noof_detections > 0 else 0
        noof_detections_confirmed = noof_detections * confirmed_detection
        
        # Adaptive averaging
        f_aver_second_slow = self.PEOPLE_AVER_SECOND
        
        if noof_detections_confirmed < self.f_noof_people:
            if self.f_noof_people < 1:
                f_aver_second_slow += (f_aver_second_slow * (1 - self.f_noof_people) * 
                                      self.PEOPLE_AVER_SECOND_SLOWDOWN_COEF)
        else:
            f_aver_second_slow = (f_aver_second_slow * 
                                 (self.f_noof_people + self.PEOPLE_AVER_SECOND_SPEEDUP_REGULAR_COEF) /
                                 (noof_detections_confirmed + self.PEOPLE_AVER_SECOND_SPEEDUP_REGULAR_COEF))
        
        f_aver_second_slow = max(min(f_aver_second_slow, self.PEOPLE_AVER_SECOND_HI_LIMIT), 
                                 self.PEOPLE_AVER_SECOND_LOW_LIMIT)
        
        average_length = int(f_aver_second_slow * self.fps + 0.5)
        average_length = max(min(average_length, self.NOOF_PEOPLE_AVER_LENGTH_HI_LIMIT), 
                            self.NOOF_PEOPLE_AVER_LENGTH_LOW_LIMIT)
        
        # Update moving averages
        if self.person_frame_count >= 0 or self.switch_relay_on_frame_fixed:
            self.f_noof_people = ((self.f_noof_people * average_length - 
                                  self.f_noof_people + noof_detections) / average_length)
            self.f_one_person = ((self.f_one_person * average_length - 
                                 self.f_one_person + confirmed_detection) / average_length)
        else:
            self.f_one_person = 0
            self.f_noof_people = 0
                
        self.person_frame_count += 1
        
        one_person = int(self.f_one_person + 0.5)
        noof_people = int(self.f_noof_people + 0.5)
        
        if one_person and noof_people > self.max_noof_people:
            self.max_noof_people = noof_people
            
        # Output logic
        if one_person == 0:
            noof_people_out = 0
            self.max_noof_people = 0
        elif self.peak_hold:
            noof_people_out = max(self.max_noof_people, noof_people)
        else:
            noof_people_out = noof_people

        return noof_people_out, noof_detections_confirmed
    
    def reset(self):
        """Reset counter state"""
        self.max_noof_people = 0
        self.f_noof_people = 0.0
        self.f_one_person = 0.0
        self.person_frame_count = 0


class PeopleCounter:
    """
    Optimized real-time people counter
    """
    
    def __init__(self, args):
        self.args = args
        
        # I/O shape dictionary
        self.io_shape_dict = {
            "idt": [DataType['UINT8']],
            "odt": [DataType['INT8']],
            "ishape_normal": [(1, 416, 416, 3)],
            "oshape_normal": [(1, 13, 13, 18)],
            "ishape_folded": [(1, 416, 416, 1, 3)],
            "oshape_folded": [(1, 13, 13, 18, 1)],
            "ishape_packed": [(1, 416, 416, 1, 3)],
            "oshape_packed": [(1, 13, 13, 18, 1)],
            "input_dma_name": ['idma0'],
            "output_dma_name": ['odma0'],
            "number_of_external_weights": 0,
            "num_inputs": 1,
            "num_outputs": 1,
        }
        
        # Setup device
        device = Device.devices[args.device]
        
        # Setup Detect head
        anchors = np.array([[10, 14, 23, 27, 37, 58]]) / np.array([32])
        detect_head = Detect(args.num_classes, anchors)
        
        # Initialize FINN accelerator
        print("Loading bitstream...")
        self.accel = FINNExampleOverlay(
            bitfile_name=args.bitfile,
            platform=args.platform,
            io_shape_dict=self.io_shape_dict,
            batch_size=1,
            runtime_weight_dir=args.runtime_weight_dir,
            device=device
        )
        print("Bitstream loaded!")
        
        # Create detector driver
        self.detector_driver = TinyYOLOv3DetectorDriver(
            self.accel,
            self.io_shape_dict,
            batch_size=1,
            detect_head=detect_head,
            scale_path=args.scale_path,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
        
        # Initialize camera
        if args.camera_id < 0:
            print("Auto-detecting camera...")
            self.cap, cam_index, self.cam_width, self.cam_height = find_working_camera(max_index=10)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if args.camera_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        else:
            print(f"Opening camera {args.camera_id}...")
            self.cap = cv2.VideoCapture(args.camera_id)

            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {args.camera_id}")

            if args.camera_width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            if args.camera_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            if args.camera_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera: {self.cam_width}x{self.cam_height} @ {args.camera_fps} FPS")
        
        # Initialize optimized HPD2 counter
        peak_hold = not getattr(args, 'no_peak_hold', False)
        self.people_counter = HPD2PeopleCounter(
            image_width=self.cam_width,
            image_height=self.cam_height,
            fps=args.camera_fps,
            threshold=args.hpd2_threshold,
            peak_hold=peak_hold,
        )

        # Motion detection
        self.motion_triggered = getattr(args, 'motion_triggered', False)
        self.motion_engine_cpp = None
        self.motion_wnd_cnt = 0
        if self.motion_triggered:
            cell = getattr(args, 'motion_cell_size', 25)
            self.motion_engine_cpp = cpp_detect.MotionEngine(
                self.cam_width, self.cam_height, cell, cell, 5
            )
            self.motion_wnd_cnt = self.motion_engine_cpp.get_window_count()
            print(f"Motion detection: {self.motion_wnd_cnt} windows (C++)")
            print("  [Motion] Status column shows MOT:<n> = active motion windows; people count updates only when motion > 0.")
            self._motion_first_frame_logged = False
        else:
            self._motion_first_frame_logged = True

        # Stats
        self.frame_count = 0
        self.start_time = time()
        self.last_print_time = time()
        
        # Results storage
        self.results = []
    
    
    def detections_to_windows(self, detections):
        """Convert YOLO detections to ResultWindow objects - pre-allocate arrays"""
        n = len(detections)
        if n == 0:
            return []
        if not isinstance(detections, np.ndarray):
            detections = np.asarray(detections, dtype=np.float32)

        xs, ys, ws, hs, vals = cpp_detect.detections_to_arrays(detections, int(self.cam_height))

        # xs, ys, ws, hs, vals are numpy int arrays
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        ws = np.asarray(ws)
        hs = np.asarray(hs)
        vals = np.asarray(vals)

        # Adaptive HPD2 thresholding
        if self.args.hpd2_adaptive_threshold and xs.size > 0:
            y_frombottom = (self.cam_height - ys - hs).astype(np.int32)
            idxs = cpp_detect.apply_hpd2_threshold(ws.astype(np.int32), hs.astype(np.int32),
                                                 y_frombottom.astype(np.int32), vals.astype(np.int32),
                                                 int(self.cam_height), int(self.people_counter.int_threshold))
            if idxs.size == 0:
                return []
            idxs = np.asarray(idxs, dtype=np.intp)
            xs = xs[idxs]
            ys = ys[idxs]
            ws = ws[idxs]
            hs = hs[idxs]
            vals = vals[idxs]

        # Custom suppression
        if self.args.hpd2_nms and xs.size > 0:
            boxes = np.stack((xs, ys, xs + ws, ys + hs), axis=1).astype(np.int32)
            sel = cpp_detect.suppress_custom_selection(boxes, vals.astype(np.int64), int(self.people_counter.WALLM_DENSITY_MAX_WNDS))
            if sel.size == 0:
                return []
            sel = np.asarray(sel, dtype=np.intp)
            xs = xs[sel]
            ys = ys[sel]
            ws = ws[sel]
            hs = hs[sel]
            vals = vals[sel]

        # Build ResultWindow objects only for final selections
        windows = []
        for x, y, w, h, v in zip(xs.tolist(), ys.tolist(), ws.tolist(), hs.tolist(), vals.tolist()):
            windows.append(ResultWindow(int(x), int(y), int(w), int(h), int(v), self.cam_height))

        return windows
    
    
    def run_pipeline(self):
        print(f"\n{'='*60}")
        print(f"Starting people counting pipeline")
        print(f"Resolution: {self.cam_width}x{self.cam_height}")
        print(f"Target FPS: {self.args.camera_fps}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        print("FRAME | TIME(s) | FPS   | RAW | PROC | PEOPLE | STATUS")
        print("-" * 60)
        
        try:
            # Prime pipeline
            ret, current_frame = self.cap.read()
            if not ret:
                raise RuntimeError("Cannot capture first frame")

            prev_frame = None  # for motion: previous frame
            last_people_count = 0
            last_confirmed = 0
            
            self.detector_driver.preproc_and_write_accel([current_frame])
            iteration = 0
            
            # Pre-allocate variables to reduce overhead
            num_outputs = self.io_shape_dict['num_outputs']
            
            while True:
                iteration += 1
                loop_start = time()
                elapsed = time() - self.start_time
                frame_time_ms = int(1000.0 / self.args.camera_fps) if self.args.camera_fps > 0 else 33

                # Motion detection (C++ only)
                active_motion_wnd_cnt = 0
                if self.motion_triggered:
                    if prev_frame is not None:
                        prev_c = np.ascontiguousarray(prev_frame)
                        curr_c = np.ascontiguousarray(current_frame)
                        active_motion_wnd_cnt = self.motion_engine_cpp.process_motion(
                            prev_c, curr_c, frame_time_ms, True, False
                        )
                        if not getattr(self, "_motion_first_frame_logged", True):
                            self._motion_first_frame_logged = True
                            print(f"  [Motion] First frame processed, active_windows={active_motion_wnd_cnt} (motion detection running).")
                    prev_frame = current_frame.copy()
                
                # Check max duration if positive (<=0 means run forever)
                if self.args.max_duration > 0 and elapsed >= self.args.max_duration:
                    print(f"\nReached max duration ({self.args.max_duration}s)")
                    break
                
                # Start FPGA execution (async)
                if iteration > 0:
                    self.accel.execute_on_buffers(asynch=True)
                
                # Capture next frame
                ret, current_frame = self.cap.read()
                if not ret:
                    print("\nEnd of stream")
                    break
                
                self.frame_count += 1
                
                # Preprocess next frame while FPGA runs
                if iteration > 0:
                    self.detector_driver.preproc_and_write_accel([current_frame])
                
                # Wait for FPGA, get results
                if iteration > 0:
                    self.accel.wait_until_finished()
                    
                    # Copy output (optimized loop)
                    for o in range(num_outputs):
                        self.accel.copy_output_data_from_device(self.accel.obuf_packed[o], ind=o)
                    
                    # Get detections
                    batch_detections = self.detector_driver.read_accel_and_postprocess()
                    detections = batch_detections[0]
                    
                    # Convert to windows
                    windows = self.detections_to_windows(detections)
                    raw_count = len(windows)
                    
                    # Apply HPD2 processing with optimized methods
                    if self.args.hpd2_adaptive_threshold and windows:
                        windows = self.people_counter.apply_threshold_by_size_and_distance_vectorized(
                            windows, self.args.hpd2_threshold
                        )
                    
                    if self.args.hpd2_nms and windows:
                        windows = self.people_counter.suppress_non_maxima_optimized(windows)
                    
                    proc_count = len(windows)
                    
                    # Always update counter state; when motion-triggered, only show new count if motion is active
                    people_count, confirmed = self.people_counter.calculate_people_count(windows)
                    if self.motion_triggered and active_motion_wnd_cnt == 0:
                        people_count = last_people_count
                        confirmed = last_confirmed
                    else:
                        last_people_count = people_count
                        last_confirmed = confirmed
                    
                    # Calculate performance
                    throughput_fps = self.frame_count / elapsed
                    
                    # Store result (only if needed)
                    if self.args.save_results:
                        loop_time = time() - loop_start
                        instant_fps = 1.0 / loop_time if loop_time > 0 else 0
                        
                        self.results.append({
                            'frame': self.frame_count,
                            'time': elapsed,
                            'fps': throughput_fps,
                            'instant_fps': instant_fps,
                            'raw': raw_count,
                            'processed': proc_count,
                            'people': people_count,
                            'confirmed': confirmed
                        })
                    
                    # Print status (reduced frequency for performance)
                    time_since_print = time() - self.last_print_time
                    if self.frame_count % self.args.print_every == 0 or time_since_print >= 1.0:
                        status = "PRESENT" if people_count > 0 else "EMPTY"
                        mot = f" | MOT:{active_motion_wnd_cnt}" if self.motion_triggered else ""
                        print(f"{self.frame_count:5d} | {elapsed:7.1f} | {throughput_fps:5.1f} | "
                              f"{raw_count:3d} | {proc_count:4d} | {people_count:6d} | {status}{mot}")
                        self.last_print_time = time()
        
        except KeyboardInterrupt:
            print("\n\nPipeline stopped by user")
        
        finally:
            self.cleanup()
    
    
    def cleanup(self):
        """Cleanup and show final stats"""
        total_runtime = time() - self.start_time
        avg_fps = self.frame_count / total_runtime if total_runtime > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total runtime: {total_runtime:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        
        if self.frame_count > 0:
            print(f"Time per frame: {total_runtime/self.frame_count*1000:.1f}ms")
        
        # People counting stats
        if self.results:
            people_counts = [r['people'] for r in self.results]
            max_people = max(people_counts)
            avg_people = np.mean(people_counts)
            frames_with_people = sum(1 for c in people_counts if c > 0)
            
            print(f"\nPEOPLE COUNTING:")
            print(f"Max people: {max_people}")
            print(f"Average: {avg_people:.2f}")
            print(f"Frames with people: {frames_with_people}/{len(self.results)}")
            print(f"Percentage: {frames_with_people/len(self.results)*100:.1f}%")
            
            # Show distribution
            if max_people > 0:
                print(f"\nDISTRIBUTION:")
                for i in range(max_people + 1):
                    count = sum(1 for c in people_counts if c == i)
                    if count > 0:
                        print(f"  {i} people: {count} frames ({count/len(self.results)*100:.1f}%)")
        
        print(f"{'='*60}")
        
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description='Optimized real-time people counter'
    )
    
    # FPGA configuration
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--platform', type=str, default='zynq-iodma')
    parser.add_argument('--bitfile', type=str, default='resizer.bit')
    parser.add_argument('--runtime_weight_dir', type=str, default='runtime_weights/')
    parser.add_argument('--scale_path', type=str, default='scale.npy')
    
    # Camera configuration
    parser.add_argument('--camera_id', type=int, default=-1)
    parser.add_argument('--camera_width', type=int, default=640)
    parser.add_argument('--camera_height', type=int, default=480)
    parser.add_argument('--camera_fps', type=int, default=30)
    
    # Detection configuration
    parser.add_argument('--conf_thres', type=float, default=0.15)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--num_classes', type=int, default=1)
    
    # HPD2 configuration
    parser.add_argument('--hpd2_threshold', type=float, default=0.5)
    parser.add_argument('--hpd2_adaptive_threshold', action='store_true')
    parser.add_argument('--hpd2_nms', action='store_true')

    # Motion detection
    parser.add_argument('--motion_triggered', action='store_true',
                        help='Only update people count when motion is detected in the scene')
    parser.add_argument('--motion_cell_size', type=int, default=25,
                        help='Motion grid cell size (default 25)')
    parser.add_argument('--no_peak_hold', action='store_true',
                        help='Let people count decrease when detections drop (default: peak-hold, count only resets when scene empty)')
    
    # Output configuration
    parser.add_argument('--print_every', type=int, default=30)
    parser.add_argument('--max_duration', type=float, default=60.0)
    parser.add_argument('--save_results', action='store_true',
                       help='Save detailed results (adds overhead)')
    
    args = parser.parse_args()
    
    # Run pipeline
    counter = PeopleCounter(args)
    counter.run_pipeline()


if __name__ == "__main__":
    main()