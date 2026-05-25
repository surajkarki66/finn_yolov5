#!/usr/bin/env python3
import os

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import argparse
import numpy as np
from time import time
from typing import List

# C++ extension
import cpp_detect


from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

from TinyYOLOv3DetectorDriver import TinyYOLOv3DetectorDriver
from models import Detect
from human_counter import ResultWindow, HPD2PeopleCounter
from utils import visualize_boxes, scale_coords


class PeopleCounterVideo:
    """
    People counter pipeline running on a video file instead of a live camera.
    """

    def __init__(self, args):
        self.args = args

        # I/O shape dictionary
        self.io_shape_dict = {
            "idt": [DataType["UINT8"]],
            "odt": [DataType["INT8"]],
            "ishape_normal": [(1, 416, 416, 3)],
            "oshape_normal": [(1, 13, 13, 18)],
            "ishape_folded": [(1, 416, 416, 1, 3)],
            "oshape_folded": [(1, 13, 13, 18, 1)],
            "ishape_packed": [(1, 416, 416, 1, 3)],
            "oshape_packed": [(1, 13, 13, 18, 1)],
            "input_dma_name": ["idma0"],
            "output_dma_name": ["odma0"],
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
            device=device,
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
            iou_thres=args.iou_thres,
        )

        # Open video file
        print(f"Opening video: {args.video}")
        self.cap = cv2.VideoCapture(args.video)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {args.video}")

        self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

        print(f"Video: {self.cam_width}x{self.cam_height} @ {self.video_fps:.2f} FPS")

        # Initialize HPD2 counter
        fps_for_hpd2 = self.video_fps if self.video_fps > 0 else args.camera_fps
        peak_hold = not getattr(args, "no_peak_hold", False)
        self.people_counter = HPD2PeopleCounter(
            image_width=self.cam_width,
            image_height=self.cam_height,
            fps=fps_for_hpd2,
            threshold=args.hpd2_threshold,
            peak_hold=peak_hold,
        )

        # Motion detection (C++ only)
        self.motion_triggered = getattr(args, "motion_triggered", False)
        self.motion_engine_cpp = None
        self.motion_wnd_cnt = 0
        if self.motion_triggered:
            cell = getattr(args, "motion_cell_size", 25)
            self.motion_engine_cpp = cpp_detect.MotionEngine(
                self.cam_width, self.cam_height, cell, cell, 5
            )
            self.motion_wnd_cnt = self.motion_engine_cpp.get_window_count()
            print(f"Motion detection: {self.motion_wnd_cnt} windows (C++)")
            print("  [Motion] Status column shows MOT:<n> = active motion windows; people count updates only when motion > 0.")
            self._motion_first_frame_logged = False
        else:
            self._motion_first_frame_logged = True  # no-op, won't log

        # Stats
        self.frame_count = 0
        self.start_time = time()
        self.last_print_time = time()

        # Results storage
        self.results = []

        # Visualize
        self.visualize = getattr(args, "visualize", False)
        if self.visualize:
            self.output_path = getattr(args, "output_path", "./video_output")
            os.makedirs(self.output_path, exist_ok=True)
            print(f"Visualize: saving annotated frames to {self.output_path}")

    def detections_to_windows(self, detections):
        """Convert YOLO detections to ResultWindow objects using C++ helpers"""
        if len(detections) == 0:
            return []

        if not isinstance(detections, np.ndarray):
            detections = np.asarray(detections, dtype=np.float32)

        xs, ys, ws, hs, vals = cpp_detect.detections_to_arrays(detections, int(self.cam_height))

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        ws = np.asarray(ws)
        hs = np.asarray(hs)
        vals = np.asarray(vals)

        # Adaptive HPD2 thresholding (C++)
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

        # Custom suppression (C++)
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

        windows: List[ResultWindow] = []
        for x, y, w, h, v in zip(xs.tolist(), ys.tolist(), ws.tolist(), hs.tolist(), vals.tolist()):
            windows.append(ResultWindow(int(x), int(y), int(w), int(h), int(v), self.cam_height))

        return windows

    def run_pipeline(self):
        """
        Pipeline processing the entire video file.
        """
        print(f"\n{'='*60}")
        print(f"Starting people counting on video")
        print(f"Resolution: {self.cam_width}x{self.cam_height}")
        print(f"Nominal FPS (from file): {self.video_fps:.2f}")
        if self.visualize:
            print(f"Visualize: saving annotated frames to {self.output_path}")
        print(f"HPD2 Threshold: {self.args.hpd2_threshold}")
        max_rt = "unlimited" if self.args.max_duration <= 0 else f"{self.args.max_duration}s"
        print(f"Max runtime: {max_rt}")
        if self.args.max_frames > 0:
            print(f"Max frames: {self.args.max_frames}")
        print(f"{'='*60}\n")

        print("FRAME | TIME(s) | FPS   | RAW | PROC | PEOPLE | STATUS")
        print("-" * 60)

        try:
            # Prime pipeline with first frame
            ret, current_frame = self.cap.read()
            if not ret:
                raise RuntimeError("Cannot read first frame from video")

            prev_frame = None
            last_people_count = 0
            last_confirmed = 0
            nominal_fps = self.video_fps if self.video_fps > 0 else self.args.camera_fps
            frame_time_ms = int(1000.0 / nominal_fps) if nominal_fps > 0 else 33

            self.detector_driver.preproc_and_write_accel([current_frame])
            iteration = 0

            while True:
                iteration += 1
                loop_start = time()
                elapsed = time() - self.start_time

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

                # Check max duration (wall-clock) only if positive
                if self.args.max_duration > 0 and elapsed >= self.args.max_duration:
                    print(f"\nReached max duration ({self.args.max_duration}s)")
                    break

                # Start FPGA execution (async)
                if iteration > 0:
                    self.accel.execute_on_buffers(asynch=True)

                # Read next frame from video
                ret, current_frame = self.cap.read()
                if not ret:
                    print("\nEnd of video stream")
                    break

                self.frame_count += 1

                # Max frames limit
                if self.args.max_frames > 0 and self.frame_count > self.args.max_frames:
                    print(f"\nReached max_frames limit ({self.args.max_frames})")
                    break

                # Preprocess next frame while FPGA runs
                if iteration > 0:
                    self.detector_driver.preproc_and_write_accel([current_frame])

                # Wait for FPGA, get results
                if iteration > 0:
                    self.accel.wait_until_finished()

                    # Copy output
                    for o in range(self.io_shape_dict["num_outputs"]):
                        self.accel.copy_output_data_from_device(
                            self.accel.obuf_packed[o], ind=o
                        )

                    # Get detections
                    batch_detections = self.detector_driver.read_accel_and_postprocess()
                    detections = batch_detections[0]

                    # Convert to windows
                    windows = self.detections_to_windows(detections)
                    raw_count = len(windows)

                    # HPD2 processing is applied inside `detections_to_windows` (C++ accelerated)

                    proc_count = len(windows)

                    # Always update counter state; when motion-triggered, only show new count if motion is active
                    people_count, confirmed = self.people_counter.calculate_people_count(
                        windows
                    )
                    if self.motion_triggered and active_motion_wnd_cnt == 0:
                        people_count = last_people_count
                        confirmed = last_confirmed
                    else:
                        last_people_count = people_count
                        last_confirmed = confirmed

                    # Calculate performance
                    throughput_fps = self.frame_count / elapsed
                    loop_time = time() - loop_start
                    instant_fps = 1.0 / loop_time if loop_time > 0 else 0

                    # Store result
                    self.results.append(
                        {
                            "frame": self.frame_count,
                            "time": elapsed,
                            "fps": throughput_fps,
                            "instant_fps": instant_fps,
                            "raw": raw_count,
                            "processed": proc_count,
                            "people": people_count,
                            "confirmed": confirmed,
                        }
                    )

                    # Visualize
                    if self.visualize and len(detections) > 0:
                        # Detections are in 416x416 letterbox space; scale to original frame size
                        boxes_detected = detections[:, :4].copy()
                        scale_coords(
                            (416, 416),
                            boxes_detected,
                            (self.cam_height, self.cam_width),
                        )
                        probs_detected = detections[:, 4]
                        class_names_list = ["person"]  # single class
                        class_names_detected = [class_names_list[int(c)] for c in detections[:, 5]]
                        image_boxes = visualize_boxes(
                            current_frame.copy(),
                            boxes_detected,
                            class_names_detected,
                            probs_detected,
                        )
                        # Draw people count on image (top-left panel)
                        count_text = f"People: {people_count}  |  Det: {proc_count}"
                        (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        cv2.rectangle(image_boxes, (0, 0), (tw + 20, th + 24), (0, 255, 0), -1)
                        cv2.rectangle(image_boxes, (0, 0), (tw + 20, th + 24), (0, 255, 0), 2)
                        cv2.putText(
                            image_boxes, count_text, (10, th + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                        )
                        output_filename = os.path.join(
                            self.output_path, f"frame_{self.frame_count:05d}.jpg"
                        )
                        cv2.imwrite(output_filename, image_boxes)
                        print(f"  Saved: {output_filename}")

                    # Print status (every N frames or every second)
                    time_since_print = time() - self.last_print_time
                    if (
                        self.frame_count % self.args.print_every == 0
                        or time_since_print >= 1.0
                    ):
                        status = "PRESENT" if people_count > 0 else "EMPTY"
                        mot = f" | MOT:{active_motion_wnd_cnt}" if self.motion_triggered else ""
                        print(
                            f"{self.frame_count:5d} | {elapsed:7.1f} | {throughput_fps:5.1f} | "
                            f"{raw_count:3d} | {proc_count:4d} | {people_count:6d} | {status}{mot}"
                        )
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
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total runtime: {total_runtime:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")

        if self.frame_count > 0:
            print(f"Time per frame: {total_runtime/self.frame_count*1000:.1f}ms")

        # People counting stats
        if self.results:
            people_counts = [r["people"] for r in self.results]
            max_people = max(people_counts)
            avg_people = np.mean(people_counts)
            frames_with_people = sum(1 for c in people_counts if c > 0)

            print("\nPEOPLE COUNTING:")
            print(f"Max people: {max_people}")
            print(f"Average: {avg_people:.2f}")
            print(f"Frames with people: {frames_with_people}/{len(self.results)}")
            print(
                f"Percentage: {frames_with_people/len(self.results)*100:.1f}%"
            )

            # Show distribution
            if max_people > 0:
                print("\nDISTRIBUTION:")
                for i in range(max_people + 1):
                    count = sum(1 for c in people_counts if c == i)
                    if count > 0:
                        print(
                            f"  {i} people: {count} frames ({count/len(self.results)*100:.1f}%)"
                        )

        print(f"{'='*60}")

        self.cap.release()


def main():
    parser = argparse.ArgumentParser(
        description="Offline people counter on video using FINN accelerator + HPD2"
    )

    # FPGA configuration
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--platform", type=str, default="zynq-iodma")
    parser.add_argument("--bitfile", type=str, default="resizer.bit")
    parser.add_argument("--runtime_weight_dir", type=str, default="runtime_weights/")
    parser.add_argument("--scale_path", type=str, default="scale.npy")

    # Video configuration
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--camera_fps",
        type=int,
        default=30,
        help="Fallback FPS for HPD2 if video metadata is missing",
    )

    # Detection configuration
    parser.add_argument("--conf_thres", type=float, default=0.15)
    parser.add_argument("--iou_thres", type=float, default=0.45)
    parser.add_argument("--num_classes", type=int, default=1)

    # HPD2 configuration
    parser.add_argument("--hpd2_threshold", type=float, default=0.5)
    parser.add_argument("--hpd2_adaptive_threshold", action="store_true")
    parser.add_argument("--hpd2_nms", action="store_true")

    # Output / runtime configuration
    parser.add_argument(
        "--print_every",
        type=int,
        default=30,
        help="Print status every N frames",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=-1.0,
        help="Maximum runtime in seconds (wall-clock). <= 0 means unlimited (run full video).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Maximum number of frames to process (-1 = all frames)",
    )
    parser.add_argument(
        "--motion_triggered",
        action="store_true",
        help="Only update people count when motion is detected in the scene",
    )
    parser.add_argument(
        "--motion_cell_size",
        type=int,
        default=25,
        help="Motion grid cell size (default 25)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Draw detection boxes and save annotated frames to output path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./video_output",
        help="Directory to save annotated frames when --visualize is set (default: ./video_output)",
    )
    parser.add_argument(
        "--no_peak_hold",
        action="store_true",
        help="Let people count decrease when detections drop (default: peak-hold until scene empty)",
    )

    args = parser.parse_args()

    counter = PeopleCounterVideo(args)
    counter.run_pipeline()


if __name__ == "__main__":
    main()

