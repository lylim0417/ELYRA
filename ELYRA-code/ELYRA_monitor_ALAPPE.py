import datetime
import os
import time

from screeninfo import get_monitors
import cv2
import torch.cuda
from ultralytics import YOLO
import pandas as pd
import pynvml
import numpy as np
import psutil


class AIoTCam:
    def __init__(self):
        # Initialize the device (GPU if available, otherwise CPU)
        self.device = self.assign_device()

        # Load YOLOv8 Pose model with ByteTrack support
        #self.pose_model = self.load_yolo_model('Yolo-Weights/yolo11n-pose.pt')  # REPLACEMENT for person detection
        self.pose_model = self.load_yolo_model('/home/lylim/AIoTCam/Yolo-Weights/yolo11n-pose_ncnn_model')  # REPLACEMENT for person detection

        # Load your custom detection model for compliance items
        #self.custom_model = self.load_yolo_model('model-5000/5000-11s/weights/best.pt')
        self.custom_model = self.load_yolo_model('/home/lylim/AIoTCam/model-5000/5000-11s/weights/best_ncnn_model')

        # Class names for your custom model
        self.class_name = ['Compliance-Shirt', 'Compliance-Pants', 'Compliance-Shoes',
                           'Non-Compliance-Shirt', 'Non-Compliance-Pants', 'Non-Compliance-Shoes']

        # Class IDs that are considered compliant and should be excluded from alerts
        self.compliance_class = [0, 1, 2]

        # Open the camera for capturing video frames
        self.capture = self.open_cam()

        # Frame size setup
        '''self.frame_size = (1920, 1080)
        self.screen = get_monitors()[0]
        self.screen_width = self.screen.width
        self.screen_height = self.screen.height'''
        self.frame_size = (int(get_monitors()[0].width / 2), get_monitors()[0].height)
        self.fps = 0
        self.frame_count = 1
        self.skip_number = 10

        # Enable/disable video recording
        self.record = True
        self.main_recording_path = '/home/lylim/AIoTCam/Output'
        self.main_logging_path = '/home/lylim/AIoTCam/Log'

        # Initialize tracking & recording helpers
        self.output = {}
        self.recording = {}
        self.frames_missing = {}
        self.max_missing_frames = 10
        self.recorded_person_id_list = []
        self.recorded_path = {}

        # Initialize non-compliance item tracking
        self.non_compliance_tracking = {}
        self.non_compliance_consecutive_frame = 3

        # Detection results
        self.roi_item_result = None

        # Speed measurement lists
        self.fps_list = []
        self.item_preprocess_list = []
        self.item_inference_list = []
        self.item_postprocess_list = []
        self.person_preprocess_list = []
        self.person_inference_list = []
        self.person_postprocess_list = []

        # GPU / CPU monitoring
        '''self.gpu_handle = self.init_gpu_monitor()
        self.gpu_util_list = []
        self.mem_usage_list = []'''
        self.cpu_percent_list = []

        # Detected item
        self.number_of_person_detected_list = []
        self.number_of_compliance_shirt_detected_list = []
        self.number_of_compliance_pants_detected_list = []
        self.number_of_compliance_shoes_detected_list = []
        self.number_of_non_compliance_shirt_detected_list = []
        self.number_of_non_compliance_pants_detected_list = []
        self.number_of_non_compliance_shoes_detected_list = []

        # Confidence tracking
        self.compliance_shirt_conf_list = []
        self.compliance_pants_conf_list = []
        self.compliance_shoes_conf_list = []
        self.non_compliance_shirt_conf_list = []
        self.non_compliance_pants_conf_list = []
        self.non_compliance_shoes_conf_list = []

    def main(self):
        """
        Capture video frames, perform full-body detection using pose + ByteTrack,
        run item detection for non-compliance, and handle recording.
        """
        while self.capture.isOpened():
            start_time = time.time()
            success, frame = self.capture.read()
            if not success:
                print('Error in reading frame')
                break

            frame, self.frame_count = self.skip_frame(frame, self.frame_count, self.skip_number)

            if frame is None:
                continue

            # frame = self.resize_frame(frame, self.frame_size)
            frame = self.enhance_image_opencv(frame)

            # -------- Pose detection and tracking (ByteTrack) --------
            pose_result = self.pose_model.track(
                source=frame,
                persist=True,
                conf=0.8,
                tracker="bytetrack.yaml",
                device=self.device
            )
            pose_frame = pose_result[0]

            # -------- Filter full-body persons --------
            person_tracker_list = []

            # Check if detections exist and attributes are not None
            if (pose_frame.boxes is not None and len(pose_frame.boxes) > 0
                    and pose_frame.boxes.id is not None and pose_frame.keypoints is not None):
                try:
                    for box, track_id, kps in zip(
                            pose_frame.boxes.xyxy.cpu().numpy(),
                            pose_frame.boxes.id.int().cpu().numpy(),
                            pose_frame.keypoints.data.cpu().numpy()):

                        if self.is_full_body(kps):
                            x1, y1, x2, y2 = box
                            person_tracker_list.append([x1, y1, x2, y2, int(track_id)])

                            # Draw keypoints with confidence coloring
                            '''for i, (x, y, conf) in enumerate(kps):
                                color = (0, 255, 0) if conf > 0.3 else (0, 0, 255)
                                cv2.circle(frame, (int(x), int(y)), 3, color, -1)'''

                except Exception as e:
                    print(f"[WARN] Pose format error: {e}")

            else:
                # No detections: skip processing this frame gracefully
                pass
            
            number_of_person_detected = 0
            number_of_compliance_shirt_detected = 0
            number_of_compliance_pants_detected = 0
            number_of_compliance_shoes_detected = 0
            number_of_non_compliance_shirt_detected = 0
            number_of_non_compliance_pants_detected = 0
            number_of_non_compliance_shoes_detected = 0

            person_roi_item_result = {}
            self.roi_item_result = None

            frame_item_preprocess_speeds = []
            frame_item_inference_speeds = []
            frame_item_postprocess_speeds = []

            # -------- Loop over tracked full-body persons --------
            for person_data in person_tracker_list:
                number_of_person_detected += 1
                x1, y1, x2, y2, track_id = person_data
                height, width = frame.shape[:2]
                y1 = max(0, min(height, int(y1)))
                y2 = max(0, min(height, int(y2)))
                x1 = max(0, min(width, int(x1)))
                x2 = max(0, min(width, int(x2)))
                person_roi = frame[y1:y2, x1:x2]
                #person_roi = self.enhance_image_opencv(person_roi)

                roi_result = self.custom_model(person_roi, conf=0.8, device=self.device)
                person_roi_item_result_single = roi_result[0]
                roi_item_detector_list = self.process_detect_results(person_roi_item_result_single)

                for roi_item_detector in roi_item_detector_list:
                    class_id = roi_item_detector[5]
                    conf = roi_item_detector[4]

                    if class_id == 0:
                        number_of_compliance_shirt_detected += 1
                        self.compliance_shirt_conf_list.append(conf)
                    elif class_id == 1:
                        number_of_compliance_pants_detected += 1
                        self.compliance_pants_conf_list.append(conf)
                    elif class_id == 2:
                        number_of_compliance_shoes_detected += 1
                        self.compliance_shoes_conf_list.append(conf)
                    elif class_id == 3:
                        number_of_non_compliance_shirt_detected += 1
                        self.non_compliance_shirt_conf_list.append(conf)
                    elif class_id == 4:
                        number_of_non_compliance_pants_detected += 1
                        self.non_compliance_pants_conf_list.append(conf)
                    elif class_id == 5:
                        number_of_non_compliance_shoes_detected += 1
                        self.non_compliance_shoes_conf_list.append(conf)


                roi_item_detector_list_cleaned = [{'bbox': item[:4], 'conf': item[4], 'class_id': item[-1]} for item in roi_item_detector_list]
                person_roi_item_result[track_id] = roi_item_detector_list_cleaned

                if person_roi_item_result_single and person_roi_item_result_single.speed:
                    frame_item_preprocess_speeds.append(person_roi_item_result_single.speed.get('preprocess', 0.0))
                    frame_item_inference_speeds.append(person_roi_item_result_single.speed.get('inference', 0.0))
                    frame_item_postprocess_speeds.append(person_roi_item_result_single.speed.get('postprocess', 0.0))

            self.number_of_person_detected_list.append(number_of_person_detected)
            self.number_of_compliance_shirt_detected_list.append(number_of_compliance_shirt_detected)
            self.number_of_compliance_pants_detected_list.append(number_of_compliance_pants_detected)
            self.number_of_compliance_shoes_detected_list.append(number_of_compliance_shoes_detected)
            self.number_of_non_compliance_shirt_detected_list.append(number_of_non_compliance_shirt_detected)
            self.number_of_non_compliance_pants_detected_list.append(number_of_non_compliance_pants_detected)
            self.number_of_non_compliance_shoes_detected_list.append(number_of_non_compliance_shoes_detected)
            
            # -------- Draw detections --------
            self.draw_detections(frame, person_tracker_list, person_roi_item_result, self.compliance_class)

            self.person_preprocess_list.append(pose_frame.speed.get('preprocess', 0.0))
            self.person_inference_list.append(pose_frame.speed.get('inference', 0.0))
            self.person_postprocess_list.append(pose_frame.speed.get('postprocess', 0.0))

            if frame_item_preprocess_speeds:
                avg_pre = sum(frame_item_preprocess_speeds) / len(frame_item_preprocess_speeds)
                avg_inf = sum(frame_item_inference_speeds) / len(frame_item_inference_speeds)
                avg_post = sum(frame_item_postprocess_speeds) / len(frame_item_postprocess_speeds)
            else:
                avg_pre = avg_inf = avg_post = 0.0

            self.item_preprocess_list.append(avg_pre)
            self.item_inference_list.append(avg_inf)
            self.item_postprocess_list.append(avg_post)

            # -------- FPS + GPU / CPU --------
            end_time = time.time()
            self.fps = self.calculate_fps(start_time, end_time)
            self.fps_list.append(self.fps)
            self.draw_fps(frame, self.fps)
            '''gpu_util, mem_usage = self.get_gpu_info(self.gpu_handle)
            self.gpu_util_list.append(gpu_util)
            self.mem_usage_list.append(mem_usage)'''
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_percent_list.append(cpu_percent)

            # -------- Show Frame --------
            frame = self.resize_frame(frame, self.frame_size)
            self.show_frame(frame, True)

            # -------- Compliance Logic --------
            non_compliance_item_in_person = self.remove_compliance_item(person_roi_item_result, self.compliance_class)
            if self.record:
                self.handle_recording(frame, person_tracker_list, non_compliance_item_in_person)

            if self.close_cam():
                self.stop_all_recordings()
                break

            self.frame_count += 1

        self.exit_cam(self.capture)

        # -------- Logging --------
        item_avg_pre = self.get_average_inference_speed(self.item_preprocess_list[1:])
        item_avg_inf = self.get_average_inference_speed(self.item_inference_list[1:])
        item_avg_post = self.get_average_inference_speed(self.item_postprocess_list[1:])
        person_avg_pre = self.get_average_inference_speed(self.person_preprocess_list[1:])
        person_avg_inf = self.get_average_inference_speed(self.person_inference_list[1:])
        person_avg_post = self.get_average_inference_speed(self.person_postprocess_list[1:])
        avg_fps = self.get_average_fps(self.fps_list[1:])

        sum_person = sum(self.number_of_person_detected_list[1:])
        print(f'sum_person: {sum_person}')
        sum_compliance_shirt = sum(self.number_of_compliance_shirt_detected_list[1:])
        print(f'sum_compliance_shirt: {sum_compliance_shirt}')
        sum_compliance_pants = sum(self.number_of_compliance_pants_detected_list[1:])
        print(f'sum_compliance_pants: {sum_compliance_pants}')
        sum_compliance_shoes = sum(self.number_of_compliance_shoes_detected_list[1:])
        print(f'sum_compliance_shoes: {sum_compliance_shoes}')
        sum_non_compliance_shirt = sum(self.number_of_non_compliance_shirt_detected_list[1:])
        print(f'sum_non_compliance_shirt: {sum_non_compliance_shirt}')
        sum_non_compliance_pants = sum(self.number_of_non_compliance_pants_detected_list[1:])
        print(f'sum_non_compliance_pants: {sum_non_compliance_pants}')
        sum_non_compliance_shoes = sum(self.number_of_non_compliance_shoes_detected_list[1:])
        print(f'sum_non_compliance_shoes: {sum_non_compliance_shoes}')
        print(f'avg_compliance_shirt_conf: {self.get_average_inference_speed(self.compliance_shirt_conf_list)}')
        print(f'avg_compliance_pants_conf: {self.get_average_inference_speed(self.compliance_pants_conf_list)}')
        print(f'avg_compliance_shoes_conf: {self.get_average_inference_speed(self.compliance_shoes_conf_list)}')
        print(f'avg_non_compliance_shirt_conf: {self.get_average_inference_speed(self.non_compliance_shirt_conf_list)}')
        print(f'avg_non_compliance_pants_conf: {self.get_average_inference_speed(self.non_compliance_pants_conf_list)}')
        print(f'avg_non_compliance_shoes_conf: {self.get_average_inference_speed(self.non_compliance_shoes_conf_list)}')

        print(f'item_average_preprocess_speed: {item_avg_pre:.2f}')
        print(f'item_average_inference_speed: {item_avg_inf:.2f}')
        print(f'item_average_postprocess_speed: {item_avg_post:.2f}')
        print(f'person_average_preprocess_speed: {person_avg_pre:.2f}')
        print(f'person_average_inference_speed: {person_avg_inf:.2f}')
        print(f'person_average_postprocess_speed: {person_avg_post:.2f}')
        print(f'average_fps: {avg_fps:.2f}')
        print('------------------')
        print(len(self.person_inference_list))
        print(len(self.item_inference_list))
        print('------------------')

        df = pd.DataFrame({
            'frame': list(range(1, len(self.fps_list))),
            'fps_roi': self.fps_list[1:],
            'person_inference': self.person_inference_list[1:],
            'item_inference': self.item_inference_list[1:],
            'cpu_percent': self.cpu_percent_list[1:],
            'person_detected': self.number_of_person_detected_list[1:],
            'compliance_shirt_detected': self.number_of_compliance_shirt_detected_list[1:],
            'compliance_pants_detected': self.number_of_compliance_pants_detected_list[1:],
            'compliance_shoes_detected': self.number_of_compliance_shoes_detected_list[1:],
            'non_compliance_shirt_detected': self.number_of_non_compliance_shirt_detected_list[1:],
            'non_compliance_pants_detected': self.number_of_non_compliance_pants_detected_list[1:],
            'non_compliance_shoes_detected': self.number_of_non_compliance_shoes_detected_list[1:],
        })
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        df.to_csv(f'{self.main_logging_path}/roi_performance_log_{timestamp}.csv', index=False)


    def process_tracking_results(self, tracking_results):
        """
        Process YOLO tracking results to extract information about detected persons.

        Args:
            tracking_results (Results): List of tracked persons, each containing bounding
                                        box coordinates, tracking ID, confidence score,
                                        and class ID.

        Returns:
            person_tracker_list (list): List of detected persons with their bounding box
                                        coordinates and tracking IDs.
        """
        person_tracker_list = []

        # Loop through each tracked object in the results
        for tracked_person in tracking_results.boxes.data.tolist():
            x1, y1, x2, y2, track_id, conf, class_id = [-1, -1, -1, -1, -1, -1, -1]

            # Handle different cases where tracking data may have 6 or 7 elements
            if len(tracked_person) == 7:
                x1, y1, x2, y2, track_id, conf, class_id = tracked_person
            elif len(tracked_person) == 6:
                x1, y1, x2, y2, track_id, class_id = tracked_person

            person_tracker = [x1, y1, x2, y2, int(track_id)]
            person_tracker_list.append(person_tracker)
            self.frames_missing.setdefault(track_id, 0)

        print(f'person_tracker_list: {person_tracker_list}')

        return person_tracker_list

    def process_detect_results(self, detect_results):
        """
        Process YOLO detect results to extract information about detected items.

        Args:
            detect_results (Results): List of detected objects, each containing bounding
                                      box coordinates, confidence score, and class ID.

        Returns:
            item_detector_list (list): List of detected persons with their bounding box
                                       coordinates, confidence score, and class ID.
        """
        item_detector_list = []

        # Loop through each detected object in the results
        for detect_object in detect_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = detect_object

            item_detector = [x1, y1, x2, y2, round(conf, 2), int(class_id)]
            item_detector_list.append(item_detector)

        print(f'item_detector_list: {item_detector_list}')

        return item_detector_list

    def draw_detections(self, frame, person_tracker_list, person_roi_item_result, compliance_class):
        """
        Draw bounding boxes and labels on the video frame for detected persons and items.

        Args:
            frame (numpy.ndarray): The current video frame.
            person_tracker_list (list): List of detected persons with their bounding box coordinates
                                        and tracking IDs.
            person_roi_item_result (dict): List of detected items in person ROI.
        """
        # Draw corner boxes for each detected person
        for person_tracker in person_tracker_list:
            person_bbox = person_tracker[:4]
            track_id = person_tracker[4]
            x1, y1, x2, y2 = map(int, person_bbox)

            border_color = (255, 0, 0)
            text_color = (255, 255, 255)
            label = f"ID: {track_id}"
            self.draw_dynamic_label_box(
                frame,
                person_bbox,
                text=label,
                border_color=border_color,
                text_color=text_color,
                thickness=5
            )

            # Draw items only for this person
            if track_id in person_roi_item_result:
                for item in person_roi_item_result[track_id]:
                    dx1, dy1, dx2, dy2 = map(int, item['bbox'])
                    item_id = item['class_id']
                    conf = item['conf']

                    # Convert ROI-relative coordinates to full-frame coordinates
                    full_x1 = dx1 + x1
                    full_y1 = dy1 + y1
                    full_x2 = dx2 + x1
                    full_y2 = dy2 + y1

                    item_bbox = [full_x1, full_y1, full_x2, full_y2]
                    item_label = self.class_name[item_id]
                    conf_text = f"{item_label} ({conf:.2f})"

                    if item_id in compliance_class:     # compliance
                        border_color = (0, 255, 0)
                        text_color = (0, 0, 0)
                    else:                               # non-compliance
                        border_color = (0, 0, 255)
                        text_color = (255, 255, 255)

                    self.draw_dynamic_label_box(
                        frame,
                        item_bbox,
                        text=conf_text,
                        border_color=border_color,
                        text_color=text_color
                    )

    def handle_recording(self, frame, person_tracker_list, person_roi_item_result):
        """
        Handle the logic for starting, updating, and stopping video recordings of detected persons.

        Args:
            frame (numpy.ndarray): The current video frame.
            person_tracker_list (list): List of detected persons with their bounding box coordinates
                                        and tracking IDs.
            person_roi_item_result (dict): A dictionary mapping person IDs to detected items in their vicinity.
        """
        # Get the list of current person IDs detected in the frame
        current_person_id_list = [person_tracker[4] for person_tracker in person_tracker_list]
        print(f'current_person_id_list: {current_person_id_list}')
        print(f'recorded_person_id_list: {self.recorded_person_id_list}')

        # Loop through detected persons and their items
        for person_id, items in person_roi_item_result.items():
            if items:
                # Initialize tracking dictionary for the person if not already initialized
                if person_id not in self.non_compliance_tracking:
                    self.non_compliance_tracking[person_id] = {}

                # Loop over each detected item for the person
                for item in items:
                    item_id = item['class_id']
                    # Check if it's a non-compliant item
                    if item_id not in self.compliance_class:  # Non-compliant item
                        
                        # Track consecutive frames of non-compliance detection
                        if item_id in self.non_compliance_tracking[person_id]:
                            self.non_compliance_tracking[person_id][item_id] += 1
                        else:
                            self.non_compliance_tracking[person_id][item_id] = 1

                        # If the item has been detected for more than 3 consecutive frames, start recording
                        if self.non_compliance_tracking[person_id][item_id] > self.non_compliance_consecutive_frame:
                            if not self.recording.get(person_id, False):  # Start recording if not already recording
                                self.start_recording_for_person(person_id)
                                print(f"Started recording for person ID: {person_id} due to non-compliance item {item_id} for {self.non_compliance_tracking[person_id][item_id]} frame")

                    else:
                        # Reset counter for compliant items
                        self.non_compliance_tracking[person_id][item_id] = 0

                # Check for items no longer detected (reset counter)
                detected_item_ids = {item['class_id'] for item in items}
                for tracked_item_id in list(self.non_compliance_tracking[person_id].keys()):
                    if tracked_item_id not in detected_item_ids:
                        self.non_compliance_tracking[person_id][tracked_item_id] = 0

        # Update recording status for each recorded person
        for recorded_person_id in self.recorded_person_id_list:
            self.update_recording_status(recorded_person_id, frame, current_person_id_list)

    def start_recording_for_person(self, person_id):
        """
        Start video recording for a detected person.

        Args:
            person_id (int): The tracking ID of the person for whom recording is being started.
        """
        self.output[person_id] = self.start_recording(self.frame_size, self.fps, self.main_recording_path, person_id=person_id)
        self.recording[person_id] = True
        self.recorded_person_id_list.append(person_id)
        print(f"Started recording for person ID: {person_id}")
        self.frames_missing[person_id] = 0  # Reset the missing frames counter

    def update_recording_status(self, recorded_person_id, frame, current_person_id_list):
        """
        Update the recording status of a tracked person, including stopping the recording if the
        person is no longer detected.

        Args:
            recorded_person_id (int): The tracking ID of the person currently being recorded.
            frame (numpy.ndarray): The current video frame.
            current_person_id_list (list): List of person IDs detected in the current frame.
        """
        if self.recording.get(recorded_person_id, False):
            # Reset missing frames count if person is detected
            if recorded_person_id in current_person_id_list:
                self.frames_missing[recorded_person_id] = 0
            # Increment missing frames count if person is not detected
            else:
                self.frames_missing[recorded_person_id] += 1

            # Continue writing frames to the video file
            self.output[recorded_person_id].write(frame)

            # Stop recording if the person is not detected for too many frames
            if self.frames_missing.get(recorded_person_id, 0) > self.max_missing_frames:
                self.stop_recording_for_person(recorded_person_id)

    def stop_recording_for_person(self, person_id):
        """
        Stop video recording for a given person.

        Args:
            person_id (int): The tracking ID of the person for whom recording is being stopped.
        """
        self.output[person_id] = self.stop_recording(self.output[person_id])
        print(f"Stopped recording for person ID: {person_id}")
        self.finalize_recording_filename(self.recorded_path.get(person_id))
        self.recording[person_id] = False
        self.recorded_person_id_list.remove(person_id)  # Remove the person from the recorded list
        if person_id in self.recorded_path:
            del self.recorded_path[person_id]

    def stop_all_recordings(self):
        """
        Stop all ongoing video recordings.
        """
        for person_id in list(self.recording.keys()):
            if self.recording[person_id]:
                self.output[person_id] = self.stop_recording(self.output[person_id])
                print(f"Stopped recording for person ID: {person_id}")
                self.finalize_recording_filename(self.recorded_path.get(person_id))
                if person_id in self.recorded_path:
                    del self.recorded_path[person_id]

    @staticmethod
    def assign_device():
        """
        Assign the device for running the YOLO model, prioritizing GPU if available.

        Returns:
            torch.device: The device (GPU or CPU) to be used for running the model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        return device

    @staticmethod
    def load_yolo_model(yolo_weight):
        """
        Load the YOLO model with the specified weights.

        Args:
            yolo_weight (str): Path to the YOLO weight file.

        Returns:
            YOLO: The loaded YOLO model.
        """
        model = YOLO(yolo_weight)
        return model

    @staticmethod
    def open_cam():
        """
        Open the default camera for video capture.

        Returns:
            cv2.VideoCapture: The video capture object for the default camera.
        """
        capture = cv2.VideoCapture(0)
        #cv2.namedWindow("AIoTCam", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("AIoTCam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return capture

    @staticmethod
    def get_frame_size(capture):
        """
        Retrieve the size of the video frames from the capture device.

        Args:
            capture (cv2.VideoCapture): The video capture object.

        Returns:
            tuple: The width and height of the video frames.
        """
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        return frame_size

    @staticmethod
    def resize_frame(frame, frame_size):
        frame = cv2.resize(frame, frame_size)
        # frame = cv2.WINDOW_FULLSCREEN
        return frame

    @staticmethod
    def skip_frame(frame, frame_count, skip_number):
        frame_count += 1
        if frame_count % skip_number == 0:
            return frame, frame_count
        else:
            return None, frame_count

    @staticmethod
    def get_fps(capture):
        """
        Retrieve the frames per second (FPS) of the video capture device.

        Args:
            capture (cv2.VideoCapture): The video capture object.

        Returns:
            int: The FPS of the video capture device.
        """
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        return fps

    @staticmethod
    def calculate_fps(start_time, end_time):
        return 1.0 / (end_time - start_time)

    @staticmethod
    def draw_fps(frame, fps):
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    @staticmethod
    def get_average_fps(fps_list):
        """
        Calculate average FPS from a list of frame rates.

        Args:
            fps_list (list): List of FPS values per frame.

        Returns:
            float: Average FPS across the video capture.
        """
        if not fps_list:
            return 0.0
        return sum(fps_list) / len(fps_list)

    @staticmethod
    def show_frame(frame, show):
        """
        Show the frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            show (bool): Show frame or not.
        """
        if show:
            cv2.imshow('AIoTCam', frame)

    @staticmethod
    def is_full_body(kps, required_ids=None, conf_threshold=0.3):
        if required_ids is None:
            required_ids = [0, 5, 6, 11, 12, 13, 14, 15, 16]  # keypoints from head to ankle
        for idx in required_ids:
            if idx >= len(kps):
                return False
            x,y, conf = kps[idx]
            if conf < conf_threshold:
                return False
        return True

    @staticmethod
    def get_item_in_person_id(item, person_tracker_list, threshold=10):
        """
        Map an item to the closest person by comparing bounding boxes.

        Args:
            item (list): The bounding box coordinates of the item.
            person_tracker_list (list): List of detected persons with their bounding box
                                        coordinates and tracking IDs.
            threshold (int): Threshold value to expand the person's bounding box, allowing
                             for a larger area to be considered.

        Returns:
            int: The tracking ID of the closest person to the item, or -1 if no person is close.
        """
        item_x1, item_y1, item_x2, item_y2, item_conf, item_class_id = item
        item_in_person_id = -1

        for person_tracker in person_tracker_list:
            person_x1, person_y1, person_x2, person_y2, person_id = person_tracker
            buffered_person_x1, buffered_person_y1 = person_x1 - threshold, person_y1 - threshold
            buffered_person_x2, buffered_person_y2 = person_x2 + threshold, person_y2 + threshold

            if (item_x1 > buffered_person_x1 and item_y1 > buffered_person_y1 and
                    item_x2 < buffered_person_x2 and item_y2 < buffered_person_y2):
                item_in_person_id = person_id
                break

        return item_in_person_id

    @staticmethod
    def remove_compliance_item(item_in_person, classes):
        """
        Remove compliance item in the item_in_person dictionary.

        Args:
            item_in_person (dict): A dictionary mapping person IDs to a dictionary of items detected in their vicinity.
                                   Each item is represented by its class ID and bounding box coordinates.
            classes (list): A list contain the classes that desire to remove.

        Returns:
            non_compliance_item_in_person (dict): A dictionary mapping person IDs to a dictionary of non-compliance
                                                  items detected in their vicinity. Each item is represented by its
                                                  class ID and bounding box coordinates.
        """
        non_compliance_item_in_person = item_in_person.copy()

        # Iterate through the dictionary
        persons_to_remove = []  # To store person_ids to be removed later

        for person_id, items in non_compliance_item_in_person.items():
            # Collect the item_ids that need to be removed
            keys_to_remove = [item_id for item_id in items if item_id in classes]

            # Remove the collected item_ids
            for item_id in keys_to_remove:
                items.pop(item_id)

            # After removing items, check if the items dictionary is empty
            if not items:  # If the dictionary is empty
                persons_to_remove.append(person_id)

        # Now, remove the empty person_ids
        for person_id in persons_to_remove:
            non_compliance_item_in_person.pop(person_id)

        return non_compliance_item_in_person

    def start_recording(self, frame_size, fps, main_folder_path, person_id=None):
        """
        Start recording a video file, creating a folder if necessary.

        Args:
            frame_size (tuple): The width and height of the video frames.
            fps (int): The frames per second for the recording.
            main_folder_path (str): Main directory of the recording.
            recorded_path (dict): File path of each recorded person id.
            person_id (int): The tracking ID of the person for whom recording is being started.

        Returns:
            cv2.VideoWriter: The video writer object for recording.
        """
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        now = datetime.datetime.now().strftime('%H-%M-%S')
        folder_path = f'{main_folder_path}/{date}'
        os.makedirs(folder_path, exist_ok=True)

        if person_id is not None:
            output_file_path = f'{folder_path}/temp_{now}_ID_{person_id}.mp4'
        else:
            output_file_path = f'{folder_path}/temp_{now}.mp4'

        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_file_path, fourcc, fps, frame_size)
        self.recorded_path[person_id] = output_file_path
        return output

    @staticmethod
    def stop_recording(output):
        """
        Stop recording and release the VideoWriter object.

        Args:
            output (cv2.VideoWriter): The video writer object to be released.

        Returns:
            cv2.VideoWriter: The released video writer object (now closed).
        """
        if output:
            output.release()
            output = None
        return output

    @staticmethod
    def finalize_recording_filename(temp_path):
        """
        Rename a temporary temp_ file to its final .mp4 filename

        Args:
            temp_path (str): The full path of the temporary temp_ file.
        """
        print("temp_path: ", temp_path)
        if temp_path and os.path.basename(temp_path).startswith('temp_'):
            final_path = temp_path.replace('temp_', '')
            try:
                os.rename(temp_path, final_path)
                print(f"[INFO] Renamed {temp_path} to {final_path}")
            except Exception as e:
                print(f"[ERROR] Failed to rename {temp_path}: {e}")

    @staticmethod
    def close_cam():
        """
        Check if the user has requested to close the camera by pressing the 'ESC' key.

        Returns:
            bool: True if 'ESC' key is pressed, False otherwise.
        """
        return cv2.waitKey(1) & 0xFF == 27

    @staticmethod
    def exit_cam(capture):
        """
        Release the camera capture device and close any OpenCV windows.

        Args:
            capture (cv2.VideoCapture): The video capture object to be released.
        """
        capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def enhance_image_opencv(img):
        """
        Apply a basic sharpening filter to enhance image clarity using OpenCV.
        Args:
            img (numpy.ndarray): Input BGR image.
        Returns:
            numpy.ndarray: Sharpened image.
        """
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def get_average_inference_speed(inference_list):
        if not inference_list:
            return 0.00
        avg = sum(inference_list) / len(inference_list)
        return avg

    @staticmethod
    def init_gpu_monitor():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # usually GPU 0
        return handle

    @staticmethod
    def get_gpu_info(handle):
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return util.gpu, mem.used / 1024 ** 2  # GPU%, Memory MB

    
    @staticmethod
    def draw_dynamic_label_box(frame, bbox, text, border_color=(255, 0, 0), text_color=(255, 255, 255), thickness=3):
        """
        Draw a dynamic label box with text for a detected item in the video frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            bbox (list): Bounding box coordinates of the item.
            text (str): The label text to display.
            border_color (tuple): The color of bounding box.
            text_color (tuple): The color of text.
            thickness (int): The color of bounding box.
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Calculate text size and background dimensions
        (text_width, text_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                              fontScale=0.8, thickness=1)
        '''(text_width, text_height), baseline = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                              fontScale=2.25, thickness=3)'''

        text_x1 = max(0, x1)
        text_y1 = max(0, y1 - text_height - baseline)
        text_x2 = text_x1 + text_width
        text_y2 = text_y1 + text_height + baseline

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=border_color, thickness=thickness)
        '''cv2.rectangle(frame, (x1, y1), (x2, y2), color=border_color, thickness=5)'''

        # Draw label background rectangle
        cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), color=border_color, thickness=-1)

        # Draw label text
        cv2.putText(frame, text, (text_x1, text_y2 - baseline), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    color=text_color, fontScale=0.75, thickness=1, lineType=cv2.LINE_AA)
        '''cv2.putText(frame, text, (text_x1, text_y2 - baseline), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    color=text_color, fontScale=2, thickness=3, lineType=cv2.LINE_AA)'''

    def run_on_image(self, image_path, show_result=True, save_path=None):
        """
        Run detection and tracking on a single image and display/save the result.

        Args:
            image_path (str): Path to the input image.
            show_result (bool): Whether to display the output image.
            save_path (str or None): If provided, save the result to this path.
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return

        # Read and resize the input image
        frame = cv2.imread(image_path)
        #frame = self.resize_frame(frame, self.frame_size)

        # Pose + tracking
        pose_result = self.pose_model.track(
            source=frame,
            persist=True,
            conf=0.5,
            tracker="bytetrack.yaml",
            device=self.device
        )
        pose_frame = pose_result[0]

        person_tracker_list = []

        if (pose_frame.boxes is not None and len(pose_frame.boxes) > 0
                and pose_frame.boxes.id is not None and pose_frame.keypoints is not None):
            try:
                for box, track_id, kps in zip(
                        pose_frame.boxes.xyxy.cpu().numpy(),
                        pose_frame.boxes.id.int().cpu().numpy(),
                        pose_frame.keypoints.xy.cpu().numpy()):

                    if self.is_full_body(kps):
                        x1, y1, x2, y2 = box
                        person_tracker_list.append([x1, y1, x2, y2, int(track_id)])

            except Exception as e:
                print(f"[WARN] Pose format error: {e}")

        person_roi_item_result = {}

        for person_data in person_tracker_list:
            x1, y1, x2, y2, track_id = person_data
            height, width = frame.shape[:2]
            y1 = max(0, min(height, int(y1)))
            y2 = max(0, min(height, int(y2)))
            x1 = max(0, min(width, int(x1)))
            x2 = max(0, min(width, int(x2)))
            person_roi = frame[y1:y2, x1:x2]
            person_roi = self.enhance_image_opencv(person_roi)

            roi_result = self.custom_model(person_roi, conf=0.5, device=self.device)
            person_roi_item_result_single = roi_result[0]
            roi_item_detector_list = self.process_detect_results(person_roi_item_result_single)

            roi_item_detector_list_cleaned = [{'bbox': item[:4], 'conf': item[4], 'class_id': item[-1]} for item in roi_item_detector_list]
            person_roi_item_result[track_id] = roi_item_detector_list_cleaned

        self.draw_detections(frame, person_tracker_list, person_roi_item_result, self.compliance_class)

        if show_result:
            cv2.imshow("AIoTCam Image Result", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, frame)

    @staticmethod
    def correct_orientation(frame):
        # Rotate 90 degrees clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    def run_on_video(self, video_path, show_result=True, save_path=None):
        """
        Run detection and tracking on a video file.

        Args:
            video_path (str): Path to the input video.
            show_result (bool): Whether to display the output video.
            save_path (str or None): If provided, save the output video to this path.
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 30, (width, height))
        else:
            out = None

        while cap.isOpened():
            ret, frame = cap.read()
            #frame = self.correct_orientation(frame)
            if not ret:
                break

            # Pose + tracking
            pose_result = self.pose_model.track(
                source=frame,
                persist=True,
                conf=0.5,
                tracker="bytetrack.yaml",
                device=self.device
            )
            pose_frame = pose_result[0]

            person_tracker_list = []
            if (pose_frame.boxes is not None and len(pose_frame.boxes) > 0
                    and pose_frame.boxes.id is not None and pose_frame.keypoints is not None):
                try:
                    for box, track_id, kps in zip(
                            pose_frame.boxes.xyxy.cpu().numpy(),
                            pose_frame.boxes.id.int().cpu().numpy(),
                            pose_frame.keypoints.xy.cpu().numpy()):

                        if self.is_full_body(kps):
                            x1, y1, x2, y2 = box
                            person_tracker_list.append([x1, y1, x2, y2, int(track_id)])
                except Exception as e:
                    print(f"[WARN] Pose format error: {e}")

            person_roi_item_result = {}
            for person_data in person_tracker_list:
                x1, y1, x2, y2, track_id = person_data
                h, w = frame.shape[:2]
                y1 = max(0, min(h, int(y1)))
                y2 = max(0, min(h, int(y2)))
                x1 = max(0, min(w, int(x1)))
                x2 = max(0, min(w, int(x2)))
                roi = frame[y1:y2, x1:x2]
                roi = self.enhance_image_opencv(roi)

                roi_result = self.custom_model(roi, conf=0.5, device=self.device, classes=[0, 1, 3, 4])
                person_roi_item_result_single = roi_result[0]
                detector_list = self.process_detect_results(person_roi_item_result_single)

                roi_item_detector_list_cleaned = [{'bbox': item[:4], 'conf': item[4], 'class_id': item[-1]} for item in roi_item_detector_list]
                person_roi_item_result[track_id] = roi_item_detector_list_cleaned

            self.draw_detections(frame, person_tracker_list, person_roi_item_result, self.compliance_class)

            if show_result:
                cv2.imshow("AIoTCam Video Result", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    break

            if out:
                out.write(frame)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    AIoTCam().main()
    '''test_image_path = r"D:\PycharmProjects\AIoTCam\AIoTCam\test_images\ly_with_kimi.jpg"
    output_path = r"D:\PycharmProjects\AIoTCam\AIoTCam\test_images\ly_with_kimi_11ss_shoes.jpg"
    AIoTCam().run_on_image(test_image_path, show_result=True, save_path=output_path)'''
    '''cam = AIoTCam()
    input_video = r"D:\PycharmProjects\AIoTCam\AIoTCam\test_images\compliance.mp4"
    output_video = r"D:\PycharmProjects\AIoTCam\AIoTCam\test_images\compliance_output_n.mp4"
    cam.run_on_video(input_video, show_result=True, save_path=output_video)'''
