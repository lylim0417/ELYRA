
# Monitor ALA PPE Setup Instructions for ELYRA System

This section explains how to set up the ALA PPE compliance monitoring system using YOLO11, Pose Tracking, ByteTrack, ROI Segmentation, and NCNN Optimization. The monitoring script runs on the client side and detects ALA PPE compliance and records it down.

## Setting Up the Cron Job for Monitor ALA PPE (on **client**)

1. **Ensure the Monitor ALA PPE Script is in Place**:
   First, make sure you have the Monitor ALA PPE script [ELYRA_monitor_ALAPPE.py](../ELYRA-code/ELYRA_monitor_ALAPPE.py) ready and placed on your **client** machine. This script will be responsible for detecting compliance items such as shirts, pants, and shoes, using YOLO11.

2. **Install Dependencies**

   Install necessary libraries and packages, please refer to [client_requirements.txt](../requirements/client_requirements.txt):
   ```bash
   pip install -r client_requirements.txt
   ```

3. **Configure the YOLO1 Model**
   The system uses a **YOLO11 model** for object detection. Please refer to [ELYRA-model/](../ELYRA-model/).

   - **Custom ALA PPE Model (YOLO11s + NCNN)**: [ELYRA-model/Custom_ALAPPE_Detection-YOLO11s/best_ncnn_model](../ELYRA-model/Custom_ALAPPE_Detection-YOLO11s/best_ncnn_model)
   - **Pose Detection Model (YOLO11n + NCNN)**: [ELYRA-model/Default_Person_Pose_Estimation-YOLO11n/yolo11n-pose_ncnn_model](../ELYRA-model/Default_Person_Pose_Estimation-YOLO11n/yolo11n-pose_ncnn_model)

4. **Check GPU Support (Optional)**

   The system can utilize a **GPU** if available for faster inference. To check for GPU support:
   - Ensure that **CUDA** is installed if you're using an NVIDIA GPU.
   - The script uses **`torch`** to detect and utilize the GPU if it's available:
     ```python
     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     ```

   - If no GPU is available, the system will fall back to the **CPU**.

5. **Camera Configuration**
   Ensure that your camera is properly connected to the system and accessible via OpenCV. The script uses OpenCV to access the camera:
   ```python
   capture = cv2.VideoCapture(0)
   ```

6. **Running the Script**

   After completing the setup, you can run the **`ELYRA_monitor_ALAPPE`** system script to start processing video frames, detect people and non-compliance items, and record the output:
   ```bash
   python3 ELYRA_monitor_ALAPPE.py
   ```

   The system will use the **YOLO11 model** with the integration of **ROI Segmentation** and **NCNN Optimization** for detecting compliance items (e.g., shirts, pants, shoes) and track them.

7. **Monitoring and Logging**

   The system collects various performance metrics during processing:
   - **FPS (Frames per Second)**
   - **CPU Usage**
   - **Detected items** (compliance vs. non-compliance)
   - The results are saved to CSV logs for further analysis.

8. **Start Recording**

   The system can also record video for detected persons, especially if they are found to be in non-compliance with PPE requirements.

---

## Auto-run Script on Boot using `@reboot` in Crontab

To ensure the script runs automatically on system startup, follow these steps:

1. **Edit Crontab**:
   Open the crontab configuration file:
   ```bash
   crontab -e
   ```

2. **Add the `@reboot` Line**:
   Add the following line to the crontab to automatically run the script on boot:
   ```bash
   @reboot /usr/bin/python3 /home/ELYRA/ELYRA-code/ELYRA_monitor_ALAPPE.py >> /home/ELYRA/ELYRA-code/ELYRA_monitor_ALAPPE_logfile.log 2>&1
   ```

   This line ensures that the **`ELYRA_monitor_ALAPPE.py`** script will start automatically after reboot, and all output (including errors) will be logged to **`ELYRA_monitor_ALAPPE.log`**.

3. **Verify Crontab**:
   To confirm that the crontab entry has been added correctly, list your crontab jobs:
   ```bash
   crontab -l
   ```

   You should see the following line in the list:
   ```bash
   @reboot /usr/bin/python3 /home/ELYRA/ELYRA-code/ELYRA_monitor_ALAPPE.py >> /home/ELYRA/ELYRA-code/ELYRA_monitor_ALAPPE_logfile.log 2>&1
   ```

4. **Check the Log File**:
   After a reboot, check the log file for any output or errors:
   ```bash
   cat /home/ELYRA/ELYRA-code/ELYRA_monitor_ALAPPE_logfile.log
   ```

---

## Additional Notes

- **OpenCV** requires that you have a proper installation of dependencies for handling video capture and processing.
- **Ultralytics YOLO** (used in this project) is a state-of-the-art object detection model, and you will need **PyTorch** for running the model.
- Ensure that **CUDA** is installed for NVIDIA GPU support, which will drastically improve inference speed. **(No difference in Raspberry Pi)**

---

## Troubleshooting

- **Issue with `cv2.VideoCapture`**: Make sure your camera is properly connected and recognized by the system. You can test this with a simple OpenCV script:
   ```python
   import cv2
   capture = cv2.VideoCapture(0)
   if not capture.isOpened():
       print("Error: Camera not detected.")
   else:
       print("Camera is working.")
   ```

- **Out of Memory (OOM) error**: If you are running the script on a machine with limited CPU memory, try reducing the image resolution or batch size in the YOLO model configuration.

---

## Conclusion

By following these setup instructions, you should be able to run the **ELYRA_monitro_ALAPPE** system, utilizing YOLO for object detection and real-time compliance monitoring and recording.
