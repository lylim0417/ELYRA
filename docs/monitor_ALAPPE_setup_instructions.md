
# ELYRA System - AIoTCam Setup Instructions

This guide outlines the steps to set up the **AIoTCam** system, including the required dependencies and setup for video processing, object detection using YOLO, and system monitoring.

## Prerequisites

### 1. **Create a Project Directory**:
   First, create a directory for the project named `ELYRA-code`:
   ```bash
   mkdir ELYRA-code
   ```

   Then, grant full permissions to the directory:
   ```bash
   sudo chmod 777 ELYRA-code
   ```

### 2. **Install Required Python Packages**

   To install the necessary dependencies, you can use the provided **`requirements.txt`** file.

   1. Create the `requirements.txt` file with the following content:
      ```plaintext
      ultralytics         # For YOLOv5 and other YOLO models
      opencv-python       # For video processing with OpenCV
      screeninfo          # For retrieving monitor information
      torch               # For running the YOLO model, support for GPU/CPU
      psutil              # For system monitoring (CPU, memory, etc.)
      pynvml              # For monitoring NVIDIA GPU status
      pandas              # For handling data frames
      numpy               # For numerical operations
      ```

   2. Install the dependencies using pip:
      ```bash
      pip install -r requirements.txt
      ```

### 3. **Configure the YOLOv5 Model**
   The system uses a **YOLOv5 model** for object detection. You should already have the model weights available, or you can download the pre-trained model.

   - **Custom YOLO Model**: `/home/lylim/AIoTCam/model-5000/5000-11s/weights/best_ncnn_model`
   - **Pose Detection Model**: `/home/lylim/AIoTCam/Yolo-Weights/yolo11n-pose_ncnn_model`

### 4. **Check GPU Support (Optional)**

   The system can utilize a **GPU** if available for faster inference. To check for GPU support:
   - Ensure that **CUDA** is installed if you're using an NVIDIA GPU.
   - The script uses **`torch`** to detect and utilize the GPU if it's available:
     ```python
     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     ```

   - If no GPU is available, the system will fall back to the **CPU**.

### 5. **Camera Configuration**
   Ensure that your camera is properly connected to the system and accessible via OpenCV. The script uses OpenCV to access the camera:
   ```python
   capture = cv2.VideoCapture(0)
   ```

### 6. **Running the Script**

   After completing the setup, you can run the **`AIoTCam`** system script to start processing video frames, detect people and non-compliance items, and record the output:
   ```bash
   python3 AIoTCam.py
   ```

   The system will use the **YOLOv5 model** for detecting compliance items (e.g., shirts, pants, shoes) and track them.

### 7. **Monitoring and Logging**

   The system collects various performance metrics during processing:
   - **FPS (Frames per Second)**
   - **CPU Usage**
   - **Detected items** (compliance vs. non-compliance)
   - The results are saved to CSV logs for further analysis.

### 8. **Start Recording**

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
   @reboot /usr/bin/python3 /home/lylim/AIoTCam/AIoTCam.py >> /home/lylim/AIoTCam/aiotcam_logfile.log 2>&1
   ```

   This line ensures that the **`AIoTCam.py`** script will start automatically after reboot, and all output (including errors) will be logged to **`aiotcam_logfile.log`**.

3. **Verify Crontab**:
   To confirm that the crontab entry has been added correctly, list your crontab jobs:
   ```bash
   crontab -l
   ```

   You should see the following line in the list:
   ```bash
   @reboot /usr/bin/python3 /home/lylim/AIoTCam/AIoTCam.py >> /home/lylim/AIoTCam/aiotcam_logfile.log 2>&1
   ```

4. **Check the Log File**:
   After a reboot, check the log file for any output or errors:
   ```bash
   cat /home/lylim/AIoTCam/aiotcam_logfile.log
   ```

---

## Additional Notes

- **OpenCV** requires that you have a proper installation of dependencies for handling video capture and processing.
- **Ultralytics YOLO** (used in this project) is a state-of-the-art object detection model, and you will need **PyTorch** for running the model.
- Ensure that **CUDA** is installed for NVIDIA GPU support, which will drastically improve inference speed.

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

- **Out of Memory (OOM) error**: If you are running the script on a machine with limited GPU memory, try reducing the image resolution or batch size in the YOLO model configuration.

---

## Conclusion

By following these setup instructions, you should be able to run the **AIoTCam** system, utilizing YOLO for object detection and real-time compliance monitoring with video recording and alerting capabilities.

Let me know if you encounter any issues or need additional support!
