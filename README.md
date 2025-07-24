# ELYRA - Edge-Optimized Lightweight YOLO with ROI Alignment for Real-Time PPE Compliance Monitoring in Workplace Safety with a Focus on Appropriate Laboratory Attire

## Overview

**ELYRA** is a real-time Personal Protective Equipment (PPE) compliance monitoring system designed for laboratories and low-risk environments. It uses **AI** and **computer vision** to automatically detect whether employees are wearing the required protective gear, such as sleeved shirts, long pants, and closed-toe shoes.

The system runs on a **Raspberry Pi 5** and uses the **YOLO11** object detection model, optimized with **NCNN** for efficient performance on edge devices. It provides real-time compliance feedback and sends **email alerts** when non-compliance is detected.

Additionally, **Region of Interest (ROI) segmentation** is used to focus the model’s detection on specific areas of the body, improving detection accuracy and speed, especially in complex environments.

## Features

- **Real-time PPE detection**: Detects appropriate laboratory attire (ALA PPE) using a camera and AI.
- **Edge computing**: Processes data locally on a Raspberry Pi to minimize latency and avoid cloud dependency.
- **Email alerts**: Sends notifications when non-compliance is detected.
- **Custom YOLO model**: Optimized for real-time processing with NCNN.
- **ROI Segmentation**: Focuses AI detection on specific regions of interest (such as the torso, legs, and feet), improving accuracy and efficiency.

## Components

- **`ELYRA_monitor_ALAPPE.py`**: Monitors PPE compliance using a connected camera and runs AI inference for object detection, utilizing **ROI segmentation** for improved accuracy and **NCNN** optimization for efficient real-time processing on the Raspberry Pi.
- **`ELYRA_export_recording.py`**: Exports recorded compliance data for analysis, utilizing **NFS** for centralized storage and easy access to the logged data across multiple devices.
- **`ELYRA_alert_email.py`**: Sends email alerts when PPE violations are detected, using **SMTP** for email delivery and **MIME** for formatting the email content with attachments.

