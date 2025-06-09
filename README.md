# Station-Based Robot Tracking

## Description
**Station-Based Robot Tracking** is a system designed to detect and track **Pudu** robot locations across various **stations** and record the data into a **database** for further monitoring. The system uses **YOLO** for real-time object detection and utilizes a **MySQL database** to store data and logs of robot movements.

The system detects robots moving through different station checkpoints, verifies if the robot is in the designated area, and records the **entry/exit** status of the robot at each checkpoint. All data is stored in a **MySQL database**, allowing for continuous monitoring and analysis of robot movements.

## Features
- **Real-Time Tracking**: Detects **Pudu** robots in multiple **stations** in real-time using YOLO.
- **Data Storage**: Records robot movement information in a **MySQL database** for further monitoring.
- **Multithreading**: Processes video frames, object detection, and data storage simultaneously.
- **Data Cleanup**: Automatically deletes outdated data to prevent memory leaks.
- **Database Connection Management**: Utilizes MySQL connection pooling for efficient database operations.

## Technologies Used
- **YOLOv8**: Used for real-time object detection of robots in video streams.
- **OpenCV**: Used for image processing and video frame display.
- **MySQL**: Used for data storage and recording robot movements.
- **Python**: The main programming language for developing the system.
- **Threading**: Allows parallel processing of video frames and database interactions.

## Installation
Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/username/Station-Based-Robot-Tracking.git
cd Station-Based-Robot-Tracking
```
