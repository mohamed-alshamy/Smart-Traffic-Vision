# 🚗 Smart Traffic Vision

<p align="center">
  <img src="https://github.com/mohamed-alshamy/Smart-Traffic-Vision/blob/main/Smart%20Traffic%20Vision.jpg" width="650" alt="MAVERICK Banner">
</p>

A real-time computer vision system for **vehicle tracking, traffic flow analysis, wrong-way detection, lane monitoring, and road blockage detection** using **YOLOv12x, OpenCV, and Python**.

The system processes traffic video feeds, tracks individual vehicles across frames, learns the dominant traffic direction in each lane, detects vehicles moving against the established flow, and identifies potential lane blockages caused by stopped vehicles moving in conflicting directions.

---

## 🎯 Project Overview

**Smart Traffic Vision** is designed to transform a conventional traffic video feed into actionable traffic intelligence.

The system divides the scene into two traffic lanes and performs independent analysis for each lane.

### Core Pipeline

```text
Traffic Video
      │
      ▼
┌─────────────────────┐
│   YOLOv12x Model    │
│ Vehicle Detection   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Object Tracking   │
│  Persistent IDs     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Lane Classification │
│    Left / Right     │
└──────────┬──────────┘
           │
           ▼
┌────────────────────────────┐
│ Traffic Direction Learning │
│   Up / Down Dominance      │
└────────────┬───────────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
┌─────────────┐ ┌────────────────┐
│ Wrong-Way   │ │ Vehicle Stop   │
│ Detection   │ │ Detection      │
└──────┬──────┘ └───────┬────────┘
       │                │
       └────────┬───────┘
                ▼
       ┌──────────────────┐
       │ Lane Blockage    │
       │    Detection     │
       └──────────────────┘
                │
                ▼
       Annotated Video
       + Analysis Frames
```

---

## 🌟 Key Features

### 🚘 Vehicle Detection & Tracking

The system uses **YOLOv12x** through the Ultralytics framework to detect and track vehicles across video frames.

Tracked vehicle classes include:

* 🚗 Cars
* 🏍️ Motorcycles
* 🚌 Buses
* 🚛 Trucks

Each detected vehicle receives a persistent **tracking ID**, allowing the system to follow its movement throughout the video.

---

### 🔢 Vehicle Counting

Vehicles are counted when they cross a predefined horizontal counting line.

The system maintains separate counters for each lane:

```text
Left Lane
├── Up
└── Down

Right Lane
├── Up
└── Down
```

This provides independent traffic-flow statistics for both sides of the road.

---

### 🛣️ Automatic Lane Division

The video frame is divided vertically into two regions:

```text
┌──────────────────────┬──────────────────────┐
│                      │                      │
│      LEFT LANE       │      RIGHT LANE      │
│                      │                      │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

The lane boundary is automatically positioned at approximately **50% of the frame width**.

---

### 🧠 Automatic Traffic Direction Learning

Instead of requiring the traffic direction to be manually configured, the system learns the **dominant movement direction** independently for each lane.

Vehicle trajectories are analyzed over time to determine whether traffic predominantly moves:

* **Up**
* **Down**

Once enough movement data has been collected, the dominant direction is established for the corresponding lane.

This allows the system to adapt to different traffic scenes without manually specifying the expected direction.

---

### ↩️ Wrong-Way Vehicle Detection

After learning the dominant traffic direction, each tracked vehicle is compared against the expected direction of its lane.

If a vehicle moves against the established traffic flow, it is classified as:

```text
⚠️ WRONG WAY
```

The system also maintains separate wrong-way counters:

```text
Left Lane  → Wrong-Way Count
Right Lane → Wrong-Way Count
```

Wrong-way vehicles are highlighted differently in the output video for visual identification.

---

### 🛑 Stopped Vehicle Detection

The system monitors the movement of every tracked vehicle.

A vehicle is considered stopped when its movement remains below a predefined pixel threshold for a specified number of consecutive frames.

Current parameters:

```python
STOP_THRESHOLD_PIXELS = 5
STOP_THRESHOLD_FRAMES = 15
```

Stopped vehicles are marked in the visualization as:

```text
(Stopped)
```

---

### 🚧 Lane Blockage Detection

The system uses stopped-vehicle information together with vehicle movement directions to determine whether a lane may be blocked.

A lane is considered **Blocked** when stopped vehicles within the same lane have conflicting movement directions.

The system displays the current status for each lane:

```text
Status: Clear
```

or

```text
Status: Blocked
```

This provides an additional traffic-level interpretation beyond individual vehicle detection.

---

## 📊 Real-Time Analytics

The processed video displays lane-specific traffic statistics directly on the frame.

### Left Lane

```text
Left Lane Stats

Correct (Up): 12
Wrong Way: 1
Status: Clear
```

### Right Lane

```text
Right Lane Stats

Correct (Down): 18
Wrong Way: 2
Status: Blocked
```

The displayed information changes dynamically as the video is processed.

---

## 🧠 Technologies Used

| Technology             | Purpose                            |
| ---------------------- | ---------------------------------- |
| **Python**             | Core application development       |
| **YOLOv12x**           | Vehicle detection                  |
| **Ultralytics**        | YOLO inference and tracking        |
| **OpenCV**             | Video processing and visualization |
| **NumPy**              | Numerical operations               |
| **Python Collections** | Tracking and state management      |

---

## ⚙️ Requirements

### Software

* Python 3.x
* Ultralytics
* OpenCV
* NumPy

Install the required Python packages:

```bash
pip install ultralytics opencv-python numpy
```

---

## 📁 Project Structure

A typical project structure is:

```text
Smart-Traffic-Vision/
│
├── vehicle_tracker.py
├── yolo12x.pt
├── vid.mp4
│
├── video_output_with_blockage_status.mp4
├── last_frame.jpg
└── last_frame_with_analysis.jpg
```

### Main Files

**`vehicle_tracker.py`**

The main application containing:

* YOLO model loading
* Vehicle detection
* Multi-object tracking
* Lane classification
* Vehicle counting
* Direction learning
* Wrong-way detection
* Stopped-vehicle detection
* Blockage detection
* Result visualization
* Output generation

**`yolo12x.pt`**

The YOLOv12x model used for vehicle detection and tracking.

**`vid.mp4`**

Input traffic video.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/mohamed-alshamy/Smart-Traffic-Vision.git
cd Smart-Traffic-Vision
```

### 2. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 3. Add the YOLO Model

Place:

```text
yolo12x.pt
```

in the project root directory.

The model path is defined in the Python script:

```python
model = YOLO("yolo12x.pt")
```

---

### 4. Add Your Traffic Video

Place your input video in the project directory.

For example:

```text
vid.mp4
```

The input path is configured using:

```python
video_path = "vid.mp4"
```

You can replace it with another video filename or path.

---

### 5. Run the System

Execute:

```bash
python vehicle_tracker.py
```

The system will begin processing the video and display the annotated traffic analysis in real time.

Press:

```text
Q
```

to stop processing.

---

## 📤 Outputs

After processing the video, the system generates several outputs.

### 🎥 Processed Video

```text
video_output_with_blockage_status.mp4
```

Contains:

* Vehicle bounding boxes
* Tracking IDs
* Vehicle classes
* Traffic direction
* Wrong-way warnings
* Stopped-vehicle indicators
* Lane statistics
* Wrong-way counters
* Lane blockage status
* Lane and counting lines

---

### 🖼️ Original Last Frame

```text
last_frame.jpg
```

Contains the final frame of the input video before analytics are drawn.

---

### 🖼️ Analyzed Last Frame

```text
last_frame_with_analysis.jpg
```

Contains the final processed frame including:

* Vehicle detections
* Tracking IDs
* Lane boundaries
* Counting line
* Traffic statistics
* Wrong-way status
* Lane blockage status

---

## 🔬 Detection & Decision Logic

The system combines several layers of computer vision logic rather than relying solely on object detection.

### 1. Detection

YOLOv12x identifies vehicles in each frame.

### 2. Tracking

The tracking system assigns persistent IDs to detected vehicles.

### 3. Motion Estimation

The system stores recent vehicle center positions to estimate movement.

### 4. Direction Classification

Vehicle movement is classified as:

```text
Up    → -1
Down  → +1
```

### 5. Traffic Flow Learning

Vehicle directions are accumulated separately for the left and right lanes.

The dominant direction is established after sufficient directional evidence is collected.

### 6. Wrong-Way Analysis

Each vehicle's learned direction is compared with the dominant direction of its current lane.

### 7. Stop Detection

Vehicle displacement is monitored across consecutive frames.

### 8. Blockage Analysis

Stopped vehicles with conflicting movement directions are used to determine whether a lane is potentially blocked.

---

## 📐 Configurable Parameters

Several parameters can be adjusted depending on the camera position, video resolution, and traffic conditions.

### Counting Line

```python
horizontal_line_y = int(frame_height * 0.8)
```

### Lane Division

```python
vertical_line_x = int(frame_width * 0.5)
```

### Stop Detection Threshold

```python
STOP_THRESHOLD_PIXELS = 5
STOP_THRESHOLD_FRAMES = 15
```

### Direction Confirmation

```python
DIRECTION_CONFIRMATION_THRESHOLD = 5
```

### Initial Direction Learning Period

```python
INITIAL_PERIOD_FRAMES = 90
```

These parameters can be tuned for different camera perspectives and traffic environments.

---

## 🎯 Applications

The system can serve as a foundation for intelligent transportation and traffic-monitoring applications such as:

* 🚦 Traffic flow monitoring
* 🚘 Vehicle counting
* ↩️ Wrong-way vehicle detection
* 🚧 Road blockage monitoring
* 🛣️ Lane-level traffic analysis
* 📹 Intelligent CCTV analytics
* 🏙️ Smart city infrastructure
* 🚨 Automated traffic violation detection

---

## 🔮 Future Improvements

Potential extensions include:

* Multi-lane detection instead of fixed two-lane division
* Automatic lane segmentation
* Vehicle speed estimation
* Traffic density estimation
* Automatic congestion detection
* License plate recognition
* Traffic violation classification
* Real-time RTSP/IP camera support
* GPU-accelerated deployment
* Web-based traffic monitoring dashboard
* Event logging and database integration
* Real-time alerts for wrong-way vehicles and blocked lanes

---

## 👨‍💻 Author

**Mohamed Alshamy**

Computer Vision & Autonomous Systems Engineer

GitHub:
https://github.com/mohamed-alshamy

Portfolio:
https://mohamed-alshamy.vercel.app

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.
