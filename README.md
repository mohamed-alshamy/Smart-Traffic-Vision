🚗 Vehicle Tracking and Blockage Detection System
This project utilizes the YOLOv12x model to track vehicles in a video feed, count them, identify wrong-way drivers, and detect traffic blockages.

🌟 Key Features
🚗 Vehicle Tracking and Counting: Tracks and counts vehicles (cars, motorcycles, buses, trucks) as they cross a virtual line.

🛣️ Lane Division: Splits the screen into two lanes (left and right) and performs separate counts for each.

🧠 Automatic Direction Learning: Automatically learns the dominant direction of traffic flow in each lane.

↩️ Wrong-Way Detection: Identifies vehicles moving against the established dominant traffic direction in their lane.

🚧 Blockage Detection: Detects if a lane is blocked based on stopped vehicles facing opposite directions.

💾 Result Saving: Saves the processed video with all analytics drawn on it, and also saves the last frame of the video.

⚙️ Requirements
To run this project, you will need the following:

1. Python Libraries
You must have Python installed on your system. You can install the necessary libraries using pip:

pip install ultralytics
pip install opencv-python
pip install numpy

2. System Software
💿 ffmpeg: Essential for video processing with OpenCV. It must be downloaded, and its executable files (ffmpeg.exe, ffplay.exe, ffprobe.exe) must be in the system's PATH.

3. Project Files
Ensure the following files are present in the main project folder:

🐍 vehicle_tracker.py: The main script for the project (or traffic.py depending on your naming).

🤖 yolo12x.pt: The pre-trained YOLO model file.

🎬 vid.mp4: The video file to be analyzed (supports formats like MP4, M4V).

🚀 How to Use
Download YOLO Model:
The script uses yolo12x.pt as the standard model. If you use a custom model, make sure to place it in the same project folder and update its name in the code.

Prepare the Video:
Place the video file you want to analyze in the same folder.
Open the vehicle_tracker.py file and change the value of the video_path variable to your video file's name.

video_path = "your_video_file.mp4"

Run the Script:
Execute the script from your terminal:

python vehicle_tracker.py

📊 Outputs
🎥 Processed Video: A new video file named video_output_with_blockage_status.mp4 will be created.

🖼️ Last Frame Images: Two images will be saved:
last_frame.jpg: The original last frame from the video.
last_frame_with_analysis.jpg: The last frame with the analytics displayed on it.