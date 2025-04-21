**Overview**
The Drowsiness Detection System is a real-time computer vision application that detects signs of driver fatigue using a webcam. It alerts the driver when drowsiness is detected by monitoring eye behavior, helping to prevent accidents caused by microsleep or lack of attention.

**Features**
  1.Real-time detection using a webcam

  2.Monitors eye aspect ratio (EAR) to detect drowsiness

  3.Sounds an alarm when eyes remain closed beyond a certain threshold

  4.Uses facial landmark detection for accurate eye tracking

**Technologies Used**
  1.Python
  2.OpenCV
  3.Dlib
  4.imutils
  5.Scipy

**How It Works**
  Captures video from the system's webcam.
  Uses facial landmarks to identify eye regions.
  Calculates the Eye Aspect Ratio (EAR).
  Triggers an alarm if the EAR stays below a predefined threshold for a continuous period.

**Installation**
Make sure Python is installed, then install the required libraries:
bash
Copy
Edit
pip install opencv-python dlib imutils scipy
Usage
**
Run the program using the command:**

python drowsiness_detection.py
Ensure your webcam is properly connected before running the script.

Notes
Tested on standard laptops with built-in webcams.


