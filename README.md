# OpticalFlowOnESP
This repository contains code on how to calculate optical flow vectors at a specific coordinate using an ESP32. A sample video is preprocess using OpenCV to send over a small 16 by 16 grayscale image over serial at 500000 Baud to the ESP32. Afterwards the ESP32 returns 4 Bytes over serial containing U and V values (Signed 16 bit ints) for the optical flow vector at the determined coordinate. This data is then read by the host PC and has the magnitude and angles calculated compared with Farneback algorithm using OpenCV.

A preview of the video with the optical flow vectors at the coordinate can be toggled to sanity check that the optical flow vectors produce by OpenCV is correct in the first place. These vectors are represented using HSV with Direction corresponding to Hue value of the image and Magnitude corresponds to Value.

# Quickstart guide
1. Create a venv
```
python3 -m venv <myenvpath>
```
2. Install requirements.txt
```
pip install -r requirements.txt
```
3. Upload esp32 code using arduino ide
4. Add a video to the src folder
5. run main.py after setting the video name variable
```
python main.py
```