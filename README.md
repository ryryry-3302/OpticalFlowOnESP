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

[Results](https://ryryry-3302.github.io/OpticalFlowOnESP/)

# Overview of processing with esp
1. #### preprocessing 
- `preprocess.py` is used to extract 16 by 16 sector of interest using openCV
- Converts from rgb to grayscale
- Performs gaussian blur
- flattens 2d array to be sent byte by byte over serial to esp32 for processing.
2. #### esp32
- Awaits receiving two frames of 256 bytes grayscale
- Changes frames back into 2d arrays
- Performs optical flow algorithm using a simplified version of lucas kanade calculating Ix Iy It for calculating of u and v using inverse matrix (if det == 0 return u and v = 0)
- After calculating u and v sends back u and v values scaled up by 100 to have precision of up to 2.dp. Transmission is done over serial with each value being sent as a signed 16 bit int

3. #### preprocessing.py
- After successfully receiving u and v values or timing out host computer stores u v in a list of flowvectors as a tuple
- Loops sending over and awaiting until all desired frames are processed

# Validation with openCV
- The same video feed and region of interest is processed using openCVFarneback algorithm
- Each processed flow vector is appended to the original video over a HSV 16 by 16 box to evaluate whether openCV is able to correctly detect motion along with direction

# Main.py
- Runs both esp32 processing and openCV processing before plotting magnitude and angle values using matplotlib.
- Moving average and median filter is used before plotting to make data more readable by reducing effect of noise

