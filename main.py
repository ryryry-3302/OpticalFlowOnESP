from preprocess import processdata
from validation import validate
import matplotlib.pyplot as plt
import numpy as np

# Process the video file and send frames to ESP32
flowListESP = processdata("DashcamFootage.mp4", 2)

# Validate the optical flow vectors
flowListOpenCV = validate("DashcamFootage.mp4", 2)

print(flowListESP)
print(flowListOpenCV)


