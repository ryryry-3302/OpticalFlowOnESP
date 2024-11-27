from preprocess import processdata
from opencvlk import flowFarneback as validate
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

video_name = "test.mp4"
# Process the video file and send frames to ESP32
flowListESP = processdata(video_name, 900)

# Validate the optical flow vectors
magnitudeOpenCV = validate(video_name, 900)

def compute_magnitude(flowList):
    magnitudes = []
    for flow in flowList:
        # Assuming each 'flow' is a 2D vector (u, v)
        _, u, v = flow
        magnitude = np.sqrt(u**2 + v**2)*10
        magnitudes.append(magnitude)
    return np.array(magnitudes)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Compute magnitudes for both flow lists
magnitudeESP = compute_magnitude(flowListESP)


magnitudeESP = moving_average(magnitudeESP, window_size=5)
magnitudeESP = median_filter(magnitudeESP, size= 8)

# Create a time axis for the frames (assuming 900 frames)
min_length = min(len(magnitudeESP), len(magnitudeOpenCV))
magnitudeESP = magnitudeESP[:min_length]
magnitudeOpenCV = magnitudeOpenCV[:min_length]

time_axis = np.arange(1, min_length + 1)

# Plotting the magnitudes of the flow vectors
plt.figure(figsize=(10, 6))

# Plot ESP32 flow vector magnitudes (after applying median filter)
plt.plot(time_axis, magnitudeESP, label="ESP32 Flow Magnitude (Filtered)", color='blue', alpha=0.7)
plt.plot(time_axis, magnitudeOpenCV, label="OpenCV FLow Magnitude Farneback", color='red', alpha=0.7)



# Add labels and title
plt.xlabel("Frame Number")
plt.ylabel("Magnitude of Flow Vector")
plt.title("Comparison of Flow Vector Magnitudes (ESP32 vs OpenCV)")

# Display legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
