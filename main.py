from preprocess import processdata
from opencvlk import flowFarneback as validate
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

def compute_magnitude_and_angle(flowList):
    """
    Separate the flow data into magnitudes and angles.
    """
    magnitudes = []
    angles = []
    for flow in flowList:
        _, u, v = flow
        magnitude = np.sqrt(u**2 + v**2) * 10  # Scale factor for visualization
        angle = np.arctan2(v, u)  
        angle = np.where(angle < 0, angle + 2*np.pi, angle)  # Ensure angle is positive
        magnitudes.append(magnitude)
        angles.append(angle)
    return np.array(magnitudes), np.array(angles)

def moving_average(data, window_size):
    """
    Apply a moving average filter to the data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def filter_data(data, moving_avg_window_size=5, median_filter_size=8):
    """
    Apply a moving average and median filter to the data.
    """
    filtered_data = moving_average(data, moving_avg_window_size)
    filtered_data = median_filter(filtered_data, size=median_filter_size)
    return filtered_data

def preprocess_video_data(video_name, video=False, max_frames=900):
    """
    Process and compute both ESP32 and OpenCV flow data.
    """
    # Process the video file and send frames to ESP32
    flowListESP = processdata(video_name, max_frames)

    # Validate the optical flow vectors using OpenCV Farneback method
    flowListOfMagAndAngOpenCV = validate(video_name, max_frames, video)

    # Compute magnitudes and angles for ESP32 data
    magnitudeESP, angleESP = compute_magnitude_and_angle(flowListESP)

    # Extract magnitudes and angles from OpenCV data
    magnitudeOpenCV = np.array([item[0] for item in flowListOfMagAndAngOpenCV])
    angleOpenCV = np.array([item[1] for item in flowListOfMagAndAngOpenCV])

    # Filter magnitudes and angles
    magnitudeESP = filter_data(magnitudeESP)
    angleESP = filter_data(angleESP)
    magnitudeOpenCV = filter_data(magnitudeOpenCV)
    angleOpenCV = filter_data(angleOpenCV)

    # Truncate or align lengths of both arrays
    min_length = min(len(magnitudeESP), len(magnitudeOpenCV), len(angleESP), len(angleOpenCV))

    # Make sure all arrays are the same length
    magnitudeESP = magnitudeESP[:min_length]
    magnitudeOpenCV = magnitudeOpenCV[:min_length]
    angleESP = angleESP[:min_length]
    angleOpenCV = angleOpenCV[:min_length]

    return magnitudeESP, magnitudeOpenCV, angleESP, angleOpenCV, min_length

def plot_data(time_axis, magnitudeESP, magnitudeOpenCV, angleESP, angleOpenCV):
    """
    Apply plotting for magnitudes and angles of optical flow vectors.
    """
    # Plotting the magnitudes in a separate window
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, magnitudeESP, label="ESP32 Flow Magnitude (Filtered)", color='blue', alpha=0.7)
    plt.plot(time_axis, magnitudeOpenCV, label="OpenCV Flow Magnitude Farneback", color='red', alpha=0.7)
    plt.xlabel("Frame Number")
    plt.ylabel("Magnitude of Flow Vector")
    plt.title("Comparison of Flow Vector Magnitudes")
    plt.legend()
    plt.grid(True)
    plt.show()  # Show the first figure

    # Plotting the angles in a separate window
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, np.degrees(angleESP), label="ESP32 Flow Angle", color='blue', alpha=0.7)
    plt.plot(time_axis, np.degrees(angleOpenCV), label="OpenCV Flow Angle Farneback", color='red', alpha=0.7)
    plt.xlabel("Frame Number")
    plt.ylabel("Angle of Flow Vector (Degrees)")
    plt.title("Comparison of Flow Vector Angles")
    plt.legend()
    plt.grid(True)
    plt.show()  # Show the second figure

def main():
    """
    Main function to run the flow processing, filtering, and plotting.
    """
    video_name = "test.mp4"

    # Step 1: Preprocess the video data
    magnitudeESP, magnitudeOpenCV, angleESP, angleOpenCV, min_length = preprocess_video_data(video_name, video=True)
    
    time_axis = np.arange(1, min_length + 1)
    plot_data(time_axis, magnitudeESP, magnitudeOpenCV, angleESP, angleOpenCV)

if __name__ == "__main__":
    main()
