import cv2
import os

# Used to process frame of video to grayscale and resize to 16x16
def preprocess_frame(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (16, 16), interpolation=cv2.INTER_AREA)
    return resized.flatten()

# Helper function to upscale frames for visualisation on host machine
def upscale_frame(frame, scale):
    upscaled = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return upscaled

src_path = os.path.join("src", "DashcamFootage.mp4")
output_path = "output"
cap = cv2.VideoCapture(src_path)


frame_count = 0
scale_factor = 5  # Adjust this to control the visualization size

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = preprocess_frame(frame)
    processed_frame_reshaped = processed_frame.reshape(16, 16)
    
    # Upscale for visualization
    upscaled_frame = upscale_frame(processed_frame_reshaped, scale_factor)
    
    # Display the upscaled frame
    cv2.imshow('Processed Frame (Upscaled)', upscaled_frame)
    
    # Save the original 16x16 frame to the output folder
    output_path = os.path.join(output_path, f"frame_{frame_count:04d}.png")
    cv2.imwrite(output_path, processed_frame_reshaped)
    
    # Save the upscaled frame for visualization (optional)
    vis_output_path = os.path.join(output_path, f"frame_{frame_count:04d}_upscaled.png")
    cv2.imwrite(vis_output_path, upscaled_frame)
    
    frame_count += 1
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
