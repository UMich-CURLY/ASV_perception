import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video and saves them as image files.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        frame_rate (int): Number of frames to save per second (default is 1).

    Returns:
        None
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video frame rate
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Frame interval to save based on desired frame rate
    frame_interval = max(1, video_fps // frame_rate)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it matches the frame interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames saved: {saved_count} in {output_folder}")

if __name__ == "__main__":
    video_path = "algae_video.mp4"  # Replace with your video file path
    output_folder = "algae_frames"  # Replace with your desired output folder
    frame_rate = 30  # Number of frames to save per second

    extract_frames(video_path, output_folder, frame_rate)