import cv2

def read_video(video_path):
    """
    Reads a video file and extracts its frames.

    Parameters:
    video_path (str): The path to the video file to be read.

    Returns:
    list: A list of frames (images) extracted from the video.
    """
    # Create a VideoCapture object to read the video from the specified path
    cap = cv2.VideoCapture(video_path)
    
    # Initialize an empty list to store the frames
    frames = []
    
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        # Check if the frame was successfully read
        if not ret:
            break  # Exit the loop if there are no more frames to read
        
        # Append the current frame to the list of frames
        frames.append(frame)
    
    # Release the VideoCapture object
    cap.release()
    
    # Return the list of frames extracted from the video
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Saves a list of frames as a video file.

    Parameters:
    output_video_frames (list): A list of frames (images) to be written to the video.
    output_video_path (str): The path where the output video file will be saved.
    """
    # Define the codec for the video writer. 'XVID' is commonly used for .avi files.
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    
    # Create a VideoWriter object to write the video to the specified output path.
    # The frame rate is set to 24 frames per second, and the size of the video is based on the dimensions of the first frame.
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    
    # Iterate through each frame in the list of output video frames
    for frame in output_video_frames:
        # Write the current frame to the video file
        out.write(frame)
    
    # Release the VideoWriter object to free up resources
    out.release()