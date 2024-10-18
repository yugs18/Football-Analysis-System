import numpy as np
import pickle
import cv2
import os
import sys

# Add the parent directory to the system path to import utility functions
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

# Define the CameraMovement class to handle camera movement calculations and tracking
class CameraMovement():
    def __init__(self, frame):
        """
        Initialize the CameraMovement class with the given frame.

        Parameters:
        frame (numpy.ndarray): The initial frame to analyze camera movement.
        """
        self.minimum_distance = 5  # Minimum distance to detect movement

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),  # Size of the search window
            maxLevel=2,        # Maximum pyramid levels for tracking
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Termination criteria
        )

        # Convert the first frame to grayscale
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask for feature detection
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1        # Left side region of interest
        mask_features[:, 900:1050] = 1    # Right side region of interest

        # Parameters for detecting good features to track
        self.features = dict(
            maxCorners=100,      # Max corners to detect
            qualityLevel=0.3,    # Quality level of detected corners
            minDistance=3,       # Minimum distance between detected corners
            blockSize=7,         # Block size for computing derivatives
            mask=mask_features    # Mask for region of interest
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        Adjusts object positions in the tracks based on camera movement.

        Parameters:
        tracks (dict): A dictionary of tracked objects.
        camera_movement_per_frame (list): List of camera movements for each frame.
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Adjust the position based on camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Store the adjusted position back in the tracks
                    tracks[object_type][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        Calculates camera movement between frames.

        Parameters:
        frames (list): List of frames to analyze.
        read_from_stub (bool): Whether to read movement data from a stub file.
        stub_path (str): Path to the stub file.

        Returns:
        list: A list of camera movements for each frame.
        """
        # Load movement data from a stub file if requested
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize movement with zeros
        movement = np.zeros((len(frames), 2))

        # Convert the first frame to grayscale and detect good features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Loop through the frames to calculate movement
        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Calculate optical flow between frames
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Variables to store the largest movement
            max_dist = 0
            movement_x, movement_y = 0, 0

            # Loop through the new and old features to calculate movement
            for new, old in zip(new_features, old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Measure the distance between the new and old feature points
                dist = measure_distance(new_features_point, old_features_point)

                # Update the movement if this distance is the maximum found
                if dist > max_dist:
                    max_dist = dist
                    movement_x, movement_y = measure_xy_distance(old_features_point, new_features_point)

            # Record the movement if it exceeds the minimum distance
            if max_dist > self.minimum_distance:
                movement[frame_num] = [movement_x, movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Update the old frame for the next iteration
            old_gray = frame_gray.copy()

            # Save movement data to a file if a path is provided
            if stub_path is not None:
                try:
                    with open(stub_path, 'wb') as f:
                        pickle.dump(movement, f)
                except IOError as e:
                    print(f"Error saving movement data to file: {e}")

        return movement

    def draw_movement(self, frames, camera_movement_per_frame):
        """
        Visualizes and draws camera movement on frames.

        Parameters:
        frames (list): List of frames to annotate.
        camera_movement_per_frame (list): List of camera movements for each frame.

        Returns:
        list: A list of frames with movement annotations.
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Create an overlay rectangle for displaying the movement text
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6  # Transparency level
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get the x and y movement for the current frame
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Draw the x and y movement as text on the frame
            frame = cv2.putText(frame, f"Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            frame = cv2.putText(frame, f"Movement Y: {y_movement:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Append the modified frame to the output frames list
            output_frames.append(frame)

        return output_frames
