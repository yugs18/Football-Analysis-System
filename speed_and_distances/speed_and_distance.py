# Import necessary libraries
import cv2
import sys
import numpy as np  # Import NumPy for numerical operations if needed

# Append the utils module to system path for utility functions like measure_distance and get_foot_position
sys.path.append('../')
from utils import measure_distance, get_foot_position

# Define a class to estimate speed and distance covered by players over time
class SpeedAndDistance():
    def __init__(self):
        # Set the window of frames to calculate speed and distance
        self.frame_window = 5
        # Frame rate of the video (frames per second)
        self.frame_rate = 24

    # Function to calculate and add speed and distance information to track data
    def add_speed_and_distance_to_tracks(self, tracks):
        # Initialize a dictionary to store the total distance covered by each object (players)
        total_distance = {}

        # Loop over each tracked object (players, referees, ball, etc.)
        for object_name, object_tracks in tracks.items():
            # Skip the ball and referees as we are only calculating player speeds
            if object_name == "ball" or object_name == "referees":
                continue
            
            number_of_frames = len(object_tracks)  # Get the total number of frames in the track

            # Loop through the frames in increments of self.frame_window
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Define the last frame in the current window, ensuring it doesn't exceed the total frames
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # For each track (player) in the current frame
                for track_id in object_tracks[frame_num]:
                    # If the player doesn't exist in the last frame of the window, skip
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Get the transformed positions of the player at the start and end of the frame window
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Skip if any of the positions are None (out of bounds)
                    if start_position is None or end_position is None:
                        continue
                    
                    # Calculate the distance covered by the player between the start and end positions
                    distance_covered = measure_distance(start_position, end_position)
                    # Calculate the time elapsed (in seconds) over the frame window
                    time_elapsed = (last_frame - frame_num) / self.frame_rate

                    # Skip the calculation if time_elapsed is zero to avoid division by zero
                    if time_elapsed <= 0:
                        continue

                    # Calculate the speed in meters per second
                    speed_meters_per_second = distance_covered / time_elapsed
                    # Convert speed to kilometers per hour
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Initialize the total distance covered by each player
                    if object_name not in total_distance:
                        total_distance[object_name] = {}
                    
                    # Initialize total distance for the specific track (player)
                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0
                    
                    # Add the distance covered in this window to the player's total distance
                    total_distance[object_name][track_id] += distance_covered

                    # Add speed and distance information to all frames in the current window
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        # Store the calculated speed and total distance in the track
                        tracks[object_name][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object_name][frame_num_batch][track_id]['distance'] = total_distance[object_name][track_id]

    # Function to draw the speed and distance of each player on the video frames
    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []  # List to store frames with speed and distance annotations
        
        # Loop through each frame of the video
        for frame_num, frame in enumerate(frames):
            # Loop through the objects (players) being tracked in each frame
            for object_name, object_tracks in tracks.items():
                # Skip the ball and referees as we are only annotating players
                if object_name == "ball" or object_name == "referees":
                    continue 

                # Loop through the track info for each player in the current frame
                for track_id, track_info in object_tracks[frame_num].items():
                    # Check if the player's speed and distance have been calculated
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)  # Get speed if available
                        distance = track_info.get('distance', None)  # Get distance if available
                        # Skip if either speed or distance is missing
                        if speed is None or distance is None:
                            continue
                        
                        # Get the bounding box of the player to determine where to display the text
                        bbox = track_info['bbox']
                        # Get the player's foot position from the bounding box
                        position = get_foot_position(bbox)
                        position = list(position)  # Convert position to a list for easy manipulation
                        position[1] += 40  # Shift the Y-coordinate to display text above the player's foot

                        # Convert position values to integers
                        position = tuple(map(int, position))
                        
                        # Draw the speed (in km/h) on the frame at the player's position
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Draw the distance (in meters) slightly below the speed
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Append the annotated frame to the output list
            output_frames.append(frame)
        
        # Return the list of annotated frames
        return output_frames
