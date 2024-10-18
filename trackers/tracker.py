from ultralytics import YOLO  # Import YOLO model from ultralytics
import supervision as sv      # Import supervision for ByteTrack tracker and object detection utilities
import pickle                 # Import pickle to save/load Python objects (like tracking data)
import os                     # Import os for file handling
import cv2                    # Import OpenCV for image and video processing
import sys                    # Import sys to modify the path for importing custom modules
import numpy as np            # Import NumPy for array manipulations
import pandas as pd           # Import pandas for handling ball position interpolation as a DataFrame

# Append the system path to include custom utilities
sys.path.append('/home/shadowkiller/Football Analysis System/')

from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # Import helper functions for bounding box handling


class Tracker:
    def __init__(self, model_path):
        """
        Initialize Tracker class with a YOLO model and ByteTrack tracker.
        :param model_path: Path to the YOLO model to be loaded for object detection.
        """
        self.model = YOLO(model_path)  # Load YOLO model from the specified path
        self.tracker = sv.ByteTrack()  # Initialize ByteTrack tracker for tracking objects across frames


    def add_position_to_tracks(self, tracks):
        """
        Add the position of objects (ball/players) based on the bounding box for each track.
        :param tracks: A dictionary of tracked objects (players, ball, referees).
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    # Use center for the ball, foot position for players/referees
                    position = get_center_of_bbox(bbox) if object_type == 'ball' else get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position


    def interpolate_ball_position(self, ball_positions):
        """
        Interpolate missing ball positions to smooth tracking.
        :param ball_positions: A list of ball position data across frames.
        :return: Interpolated list of ball positions.
        """
        # Extract bounding box for each frame where the ball is detected
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        # Convert to a DataFrame for easier interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Perform linear interpolation on missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()  # Backfill in case there are NaN values at the start

        # Convert the interpolated DataFrame back to the required list format
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions


    def detect_frames(self, frames):
        """
        Detect objects in the provided frames using the YOLO model in batches.
        :param frames: List of frames (images) to process.
        :return: List of detection results for all frames.
        """
        batch_size = 20  # Process frames in batches to manage memory usage
        detections = []  # Store all detections

        # Process each batch of frames and append the detection results
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)  # Predict with confidence threshold 0.1
            detections.extend(detection_batch)  # Add detection results to the list

        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track objects across frames. Optionally, read from a pre-saved stub file.
        :param frames: List of frames to process for tracking.
        :param read_from_stub: Boolean flag to read tracks from a pre-saved stub.
        :param stub_path: Path to the pre-saved pickle file with tracking data.
        :return: Dictionary containing tracked objects (players, ball, referees).
        """
        # Load tracks from a stub file if it exists and the flag is set
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)  # Load and return pre-saved tracks

        # Otherwise, perform detection on the given frames
        detections = self.detect_frames(frames)

        # Initialize an empty dictionary for storing tracks
        tracks = {"players": [], "referees": [], "ball": []}

        # Process each frame's detection results
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # Get class names (e.g., player, ball, referee)
            cls_name_inv = {value: key for key, value in cls_names.items()}  # Reverse mapping for class ID lookup

            # Convert YOLO detections to Supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper class to player class
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_name_inv["player"]

            # Update the tracker with the current frame's detections
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Prepare empty dictionaries for each frame's tracked objects
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Handle tracked objects (players/referees)
            for frame_detection in detection_with_tracks:
                bounding_box = frame_detection[0].tolist()  # Extract bounding box
                cls_id = frame_detection[3]  # Get object class ID
                track_id = frame_detection[4]  # Get track ID for this object

                # Add tracked players to the list
                if cls_id == cls_name_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bounding_box}

                # Add tracked referees to the list
                elif cls_id == cls_name_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bounding_box}

            # Handle ball detections separately (without tracking ID)
            for frame_detection in detection_supervision:
                bounding_box = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                # Add detected ball to the list
                if cls_id == cls_name_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bounding_box}

        # Optionally save tracks to a stub file
        if stub_path:
            with open(stub_path, "wb") as file:
                pickle.dump(tracks, file)

        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse around a tracked object and optionally label it with the track ID.
        :param frame: The video frame to draw on.
        :param bbox: Bounding box of the object.
        :param color: Color of the ellipse.
        :param track_id: Optional track ID to label the object.
        :return: Modified frame with the drawn ellipse.
        """
        y2 = int(bbox[3])  # Get the bottom y-coordinate
        x_center, _ = get_center_of_bbox(bbox)  # Get the center x-coordinate of the bounding box
        width = get_bbox_width(bbox)  # Calculate the width of the bounding box

        # Draw an ellipse to represent the object
        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)), 0, -45, 235, color, 2)

        # Draw a rectangle to label the object with its track ID (if available)
        if track_id is not None:
            rect_top_left = (x_center - 20, y2 + 15)
            rect_bottom_right = (x_center + 20, y2 + 35)
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x_center - 10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame
    

    def draw_triangle(self, frame, bbox, color):
        """
        Draws a triangle on the frame based on the bounding box provided.

        :param frame: The video frame on which the triangle will be drawn.
        :param bbox: The bounding box coordinates, used to determine the position of the triangle.
        :param color: The color of the triangle to be drawn.
        :return: The modified frame with the triangle drawn on it.
        """
        y1 = int(bbox[1])  # Get the y-coordinate from the bounding box
        x_center, _ = get_center_of_bbox(bbox)  # Get the x-coordinate of the center of the bounding box

        # Define the triangle's vertices
        triangle_points = np.array([
            [x_center, y1],              # Top vertex of the triangle
            [x_center - 10, y1 - 20],    # Bottom-left vertex
            [x_center + 10, y1 - 20],    # Bottom-right vertex
        ])

        # Fill the triangle with the specified color
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED,
        )

        # Draw a border around the triangle in black
        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            2,
        )

        return frame  # Return the modified frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draws the ball control information for both teams on the frame.

        :param frame: The video frame on which the ball control information will be drawn.
        :param frame_num: The current frame number.
        :param team_ball_control: An array indicating which team has ball control for each frame.
        :return: The modified frame with ball control annotations.
        """
        overlay = frame.copy()  # Create a copy of the frame for overlaying
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)  # Draw a white rectangle as a background for text
        alpha = 0.4  # Define the transparency for the overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Blend the overlay with the original frame

        # Get ball control information until the current frame
        control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = control_till_frame[control_till_frame == 1].shape[0]  # Count frames controlled by Team 1
        team_2_num_frames = control_till_frame[control_till_frame == 2].shape[0]  # Count frames controlled by Team 2

        # Calculate ball control percentages
        total_frames = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total_frames if total_frames > 0 else 0
        team_2 = team_2_num_frames / total_frames if total_frames > 0 else 0

        # Annotate the frame with ball control percentages
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame  # Return the modified frame


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draws annotations on the video frames based on object tracking and ball control.

        :param video_frames: A list of video frames to be annotated.
        :param tracks: A dictionary containing tracked objects (players, ball, referees).
        :param team_ball_control: An array indicating which team has ball control for each frame.
        :return: A list of annotated video frames.
        """
        output_video_frames = []  # Initialize a list to store annotated frames

        # Loop through each frame for annotation
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Copy the current frame to avoid modifying the original

            player_dict = tracks["players"][frame_num]  # Get players' tracking data for the current frame
            ball_dict = tracks["ball"][frame_num]        # Get ball tracking data for the current frame
            referee_dict = tracks["referees"][frame_num]  # Get referees' tracking data for the current frame

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))  # Default to red if team color is not specified
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)  # Draw player's bounding box

                # Draw a triangle above the player if they have the ball
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))  # Draw triangle in red for ball possession

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))  # Draw referee's bounding box in yellow

            # Draw ball
            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))  # Draw ball indicator in green

            # Draw team ball control information
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)  # Add the annotated frame to the output list

        return output_video_frames  # Return the list of annotated frames
