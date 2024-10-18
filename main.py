# Import necessary modules for reading, saving videos, tracking, assigning teams, estimating camera movement, etc.
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssign
from camera_movement_estimator import CameraMovement
from view_transformer import ViewTransformer
from speed_and_distances import SpeedAndDistance
import numpy as np
import cv2

# Main function to orchestrate the overall process of tracking, camera movement estimation, team assignment, etc.
def main():
    """
    Main function to orchestrate the overall video processing pipeline.
    It reads the video, tracks objects, estimates camera movement, assigns teams,
    and saves an annotated output video.
    """
    # Read the input video file and extract frames
    video_frames = read_video('Input_Videos/08fd33_4 - Trim.mp4')

    # Initialize the object tracker using the specified model
    tracker = Tracker('model/best.pt')

    # Get object tracks from the video frames (may read from a pre-saved stub to avoid recalculating)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,  # Read from a previously saved stub to save computation time
        stub_path="stubs/track_stub.pkl"
    )

    # Optionally save a cropped image of a tracked player (commented out for now)
    # Uncomment and customize this block to save player images if needed.
    # for track_id, player in tracks["players"][0].items():
    #     bounding_box = player["bbox"]
    #     frame = video_frames[0]
    #     x1, y1, x2, y2 = bounding_box
    #     cropped_image = frame[int(y1): int(y2), int(x1): int(x2)]
    #     cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)
    #     break  # Stop after saving the first player's cropped image

    # Add positions (bounding boxes) to the tracked objects
    tracker.add_position_to_tracks(tracks)

    # Initialize the camera movement estimator using the first video frame
    camera_movement_estimator = CameraMovement(video_frames[0])
    
    # Estimate camera movement across frames (reading from a stub to optimize performance if available)
    camera_movement_per_frame = camera_movement_estimator.get_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    
    # Adjust the positions of tracked objects according to the camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Initialize the view transformer to correct for perspective distortion
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate the ball's position to smooth its movement between frames
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Initialize the speed and distance estimator for tracked players
    speed_and_distance_estimator = SpeedAndDistance()
    
    # Add calculated speed and distance data to the player tracks
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Initialize the team assigner for determining which player belongs to which team
    team_assigner = TeamAssigner()
    
    # Assign team colors based on the first frame and track data
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    # Loop through all frames and assign teams to players based on bounding box and appearance
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
       
    # Initialize the player-ball assigner to detect which player has possession of the ball
    player_ball_assigner = PlayerBallAssign()
    team_ball_control = []

    # Loop through each frame to assign ball possession to a player
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox)

        # If a player is assigned, mark them as having possession of the ball
        if assigned_player is not None:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            # If no player is assigned, assume the previous frame's team still has possession
            team_ball_control.append(team_ball_control[-1])

    # Convert the ball control list to a numpy array for easier processing later
    team_ball_control = np.array(team_ball_control)

    # Draw annotations (like bounding boxes, player IDs, ball possession, etc.) on the video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Overlay camera movement data onto the video frames
    output_video_frames = camera_movement_estimator.draw_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance information on the video frames
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the final annotated video to an output file
    save_video(output_video_frames, 'output_videos/video.avi')


# Entry point for the program
if __name__ == "__main__":
    main()
