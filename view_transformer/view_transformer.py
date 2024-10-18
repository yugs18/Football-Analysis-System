# Import necessary libraries
import numpy as np 
import cv2

# Define a class for transforming points based on perspective view
class ViewTransformer():
    def __init__(self):
        # Set real-world dimensions of the court (in meters)
        court_width = 68  # Width of the court
        court_length = 23.32  # Length of the court

        # Define the pixel coordinates of the court's vertices in the image
        self.pixel_vertices = np.array([
            [110, 1035],  # Bottom-left corner
            [265, 275],   # Top-left corner
            [910, 260],   # Top-right corner
            [1640, 915]   # Bottom-right corner
        ])
        
        # Define the real-world coordinates of the court's vertices
        self.target_vertices = np.array([
            [0, court_width],         # Bottom-left in real-world coordinates
            [0, 0],                   # Top-left in real-world coordinates
            [court_length, 0],        # Top-right in real-world coordinates
            [court_length, court_width] # Bottom-right in real-world coordinates
        ])

        # Convert vertices to floating-point precision for use with OpenCV
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Create a perspective transformation matrix from image to real-world coordinates
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    # Function to transform a point from the image view to the real-world view
    def transform_point(self, point):
        # Convert the point to integer format
        p = (int(point[0]), int(point[1]))
        # Check if the point lies within the court's pixel vertices
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0 
        if not is_inside:
            return None  # If the point is outside the court, return None

        # Reshape the point for perspective transformation
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        # Apply the perspective transformation to the point
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)
        # Return the transformed point reshaped to a 2D array
        return tranform_point.reshape(-1, 2)

    # Method to add the transformed positions (real-world coordinates) to track objects
    def add_transformed_position_to_tracks(self, tracks):
        # Loop through each object type in the tracks (players, ball, etc.)
        for object, object_tracks in tracks.items():
            # Loop through frames and tracks of each object
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Get the adjusted position of the object (in image coordinates)
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    # Transform the position to real-world coordinates
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        # If the transformation is successful, reshape and convert to a list
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    # Store the transformed position back into the tracks
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
