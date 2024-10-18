import sys

# Append the parent directory to the system path to allow importing modules from it.
sys.path.append("../")

# Import helper functions for calculating the center of a bounding box and measuring distances.
from utils import get_center_of_bbox, measure_distance

# Define a class that assigns the ball to a player based on proximity.
class PlayerBallAssign():
    def __init__(self):
        # Set the maximum allowable distance between a player and the ball for assignment.
        self.max_player_ball_distance = 70

    # Function to assign the ball to the closest player within a certain distance.
    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assigns the ball to the closest player within a specified distance.

        Parameters:
        players (dict): A dictionary of players with their IDs as keys and bounding boxes as values.
        ball_bbox (tuple): A tuple representing the bounding box of the ball.

        Returns:
        int or None: The ID of the closest player to the ball, or None if no player is close enough.
        """
        # Get the center position of the ball's bounding box.
        ball_position = get_center_of_bbox(ball_bbox)

        # Initialize variables to track the minimum distance and the assigned player.
        min_dis = float('inf')  # Use infinity to ensure any valid distance is smaller.
        assigned_player = None  # None indicates no player is assigned initially.

        # Loop over all players to calculate the distance between each player and the ball.
        for player_id, player in players.items():
            player_bbox = player["bbox"]  # Get the player's bounding box.

            # Calculate the distance between the ball and the player's left foot.
            left_foot_distance = measure_distance((player_bbox[0], player_bbox[3]), ball_position)

            # Calculate the distance between the ball and the player's right foot.
            right_foot_distance = measure_distance((player_bbox[2], player_bbox[3]), ball_position)

            # Select the smaller distance between the two feet and the ball.
            distance = min(left_foot_distance, right_foot_distance)

            # If the distance is less than the maximum threshold and less than the current minimum distance:
            if distance < self.max_player_ball_distance:
                if distance < min_dis:
                    # Update the minimum distance and assign the player as the closest one to the ball.
                    min_dis = distance
                    assigned_player = player_id

        # Return the ID of the assigned player who is closest to the ball.
        return assigned_player
