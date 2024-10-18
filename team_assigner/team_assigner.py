from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # Dictionary to store the colors of the two teams
        self.player_team_dict = {}  # Dictionary to map player IDs to their assigned team

    def get_clustering_model(self, image):
        """
        Fit a KMeans model to the image data for color clustering.

        :param image: The image from which to extract color clusters.
        :return: The fitted KMeans model.
        """
        image_2d = image.reshape(-1, 3)  # Reshape image to a 2D array of pixels
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)  # Initialize KMeans for 2 clusters
        kmeans.fit(image_2d)  # Fit the model to the pixel data

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color of a player based on their bounding box.

        :param frame: The video frame containing the player.
        :param bbox: The bounding box coordinates of the player.
        :return: The RGB color of the player.
        """
        x1, y1, x2, y2 = bbox
        image = frame[int(y1):int(y2), int(x1):int(x2)]  # Crop the player from the frame

        # Take only the top half of the player's image for color clustering
        top_half_image = image[0:int(image.shape[0] / 2), :]

        kmeans = self.get_clustering_model(top_half_image)  # Get KMeans model for clustering colors

        labels = kmeans.labels_  # Get labels for the pixels
        cluster_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])  # Reshape labels to image shape

        # Identify the cluster labels of the corners
        corner_cluster = [
            cluster_image[0, 0],       # Top-left corner pixel
            cluster_image[-1, 0],      # Bottom-left corner pixel
            cluster_image[0, -1],      # Top-right corner pixel
            cluster_image[-1, -1]      # Bottom-right corner pixel
        ]

        # Determine the non-player cluster by majority vote among corner clusters
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster  # The other cluster is the player's

        player_color = kmeans.cluster_centers_[player_cluster]  # Get the RGB color of the player

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assign colors to teams based on the detected players.

        :param frame: The video frame containing player detections.
        :param player_detections: A dictionary of detected players with their bounding boxes.
        """
        player_colors = []

        # Extract colors for each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)  # Get the player's color
            player_colors.append(player_color)  # Append the color to the list

        # Fit KMeans to the player colors to determine team colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)  # Fit the model to the player colors

        self.kmeans = kmeans  # Store the KMeans model

        # Assign team colors based on the cluster centers
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Retrieve or assign a team ID to a player based on their color.

        :param frame: The video frame containing the player.
        :param player_bbox: The bounding box of the player.
        :param player_id: The unique identifier for the player.
        :return: The team ID assigned to the player.
        """
        # Check if the player has already been assigned a team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)  # Get the player's color

        # Predict the team based on the player's color
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Convert from 0-indexed to 1-indexed

        # Special case for a specific player ID
        if player_id == 91:
            team_id = 1  # Force player with ID 91 to be assigned to team 1

        self.player_team_dict[player_id] = team_id  # Map player ID to team ID

        return team_id  # Return the assigned team ID
