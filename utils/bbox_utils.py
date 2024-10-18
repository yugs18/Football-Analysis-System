# Function to calculate the center point of a bounding box (bbox).
# The bbox is assumed to be in the format [x1, y1, x2, y2], where:
# (x1, y1) represents the top-left corner and (x2, y2) represents the bottom-right corner of the box.
def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates
    # Return the midpoint of the bbox by averaging x1 and x2 for the x-coordinate, and y1 and y2 for the y-coordinate
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# Function to calculate the width of a bounding box.
# The width is determined by the difference between the x-coordinates (x2 - x1).
def get_bbox_width(bbox):
    x1, _, x2, _ = bbox  # Extract the x1 and x2 coordinates (ignore y values using underscores)
    # Return the difference between the right (x2) and left (x1) coordinates
    return x2 - x1

# Function to measure the Euclidean distance between two points, p1 and p2.
# The points p1 and p2 are expected to be in the form (x, y).
def measure_distance(p1, p2):
    # Compute the Euclidean distance using the distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Function to measure the difference in x and y coordinates between two points.
# The points p1 and p2 are expected to be in the form (x, y).
# This function returns the horizontal (x) and vertical (y) differences separately.
def measure_xy_distance(p1, p2):
    # Subtract the x-coordinate of p2 from the x-coordinate of p1 to get the horizontal difference
    # Subtract the y-coordinate of p2 from the y-coordinate of p1 to get the vertical difference
    return p1[0] - p2[0], p1[1] - p2[1]


# Function to compute the foot position of an object based on its bounding box.
# The foot position is taken as the midpoint of the x-coordinates and the bottom y-coordinate (y2) of the bbox.
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox  # Unpack the bounding box coordinates
    # Return the midpoint of the bottom edge for the x-coordinate, and the bottom y-coordinate (y2)
    return int((x1 + x2) / 2), int(y2)
