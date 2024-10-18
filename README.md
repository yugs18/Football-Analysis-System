# Football Analysis System

## Introduction ##

    The Football Analysis System is an advanced video analysis tool that uses computer vision and machine learning to extract key insights from football match videos. This system automates the detection and tracking of players and the ball, estimates camera movement, assigns teams, calculates player speed and distance, and determines ball possession.

    I am new to this field, and while this may be the best football analysis model I've developed so far and only one, I hope you enjoy exploring it. Feel free to improve upon this project and take it to the next level!

## Key Features: ##

    1.Player and Ball Tracking: Utilizes a pre-trained YOLO model to track players and the ball throughout the match.
    
    2.Camera Movement Estimation: Adjusts for camera shifts to maintain accurate tracking.
    
    3.Team Assignment: Assigns players to teams based on color and appearance data from the video.
    
    4.Ball Possession Detection: Automatically identifies which player is in control of the ball during the game.
    
    5.Speed and Distance Calculation: Computes the speed and distance covered by each player.

## How to Run: ##

    1. Insert your video inside the input_videos folder.

    2. Open main.py and modify line 20 to point to your video file.
        Example: Replace 'Input_Videos/08fd33_4 - Trim3.mp4' with the path to your video.
    
    3. Delete all files inside the stubs/ and output_videos/ folders to reset any previously stored data.
    
    4. Run main.py to start the video analysis process.
    
    After following these steps, the system will process your video, track player movements, estimate camera adjustments, and generate an annotated output video in the output_videos folder.