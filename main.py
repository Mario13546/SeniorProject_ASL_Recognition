# Created by Alex Pereira

# Import Libraries
import os
import cv2 as cv
import numpy as np

# Import Classes
from camera    import USBCamera
from holistic  import HolisticDetector
from directory import CreateStorage

# Gets variables for data collection
DATA_PATH    = os.path.join("MP_Data")
action_list  = ["hello", "thanks", "iloveyou"] # List of gestures
num_videos   = 30                              # Number of videos
video_frames = 30                              # Frames per video

# Create a video capture
camera = USBCamera(0)
cap    = camera.getCapture()

# Instance creation
holistic  = HolisticDetector(cap)
directory = CreateStorage(action_list, DATA_PATH, num_videos, video_frames)

# Loop through actions
for id, action in enumerate(action_list):
    # Check if data collection has been completed
    try:
        test_path = os.path.join(DATA_PATH, action, str(num_videos - 1), str(video_frames - 1) + ".npy")
        np.load(test_path)
        print("Data present for " + action)
        collect = input("Would you like to redo data collection for " + action + "? (Y/n) ")
    except:
        print("No data present for " + action)
        collect = "Y"
        pass

    # Collects data
    if (collect == "Y"):
        # Loop through videos
        for video in range(0, num_videos):
            # Loop thorugh frames
            for frame in range(0, video_frames):
                # Runs the holistic methods
                stream = holistic.readCapture()

                # Does the detection
                stream = holistic.mpDetection(stream)

                # Extracts Landmarks
                holistic.extractKeypoints()

                # Draws the positions on the image
                holistic.drawLandmarks(stream)

                # Frame collection
                holistic.collectFrames(stream, action, video, frame)

                # Exports the data
                npy_path = os.path.join(DATA_PATH, action, str(video), str(frame))
                holistic.exportData(npy_path)

                # Press q to end the program
                if ( cv.waitKey(1) == ord("q") ):
                    print("Process Ended by User")
                    cv.destroyAllWindows()
                    cap.release()

    # Asks if the user would like to continue with data selection 
    cv.destroyAllWindows()

    try:
        retry = input("Would you like to collect data for "+ action_list[id + 1] +"? (Y/n) ")
    except:
        print("Data collected for", len(action_list), "actions.")
        retry = "n"
        pass

    if (retry == "Y"):
        continue
    elif (retry == "n"):
        print("Thank you for your coperation!")
        break