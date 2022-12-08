# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from camera    import USBCamera
from holistic  import HolisticDetector

# Create a video capture
camera = USBCamera(0)
cap    = camera.getCapture()

# Instance creation
holistic  = HolisticDetector(cap, True)

# Main loop
while (cap.isOpened() == True):
    # Runs the holistic methods
    stream = holistic.readCapture()

    # Does the detection
    stream = holistic.mpDetection(stream)

    # Extracts Landmarks
    keypoints = holistic.extractKeypoints()

    # Runs the model predictions
    stream = holistic.modelPredictions(keypoints, stream)

    # Draws the positions on the image
    holistic.drawLandmarks(stream)

    # Shows the stream
    holistic.showFrame(stream)

    # Press q to end the program
    if ( cv.waitKey(1) == ord("q") ):
        print("Process Ended by User")
        cv.destroyAllWindows()
        cap.release()