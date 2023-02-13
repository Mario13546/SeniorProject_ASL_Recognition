# Created by Alex Pereira

# Import Libraries
import cv2 as cv

# Import Classes
from hands  import HandDetector
from camera import USBCamera

# Create a video capture
camera = USBCamera(0)
cap    = camera.getCapture()

# Instance creation
hands = HandDetector(maxHands = 1, detectionCon = 0.75, minTrackCon = 0.75, detection = True)

# Main loop
while (cap.isOpened() == True):
    # Runs the hand methods
    stream = hands.readCapture(cap)

    # Does the detection
    allHands, stream = hands.findHands(stream)

    # Makes a prediction
    hands.makePrediction(stream, allHands)

    # 
    hands.showFrame(stream)

    # Press q to end the program
    if ( cv.waitKey(1) == ord("q") ):
        print("Process Ended by User")
        cv.destroyAllWindows()
        cap.release()