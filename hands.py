# Created by Alex Pereira

# Import Libraries
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from   time import sleep

# Creates the HandDetector Class
class HandDetector:
    # Constructor
    def __init__(self, maxHands = 2, detectionCon = 0.5, minTrackCon = 0.5, detection = False):
        """
        Constructor for HandDetector.
        @param maxHands
        @param modelComplexity
        @param detectionConfidence
        @param minimumTrackingConfidence
        """
        # Initiaizes the MediaPipe Hands solution
        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(static_image_mode = False,
                                        max_num_hands = maxHands,
                                        min_detection_confidence = detectionCon,
                                        min_tracking_confidence = minTrackCon)

        # Creates the drawing objects
        self.mp_drawing        = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Creates the ID lists
        self.tipIds  = [4, 8, 12, 16, 20]
        self.baseIds = [1, 5, 9, 13, 17]

        # Creates the variables for length calculations
        self.HandDistance    = 0
        self.maxHandWidth    = 0
        self.fingerDistance  = [0] * 5
        self.maxFingerLength = [0] * 5

        # Variables
        self.width, self.height, self.center = 0, 0, 0

        # Creates blank arrays
        self.sequence, self.sentence, self.predictions = [], [], []

        # Builds a TensorFlow model if told to do so
        if (detection == True):
            sleep(1.5)
            self.buildModel()

    def readCapture(self, capture):
        """
        Reads the VideoCapture capture.
        @param capture
        @return videoStream
        """
        # Reads the capture
        success, stream = capture.read()

        # If read fails, raise an error
        if not success:
            raise OSError("Camera error! Failed to start!")
        
        return stream

    def showFrame(self, frame):
        """
        Displays the frame to the screen.
        @param frame
        """
        cv.imshow("Stream", frame)

    def findHands(self, stream):
        """
        Finds hands in a stream.
        @param stream
        @return allDetectedHands
        @return annotatedStream
        """
        # Creates a blank array
        allHands = []

        # Flips the image
        stream = cv.flip(stream, 1)

        # Converts the image to RGB
        streamRGB = cv.cvtColor(stream, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(streamRGB)
        stream.flags.writeable = False

        # Gets the shape of the image
        self.height, self.width, self.center = stream.shape

        # If the results.multi_hand_landmarks array is not empty
        if (self.results.multi_hand_landmarks is not None):
            for handType, handLandmarks in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                # Generates an empty dictionary
                myHand = {}

                # Generates some empty lists
                myLandmarkList, xList, yList = [], [], []

                # Adds cordinates to the landmarkList
                for ind, landmarkList in enumerate(handLandmarks.landmark):
                    px, py, pz = int(landmarkList.x * self.width), int(landmarkList.y * self.height), int(landmarkList.z * self.width)
                    myLandmarkList.append([px, py, pz, ind])
                    xList.append(px)
                    yList.append(py)

                # Bounding Box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                boundingBox = xmin, ymin, boxW, boxH
                cx, cy = boundingBox[0] + (boundingBox[2] / 2), boundingBox[1] + (boundingBox[3] / 2)

                # Adds the values to myHand
                myHand["landmarkList"] = myLandmarkList
                myHand["boundingBox"]  = boundingBox
                myHand["center"]       = (cx, cy)
                myHand["type"]         = str(handType.classification[0].label)

                # Adds the hand data to allHands[]
                allHands.append(myHand)

                # Draw the Hand landmarks
                stream.flags.writeable = True
                self.mp_drawing.draw_landmarks( stream,
                                                handLandmarks,
                                                self.mpHands.HAND_CONNECTIONS,
                                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                self.mp_drawing_styles.get_default_hand_connections_style())

                # Draw the bounding box
                cv.rectangle(stream, (boundingBox[0] - 20, boundingBox[1] - 20),
                                    (boundingBox[0] + boundingBox[2] + 20, boundingBox[1] + boundingBox[3] + 20),
                                    (255, 0, 255), 2)

                # Writes the identifier
                cv.putText(stream, myHand["type"], (boundingBox[0] - 30, boundingBox[1] - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                # Draws on corners to the rectangle because they're cool
                colorC    = (0, 255, 0)
                l, t, adj = 30, 5, 20
                xmin, ymin, xmax, ymax = xmin - adj, ymin - adj, xmax + adj, ymax + adj
                cv.line(stream, (xmin, ymax), (xmin + l, ymax), colorC, t) # Bottom Left   (xmin, ymax)
                cv.line(stream, (xmin, ymax), (xmin, ymax - l), colorC, t) # Bottom Left   (xmin, ymax)
                cv.line(stream, (xmax, ymax), (xmax - l, ymax), colorC, t) # Bottom Right  (xmax, ymax)
                cv.line(stream, (xmax, ymax), (xmax, ymax - l), colorC, t) # Bottom Right  (xmax, ymax)
                cv.line(stream, (xmin, ymin), (xmin + l, ymin), colorC, t) # Top Left      (xmin, ymin)
                cv.line(stream, (xmin, ymin), (xmin, ymin + l), colorC, t) # Top Left      (xmin, ymin)
                cv.line(stream, (xmax, ymin), (xmax - l, ymin), colorC, t) # Top Right     (xmax, ymin)
                cv.line(stream, (xmax, ymin), (xmax, ymin + l), colorC, t) # Top Right     (xmax, ymin)

        return allHands, stream

    def buildModel(self):
        """
        Builds the model to use for predictions.
        """
        # Enter the model path
        # model_id = input("Enter model identifiers. Do not include quotation marks or the file extension. ")
        model_id = ""

        # Default model to load
        if (len(model_id) < 5):
            model_id = "default"

        # Loads the default model
        self.model = tf.keras.models.load_model("Selected Models/mnist_detection.default.h5")
        return

        # Loads the model
        try:
            self.model = tf.keras.models.load_model("Selected Models/" + model_id + ".h5")
        except:
            self.model = tf.keras.models.load_model("Selected Models/mnist_detection.default.h5")
            pass

    def makePrediction(self, stream, allHands):
        """
        Makes a prediction off the incoming stream.
        @param stream
        @param allHands
        """
        # Gets the actions with valid data
        labels = np.load("./labels.npy")

        # Checks if the allHands array is valid
        if (allHands is not None):
            # Iterates through all the hands
            for hand in allHands:
                # Extracts the bounding box size
                bb = hand["boundingBox"]
                corner    = (bb[0], bb[1])
                boxWidth  = bb[2]
                boxHeight = bb[3]

                # Makes the box a square
                if (boxWidth > boxHeight):
                    boxHeight = boxWidth
                elif (boxHeight > boxWidth):
                    boxWidth = boxHeight

                # Extracts the roi from the stream
                roi = self.extractRoi(stream, corner, boxWidth, boxHeight)

                # Grayscales and Resizes the roi
                roiGRAY = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                frame   = cv.resize  (roiGRAY, (28, 28))

                # Adds the keypoints to the list
                self.sequence.append(frame)
                #self.sequence = self.sequence[-30:]
                self.sequence = self.sequence[-1:]

                # Resizes the array to what the model is expecting
                resizedFrame = np.array(frame).flatten().reshape((28, 28, 1))
                resizedFrame = np.expand_dims(resizedFrame, axis = 0)

                # Once the sequence length is 30
                if (len(self.sequence) == 1):
                    #res = self.model.predict(np.expand_dims(self.sequence, axis = 0))[0]
                    res = self.model.predict(resizedFrame)[0]
                    print(labels[np.argmax(res)])
                    self.predictions.append(np.argmax(res))

                    # Visualizaton logic
                    if (np.unique(self.predictions[-10:])[0] == np.argmax(res)):
                        # Checks if a result is higher than a threshold value 
                        if res[np.argmax(res)] > .25: 
                            # 
                            if len(self.sentence) > 0: 
                                if labels[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(labels[np.argmax(res)])
                            else:
                                self.sentence.append(labels[np.argmax(res)])

                    if len(self.sentence) > 5: 
                        self.sentence = self.sentence[-5:]

    def extractRoi(self, img, corner: tuple, width, height):
        """
        Extracts an roi from an image given a center, width, and height.
        @param image
        @param corner (cx, cy)
        @param width
        @param height
        @return roi
        """
        # Extracts the corner point
        x, y = corner

        # Extracts the roi
        roi = img[y : y + height, x : x + width]

        return roi