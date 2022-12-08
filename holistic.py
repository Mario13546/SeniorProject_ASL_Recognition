# Created by Alex Perira

# Import Libraries
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from   time import sleep

class HolisticDetector:
    def __init__(self, capture, detection = False) -> None:
        """
        Constructor for the HolisticDetector class.
        @param
        """
        # Localizes the capture
        self.cap = capture

        # Creates variables
        self.sequence, self.sentence, self.predictions = [], [], []

        # Initializes the MediaPipe Holistic Solution
        self.mp_holistic = mp.solutions.holistic
        self.holistic    = self.mp_holistic.Holistic(static_image_mode = False,
                                                    model_complexity = 1,
                                                    smooth_landmarks = True,
                                                    enable_segmentation = False,
                                                    smooth_segmentation = True,
                                                    refine_face_landmarks = False,
                                                    min_detection_confidence = 0.5,
                                                    min_tracking_confidence = 0.5)

        # Initializes the MediaPipe Drawing Solutions
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing        = mp.solutions.drawing_utils

        # Builds a TensorFlow model if told to do so
        if (detection == True):
            sleep(1.5)
            self.buildModel()

    def readCapture(self):
        """
        Reads the VideoCapture capture.
        @return videoStream
        """
        # Reads the capture
        success, stream = self.cap.read()
        
        return stream
    
    def showFrame(self, stream):
        """
        Displays the stream.
        @param stream
        """
        # Shows the frame
        cv.imshow("MediaPipe Holistic", stream)
    
    def mpDetection(self, stream):
        """
        Runs the MediaPipe holistic detection.
        @param
        """
        # Mirrors the stream
        stream = cv.flip(stream, 1)

        # 
        stream = cv.cvtColor(stream, cv.COLOR_BGR2RGB)
        stream.flags.writeable = False

        # 
        self.results = self.holistic.process(stream)
        stream.flags.writeable = True

        #
        stream = cv.cvtColor(stream, cv.COLOR_RGB2BGR)

        return stream

    def drawLandmarks(self, stream):
        """
        @param
        """
        # Draws the landmarks onto stream
        self.mp_drawing.draw_landmarks(stream,
                                    self.results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                    self.mp_drawing.DrawingSpec(color = (80, 110, 10) , thickness = 1, circle_radius = 1),
                                    self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                                    )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(stream,
                                    self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color =(80, 22, 10) , thickness = 2, circle_radius = 4),
                                    self.mp_drawing.DrawingSpec(color =(80, 44, 121), thickness = 2, circle_radius = 2)
                                    )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(stream,
                                    self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style()
                                    )
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(stream,
                                    self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style()
                                    )
    
    def extractKeypoints(self):
        """
        @param
        """
        # Extracts the keypoint data
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in self.results.pose_landmarks.landmark]).flatten() if self.results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in self.results.face_landmarks.landmark]).flatten() if self.results.face_landmarks else np.zeros(468*3)
        lh   = np.array([[res.x, res.y, res.z] for res in self.results.left_hand_landmarks.landmark]).flatten() if self.results.left_hand_landmarks else np.zeros(21*3)
        rh   = np.array([[res.x, res.y, res.z] for res in self.results.right_hand_landmarks.landmark]).flatten() if self.results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])

    def collectFrames(self, stream, action = "hello", video = 0, frame = 0, delay = 0):
        """
        @param
        """
        # 
        if (frame == 0):
            # Prints a visual cue
            cv.putText(stream, 'Beginning Collection', (120,200), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
            cv.putText(stream, 'Collecting frames for {} Video Number {}'.format(action, video), (15, 12), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

            # Displays stream
            self.showFrame(stream)

            # Waits
            cv.waitKey(int(delay * 1000))
        else:
            # Prints a visual marker
            cv.putText(stream, 'Collecting frames for {} Video Number {}'.format(action, video), (15, 12), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

            # Displays stream
            self.showFrame(stream)

    def exportData(self, path):
        """
        @param
        """
        keypoints = self.extractKeypoints()
        np.save(path, keypoints)
    
    def buildModel(self):
        """
        @param
        """
        # Enter the model path
        model_id = input("Enter model identifiers (after the asl_detection.). Do not include quotation marks or the file extension. ")

        # Default model to load
        if (len(model_id) < 5):
            model_id = "default"

        # Loads the model
        try:
            self.model = tf.keras.models.load_model("Selected Models/" + "asl_detection." + model_id + ".h5")
        except:
            self.model = tf.keras.models.load_model("Selected Models/asl_detection.default.h5")
            pass
    
    def modelPredictions(self, keypoints, stream):
        """
        @param
        """
        # Gets the actions with valid data
        actions = np.load("actions.npy")

        # Adds the keypoints to the list
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-30:]

        # 
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis = 0))[0]
            print(actions[np.argmax(res)])
            self.predictions.append(np.argmax(res))

            # Visualizaton logic
            if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                # Checks if a result is higher than a threshold value 
                if res[np.argmax(res)] > .25: 
                    # 
                    if len(self.sentence) > 0: 
                        if actions[np.argmax(res)] != self.sentence[-1]:
                            self.sentence.append(actions[np.argmax(res)])
                    else:
                        self.sentence.append(actions[np.argmax(res)])

            if len(self.sentence) > 5: 
                self.sentence = self.sentence[-5:]

            # Displays probabilities
            stream = self.probDisplay(res, actions, stream)
        
        return stream

    def probDisplay(self, res, actions, stream):
        """
        @param
        """
        # The all important color scheme
        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

        # Adds the probabiltiy values
        for num, prob in enumerate(res):
            cv.rectangle(stream, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv.putText  (stream, actions[num], (0, 85 + num * 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            
        return stream