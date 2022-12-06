# Created by Alex Perira

# Import Libraries
import cv2 as cv
import numpy as np
import mediapipe as mp

class HolisticDetector:
    def __init__(self, capture) -> None:
        """
        Constructor for the HolisticDetector class.
        @param
        """
        # Localizes the capture
        self.cap = capture

        # Initializes the MediaPipe Holistic Solution
        self.mp_holistic       = mp.solutions.holistic
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
        @param
        """
        # Shows the frame
        cv.imshow("MediaPipe Holistic", stream)
    
    def mpDetection(self, stream):
        """
        Runs
        @param
        """
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

    def collectFrames(self, stream, action, video, frame, delay):
        # Flips the stream for display
        stream = cv.flip(stream, 1)

        # 
        if (frame == 0):
            cv.putText(stream, 'Beginning Collection', (120,200), 
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv.LINE_AA)
            cv.putText(stream, 'Collecting frames for {} Video Number {}'.format(action, video), (15, 12), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

            # Displays stream
            self.showFrame(stream)

            # Waits
            cv.waitKey(int(delay * 1000))
        else: 
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