import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

class HandTrackingApp:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.cap = cv2.VideoCapture(2)  # 2 for external

    def run(self):
        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                # BGR 2 RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Flip on horizontal
                image = cv2.flip(image, 1)
                
                # Set flag
                image.flags.writeable = False
                
                # Detections
                results = hands.process(image)
                
                # Set flag to true
                image.flags.writeable = True
                
                # RGB 2 BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Detections
                #print(results)
                
                # Rendering results
                if results.multi_hand_landmarks:
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        self.mp_drawing.draw_landmarks(image, hand, self.mp_hands.HAND_CONNECTIONS, 
                                                      self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                                      self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                     )
                
                #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)      #saving img

                cv2.imshow('Hand Tracking', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = HandTrackingApp()
    app.run()
