import cv2
import mediapipe as mp
import numpy as np
import os

dataPath = os.path.join('data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
noSequences = 30
sequenceLen = 30

def mediapipeDetection(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    return results

def getKeyPoints(result):
    face = np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(1404)
    leftHandLandMarks = np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rightHandLandMarks = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten()if result.pose_landmarks else np.zeros(132)
    return np.concatenate([face, leftHandLandMarks, rightHandLandMarks, pose])

if __name__ == "__main__":

    mpHolistics = mp.solutions.holistic
    mpDrawing = mp.solutions.drawing_utils

    def drawLandmarks(img, results):
        mpDrawing.draw_landmarks(img, results.face_landmarks, mpHolistics.FACEMESH_TESSELATION)
        mpDrawing.draw_landmarks(img, results.pose_landmarks, mpHolistics.POSE_CONNECTIONS)
        mpDrawing.draw_landmarks(img, results.left_hand_landmarks, mpHolistics.HAND_CONNECTIONS)
        mpDrawing.draw_landmarks(img, results.right_hand_landmarks, mpHolistics.HAND_CONNECTIONS)

    for action in actions:
        for sequence in range(noSequences):
            try:
                os.makedirs(os.path.join(dataPath, action, str(sequence)))
            except Exception as e:
                print("error happened: ", e)

    cap = cv2.VideoCapture(0)

    with mpHolistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(noSequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequenceLen):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    results = mediapipeDetection(frame, holistic)

                    # Draw landmarks
                    drawLandmarks(frame, results)
                    
                    # NEW Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(frame, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', frame)
                    
                    # NEW Export keypoints
                    keypoints = getKeyPoints(results)
                    npy_path = os.path.join(dataPath, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
    cap.release()
    cv2.destroyAllWindows()