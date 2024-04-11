import os
import sys
from datetime import datetime
import cv2
import numpy as np
from src.logger import logging  
from src.exception import CustomException 
from mtcnn.mtcnn import MTCNN
from insightface.src.common import face_preprocess

class TrainDatacollection:
    def __init__(self, args):
        self.args = args
        self.detector = MTCNN()

    def collectimagefromcam(self):
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():  
                raise CustomException("Error: Could not open camera")

            faces = 0
            frames = 0
            max_faces = int(self.args["faces"])

            if not (os.path.exists('artifacts')):
                os.makedirs(self.args["artifacts"])

            output_folder = os.path.join("artifacts", "images")
            if not (os.path.exists(output_folder)):
                os.makedirs(output_folder)
            
            while faces < max_faces:
                ret, frame = cam.read()
                if not ret: 
                    raise CustomException("Error: Could not read frame")

                frames += 1
                dtString = str(datetime.now().microsecond)
                bboxes = self.detector.detect_faces(frame)

                if len(bboxes) != 0:
                    max_area = 0
                    max_bbox = np.zeros(4)  
                    for box in bboxes:
                        bbox = box['box']
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) 
                        keypoints = box["keypoints"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_bbox = bbox
                            landmarks = keypoints
                            max_area = area
                    max_bbox = max_bbox[0:4]

                if frames % 3 == 0:
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((5, 2))  
                    nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                    cv2.imwrite(os.path.join(output_folder, "{}.jpg".format(dtString)), nimg)  
                    cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                    print("[INFO] {} Image Captured".format(faces + 1))
                    faces += 1
                cv2.imshow("Face detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            logging.info("Data collection completed")
        except CustomException as e:
            logging.error(str(e), exc_info=sys.exc_info())
            raise CustomException(e,sys)
        finally:
            cam.release() 
            cv2.destroyAllWindows()  



    
  
