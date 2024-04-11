import os
import sys
import cv2
import numpy as np
import pickle
from imutils import paths
from src.logger import logging
from src.exception import CustomException
from insightface.deploy import face_model

class GenerateFaceEmbedding:
    def __init__(self, args):
        self.args = args
        self.image_size = '112,112'
        self.model = "insightface/models/model-y1-test2/model,0"
        self.threshold = 1.24
        self.det = 0
    
    def genfaceEmbedding(self):
        try:
            logging.info("GenFaceEmbedding started")
            imagePaths = list(paths.list_images(self.args['dataset']))
            embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

            knownEmbeddings = []
            knownNames= []
            total = 0

            for i, imagePath in enumerate(imagePaths):
                logging.info("[INFO] Processing image {}/{}".format(i + 1, len(imagePaths)))
                name = os.path.basename(os.path.dirname(imagePath))
                image = cv2.imread(imagePath)
                nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))
                face_embedding = embedding_model.get_feature(nimg)
                knownNames.append(name)
                knownEmbeddings.append(face_embedding)
                total += 1

            logging.info("{} faces embedded".format(total))
            embeddings_path = os.path.join(self.args['artifacts'], "embeddings.pkl")
            with open(embeddings_path, "wb") as f:
                data = {"embeddings": knownEmbeddings, "names": knownNames}
                pickle.dump(data, f)

        except Exception as e:
            logging.error("Error occurred: {}".format(e))
            raise CustomException(e, sys)


   