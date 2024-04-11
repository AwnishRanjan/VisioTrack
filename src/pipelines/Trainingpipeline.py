from src.components.datacollection import TrainDatacollection
from src.components.embedding import GenerateFaceEmbedding
import sys
sys.path.append('/Users/awnishranjan/Developer/currentproject')


if __name__=='__main__':

    args = { "faces": 10, "artifacts": "artifacts" }
    data_collector = TrainDatacollection(args)
    data_collector.collectimagefromcam()

    args1 = {"dataset": "artifacts/images", "artifacts": "artifacts"}
    face_embedding_generator = GenerateFaceEmbedding(args1)
    face_embedding_generator.genfaceEmbedding()  # somelibraries issues are here I am on it !!!!! 

    # check the full project later 
    

