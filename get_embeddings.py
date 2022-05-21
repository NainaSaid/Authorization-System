from deepface import DeepFace
import  embeddings
import os
import lightgbm as lgb
import pandas


result = embeddings.find(db_path = r"D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//", model_name="Facenet", enforce_detection = False)
print(result)
