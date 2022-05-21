from deepface import DeepFace
import V_Recoginizer
import os
import lightgbm as lgb
import pandas


result = V_Recoginizer.recoginize(db_path = r"D://Projects//Intership//Face_Recognittion//face_recognition_v4//dataset//", model_name="Facenet", enforce_detection = True)