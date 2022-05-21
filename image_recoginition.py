from deepface import DeepFace
import I_Recoginizer
import os
import lightgbm as lgb
import pandas


result = I_Recoginizer.find(img_path = "./ime3.jpg", db_path = r"D://Projects//Intership//Face_Recognittion//face_recognition_v4//dataset//", model_name="Facenet", enforce_detection = False)