from deepface import DeepFace
import  embeddings
import os
import lightgbm as lgb
import pandas


result = embeddings.find(db_path = r"D://Projects//Intership//Face_Recognittion//face_recognition_v4//dataset//", model_name="Facenet", enforce_detection = False)
print(result)

# if os.path.isdir("D://Projects//Intership//Face_Recognittion//deepface-master//deepface//dataset") == True:
#     print("yes")