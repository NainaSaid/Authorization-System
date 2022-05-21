from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace


img = cv2.imread("D://Projects//Intership//Face_Recognittion//Face-Recognition-using -opencv-deeplearning//images//3.jpg")
obj = RetinaFace.detect_faces("D://Projects//Intership//Face_Recognittion//Face-Recognition-using -opencv-deeplearning//images//3.jpg")

for key in obj.keys():
    identity = obj[key]
    face_area = identity['facial_area']

    cv2.rectangle(img,(face_area[2],face_area[3]),(face_area[0],face_area[1]),(0,255,0),1)


plt.imshow(img[:,:,::-1])
plt.show()

obj1 = DeepFace.verify(img_path="D://Projects//Intership//Face_Recognittion//Face-Recognition-using -opencv-deeplearning//images//3.jpg",db_path="D://Projects//Intership//Face_Recognittion//deepface-master//deepface//dataset",model_name="ArcFace",detector_backend="retinaface")
print(obj1)