from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle



print("[INFO] loading face embeddings...")
data = pickle.loads(open("D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//representations_facenet.pkl", "rb").read())
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print(data["embeddings"][0])


print("[INFO] training model...")
# recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer = KNeighborsClassifier()
# recognizer = RandomForestClassifier()
recognizer.fit(data["embeddings"], labels)

f = open("D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()
# write the label encoder to disk
f = open("D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()