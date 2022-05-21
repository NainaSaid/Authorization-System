import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from commons import functions, realtime, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)



def build_model(model_name):

	global model_obj #singleton design pattern

	models = {
		'VGG-Face': VGGFace.loadModel,
		'OpenFace': OpenFace.loadModel,
		'Facenet': Facenet.loadModel,
		'Facenet512': Facenet512.loadModel,
		'DeepFace': FbDeepFace.loadModel,
		'DeepID': DeepID.loadModel,
		'Dlib': DlibWrapper.loadModel,
		'ArcFace': ArcFace.loadModel
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model()
			model_obj[model_name] = model
			#print(model_name," built")
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]




def find(img_path, db_path, model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):

	tic = time.time()
    
    

	img_paths, bulkProcess = functions.initialize_input(img_path)


	l = []

	recognizer = pickle.loads(open("D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//recognizer.pickle", "rb").read())
	le = pickle.loads(open("D://Projects//Unauthorized_Person_Recognittion//Unauthorized_person_recognition_v4//dataset//le.pickle", "rb").read())

	representation, region = represent(img_path=img_path , model_name=model_name, model=None , enforce_detection=enforce_detection, detector_backend=detector_backend , align=align , normalization=normalization)
	l.append(representation)
	preds = recognizer.predict_proba(l)
	j = np.argmax(preds)
	# proba = preds[j]
	name = le.classes_[j]
	print(name)

	img = cv2.imread(img_path)
	x, y, w, h = region
	x1, y1 = x + w, y + h
	cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
	cv2.putText(img, name, (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=1)
	cv2.imshow('Image', img)
	cv2.waitKey(0)



def represent(img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base'):


	if model is None:
		model = build_model(model_name)

	#---------------------------------

	#decide input shape
	input_shape_x, input_shape_y = functions.find_input_shape(model)

	#detect and align
	img , region = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, return_region = True
		, align = align)

	#---------------------------------
	#custom normalization

	img = functions.normalize_input(img = img, normalization = normalization)

	#---------------------------------

	#represent
	embedding = model.predict(img)[0].tolist()

	# print(reg)
	# print(embedding)

	return embedding , region