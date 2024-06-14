import numpy as np
import pandas as pd
import tensorflow
from matplotlib.pyplot import matplotlib
from tensorflow import keras
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Sequential
from mtcnn import MTCNN
import os
import cv2
import pickle
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import gradio as gr

filenames = pickle.load(open('Filenames.pkl', 'rb'))

model = load_model('model.h5')


# Define functions for facial feature extraction and face detection
def facial_feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result


def face_extractor(img_path):
    detector = MTCNN()
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    img = img[y:y + height + 25, x:x + width + 25]

    return img


# Load pre-computed features
features = pickle.load(open("features.pkl", 'rb'))


def predict(img):
    detector = MTCNN()
    # load img -> facload_img detection
    sample_img = cv2.imread(img)
    results = detector.detect_faces(sample_img)

    x, y, width, height = results[0]['box']

    face = sample_img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    # print(result)
    # print(result.shape)
    # find the cosine distance of current image with all the 8655 features
    similarity = []
    for i in range(len(features)):
        similarity.append(cosine_similarity(result.reshape(1, -1), features[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    #     print(max(similarity))
    temp_img = cv2.imread(filenames[index_pos])
    temp_img = cv2.resize(temp_img, (224, 224))

    cv2.putText(temp_img, 'Matched !! [{}]'.format(filenames[index_pos].split('\\')[-2].upper()), (10, 10),
                cv2.FONT_HERSHEY_TRIPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    return temp_img

    #     cv2.imshow('output',temp_img)
    cv2.waitKey(0)


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='filepath'),
    outputs=gr.Image(type='filepath'),
    title="Attendance Systems with Face Recognition",
    description='Load an image and see prediction'
)

interface.launch(inline=False)