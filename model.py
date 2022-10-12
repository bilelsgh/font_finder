# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from generate_dataset import get_images_labels


import os
import numpy as np
from dotenv import load_dotenv

load_dotenv("./env")
WIDTH = int( os.getenv("width") )
HEIGHT = int( os.getenv("height") )

class Model :

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.int_to_class = None

        self.load_data()
        self.create_model()


    def load_data(self):

        self.train_images, self.train_labels, self.int_to_class = get_images_labels(f"{self.dataset_path}/train")
        self.test_images, self.test_labels, _ = get_images_labels(f"{self.dataset_path}/test")

        self.train_images /= 255.0
        self.test_images /= 255.0

        print(self.train_labels[:20])

    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense( len(self.int_to_class) ))

        # Need to compile the model before using it
        self.model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
    def train_model(self):
        self.model.fit(self.train_images, self.train_labels, epochs=5)
        self.model.save_weights("data/model_weights/cp.ckpt")

    def load_model(self):
        self.model.load_weights("data/model_weights/cp.ckpt")

    def eval(self):        
        test_loss, test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
        print('\n>>> Model accuracy: {} <<'.format(test_acc)) 
        return test_acc

    def get_weights(self):
        return self.model.get_weights()

    def predict(self,img):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        prediction = probability_model.predict(img)

        index = np.where( prediction == max(prediction[0]) )[1][0] # Get the index of the maximum value

        return  self.int_to_class[index]
        

    




