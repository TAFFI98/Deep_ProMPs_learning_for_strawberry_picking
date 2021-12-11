import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
import tensorflow_probability as tfp
tfd = tfp.distributions
from  tensorflow.keras.applications.efficientnet import EfficientNetB0,EfficientNetB1
tf.keras.backend.set_floatx('float64')



class deep_ProMPs_2dmodel_RGBD(tf.keras.Model):

    def __init__(self):
        super(deep_ProMPs_2dmodel_RGBD, self).__init__()
        # Define layers.

        self.x1 = layers.Conv2D(16, (3, 3), activation='relu', padding="same",kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.x3 = layers.Conv2D(8, (3, 3), padding="same", activation='relu',kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x4 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.x5 = layers.Conv2D(3, (3, 3), activation='relu', padding="same",kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x6 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        self.x7 = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x8= layers.MaxPooling2D(pool_size=(2, 2))
        self.x9 = layers.Dropout(0.25)
        self.x10= layers.Conv2D(16, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x11 = layers.MaxPooling2D(pool_size=(2, 2))
        self.x12 = layers.Dropout(0.25)
        self.x13= layers.Conv2D(8, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x14= layers.MaxPooling2D(pool_size=(2, 2))
        self.x15 = layers.Dropout(0.25)
        self.x16 = layers.Conv2D(4, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.x17= layers.Flatten()

        # Mean prediction.
        self.xa = tf.keras.layers.Dense(units=64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xb = tf.keras.layers.Dense(units=32, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xc = tf.keras.layers.Dense(units=16, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.mean_weights = tf.keras.layers.Dense(units=8, activation="linear")
        #Fake L prediciton
        self.xd = tf.keras.layers.Dense(units=82, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xe = tf.keras.layers.Dense(units=52, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.fakeL = tf.keras.layers.Dense(units=36, activation="linear")

        self.concatenate=Concatenate(axis=1)


    def call(self, inputs):


        R = inputs[0]
        D = inputs[1]

        R=self.x1(R)                        # (256, 256, 16)
        R=self.x2(R)                        # (128, 128, 16)
        R=self.x3(R)                        # (128, 128, 8)
        R=self.x4(R)                        # (64, 64, 8)
        R=self.x5(R)                        # (64, 64, 3)
        R=self.x6(R)                        # (32, 32, 3)
        R=self.x7(R)                        # (32, 32, 32)
        R=self.x8(R)                        # (16, 16, 32)
        #R=self.x9(R)                        # (16, 16, 32)
        R=self.x10(R)                       # (16, 16, 16)
        R=self.x11(R)                       # (8, 8, 16)
        #R=self.x12(R)                       # (8, 8, 16)
        R=self.x13(R)                       # (8, 8, 8)
        R=self.x14(R)                       # (4, 4, 8)
        #R=self.x15(R)                       # (4, 4, 8)
        R=self.x16(R)                       # (4, 4, 4)
        R=self.x17(R)                       # (64)


        D=self.x1(D)                        # (256, 256, 16)
        D=self.x2(D)                        # (128, 128, 16)
        D=self.x3(D)                        # (128, 128, 8)
        D=self.x4(D)                        # (64, 64, 8)
        D=self.x5(D)                        # (64, 64, 3)
        D=self.x6(D)                        # (32, 32, 3)
        D=self.x7(D)                        # (32, 32, 32)
        D=self.x8(D)                        # (16, 16, 32)
        #D=self.x9(D)                        # (16, 16, 32)
        D=self.x10(D)                       # (16, 16, 16)
        D=self.x11(D)                       # (8, 8, 16)
        #D=self.x12(D)                       # (8, 8, 16)
        D=self.x13(D)                       # (8, 8, 8)
        D=self.x14(D)                       # (4, 4, 8)
        #D=self.x15(D)                       # (4, 4, 8)
        D=self.x16(D)                       # (4, 4, 4)
        D=self.x17(D)                       # (64)


        concatenated = self.concatenate([R, D]) #(128)

        # 1 branch
        x1=self.xa(concatenated)                       # (16)
        x1=self.xb(x1)                                 # (16)
        x1=self.xc(x1)                                 # (16)
        mean_weights=self.mean_weights(x1)             # (8)


        return mean_weights

    def summary(self):
        R = layers.Input(shape=(480, 640, 3))
        D = layers.Input(shape=(480, 640, 3))
        model = tf.keras.Model(inputs=[R,D], outputs=self.call([R,D]))

        return model.summary()


if __name__ == "__main__":
    model=deep_ProMPs_2dmodel_RGBD()
    model.summary()