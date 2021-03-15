import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
(images,labels),(_,_)=tf.keras.datasets.mnist.load_data()
images=np.expand_dims(images,-1)
datasets=tf.data.Dataset.from_tensor_slices((images,labels))

BATCH_SIZE=256
noise_dim=50
dataset=datasets.shuffle(60000).batch(BATCH_SIZE)
def generate():
    seed=tf.keras.layers.Input(shape=((noise_dim)))
    label=tf.keras.layers.Input(shape=(()))#代表单个的值
    x=tf.keras.layers.Embedding(10,50,input_length=1)(label)
    x=tf.keras.layers.concatenate([seed,x])
    x=tf.keras.layers.Dense(3*3*128,use_bias=False)(x)
    x=tf.keras.layers.Reshape((3,3,128))(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.ReLU()(x)
    x=tf.keras.layers.Conv2DTranspose(64,(3,3),padding='same',strides=(2,2),use_bias=False)#因为默认是valid，根据公式，存疑（？）
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.Relu()(x)
    x=tf.keras.layers.Conv2DTranspose(1,(3,3),strides=(2,2),padding='same',use_bias=False)
    x=layers.Activation('tanh')(x)
    model=tf.keras.Model(inputs=[seed,label],outputs=x)
    return model
gen=generate()





