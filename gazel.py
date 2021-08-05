# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
target="/Users/user/POG/MLKitGazeDataCollectingButton/vision-quickstart/CaptureApp/RGBData/"

left_eye_right_top = np.load(target+"left_eye_right_top.npy")
left_eye_left_bottom = np.load(target+"left_eye_left_bottom.npy")
right_eye_right_top = np.load(target+"right_eye_right_top.npy")
right_eye_left_bottom = np.load(target+"right_eye_left_bottom.npy")


left_eye_right_top[:,1] = left_eye_right_top[:,1] - left_eye_left_bottom[:,1]
left_eye_right_top[:,0] = left_eye_left_bottom[:,0] - left_eye_right_top[:,0]

right_eye_right_top[:,1] = right_eye_right_top[:,1] - right_eye_left_bottom[:,1]
right_eye_right_top[:,0] = right_eye_left_bottom[:,0] - right_eye_right_top[:,0]


# %%
left_eye_size = left_eye_right_top
left_eye_size


# %%
right_eye_size = right_eye_right_top
right_eye_size


# %%
import json
import glob
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import concatenate, ZeroPadding2D, Add, add, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model
import gc
from pathlib import Path

dropout_rate=0
resolution = 64
channels = 1
target="/special/jbpark/TabS6LData/Joonbeom/train_dataset/"
model_dir = "/special/jbpark/EvalModel"

def custom_loss(y_true, y_pred): 
    #euclidean loss
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) # /205.7
    #squared loss
    #return K.sum(K.square(y_pred - y_true), axis=-1) 
    
SConv1 = Conv2D(32, (3, 3), padding = 'same', activation = 'relu')
SConv2 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')
SConv3 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu')

# Left Eye
input1 = Input(shape=(resolution, resolution,channels), name='left_eye')
x = SConv1(input1)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv2(x)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv3(x)
left_eye = MaxPooling2D(pool_size = (2, 2))(x)
left_eye = Flatten()(left_eye)

# Right Eye
input2 = Input(shape=(resolution, resolution,channels), name='right_eye')
x = SConv1(input2)
x = MaxPooling2D(pool_size = (2, 2))(x)

x = SConv2(x)
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

x = SConv3(x)
right_eye = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
right_eye = Flatten()(right_eye)

# Eyes
eyes = concatenate([left_eye, right_eye])
fc1 = Dense(64, activation='relu')(eyes)
fc2 = Dense(16, activation='relu')(fc1)
fc2 = Dropout(rate=dropout_rate)(fc2)

# Facepos
input4 = Input(shape=(1, 1, 2), name='facepos')
facepos = Flatten()(input4)

#Euler
input3 = Input(shape=(1, 1, 3), name='euler')
euler = Flatten()(input3)

# Eye size
input5 = Input(shape=(1, 1, 2), name='left_eye_size')
input6 = Input(shape=(1, 1, 2), name='right_eye_size')
left_eye_size = Flatten()(input5)
right_eye_size = Flatten()(input6)
eye_sizes = concatenate([left_eye_size, right_eye_size])

head_pose = concatenate([euler, facepos, eye_sizes])
fc_f1 = Dense(16, activation='relu')(head_pose)

# FC2, FC3
fc2 = concatenate([fc2, fc_f1])
fc2 = Dense(16, activation='relu')(fc2)
fc3 = Dense(2, activation='linear', name='pred')(fc2)

fc3 = add([fc3,facepos])
pred = fc3

model = Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=[pred])

tf.distribute.MirroredStrategy()

model.compile(loss=custom_loss, optimizer=Adam(lr=1e-3))
model.summary()

gaze_point = np.load(target+"gaze_point.npy").astype(float)
left_eye = np.load(target+"left_eye.npy").reshape(-1,resolution,resolution,channels)
right_eye = np.load(target+"right_eye.npy").reshape(-1,resolution,resolution,channels)
euler = np.load(target+"euler.npy").reshape(-1,1,1,3)
facepos = np.load(target+"facepos.npy").reshape(-1,1,1,2)
left_eye_right_top = np.load(target+"left_eye_right_top.npy")
left_eye_left_bottom = np.load(target+"left_eye_left_bottom.npy")
right_eye_right_top = np.load(target+"right_eye_right_top.npy")
right_eye_left_bottom = np.load(target+"right_eye_left_bottom.npy")


left_eye_right_top[:,1] = left_eye_right_top[:,1] - left_eye_left_bottom[:,1]
left_eye_right_top[:,0] = left_eye_left_bottom[:,0] - left_eye_right_top[:,0]

right_eye_right_top[:,1] = right_eye_right_top[:,1] - right_eye_left_bottom[:,1]
right_eye_right_top[:,0] = right_eye_left_bottom[:,0] - right_eye_right_top[:,0]

left_eye_size = left_eye_right_top.reshape(-1,1,1,2)
right_eye_size = left_eye_left_bottom.reshape(-1,1,1,2)

epoch = 1000
Path(model_dir+'/checkpoint').mkdir(parents=True, exist_ok=True)
mc = ModelCheckpoint(model_dir+'/checkpoint/gazel_shared_ver.h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
#hist = model.fit([left_eye,right_eye, euler, facepos],gaze_point, validation_split=0.1,epochs=epoch, callbacks=[es, mc])
hist = model.fit([left_eye, right_eye, euler, facepos, left_eye_size, right_eye_size,],gaze_point, validation_split=0.1,epochs=epoch, callbacks=[es, mc,reduce_lr])

Path(model_dir+'/Lmodels').mkdir(parents=True, exist_ok=True)
model.save(model_dir+'/Lmodels/gazel_shared_ver.h5')

K.clear_session()
get_ipython().run_line_magic('xdel', '-n gaze_point')
get_ipython().run_line_magic('xdel', '-n left_eye')
get_ipython().run_line_magic('xdel', '-n right_eye')
get_ipython().run_line_magic('xdel', '-n euler')
get_ipython().run_line_magic('xdel', '-n facepos')
get_ipython().run_line_magic('xdel', '-n left_eye_right_top')
get_ipython().run_line_magic('xdel', '-n left_eye_left_bottom')
get_ipython().run_line_magic('xdel', '-n right_eye_right_top')
get_ipython().run_line_magic('xdel', '-n right_eye_left_bottom')

gc.collect()
get_ipython().run_line_magic('reset', '-f')

# %% [markdown]
# ### TensorFlow Lite Conversion

# %%
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model


def custom_loss(y_true, y_pred): 
    #euclidean loss
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) # /205.7
    #squared loss
    #return K.sum(K.square(y_pred - y_true), axis=-1) 

model_dir = "/special/jbpark/EvalModel"
keras_file = model_dir+'/Lmodels/gazel_shared_ver.h5'
# Convert to TensorFlow Lite model.
model = load_model(keras_file,custom_objects={'custom_loss': custom_loss})

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model.
Path(model_dir+'/tflite').mkdir(parents=True, exist_ok=True)
with tf.io.gfile.GFile(model_dir+'/tflite/gazel_shared_ver.tflite', 'wb') as f:
    f.write(tflite_model)

# %% [markdown]
# ##### if you get error with toco_protos or toco run the above python code with terminal or convert whole file to .py file then run
# %% [markdown]
# ##### probably this error is due to my conda environment, if you use docker environment and latest tensorflow version 2.4.X you will not receive such error

# %%
import glob

model_dir = "/special/jbpark/EvalModel"
path = glob.glob(model_dir+'/tflite/gazel_shared_ver.tflite')


# %%
import tensorflow as tf
print(path[0])
tflite = tf.lite.Interpreter(model_path=path[0])
tflite.get_input_details()


# %%



