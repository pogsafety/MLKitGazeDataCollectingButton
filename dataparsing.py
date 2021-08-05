# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Parsing

# %%
import glob
# import dlib
import numpy as np
import math
# import face_recognition
import random
# import cv2        
from pathlib import Path
import random
import pandas as pd
from PIL import Image, ImageDraw

dir_dist  = "/Users/user/POG/MLKitGazeDataCollectingButton/vision-quickstart/CaptureApp/"

dir_list = glob.glob("/Users/user/POG/MLKitGazeDataCollectingButton/vision-quickstart/CaptureApp/")
print(dir_list)


# %%
df = pd.read_csv(dir_list[0]+"log.csv")
df


# %%
file_name = df["count"].tolist()
len(file_name)

leftEyeleft = df["leftEyeleft"].tolist()
leftEyetop = df["leftEyetop"].tolist()
rightEyeright = df["rightEyeright"].tolist()
rightEyebottom = df["rightEyebottom"].tolist()


# %%
# rightEyebottom


# %%
import glob
# import dlib
import numpy as np
import pandas as pd
import math
from PIL import Image, ImageDraw
# import face_recognition
import random
# import cv2     
from pathlib import Path
import random
from tqdm import tqdm

################################################################################
# Config Values
#
# resolution: decides in what resolution 
#           you want to save face, lefteye, righteye
#           Ex) 64, 224
# image_type: decide in what type (in Pillow lib) you want to store
#           your face, lefteye, righteye image
#           Ex) "RGB", "L"
################################################################################

resolution=64
image_type = "RGB"
basedir = dir_dist
target = '/'

left_eye = []
right_eye = []
gaze_point = []
left_eye_right_top = []
left_eye_left_bottom = []
right_eye_right_top = []
right_eye_left_bottom = []
euler = []
face_grid = []
left_eye_grid = []
right_eye_grid = []
facepos = []

dir_name = basedir + target
df = pd.read_csv(dir_name+"log.csv")
file_name = df["count"].tolist()
im = Image.open(dir_name+"lefteye/"+str(file_name[0]).zfill(5)+".jpg").convert(image_type)
gazeX = df["gazeX"].tolist()
gazeY = df["gazeY"].tolist()
eulerX = df["eulerX"].tolist()
eulerY = df["eulerY"].tolist()
eulerZ = df["eulerZ"].tolist()
faceX = df["faceX"].tolist()
faceY = df["faceY"].tolist()
leftEyeleft = df["leftEyeleft"].tolist()
leftEyetop = df["leftEyetop"].tolist()
leftEyeright = df["leftEyeright"].tolist()
leftEyebottom = df["leftEyebottom"].tolist()
rightEyeleft = df["rightEyeleft"].tolist()
rightEyetop = df["rightEyetop"].tolist()
rightEyeright = df["rightEyeright"].tolist()
rightEyebottom = df["rightEyebottom"].tolist()


for i in tqdm(range(len(file_name))):
    left_eye_image = np.asarray(Image.open(dir_name+"lefteye/"+str(file_name[i]).zfill(5)+".jpg").convert(image_type).resize((resolution,resolution)))/255
    right_eye_image = np.asarray(Image.open(dir_name+"righteye/"+str(file_name[i]).zfill(5)+".jpg").convert(image_type).resize((resolution,resolution)))/255
    left_eye.append(left_eye_image)
    right_eye.append(right_eye_image)
    facegrid = np.genfromtxt (dir_name+"facegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    face_grid.append(facegrid)
    lefteyegrid = np.genfromtxt (dir_name+"lefteyegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    left_eye_grid.append(lefteyegrid)
    righteyegrid = np.genfromtxt (dir_name+"righteyegrid/"+str(file_name[i]).zfill(5)+".csv", delimiter=",")
    right_eye_grid.append(righteyegrid)

    gaze_point.append([float(gazeX[i]),float(gazeY[i])])
    euler.append([float(eulerX[i]), float(eulerY[i]), float(eulerZ[i])])
    facepos.append([float(faceX[i]), float(faceY[i])])
    left_eye_right_top.append([float(leftEyeright[i]), float(leftEyetop[i])])
    left_eye_left_bottom.append([float(leftEyeleft[i]), float(leftEyebottom[i])])
    right_eye_right_top.append([float(rightEyeright[i]), float(rightEyetop[i])])
    right_eye_left_bottom.append([float(rightEyeleft[i]), float(rightEyebottom[i])])
        
left_eye = np.asarray(left_eye)
right_eye = np.asarray(right_eye)
gaze_point = np.asarray(gaze_point)
face_grid = np.asarray(face_grid)
left_eye_grid = np.asarray(left_eye_grid)
right_eye_grid = np.asarray(right_eye_grid)
euler = np.asarray(euler)
facepos = np.asarray(facepos)
left_eye_right_top = np.asarray(left_eye_right_top)
left_eye_left_bottom = np.asarray(left_eye_left_bottom)
right_eye_right_top = np.asarray(right_eye_right_top)
right_eye_left_bottom = np.asarray(right_eye_left_bottom)

save_dir="/Users/user/POG/MLKitGazeDataCollectingButton/vision-quickstart/CaptureApp/"+image_type+"Data/"+target
Path(save_dir).mkdir(parents=True, exist_ok=True)
             
#save to File
np.save(save_dir+"gaze_point.npy",gaze_point)
np.save(save_dir+"left_eye.npy",left_eye)
np.save(save_dir+"right_eye.npy",right_eye)
np.save(save_dir+"face_grid.npy",face_grid)
np.save(save_dir+"left_eye_grid.npy",left_eye_grid)
np.save(save_dir+"right_eye_grid.npy",right_eye_grid)
np.save(save_dir+"euler.npy",euler)
np.save(save_dir+"facepos.npy",facepos)
np.save(save_dir+"left_eye_right_top.npy",left_eye_right_top)
np.save(save_dir+"left_eye_left_bottom.npy",left_eye_left_bottom)
np.save(save_dir+"right_eye_right_top.npy",right_eye_right_top)
np.save(save_dir+"right_eye_left_bottom.npy",right_eye_left_bottom)


# %%
left_eye_right_top


# %%
print(right_eye_grid.shape)
print(left_eye_grid.shape)
print(face_grid.shape)
print(facepos.shape)
print(left_eye_right_top.shape)


# %%
split_length = int(len(gaze_point)*0.9)
target_list = []
for i in range(len(gaze_point)):
    target_list.append([
        left_eye[i], right_eye[i], gaze_point[i], euler[i],
        face_grid[i], left_eye_grid[i], right_eye_grid[i], facepos[i],
        left_eye_right_top[i], left_eye_left_bottom[i], right_eye_right_top[i], right_eye_left_bottom[i]
    ])

random.shuffle(target_list)

train_data = target_list[:split_length]
test_data = target_list[split_length:]
np_train_data = np.asarray(train_data)
np_test_data = np.asarray(test_data)

train_left_eye_list=[]
train_right_eye_list=[]
train_gaze_point_list=[]
train_euler_list=[]
train_face_grid=[]
train_left_eye_grid=[]
train_right_eye_grid=[]
train_facepos=[]
train_left_eye_right_top = []
train_left_eye_left_bottom = []
train_right_eye_right_top = []
train_right_eye_left_bottom = []

for i in range(len(np_train_data)):
    train_left_eye_list.append(np_train_data[i][0])
    train_right_eye_list.append(np_train_data[i][1])
    train_gaze_point_list.append(np_train_data[i][2])
    train_euler_list.append(np_train_data[i][3])
    train_face_grid.append(np_train_data[i][4])
    train_left_eye_grid.append(np_train_data[i][5])
    train_right_eye_grid.append(np_train_data[i][6])
    train_facepos.append(np_train_data[i][7])
    train_left_eye_right_top.append(np_train_data[i][8])
    train_left_eye_left_bottom.append(np_train_data[i][9])
    train_right_eye_right_top.append(np_train_data[i][10])
    train_right_eye_left_bottom.append(np_train_data[i][11])
    
test_left_eye_list=[]
test_right_eye_list=[]
test_gaze_point_list=[]
test_euler_list=[]
test_face_grid=[]
test_left_eye_grid=[]
test_right_eye_grid=[]
test_facepos=[]
test_left_eye_right_top = []
test_left_eye_left_bottom = []
test_right_eye_right_top = []
test_right_eye_left_bottom = []

for i in range(len(np_test_data)):
    test_left_eye_list.append(np_test_data[i][0])
    test_right_eye_list.append(np_test_data[i][1])
    test_gaze_point_list.append(np_test_data[i][2])
    test_euler_list.append(np_test_data[i][3])
    test_face_grid.append(np_test_data[i][4])
    test_left_eye_grid.append(np_test_data[i][5])
    test_right_eye_grid.append(np_test_data[i][6])
    test_facepos.append(np_test_data[i][7])
    test_left_eye_right_top.append(np_test_data[i][8])
    test_left_eye_left_bottom.append(np_test_data[i][9])
    test_right_eye_right_top.append(np_test_data[i][10])
    test_right_eye_left_bottom.append(np_test_data[i][11])

np_train_gaze_point_list = np.asarray(train_gaze_point_list)
np_train_right_eye_list = np.asarray(train_right_eye_list)
np_train_left_eye_list = np.asarray(train_left_eye_list)
np_train_euler_list = np.asarray(train_euler_list)
np_train_face_grid = np.asarray(train_face_grid)
np_train_left_eye_grid = np.asarray(train_left_eye_grid)
np_train_right_eye_grid = np.asarray(train_right_eye_grid)
np_train_facepos = np.asarray(train_facepos)
np_train_left_eye_right_top = np.asarray(train_left_eye_right_top)
np_train_left_eye_left_bottom = np.asarray(train_left_eye_left_bottom)
np_train_right_eye_right_top = np.asarray(train_right_eye_right_top)
np_train_right_eye_left_bottom = np.asarray(train_right_eye_left_bottom)

np_test_gaze_point_list = np.asarray(test_gaze_point_list)
np_test_right_eye_list = np.asarray(test_right_eye_list)
np_test_left_eye_list = np.asarray(test_left_eye_list)
np_test_euler_list = np.asarray(test_euler_list)
np_test_face_grid = np.asarray(test_face_grid)
np_test_left_eye_grid = np.asarray(test_left_eye_grid)
np_test_right_eye_grid = np.asarray(test_right_eye_grid)
np_test_facepos = np.asarray(test_facepos)
np_test_left_eye_right_top = np.asarray(test_left_eye_right_top)
np_test_left_eye_left_bottom = np.asarray(test_left_eye_left_bottom)
np_test_right_eye_right_top = np.asarray(test_right_eye_right_top)
np_test_right_eye_left_bottom = np.asarray(test_right_eye_left_bottom)

train_dir=basedir+image_type+"Data/"+target+"train_dataset/"
Path(train_dir).mkdir(parents=True, exist_ok=True)
test_dir=basedir+image_type+"Data/"+target+"test_dataset/"
Path(test_dir).mkdir(parents=True, exist_ok=True)

np.save(train_dir+"gaze_point.npy",np_train_gaze_point_list)
np.save(train_dir+"left_eye.npy",np_train_left_eye_list)
np.save(train_dir+"right_eye.npy",np_train_right_eye_list)
np.save(train_dir+"euler.npy",np_train_euler_list)
np.save(train_dir+"face_grid.npy",np_train_face_grid)
np.save(train_dir+"left_eye_grid.npy",np_train_left_eye_grid)
np.save(train_dir+"right_eye_grid.npy",np_train_right_eye_grid)
np.save(train_dir+"facepos.npy",np_train_facepos)
np.save(train_dir+"left_eye_right_top.npy",np_train_left_eye_right_top)
np.save(train_dir+"left_eye_left_bottom.npy",np_train_left_eye_left_bottom)
np.save(train_dir+"right_eye_right_top.npy",np_train_right_eye_right_top)
np.save(train_dir+"right_eye_left_bottom.npy",np_train_right_eye_left_bottom)


np.save(test_dir+"gaze_point.npy",np_test_gaze_point_list)
np.save(test_dir+"left_eye.npy",np_test_left_eye_list)
np.save(test_dir+"right_eye.npy",np_test_right_eye_list)
np.save(test_dir+"euler.npy",np_test_euler_list)
np.save(test_dir+"face_grid.npy",np_test_face_grid)
np.save(test_dir+"left_eye_grid.npy",np_test_left_eye_grid)
np.save(test_dir+"right_eye_grid.npy",np_test_right_eye_grid)
np.save(test_dir+"facepos.npy",np_test_facepos)
np.save(test_dir+"left_eye_right_top.npy",np_test_left_eye_right_top)
np.save(test_dir+"left_eye_left_bottom.npy",np_test_left_eye_left_bottom)
np.save(test_dir+"right_eye_right_top.npy",np_test_right_eye_right_top)
np.save(test_dir+"right_eye_left_bottom.npy",np_test_right_eye_left_bottom)


# %%
loaded_test_gaze_point = np.load(test_dir+"gaze_point.npy")
loaded_test_gaze_point


# %%



