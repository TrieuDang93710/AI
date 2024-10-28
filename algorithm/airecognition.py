# %%
import tensorflow as tf

# %%
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
# Import TensorFlow and tf.keras

from matplotlib.pyplot import imshow
from numpy.random import default_rng

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from tensorflow import keras


from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.python.keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Model

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras._tf_keras.keras.optimizers import Adam, SGD
from keras._tf_keras.keras.applications import mobilenet
from keras._tf_keras.keras.applications.mobilenet import preprocess_input

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

# %%
inputPath = r"D:\DoAnChuyenNganh_TTNT"

# %%
#load only persons with more than "number_of_images"
number_of_images = 50
aug_multiplier = 10
iter_csv = pd.read_csv(inputPath+"/lfw_allnames.csv", iterator=True, chunksize=1000)
df = pd.concat([chunk[chunk['images'] > number_of_images] for chunk in iter_csv])
df.head()

# %%
df.shape[0]

# %%
imagesFolder = inputPath+"/lfw_funneled/"
imagesFolder

# %%
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)

    y = onehot_encoded
    print(y.shape)
    return y

# %%
tr_len = (df.shape[0]*number_of_images*aug_multiplier)-(number_of_images*df.shape[0])
y_tr = ["" for x in range(tr_len)]

ts_len = df.shape[0]*number_of_images
y_ts = ["" for x in range(ts_len)]

# %%
df['name']

# %%
tr_len

# %%
rng = default_rng()
X_rand = rng.choice(number_of_images*aug_multiplier, size=number_of_images, replace=False)
X_rand.sort()
X_rand

# %%
def prepareImages(df, m):
    print("Preparing images")
    X_train = np.zeros((tr_len, 250, 250, 3))
    X_test = np.zeros((ts_len, 250, 250, 3))
    
    count = 0
    ts_idx = 0
    tr_idx = 0
    
    for personFolder in df['name']:
        print(personFolder)
        #count = 0
        rng = default_rng()
        X_rand = rng.choice(number_of_images*aug_multiplier, size=number_of_images, replace=False)
        X_rand.sort()
        rand_idx = 0
        person_images_idx = 0
        for img in os.listdir(imagesFolder+"/"+personFolder):
            
            #load images into images of size 100x100x3
            img = image.load_img(imagesFolder+personFolder+"/"+img, target_size=(250, 250, 3))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            if (person_images_idx == X_rand[rand_idx]):
                X_test[ts_idx] = x
                y_ts[ts_idx] = personFolder
                ts_idx += 1
                #print(ts_idx)
                if rand_idx<number_of_images-1:
                    rand_idx += 1
                
            else:
                X_train[tr_idx] = x
                #print("else\n")
                y_tr[tr_idx] = personFolder
                tr_idx += 1
            count += 1
            person_images_idx += 1
            #print(count)
            if (count % ((number_of_images*aug_multiplier)) == 0):
                print("Processing image: ", count, ", ", img)
                break
            
    
    return X_train, X_test

# %%
X_train,X_test = prepareImages(df, df.shape[0]*number_of_images*aug_multiplier)

y_train = prepare_labels(y_tr)

y_test = prepare_labels(y_ts)

df['name']

# %%
y_train.shape

# %%
X_train.shape

# %%
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator()
print(datagen)

# %%
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# %%
nr_classes = y_train.shape[1]
model = mobilenet.MobileNet(input_shape=(250, 250, 3), weights=None, include_top=True, alpha=1., classes=nr_classes)
model.compile(optimizer=Adam(learning_rate=0.0008), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy])
#model.load_model("AI\algorithm\working/model_25gpunotop.h5")
print(model.summary())

# %%
len(X_train)

# %%
BS=100
steps = len(X_train) / BS
steps

# %%
steps = len(X_train) / BS

# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=1)
# history = model.fit(datagen.flow(X_train, y_train, batch_size=BS), validation_data=(X_test, y_test), epochs=1, verbose=1)
# model.save(r"../working/model_25gpunotop.h5")
history = model.fit(datagen.flow(X_train, y_train, batch_size=BS), validation_data=(X_test, y_test), epochs=1, verbose=1)

# %%
import cv2
from tensorflow.python.keras.models import load_model  
import numpy as np

# %%
face_deteactor = cv2.CascadeClassifier(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\haarcascades\haarcascade_frontalface_alt.xml")

# Load model  
model = load_model(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\AI\algorithm\working\model_25gpunotop.h5")  

# Mở camera  
camera = cv2.VideoCapture(0)  

while True:  
    ok, frame = camera.read()  
    if not ok:  
        print("Failed to capture video")  
        break  
    
    # Phát hiện khuôn mặt  
    faces = face_deteactor.detectMultiScale(frame, 1.3, 5)  

    for (x, y, w, h) in faces:  
        roi = cv2.resize(frame[y:y+h, x:x+w], (250, 250))  # Resize mẫu khuôn mặt  
        roi = roi / 255.0  # Chuẩn hóa dữ liệu (nếu mô hình đã được huấn luyện với dữ liệu chuẩn hóa)  
        roi = np.reshape(roi, (-1, 250, 250, 3))  # Thay đổi kích thước về dạng (batch_size, height, width, channels)  
        
        # Dự đoán  
        result = model.predict(roi)  
        
        # Xử lý kết quả dự đoán  
        predicted_class = np.argmax(result, axis=1)  # Lấy lớp dự đoán cao nhất  
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 1)  
        cv2.putText(frame, str(predicted_class), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Vẽ nhãn  

    # Hiện thị khung hình  
    cv2.imshow('frame', frame)  
    
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break  

# Giải phóng camera  
camera.release()  
cv2.destroyAllWindows()  

# %%
models = models.load_model(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\AI\algorithm\working\model_25gpunotop.h5")

# %%
plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('Epoch')
plt.show()

# %%
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model validation categorical accuracy')
plt.ylabel('Val categorical accuracy')
plt.xlabel('Epoch')
plt.show()

# %%
os.listdir(r"D:\DoAnChuyenNganh_TTNT\testimage")

# %%
img = image.load_img(r"D:\DoAnChuyenNganh_TTNT\lfw_funneled\Serena_Williams\Serena_Williams_0052.jpg", target_size=(250, 250, 3))
x = image.img_to_array(img)
Xnew = x
#Xnew = preprocess_input(x)
Xtest = np.expand_dims(Xnew, axis=0)
ynew = model.predict(Xtest)
ynew

# %%
Xnew.shape

# %%
#model = mobilenet.MobileNet(input_shape=(224, 224, 3),include_top=True, alpha=1., weights=None, classes=y.shape[1])
#model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy',
#              metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

# %%
out_name = "model_5adamaug00002knopre"

# model.load_weights("../input/model-50/model_50.h5")

# %%
Xtest = np.expand_dims(Xnew, axis=0)
ynew = model.predict(Xtest)
ynew

# %%
ynew

# %%
# Create a converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Set quantize to true 
#converter.post_training_quantize=True
# Convert the model
tflite_model = converter.convert()
# Create the tflite model file
tflite_model_name = out_name+".tflite"
open(tflite_model_name, "wb").write(tflite_model)

# %%
#label_encoder.inverse_transform(df['name'])

# %%
model_json = model.to_json()
with open(out_name+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(out_name+".weights.h5")

# %%
tf.saved_model.save(model, out_name+"_saved")

# %%
import tarfile

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        
make_tarfile(out_name+"Tar", r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\model_50adamaug00002knopre_saved")

# %%
# Create a converter
converter = tf.lite.TFLiteConverter.from_saved_model(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\model_50adamaug00002knopre_saved")
# Set quantize to true 
#converter.post_training_quantize=True
# Convert the model
tflite_model = converter.convert()
# Create the tflite model file
tflite_model_name = out_name+"fromSaved.tflite"
open(tflite_model_name, "wb").write(tflite_model)

# %%
model.save(out_name+".h5", include_optimizer=True)

# %%
import cv2
from tensorflow.python.keras import layers, models

# %%
model = models.load_model(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\model_50adamaug00002knopre.keras")

# %%
print(model)

# %%
file_path = r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\model_5adamaug00002knopre.h5"
models.load_model(filepath=file_path, compile=True)

# %%
face_deteactor = cv2.CascadeClassifier(r"D:\DoAnChuyenNganh_TTNT\AI\algorithm\haarcascades\haarcascade_frontalface_alt.xml")

camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    faces = face_deteactor.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y: y+h, x: x+w], (250,250))
        result = np.argmax(model.predict(roi.resize((-1, 250, 250, 3))))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 50), 1)
        cv2.putText(frame, str(result))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

# %%



