import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten , Dense , Lambda , Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split 
import sklearn
from sklearn.utils import shuffle
from keras.models import load_model
samples = []

model = load_model('model.h5')
print('Model loaded')

with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

train_samples , validation_samples = train_test_split(samples, test_size = 0.2 )

def generator(samples, batch_size = 32):
	num_samples = len(samples)
	while(1):
		shuffle(samples)
		for offset in range(0, num_samples , batch_size):
			batch_samples =samples[offset:offset+batch_size]
			images = []
			angles = []
			correction = 0.25
			for batch_sample in batch_samples:
				image = batch_sample[0]
				img = cv2.imread(image)
				center_angle = float(batch_sample[3])
				images.append(img)
				angles.append(center_angle)
				
				left = batch_sample[1]
				right = batch_sample[2]
				
				left_image =cv2.imread(left)
				images.append(left_image)
				angles.append(center_angle + correction)
				
				right_image =cv2.imread(right)
				images.append(right_image)
				angles.append(center_angle  -correction)
			augmented_images = []
			augmented_angles = []
			for image , angle in zip(images,angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle*-1.0)
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train , y_train)


train_generator = generator(train_samples ,batch_size =32)
validation_generator = generator(validation_samples , batch_size = 32)

ch,row,col = 3,80,320

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0 ,input_shape= (160,320,3)))
model.add(Cropping2D(cropping = ((50,20) ,(0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(46,5,5,subsample = (2,2) ,activation ='relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')
history_object  = model.fit_generator(train_generator , samples_per_epoch = len(train_samples)*6 ,validation_data = validation_generator , nb_val_samples = len(validation_samples) ,   nb_epoch =2)
model.save('model.h5')

print('Model saved successfully')
print ('History Keys')
print (history_object.history.keys())

#plot
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mse loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training_set','validation_set'],loc = 'upper right')
plt.show()


