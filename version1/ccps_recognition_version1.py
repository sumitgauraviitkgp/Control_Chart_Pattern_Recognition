

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D,AveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt

data=pd.read_csv('/content/ccps_dataset.csv')

x=data.iloc[:,0:2].values
y=data.iloc[:,2].values


print(x.shape)

x = x.reshape(x.shape[0],x.shape[1] , 1)
print(x.shape)

print(unique(y))

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15)


"""Simple 1D-CNN network"""

model1 = Sequential()
model1.add(Conv1D(64,1 , activation="relu", input_shape=(2,1)))
model1.add(MaxPooling1D(pool_size=1))
model1.add(Flatten())
model1.add(Dense(128, activation="relu"))
model1.add(Dense(8, activation = 'softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )

print(model1.summary())

batch_size=32
epochs=100
model1_history=model1.fit(xtrain, ytrain,
                          batch_size,
                          epochs=epochs,
                          validation_data=(xtest, ytest)
                          )

acc1 = model1.evaluate(xtrain, ytrain)

print("Loss:", acc1[0], " Accuracy:", acc1[1])

pred = model1.predict(xtest)
pred_y = pred.argmax(axis=-1)

cm = confusion_matrix(ytest, pred_y)
print(cm)

plt.plot(model1_history.history['loss'])
plt.plot(model1_history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

"""Conv1D with inception layer"""

input_layer=Input(shape=(2,1))
conv1=Conv1D(32,1,activation='relu',padding='same')(input_layer)
conv2=Conv1D(32,2,activation='relu',padding='same')(input_layer)
inception_layer=concatenate([conv1,conv2],axis=-1)
conv3=Conv1D(64,1,activation='relu')(inception_layer)
pool=AveragePooling1D(pool_size=2)(conv3)
flatten=Flatten()(pool)
layer=Dense(64,activation='relu')(flatten)
outer_layer=Dense(8,activation='softmax')(layer)
model2 = Model(inputs=input_layer, outputs=outer_layer)

model2.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )

print(model2.summary())

batch_size=32
epochs=100
model2_history=model2.fit(xtrain, ytrain,
                          batch_size,
                          epochs=epochs,
                          validation_data=(xtest, ytest)
                          )

acc2 = model2.evaluate(xtrain, ytrain)

print("Loss:", acc2[0], " Accuracy:", acc[1])

plt.plot(model2_history.history['loss'])
plt.plot(model2_history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

