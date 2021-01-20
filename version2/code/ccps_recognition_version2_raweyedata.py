
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D,AveragePooling1D,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from numpy import unique
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt

X=pd.read_csv('/content/x.csv')
Patterns=pd.read_csv('/content/y.csv')

encode_dic = {'nor': 0, 
              'cyc': 1, 
              'sys': 2, 
              'str': 3, 
              'ut': 4,
              'dt': 5, 
              'us': 6, 
              'ds': 7}
decode_dic = {0: 'nor',
              1: 'cyc',
              2: 'sys',
              3: 'str',
              4: 'ut',
              5: 'dt',
              6: 'us',
              7: 'ds'}

Y= Patterns['pattern'].map(encode_dic).astype(int)


x_train=X.iloc[0:76800,1:].values
x_test=X.iloc[76800:87552,1:].values
y_train=Y.iloc[0:2400,].values
y_test=Y.iloc[2400:2736,].values



x_train= x_train.reshape((2400,32, 2))
x_test=x_test.reshape((336,32,2))

model1 = Sequential()
model1.add(Conv1D(112,3 , activation="relu",strides=1, input_shape=(32,2)))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Flatten())
model1.add(Dense(80, activation="relu"))
model1.add(Dense(8, activation = 'softmax'))
model1.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )

batch_size=64
epochs=250
model1_history=model1.fit(x_train, y_train,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )

acc1 = model1.evaluate(x_train, y_train)

print("Loss:", acc1[0], " Accuracy:", acc1[1])

val_acc1 = model1.evaluate(x_test, y_test)

print("Loss:", val_acc1[0], " Accuracy:", val_acc1[1])

pred = model1.predict(x_test)
pred_y = pred.argmax(axis=-1)

cm = confusion_matrix(y_test, pred_y)
print(cm)

import seaborn as sns
sns.heatmap(cm, annot=True)

plt.plot(model1_history.history['loss'])
plt.plot(model1_history.history['val_loss'])
plt.title('1-L-1D-CNN train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model1_history.history['accuracy'])
plt.plot(model1_history.history['val_accuracy'])
plt.title('1-L-1D-CNN train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

"""inception layer"""

input_layer=Input(shape=(32,2))
conv1=Conv1D(16,10,activation='relu',padding='same')(input_layer)
conv2=Conv1D(32,10,activation='relu',padding='same')(input_layer)
conv3=Conv1D(64,10,activation='relu',padding='same')(input_layer)
inception_layer=concatenate([conv1,conv2,conv3],axis=-1)
pool=MaxPooling1D(pool_size=2)(inception_layer)
conv4=Conv1D(112,10,activation='relu')(pool)
pool=MaxPooling1D(pool_size=2)(conv4)
flatten=Flatten()(pool)
layer=Dense(80,activation='relu')(flatten)
outer_layer=Dense(8,activation='softmax')(layer)
model2 = Model(inputs=input_layer, outputs=outer_layer)

model2.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )



batch_size=64
epochs=250
model2_history=model2.fit(x_train, y_train,
                          batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )

acc2 = model2.evaluate(x_train, y_train)

print("Loss:", acc2[0], " Accuracy:", acc2[1])

val_acc2 = model2.evaluate(x_test, y_test)

print("Loss:", val_acc2[0], " Accuracy:", val_acc2[1])

pred2 = model2.predict(x_test)
pred_y2 = pred2.argmax(axis=-1)

cm2 = confusion_matrix(y_test, pred_y2)
print(cm2)

import seaborn as sns
sns.heatmap(cm2, annot=True)

plt.plot(model2_history.history['loss'])
plt.plot(model2_history.history['val_loss'])
plt.title('Improved 1D-CNN train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model2_history.history['accuracy'])
plt.plot(model2_history.history['val_accuracy'])
plt.title('improved 1D-CNN train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

"""2-L-1D-CNN"""

model3 = Sequential()
model3.add(Conv1D(128,10 , activation="relu",strides=1,input_shape=(32,2)))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Conv1D(128,10 , activation="relu",strides=1))
model3.add(MaxPooling1D(pool_size=2))
model3.add(Flatten())
model3.add(Dense(80, activation="relu"))
model3.add(Dense(8, activation = 'softmax'))
model3.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy']
              )



batch_size=64
epochs=225
model3_history=model3.fit(x_train, y_train,
                          batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )

acc3 = model3.evaluate(x_train, y_train)

print("Loss:", acc3[0], " Accuracy:", acc3[1])

val_acc3 = model3.evaluate(x_test, y_test)

print("Loss:", val_acc3[0], " Accuracy:", val_acc3[1])

pred3 = model3.predict(x_test)
pred_y3 = pred3.argmax(axis=-1)

cm3 = confusion_matrix(y_test, pred_y3)
print(cm3)

import seaborn as sns
sns.heatmap(cm3, annot=True)

plt.plot(model3_history.history['loss'])
plt.plot(model3_history.history['val_loss'])
plt.title('2-L 1D-CNN train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model3_history.history['accuracy'])
plt.plot(model3_history.history['val_accuracy'])
plt.title('2-L 1D-CNN train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

"""3-L-1D-CNN"""

model5 = Sequential()
model5.add(Conv1D(80,3 , activation="relu",input_shape=(32,2)))
model5.add(MaxPooling1D(pool_size=2))
model5.add(Conv1D(112,3 , activation="relu"))
model5.add(MaxPooling1D(pool_size=2))
model5.add(Conv1D(64,3 , activation="relu"))
model5.add(MaxPooling1D(pool_size=1))
model5.add(Flatten())
model5.add(Dense(48,activation="relu"))
model5.add(Dense(8, activation = 'softmax'))
model5.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy']
              )



batch_size=64
epochs=250
model5_history=model5.fit(x_train, y_train,
                          batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )

acc5 = model5.evaluate(x_train, y_train)

print("Loss:", acc5[0], " Accuracy:", acc5[1])

val_acc5 = model5.evaluate(x_test, y_test)

print("Loss:", val_acc5[0], " Accuracy:", val_acc5[1])

pred5 = model5.predict(x_test)
pred_y5 = pred5.argmax(axis=-1)

cm5 = confusion_matrix(y_test, pred_y5)
print(cm5)

import seaborn as sns
sns.heatmap(cm5, annot=True)

plt.plot(model5_history.history['loss'])
plt.plot(model5_history.history['val_loss'])
plt.title('3-L 1D-CNN train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model5_history.history['accuracy'])
plt.plot(model5_history.history['val_accuracy'])
plt.title('3-L 1D-CNN train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

"""ANN"""

model4=Sequential()
model4.add(Flatten(input_shape=(32,2)))
model4.add(Dense(448,kernel_initializer='he_uniform',activation='relu',input_dim=2))
model4.add(Dense(448,kernel_initializer='he_uniform',activation='relu'))
model4.add(Dense(8,activation='softmax'))
model4.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size=64
epochs=300
model4_history=model4.fit(x_train, y_train,
                          batch_size=32,
                          epochs=epochs,
                          validation_data=(x_test, y_test)
                          )

acc4 = model4.evaluate(x_train, y_train)

print("Loss:", acc4[0], " Accuracy:", acc4[1])

val_acc4 = model4.evaluate(x_test, y_test)

print("Loss:", val_acc4[0], " Accuracy:", val_acc4[1])

pred4 = model4.predict(x_test)
pred_y4 = pred4.argmax(axis=-1)

cm4 = confusion_matrix(y_test, pred_y4)
print(cm4)

import seaborn as sns
sns.heatmap(cm4, annot=True)

plt.plot(model4_history.history['loss'])
plt.plot(model4_history.history['val_loss'])
plt.title('ANN train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(model4_history.history['accuracy'])
plt.plot(model4_history.history['val_accuracy'])
plt.title('ANN train vs validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

