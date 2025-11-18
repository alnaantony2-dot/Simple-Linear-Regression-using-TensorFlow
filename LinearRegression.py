import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
X=[1,2,3,4,5,6,7,8,9,10]
y=[2.5,4.5,6.5,8.5,10.5,12.5,14.5,16.5,18.5,20.5]
X=np.array(X).reshape(-1,1)
y=np.array(y).reshape(-1,1)
plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Provided data for regression")
plt.show()
model=keras.Sequential([keras.Input(shape=(1,)),layers.Dense(1)])
model.compile(optimizer='adam',loss='mean_squared_error')
history=model.fit(X,y,epochs=100,verbose=0)
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.show()
predictions=model.predict(X)
plt.scatter(X,y,label="Original Data")
plt.plot(X,predictions,color='red',label="predictions",linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regression model predictions")
plt.legend()
plt.show()
