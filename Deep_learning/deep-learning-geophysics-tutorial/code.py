import h5py
import tensorflow
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,ReLU,Subtract
from tensorflow.keras.models import Model
print(tensorflow.__version__)
ftrain = h5py.File(r'E:\Datasets\noise_dataset.h5','r')
X, Y=ftrain['/X'][()],ftrain['/Y'][()]
input = Input(shape=(None,None,1))
x=Conv2D(64,3,padding='same',activation='relu')(input)
for i in range(15):
    x= Conv2D(64,3,padding='same',use_bias = False)(x)
    x= ReLU()(BatchNormalization(axis=3,momentum=0.0,epsilon=0.0001)(x)) 
x= Conv2D(1,3,padding='same',use_bias = False)(x)
model = Model(inputs=input,outputs=Subtract()([input,x]))
model.compile(optimizer="rmsprop",loss="mean_squared_error")
model.fit(X[:-1000],Y[:-1000],batch_size=32,epochs=50,shuffle=True)
Y_ = model.predict(X[-1000:])