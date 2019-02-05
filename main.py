from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras import optimizers
import tensorflow as tf
from settings import MODEL_NAME

def loss(true_y, pred_y):
    # Temporary placeholder, create the loss function here.
    return tf.square(true_y + pred_y)


# Takes a pre-trained model from yolov2
model = load_model(MODEL_NAME)

model.layers.pop()

last_layer = model.layers[-1].output

last_layer = Conv2D(26,kernel_size=19, padding='same', name='last_layer')(last_layer)

new_model = Model(inputs = model.layers[0].input, outputs = last_layer)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

new_model.compile(loss=loss, optimizer=sgd)


