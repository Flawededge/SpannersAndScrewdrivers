#%%
import cv2 as cv
import numpy as np
import os
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

if os.path.exists('./tf_mnist/model.h5'):

    model = tf.keras.models.load_model('./tf_mnist/model.h5')

else:

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

if os.path.exists('./tf_mnist/model_weights.h5'):

    model.load_weights('./tf_mnist/model_weights.h5')

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

model.save('./tf_mnist/model.h5')
model.save_weights('./tf_mnist/model_weights.h5')

#%%

if os.path.exists('./tf_mnist/model.h5'):

    model = tf.keras.models.load_model('./tf_mnist/model.h5')

    prediction = model.predict(x_test)

    index = 5

    print('prediction: {}'.format(np.argmax(prediction[index])))

    img = x_test[index]
    img = cv.resize(img, (256, 256))
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

#%%
