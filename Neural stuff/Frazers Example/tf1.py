import tensorflow as tf


# The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers.

model = tf.keras.models.Sequential()

# Stacking layers is as easy as .add():

model.add(tf.keras.layers.Dense(units=64, activation='relu', input_dim=100))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Once a model is defined, configure its learning process with .compile():

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

# You can now iterate on your training data in batches:

model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate your performance in one line:

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# Get predictions on new data

classes = model.predict(x_test, batch_size=128)