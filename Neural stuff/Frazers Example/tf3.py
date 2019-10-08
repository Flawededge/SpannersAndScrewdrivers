import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

input_1 = tf.keras.Input(shape=(28, 28))
flat_1 = tf.keras.layers.Flatten()(input_1)
dense_1 = tf.keras.layers.Dense(128, activation='relu')(flat_1)
dropout_1 = tf.keras.layers.Dropout(0.2)(dense_1)
output_1 = tf.keras.layers.Dense(10, activation='softmax')(dropout_1)

model = tf.keras.models.Model(inputs=input_1, outputs=output_1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)