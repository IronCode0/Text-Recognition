import tensorflow as tf
import time

# get dataset
mnist =tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# normalize dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test =  tf.keras.utils.normalize(x_test , axis=1)

# create model
model = tf.keras.Sequential()

# add layers to model
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense( 10, activation = tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=9)

# test loss and accuracy
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

# save model
model.save('save.model')
print(" model saved as 'save.model'")
for i in range(1,3):
    print('exit in ' + str(3 - i + 1))
    time.sleep(0.8)