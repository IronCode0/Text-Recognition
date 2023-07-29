import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
# get dataset
print('getting dataset')
mnist =tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# normalize dataset
print('normalize dataset')
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test =  tf.keras.utils.normalize(x_test , axis=1)

# load model
print('loading model')
new_model = tf.keras.models.load_model('save2.model')
predictions = new_model.predict(x_test)



# testing

#while True :
print('------------------------------------')
for i in range(10):
    print('enter X to exit')
    p = input('Enter a Index (number) to Predict its value : ')
    if (p == 'X') or (p == 'x'):
        break
    plt.imshow(x_test[int(p)],cmap=plt.cm.binary)
    plt.show()
    result = np.argmax(predictions[int(p)])
    print('predictions of index ' + str(p) + ' is ' + str(result))
    print('------------------------------------')


