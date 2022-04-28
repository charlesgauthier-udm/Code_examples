##############################################################################
#   CHARLES GAUTHIER
##############################################################################
#=============================================================================
#   IMPORTATION DES MODULES
#=============================================================================
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import json
# IMPORTATION DES DONNÉES

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')

test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
#%% Visualisation des données
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel('({:.0f},{:.0f},{:.0f})'.format(train_labels[i][0],
                                                  train_labels[i][1],
                                                  train_labels[i][2]))
    plt.show()
#%% construction du modèle
#out_dim = train_labels[0].shape()

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation = 'relu',
                              input_shape=(32,32,3),strides = (2,2)))
#model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation = 'relu',strides = (2,2)))
#model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation = 'relu',strides = (2,2),padding='same'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation = 'relu'))
model.add(keras.layers.Dense(3,activation='linear'))


# Affichage du modèle
model.summary()
keras.utils.plot_model(model,to_file='model_cercle.png',show_shapes=True, show_layer_names=True,expand_nested=True)
#%% compilation et entrainement du modèle
model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics=['accuracy'])                   

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images,test_labels))
#with open('file.json', 'w') as f:
#    json.dump(history.history, f)
#%%
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n Accuracy = ', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#%%
# Plot training & validation accuracy values
plt.plot(history.history['loss'],'b')
plt.plot(history.history['loss'],'ob',label='Entrainement')
plt.plot(history.history['val_loss'],'r')
plt.plot(history.history['val_loss'],'or',label = 'test')
plt.title('Évolution de la fonction coût du modèle')
plt.ylabel('Valeur fonction coût')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
