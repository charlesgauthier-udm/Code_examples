# Charles Gauthier
#=============================================================================
#   IMPORTATION DES MODULES NÉCESSAIRES
#=============================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#=============================================================================
#   DÉFINITION DE FONCTIONS
#=============================================================================


#=============================================================================
#   CODE PRINCIPAL
#=============================================================================
# Importation des données
(train_images,train_labels),(test_images,test_labels) = keras.datasets.cifar10.load_data()

# On normalize les données
train_images,test_images = train_images/255.0,test_images/255.0

#%% Visualisation des données
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#%% construction du modèle
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation = 'relu',
                              input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation = 'relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation = 'relu'))
model.add(keras.layers.Dense(24,activation = 'relu'))
model.add(keras.layers.Dense(10,activation = 'softmax'))

# Affichage du modèle
model.summary()
keras.utils.plot_model(model,to_file='model_cifar10.png',show_shapes=True, show_layer_names=True,expand_nested=True)
#%% compilation et entrainement du modèle
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])                   

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images,test_labels))

# Évaluation de la performance du modèle

test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)
