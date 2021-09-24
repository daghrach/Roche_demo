#===========================================================================
# #===========================================================================
# Ce modèle est un classifieur (un CNN) entrainé sur un ensemble de données 
# afin de distinguer entre les images de 6 differents animaux.
#
# Données:
# ------------------------------------------------
# Entrainement : 3000 images par classe : 18000 images en total
# Validation : 1000 images par classe : 6000 images en total
# Test : 1000 images par classe : 6000 images en total
# ------------------------------------------------
# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

import os
import time
# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Le Type de notre modéle (séquentiel)
from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilisé dans notre modèle est adam
# L'optimisateur ajuste les poids de notre modèle par descente du gradient
# Chaque optimisateur a ses propres paramètres
# On a testé plusieurs et ajuster les paramètres afin d'avoir les meilleurs résultats
from keras.optimizers import Adam

# Les types des couches utlilisées dans notre modèle
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense

# Des outils pour suivre et gérer l'entrainement de notre modèle
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, Callback

# Configuration du GPU
# la version par defaut de tensorflow de Google colab est 2, on change a la version 1
#!pip uninstall tensorflow
#!pip install tensorflow==1.15
import tensorflow as tf
#print(tf.__version__)

from keras import backend as K

# Sauvegarde du modèle
from keras.engine.saving import load_model

# Affichage des graphes 
import matplotlib.pyplot as plt

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ================VARIABLES=================
# ==========================================
# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les données
mainDataPath = "data/"
# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "train"
# Le dossier contenant les images de validation
validationPath = mainDataPath + "valid"
# Le dossier contenant les images de test
testPath = mainDataPath + "test"
# Le nom du fichier du modèle à sauvegarder
modelsPath = "Model.hdf5"


# Le nombre d'images d'entrainement et de validation
# Il faut en premier lieu identifier les paramètres du CNN qui permettent d’arriver à des bons résultats. À cette fin, la démarche générale consiste à utiliser une partie des données d’entrainement et valider les résultats avec les données de validation. Les paramètres du réseaux (nombre de couches de convolutions, de pooling, nombre de filtres, etc) devrait etre ajustés en conséquence.  Ce processus devrait se répéter jusqu’au l’obtention d’une configuration (architecture) satisfaisante. 
# Si on utilise l’ensemble de données d’entrainement en entier, le processus va être long car on devrait ajuster les paramètres et reprendre le processus sur tout l’ensemble des données d’entrainement.

training_batch_size = 32  # mini-batch de 32 a partir d'un total de 18000
validation_batch_size = 32  #  mini-batch de 32 a partir d'un total de 6000

# Configuration des  images 
image_scale = 224 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs (1: pour les images noir et blanc; 3 pour les images en couleurs (rouge vert bleu) )
images_color_mode = "rgb"  # grayscale pour les image noir et blanc; rgb pour les images en couleurs 
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entrées, ce qui correspond à la couche d'entrée du réseau

# Configuration des paramètres d'entrainement
fit_batch_size = 32 # le nombre d'images entrainées ensemble: un batch
fit_epochs = 50 # Le nombre d'époques 

# ==========================================
# ==================MODÈLE==================
# ==========================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS DU TP
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Ajuster les deux fonctions:
# 2) feature_extraction
# 3) fully_connected
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Couche d'entrée:
# Cette couche prend comme paramètre la forme des images (image_shape)
input_layer = Input(shape=image_shape)

# Partie feature extraction (ou cascade de couches d'extraction des caractéristiques)
def feature_extraction(input):

    # 1-couche de convolution avec nombre de filtre  (32)  avec la taille de la fenetre de ballaiage : 3x3 
    # 2-fonction d'activation relu 
    # 3-couche d'echantillonage (pooling) pour reduire la taille avec la taille de la fenetre de ballaiage :2x2  
    # 3-couche dropout avec 20% de probabilité
  
    x = Conv2D(32, (3, 3), padding='same')(input) 
    x = Activation("relu")(x)
    x = Conv2D(32, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x= Dropout(0.2)(x)

    # **** On répète avec un  nombre de fitres de 64 **** 
    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x= Dropout(0.2)(x)
    
    # **** On répète avec un nombre fitre de 64 **** 
    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x= Dropout(0.2)(x)

    encoded=x
    return encoded


# Partie complètement connectée (Fully Connected Layer)
def fully_connected(encoded):
    # Flatten: pour convertir les matrices en vecteurs pour la couche MLP
    # Dense: une couche neuronale simple avec le nombre de neurone ( 64)
    # fonction d'activation exp: relu
    x = Flatten(input_shape=image_shape)(encoded)
    x = Dense(64)(x)
    x = Activation("relu")(x)        
    

    # Puisque'on a une classification multi classes, la dernière couche doit être formée de 6 neurones avec une fonction d'activation softmax
    # La fonction softmax nous donne 6 valeurs une de  0 ou 1  qui correspondent au encodage onehot de la classe    
    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


# Déclaration du modèle:
# La sortie de l'extracteur des features sert comme entrée à la couche complétement connectée
model = Model(input_layer, fully_connected(feature_extraction(input_layer)))

# Affichage des paramétres du modèle
# Cette commande affiche un tableau avec les détails du modèle 
# (nombre de couches et de paramétrer ...)
model.summary()

# Compilation du modèle :
# On définit la fonction de perte ( loss='categorical_crossentropy')
# L'optimisateur utilisé avec ses paramétres (optimizer=adam(learning_rate=0.001) )
# La valeur à afficher durant l'entrainement, metrics=['accuracy'] 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les données d'entrainement en mémoire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...) 
# aléatoirement afin de rendre le modèle plus robuste à la position du sujet dans les images
# Note: On peut utiliser cette méthode pour augmenter le nombre d'images d'entrainement (data augmentation)

training_data_generator = ImageDataGenerator(
                                   rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# validation_data_generator: charge les données de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la méthode de chargement des données d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size=training_batch_size, # nombre d'images à entrainer (batch size)
    class_mode="categorical", # classement multi classe (problème de 6 classes)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage

# validation_generator: indique la méthode de chargement des données de validation
validation_generator = validation_data_generator.flow_from_directory(
    validationPath, # Place des images de validation
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images à valider
    class_mode="categorical",  # classement multi classe (problème de 6 classes)
    shuffle=True) # on "brasse" (shuffle) les données -> pour prévenir le surapprentissage


# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le modèle avec la meilleure validation accuracy ('val_acc') 
# Note: on sauvegarder le modèle seulement quand la précision de la validation s'améliore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,monitor='val_accuracy', verbose=0, save_best_only=True, mode='auto')

#on introuduit la classe qui nous permet de calculer le temps d'execution de chaque epoque
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# entrainement du modèle
classifier = model.fit(training_generator,
                       epochs=fit_epochs, # nombre d'époques
#                       batch_size=fit_batch_size, # nombre d'images entrainées ensemble
                       validation_data=validation_generator, # données de validation
                       verbose=1, # mets cette valeur ‡ 0, si vous voulez ne pas afficher les détails d'entrainement
                       callbacks=[modelcheckpoint,time_callback], # les fonctions à appeler à la fin de chaque époque (dans ce cas modelcheckpoint: qui sauvegarde le modèle,time_callback pour le temps d'execution)
                       shuffle=True)# shuffle les images 

# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'execution
#
# ***********************************************
# on calcule le temps total d'entrainement on faisant la somme de toute les epoques
m, s = divmod(sum(time_callback.times), 60)
h, m = divmod(m, 60)
print('la durée du training est de {} heures {} minutes {} secondes '.\
      format(int(h), int(m), int(s)))

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Afficher la courbe d’exactitude par époque (Training vs Validation) ainsi que la courbe de perte (loss)
#
# ***********************************************
# Plot accuracy over epochs (precision par époque)
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.axhline(y=0.9,ls= '--')
fig = plt.gcf()
plt.show()
plt.savefig('accuracy.png', dpi=300)

# Plot loss over epochs (perte par époque)
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model error')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()
plt.savefig('error.png', dpi=300)
