
#===========================================================================
# Dans ce script, on évalue le modèle entrainé dans 1_Modele.py
# On charge le modèle en mémoire; on charge les images; et puis on applique le modèle sur les images afin de prédire les classes

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================
# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

import os
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools    
#module pour lire les images a partir d'un path
import cv2 

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilisé pour le calcul des métriques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

#Utlilisé pour encoder les classes au format onehot
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.Session(config=config)
K.set_session(sess)

# ==========================================
# ==================MODÈLE==================
# ==========================================

#Chargement du modéle sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
Classifier: Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath         
# - number_images         (pas necessaire dans notre cas)
# - number_images_class_x (pas necessaire dans notre cas)
# - image_scale          
# - images_color_mode    
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images de test
mainDataPath = "data/"
testPath = mainDataPath + "test"
# La taille des images à classer
image_scale = 224
# La couleur des images à classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255)

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une à la fois
    color_mode=images_color_mode)# couleur des images


# ==========================================
# ===============ÉVALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth.
# Dans cette partie on lis les images a partir du disque, on code chaque image au format onehot de la classe correspondante.
# On fait un sort alphabetique de la liste des classes lu pour simuler la facon dont l'iterateur test fonctionne.
# On garde aussi le path des images dans une liste qu'on peut utiliser pour afficher les images plus tard.

y_true = []
classes=os.listdir(testPath)
classes.remove('.DS_Store')

image_paths=[]
for c in sorted(classes):                       #on fait sort alphabetique
    l = [c]*len(os.listdir(testPath+'/'+c+'/'))
    image_paths.extend([testPath+'/'+c+'/'+x for x in os.listdir(testPath+'/'+c+'/')])
    y_true.extend(l)

# the ground truth au foramat onehot
onehot_y_true = to_categorical(LabelEncoder().fit_transform(y_true))

# evaluation du modele
test_eval = Classifier.evaluate_generator(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test précision:', test_eval[1])

# Prédiction des classes des images de test
predicted_classes = Classifier.predict_generator(test_itr, verbose=1)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
predicted_classes = np.round(predicted_classes) # on arrondie le output

# Cette list contient les images bien classées
correct = []
for i in range(0, len(predicted_classes) ):
    if np.array_equal(predicted_classes[i], onehot_y_true[i]):
        correct.append(i)

# Nombre d'images bien classées
print("> %d  Ètiquettes bien classÈes" % len(correct))

# Cette list contient les images mal classées
incorrect = []
for i in range(0, len(predicted_classes) ):
    if not (np.array_equal(predicted_classes[i], onehot_y_true[i])):
        incorrect.append(i)

# Nombre d'images mal classées
print("> %d Ètiquettes mal classÈes" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 1) Afficher la matrice de confusion

#fonction pour afficher la matrice de confusion
def plot_confusion_matrix(cm, classes,title='Confusion matrix',
   cmap=plt.cm.Reds):
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center",\
               color="white" if cm[i, j] > thresh else "black")
 
   plt.gcf().subplots_adjust(bottom=0.3)
   plt.ylabel('True label')
   plt.xlabel('Predicted label') 
   plt.savefig('confusion_matrix.png', dpi=300)

# On converti en panda la liste des classe vrai et la liste des classe predite
categorical_test_labels = pd.DataFrame(onehot_y_true).idxmax(axis=1)
categorical_preds = pd.DataFrame(predicted_classes).idxmax(axis=1)
# On calcule la matrice
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)
# On affiche la matrice
plot_confusion_matrix(confusion_matrix,sorted(classes))

# 2) Extraire une image mal-classée pour chaque combinaison d'espèces - Voir l'exemple dans l'énoncé.
# ***********************************************

# Extraire une image mal classé dans chaque classe
images_to_plot=[]
# l'index a partir duquelle chaque classe commence
index=[0,1001,2001,3001,4001,5001]
for a in range(0,6):
    for b in range(0,6):
        for i in range(0, len(predicted_classes) ):          
            if (np.array_equal(onehot_y_true[i],onehot_y_true[index[a]])\
                and np.array_equal(predicted_classes[i],onehot_y_true[index[b]])):
                # On lis l'image a partir de la liste des paths
                img = cv2.imread(image_paths[i])
                # les couleurs de cv et de matplotlib sont differents
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_to_plot.append(RGB_img)
                break
# On plot

plt.figure(figsize = (10,10))
for i in range(36):
    ax = plt.subplot(6,6,i+1)
    if i  in range (30,36):
        print (i,i-30)
        ax.set_xlabel(sorted(classes)[i-30])
    if i  in  [0,6,12,18,24,30]:
        print (i,i/6)
        ax.set_ylabel(sorted(classes)[int(i/6)])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_to_plot[i])

plt.savefig('image.png', dpi=300) 
plt.show()
