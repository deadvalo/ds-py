# Détection de dommages sur une voiture
Ce projet utilise différentes approches de machine learning pour détecter les dommages sur une voiture à partir d'images.
Trois modèles de classification d'images (CNN de PyTorch, CNN de Keras et Sklearn) et une API de détection d'objets de TensorFlow sont utilisés.
Le data est récupérable à l'adresse suivante : https://www.kaggle.com/datasets/anujms/car-damage-detection

# Requirements.txt

Les paquets suivants sont nécessaires pour exécuter ce projet :

numpy==1.19.2
pandas==1.1.1
scikit-learn==0.23.2
tensorflow==2.4.0
pytorch==1.7.0
keras==2.4.3

Vous pouvez installer ces paquets en exécutant la commande suivante :

pip install -r requirements.txt

# Pytorch
PyTorch est un framework de deep learning open source qui fournit des outils pour la création et l'entraînement de modèles de réseaux de neurones..

Dans notre projet de détection de dommages sur une voiture, nous avons utilisé un modèle de réseau de neurones convolutionnel (CNN) de PyTorch pour la classification d'images. Le modèle a été entraîné sur un jeu de données de test et de validation comprenant deux classes : "dommage" et "intact".

Pour optimiser les performances de notre modèle, nous avons utilisé CUDA pour entraîner notre modèle sur le GPU . CUDA permet d'accélérer les calculs en utilisant le GPU pour traiter les opérations de calcul intensives, ce qui peut améliorer considérablement les performances de l'entraînement de modèles de deep learning.

Voici comment intaller PyTorch et CUDA pour entraîner un modèle CNN sur votre jeu de données :

Pour voir le code regarder le notebook torch_CNN.ipynb

# Tensorflow object detection api

La TensorFlow Object Detection API est un outil open source qui permet de détecter et d'étiqueter les objets dans des images et des vidéos. Il s'appuie sur des modèles de deep learning pré-entraînés qui peuvent être utilisés pour la détection d'objets en utilisant le transfert d'apprentissage.

Dans notre projet de détection de dommages sur une voiture, Nous avons utilisé l'API Object Detection de TensorFlow pour détecter les parties endommagées sur les images de voitures. Pour cela, Nous avons utilisé un outil en ligne appelé Roboflow pour étiqueter les images en ajoutant des boîtes autour des parties endommagées.

Ensuite, nous avons utilisé un modèle pré-entraîné de TensorFlow 2.0 pour entraîner votre modèle de détection d'objets en utilisant le transfert d'apprentissage.

Voici comment utiliser l'API Object Detection de TensorFlow pour détecter et étiqueter des objets dans des images :

  pip install tensorflow-object-detection
  
Pour voir le code regarder le notebook tf_object_detect_api.ipynb

# SKlearn

SKlearn est une bibliothèque de machine learning, très pratique pour la prise en main et la mise en pratique de modèles d'apprentissage. Si nous avons décidé d'inclure l'usage de cette bibliothèque dans notre projet, c'est bien pour son caractère incontournable pour toute utilisateur de machine learning. De plus, nous voulions également mettre en exergue la pertinence des CNNs comparativement à des modèles plus classiques.

Cette partie sur SKlearn assume donc le rôle d'une mesure étalon des performances d'algorithmes de machine learning.

Veuillez noter que certaines images, pour des soucis de compatibilité, ont été supprimées du dataset original. Pour reproduire à l'identique les expériences, veuillez utiliser les datasets du dossier "SK_Keras_data".

(Voir notebook "Sklearn pipeline")

Sources : <br>
-https://scikit-learn.org/stable/supervised_learning.html#supervised-learning (choix et aide à prendre en main les différents modèles de classification) <br>
-https://github.com/shukkkur/Predict-Species-from-Images (aide pour l'ajout de features)

# Keras

Keras est une branche de la blibliothèque tensorflow destinée à l'entrainement de réseaux de neurones. Elle était donc toute désignée pour notre projet de reconnaissance d'objet.

Nous avons decidé d'entraîner le modèle de CNN nommé LeNet-5 pour deux raisons:

La première est que c'est un des premiers CNNs développé pour la reconnaissance d'objet. Nous voulions donc avoir un modèle moins poussé qu'avec Pytorch pour attester non-seulement de la pertinence de ces algorithmes dans ce domaine mais pour également avoir la possibilité de comparer les performances avec des algorithmes de machine learning plus communs (cf la partie sklearn).

La seconde est que nous avons pensé qu'utiliser ce modèle serait un sympathique clin d'oeil à un grand chercheur français, M. Yann Le Cun.

Veuillez noter que certaines images, pour des soucis de compatibilité, ont été supprimées du dataset original. Pour reproduire à l'identique les expériences, veuillez utiliser les datasets du dossier "SK_Keras_data".

(Voir notebook "Keras pipeline")

Sources :<br>
-https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5082166-quest-ce-quun-reseau-de-neurones-convolutif-ou-cnn (découverte de la notion de CNN) <br>
-https://keras.io/api/ (exploration de la documentation keras pour maitrîser la bibliothèque) <br>
-https://towardsdatascience.com/convolutional-neural-network-champions-part-1-lenet-5-7a8d6eb98df6 (explication de l'architecture LeNet-5) <br>
-https://gist.github.com/samyumobi/8144b8eb041046df820531e5fe982524#file-car-damage-detection-colab-ipynb (un exemple de notebook nous permettant de nous projeter sur la démarche à suivre) <br>
