# Détection de dommages sur une voiture
Ce projet utilise différentes approches de machine learning pour détecter les dommages sur une voiture à partir d'images.
Trois modèles de classification d'images (CNN de PyTorch, CNN de Keras et Sklearn) et une API de détection d'objets de TensorFlow sont utilisés.

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
PyTorch est un framework de deep learning open source qui fournit des outils pour la création et l'entraînement de modèles de réseaux de neurones. .

Dans notre projet de détection de dommages sur une voiture, nous avons utilisé un modèle de réseau de neurones convolutionnel (CNN) de PyTorch pour la classification d'images. Le modèle a été entraîné sur un jeu de données de test et de validatiob comprenant deux classes : "dommage" et "intact".

Pour optimiser les performances de notre modèle, vous avons utilisé CUDA (Compute Unified Device Architecture) pour entraîner votre modèle sur le GPU . CUDA permet d'accélérer les calculs en utilisant le GPU pour traiter les opérations de calcul intensives, ce qui peut améliorer considérablement les performances de l'entraînement de modèles de deep learning.

Voici comment intaller PyTorch et CUDA pour entraîner un modèle CNN sur votre jeu de données :
  
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  
# tensorflow object detection api
La TensorFlow Object Detection API est un outil open source TensorFlow par Google qui permet de détecter et d'étiqueter les objets dans des images et des vidéos. Il s'appuie sur des modèles de deep learning pré-entraînés qui peuvent être utilisés pour la détection d'objets en utilisant le transfert d'apprentissage.

Dans notre projet de détection de dommages sur une voiture, Nous avons utilisé l'API Object Detection de TensorFlow pour détecter les parties endommagées sur les images de voitures. Pour cela, Nous avons utilisé un outil en ligne appelé Roboflow pour étiqueter les images en ajoutant des boîtes autour des parties endommagées.

Ensuite, nous avons utilisé un modèle pré-entraîné de TensorFlow 2.0 pour entraîner votre modèle de détection d'objets en utilisant le transfert d'apprentissage.

Voici comment utiliser l'API Object Detection de TensorFlow pour détecter et étiqueter des objets dans des images :

  pip install tensorflow-object-detection

# sklearn

# keras

