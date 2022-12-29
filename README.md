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

#tensorflow object detection api

# SKlearn

SKlearn est une bibliothèque de machine learning, très pratique pour la prise en main et la mise en pratique de modèles d'apprentissage. Si nous avons décidé d'inclure l'usage de cette bibliothèque dans notre projet, c'est bien pour son caractère incontournable pour toute utilisateur de machine learning. De plus, nous voulions également mettre en exergue la pertinence des CNNs comparativement à des modèles plus classiques.

Cette partie sur SKlearn assume donc le rôle d'une mesure étalon des performances d'algorithmes de machine learning.

(Voir notebook "Sklearn pipeline")

# Keras

Keras est une branche de la blibliothèque tensorflow destinée à l'entrainement de réseaux de neurones. Elle était donc toute désignée pour notre projet de reconnaissance d'objet.

Nous avons decidé d'entraîner le modèle de CNN nommé LeNet-5 pour deux raisons:

La première est que c'est un des premiers CNNs développé pour la reconnaissance d'objet. Nous voulions donc avoir un modèle moins poussé qu'avec Pytorch pour attester non-seulement de la pertinence de ces algorithmes dans ce domaine mais pour également avoir la possibilité de comparer les performances avec des algorithmes de machine learning plus communs (cf la partie sklearn).

La seconde est que nous avons pensé qu'utiliser ce modèle serait un sympathique clin d'oeil à un grand chercheur français, M. Yann Le Cun.

(Voir notebook "Keras pipeline")

