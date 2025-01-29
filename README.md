# 🎯 Emotion Detection Project  

## 📘 Project Overview  

Ce projet vise à détecter les émotions faciales à l'aide d'un modèle de deep learning basé sur CNN. Il utilise OpenCV pour la capture vidéo et TensorFlow/Keras pour l'entraînement et l'inférence en temps réel.  

---

## 🚀 Features  

1. **Prétraitement des données :**  
   - Chargement des images depuis des dossiers représentant chaque émotion.  
   - Conversion en niveaux de gris et normalisation des images.  
   - Redimensionnement des images à 48x48 pixels.  

2. **Modèle de Deep Learning :**  
   - Architecture CNN avec plusieurs couches convolutives et de pooling.  
   - Fonctions d'activation ReLU et softmax pour la classification.  
   - Optimisation avec Adam et fonction de perte categorical crossentropy.  
   - Utilisation de dropout pour éviter l'overfitting.  

3. **Entraînement et Sauvegarde du Modèle :**  
   - Entraînement sur un dataset d'émotions avec 6 classes.  
   - Sauvegarde du modèle au format `.h5`.  

4. **Détection en temps réel :**  
   - Utilisation d’OpenCV pour capturer des images à partir de la webcam.  
   - Détection des visages avec un classificateur Haarcascade.  
   - Prédiction de l’émotion en temps réel et affichage avec la confiance du modèle.  
   - Possibilité d’arrêter l’inférence avec la touche 'q'.  

---

## 📂 Project Structure  

```
/Emotion-Detection/
├── data/               # Dossier contenant les images d'entraînement
├── src/                # Scripts Python pour le traitement et l'entraînement
├── models/             # Modèle entraîné et sauvegardé
├── requirements.txt    # Dépendances du projet
├── README.md           # Documentation du projet
├── .gitignore          # Fichiers et dossiers ignorés
```

---

## 🔀 Dataset Overview  

Le dataset utilisé contient des images de visages annotées avec les six émotions suivantes :  

| Emotion   | Label |
|-----------|------|
| Angry     | 😠   |
| Disgust   | 🤢   |
| Fear      | 😨   |
| Happy     | 😀   |
| Neutral   | 😐   |
| Sad       | 😢   |

Les images sont stockées dans des sous-dossiers correspondant à chaque émotion.  

---

## ⚙ Setup and Usage  

### Prérequis  

- *Python 3.8+*  
- Bibliothèques : OpenCV, NumPy, TensorFlow, Keras  

### Installation Steps  

1. **Cloner le projet :**  
   ```bash
   git clone https://github.com/ouissal-coder/Emotion-Detection.git
   cd Emotion-Detection
   ```  

2. **Créer un environnement virtuel et l’activer :**  
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   ```  

3. **Installer les dépendances :**  
   ```bash
   pip install -r requirements.txt
   ```  

4. **Lancer l'entraînement du modèle :**  
   ```bash
   python main.py
   ```  

5. **Démarrer la détection en temps réel :**  
   ```bash
   python main.py --webcam
   ```  

---

## 📢 Contact  

Pour toute question, contactez-moi à : ouissal.aitourab@gmail.com.
