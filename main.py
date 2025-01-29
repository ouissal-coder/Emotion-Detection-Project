import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class EmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')  # Modifié à 6 classes
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def load_data(self, data_dir):
        images = []
        labels = []
        
        # Parcourir les dossiers des émotions
        for emotion_idx, emotion in enumerate(self.emotions):
            path = os.path.join(data_dir, emotion)
            for img_file in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (48, 48))
                        images.append(img)
                        labels.append(emotion_idx)
                except Exception as e:
                    print(f"Erreur lors du chargement de {img_path}: {str(e)}")
        
        # Convertir en numpy arrays
        X = np.array(images, dtype='float32')
        X = X.reshape(X.shape[0], 48, 48, 1)
        X /= 255.0  # Normalisation
        
        y = to_categorical(labels, num_classes=6)  # Modifié à 6 classes
        
        return X, y

    def train(self, train_dir, epochs=10, batch_size=32):
        print("Chargement des données d'entraînement...")
        X_train, y_train = self.load_data(train_dir)
        
        print("Début de l'entraînement...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        return history

    def save_model(self, filename='emotion_model.h5'):
        self.model.save(filename)
        print(f"Modèle sauvegardé sous {filename}")

    def start_webcam(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype('float32') / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = np.expand_dims(roi_gray, axis=-1)

                prediction = self.model.predict(roi_gray)
                emotion_label = self.emotions[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{emotion_label} ({confidence:.1f}%)',
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                          (0, 255, 0), 2)

            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Créer l'instance du détecteur
    detector = EmotionDetector()
    
    # Entraîner le modèle
    train_dir = r"C:\Users\Wessy\Desktop\emotion_detection\train"
    detector.train(train_dir, epochs=10)
    
    # Sauvegarder le modèle
    detector.save_model()
    
    # Démarrer la détection en temps réel
    detector.start_webcam()
