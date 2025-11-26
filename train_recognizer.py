import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from sklearn.svm import SVC
from insightface.app import FaceAnalysis

DATASET = "dataset/"   # folder ze zdjęciami


def load_images():
    images = []
    labels = []
    persons = os.listdir(DATASET)

    for person in persons:
        folder = os.path.join(DATASET, person)
        if not os.path.isdir(folder):
            continue

        for img_name in os.listdir(folder):
            path = os.path.join(folder, img_name)
            img = cv2.imread(path)
            if img is None:
                continue

            images.append(img)
            labels.append(person)

    return images, labels


def extract_embeddings(images):
    detector = MTCNN()

    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    embeddings = []

    for img in images:
        faces = detector.detect_faces(img)

        if len(faces) == 0:
            embeddings.append(None)
            continue

        # wycinanie twarzy
        x, y, w, h = faces[0]["box"]
        face = img[y:y+h, x:x+w]

        # embedding
        face_embedding = app.get(face)[0].embedding
        embeddings.append(face_embedding)

    return embeddings


images, labels = load_images()
embeddings = extract_embeddings(images)

# filtrujemy zdjęcia bez twarzy
X = [e for e in embeddings if e is not None]
y = [labels[i] for i, e in enumerate(embeddings) if e is not None]

# trenujemy klasyfikator
clf = SVC(kernel="linear", probability=True)
clf.fit(X, y)

# zapisujemy do plików
pickle.dump(X, open("embeddings.pkl", "wb"))
pickle.dump(clf, open("classifier.pkl", "wb"))

print("🎉 Trening zakończony!")
