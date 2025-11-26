import cv2
import pickle
from mtcnn import MTCNN
from insightface.app import FaceAnalysis

# wczytanie modelu i klasyfikatora
clf = pickle.load(open("classifier.pkl", "rb"))

app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face["box"]
        crop = frame[y:y+h, x:x+w]

        # embedding
        info = app.get(crop)
        if len(info) == 0:
            continue

        emb = info[0].embedding.reshape(1, -1)

        # klasyfikacja osoby
        pred = clf.predict(emb)[0]
        prob = clf.predict_proba(emb).max()

        # rysujemy na ekranie
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"{pred} ({prob:.2f})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
