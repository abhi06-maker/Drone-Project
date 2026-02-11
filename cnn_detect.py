import cv2
import numpy as np
import tensorflow as tf
import os


model_file = [f for f in os.listdir() if f.endswith(".keras")][-1]
model = tf.keras.models.load_model(model_file)

print("Loaded model:", model_file)

class_names = sorted(os.listdir("dataset"))
print("Classes:", class_names)


cap = cv2.VideoCapture(0)

IMG_SIZE = 128

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)

    
    h, w, _ = frame.shape
    size = min(h, w)
    x1 = w//2 - size//4
    y1 = h//2 - size//4
    x2 = w//2 + size//4
    y2 = h//2 + size//4

    roi = frame[y1:y2, x1:x2]

    
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    
    preds = model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    label = f"{class_names[class_id]} ({confidence:.2f})"

    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(frame, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("CNN Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
