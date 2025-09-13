from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import os

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

frame_count = 0
emotion = "detecting..."
face_box = None  # to store face coordinates

def generate_frames():
    global frame_count, emotion, face_box
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 480))

        if frame_count % 3 == 0:  # analyze every 3rd frame
            try:
                analysis = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    detector_backend='opencv',
                    enforce_detection=False
                )

                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotion = analysis['dominant_emotion']
                face_box = analysis['region']

            except Exception as e:
                print("DeepFace Error:", e)
                emotion = "error"
                face_box = None

        # Draw results
        if face_box is not None:
            x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)
