from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load model (you can replace with your own trained model)
model = YOLO("yolov8n.pt")  # Replace with your rice model path

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run detection
            results = model(filepath)[0]
            image = cv2.imread(filepath)

            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                if "broken" in label.lower():
                    color = (0, 0, 255)  # red box for broken
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, image)

            return render_template("result.html", result_image=result_filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
