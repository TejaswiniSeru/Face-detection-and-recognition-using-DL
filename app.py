from flask import Flask, render_template, request, send_file
from PIL import Image
from io import BytesIO
from src.model_utils import recognize_and_visualize, ModifiedInceptionResnetV1
import torch
import json
import base64

import os

app = Flask(__name__)

model = ModifiedInceptionResnetV1(embedding_size=128)
model.load_state_dict(torch.load('face_embedding_model.pth', map_location=torch.device('cpu')))
model.eval()

json_path = 'aggregated_embeddings.json'
with open(json_path, 'r') as json_file:
    aggregated_embeddings_dict = json.load(json_file)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        image = Image.open(image_file)
        image_path = f"static/uploads/{image_file.filename}"
        image.save(image_path)

        prediction_image = recognize_and_visualize(image_path, model, aggregated_embeddings_dict)
        prediction_path = "static/prediction.png"
        prediction_image.save(prediction_path)
        return render_template("index.html", prediction_image=prediction_image)
    else:
        return render_template("index.html")

@app.route("/static/uploads/<filename>")
def send_image(filename):
    return send_file(f"static/uploads/{filename}", mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)