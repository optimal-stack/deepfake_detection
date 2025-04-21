from flask import Flask, render_template, request, jsonify
from model_utils import predict
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Make sure templates/index.html exists!

@app.route("/predict", methods=["POST"])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    try:
        result, overlay = predict(img)

        # Convert image to base64
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "label": result["label"],
            "real": result["real"],
            "fake": result["fake"],
            "image": img_str
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
