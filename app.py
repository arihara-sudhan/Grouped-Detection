from detector import Detector
from flask import Flask, request, render_template, send_from_directory
import json
import os

app = Flask(__name__)
detector = Detector()

UPLOAD_FOLDER = './uploads'
DETECTED_FOLDER = './detected'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
boxes = None
cluster_assignments = None

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(DETECTED_FOLDER):
    os.makedirs(DETECTED_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    global boxes, cluster_assignments
    file = request.files["file"]

    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        boxes, labels, confs = detector.infer(filename)
        embeddings = detector.crop_and_get_embeddings(filename, boxes)
        cluster_assignments = detector.cluster_embeddings(embeddings)
        image_path = detector.draw_boxes_with_clusters(filename, boxes, cluster_assignments)
        return render_template("result.html", image_url=image_path.split(DETECTED_FOLDER + '/')[1]) 

    return "INVALID FILE TYPE", 400

@app.route('/detected/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)


@app.route("/get_json", methods=["GET"])
def get_json():
    global boxes, cluster_assignments
    if boxes is None or cluster_assignments is None:
        return {"error": "NO PROCESSED IMAGE FOR NOW"}, 400
    
    json_data = [
        {
            "xmin": int(box[0]),
            "ymin": int(box[1]),
            "xmax": int(box[2]),
            "ymax": int(box[3]),
            "id": int(cluster_id)
        }
        for box, cluster_id in zip(boxes, cluster_assignments)
    ]
    return json.dumps(json_data), 200

if __name__ == "__main__":
    app.run(debug=True)
