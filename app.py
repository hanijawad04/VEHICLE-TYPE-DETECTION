from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from yolo_detect import detect_vehicles
from video_detect import detect_vehicles_in_video  # or from yolo_detect if merged


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/results"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/view')
def view_images():
    return render_template('view.html', images=os.listdir(app.config['UPLOAD_FOLDER']))

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file.filename == '':
        return 'No file selected'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    file.save(filepath)

    detections = detect_vehicles(filepath, result_path)

    return render_template('result.html', filename=filename, detections=detections)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    if file.filename == '':
        return 'No video selected'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result_path = os.path.join(app.config['OUTPUT_FOLDER'], "processed_" + filename)
    file.save(filepath)

    detect_vehicles_in_video(filepath, result_path)

    return render_template('video_result.html', video_path=result_path)


if __name__ == '__main__':
    app.run(debug=True)
