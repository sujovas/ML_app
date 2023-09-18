from flask import Flask, render_template, request, send_file
import time
import os
import numpy as np
import cv2
import base64
from gui import process_file

app = Flask(__name__)

def generate_table_data(shot_points):
    table_data = []
    for i, point in enumerate(shot_points):
        data = {
            'name': '#' + str(i + 1),
            'value': point
        }
        table_data.append(data)
    datasum = {
        'name': 'SUM(1:' + str(i + 1) + ')',
        'value': sum(shot_points)
    }
    table_data.append(datasum)
    return table_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return "No image file uploaded."

    image = request.files['image']

    original_filename, extension = os.path.splitext(image.filename)
    processed_image_filename = original_filename + '_processed' + '.png'

    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Add your image processing code here
    # For simplicity, let's just simulate a delay
    processed_image, shot_points = process_file(image)

    # Save the processed image to a temporary file
    processed_image_path = os.path.join('static', processed_image_filename)
    cv2.imwrite(processed_image_path, processed_image)

    # Convert the processed image to base64
    with open(processed_image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    # Generate the table data based on the processed image
    tableData = generate_table_data(shot_points)

    return {
        'encoded_image': encoded_image,
        'image_filename': processed_image_filename,
        'tableData': tableData
    }


@app.route('/result')
def result():
    processed_image_filename = request.args.get('filename')
    if processed_image_filename:
        processed_image_path = os.path.join('static', processed_image_filename)
        if os.path.isfile(processed_image_path):
            return send_file(processed_image_path, mimetype='image/png')
    return "Image not found"


if __name__ == '__main__':
    app.run(debug=True)
