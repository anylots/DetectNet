import base64
import io
from io import BytesIO

import requests as req
from PIL import Image
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap

import efficientService as service

# flask web service
app = Flask(__name__, template_folder="web")

# web UI
bootstrap = Bootstrap(app)


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/detect/process', methods=['post'])
def upload():
    # step 1. receive image
    file = request.files['file']
    image_link = request.form.get("imageLink")
    model_name = request.form.get("model")

    if not image_link.strip() and not file.filename:
        return render_template('error.html')  # check request

    if image_link.strip():
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(file)

    # step 2. detect image
    image_array = service.detect(image)

    # step 3. convert image_array to byte_array
    img = Image.fromarray(image_array, 'RGB')
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return render_template('detectOut.html', img=image_info)


@app.route('/detect/imageDetect', methods=['post'])
def process():
    # step 1. receive image url
    image_link = request.form.get("imageLink")

    if not image_link.strip():
        return "error"  # check request

    response = req.get(image_link)
    image = Image.open(BytesIO(response.content))

    # step 2. detect image
    image_array = service.detect(image)

    # step 3. convert image_array to byte_array
    img = Image.fromarray(image_array, 'RGB')
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='JPEG')

    # step 4. return image_info to page
    image_info = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
    return image_info


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8081)
