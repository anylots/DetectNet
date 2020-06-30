import base64
import io
from io import BytesIO

import requests as req
from PIL import Image
from flask import Flask, request, render_template

import efficientService as service

# flask web service
app = Flask(__name__, template_folder="web")


@app.route('/detect')
def detect():
    return render_template('detect.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/detect/imageDetect', methods=['post'])
def upload():
    # step 1. receive image
    file = request.form.get('imageBase64Code')
    image_link = request.form.get("imageLink")

    if image_link:
        response = req.get(image_link)
        image = Image.open(BytesIO(response.content))
    # else:
    # image = base64.b64decode(file)

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
