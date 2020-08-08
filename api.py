import os
os.system("apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender1")
import cv2
import flask
import base64
import json
import re
import numpy as np
import tensorflow as tf
from flask import request, jsonify, send_from_directory, render_template
from main import preprocess, morph
from mtcnn.mtcnn import MTCNN

app = flask.Flask(__name__)
app.config["DEBUG"] = False

global graph
graph = tf.get_default_graph()
global model
model = MTCNN()
global filename
filename = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/preprocess/', methods=['POST'])
def pProcess():
    try:
        f = request.files['img']
        ext = f.filename.split('.')[-1]
        f.save('temp.'+ext)

        resp = preprocess('temp.'+ext, model)
    except:
        return "Some error occurred"
    return jsonify(resp)

@app.route('/api/morph/', methods=['POST'])
def mMorph():
    try:
        s = json.dumps(request.form)
        s = re.sub("\"\[","[",s)
        s = re.sub("\]\"","]",s)
        s = re.sub("\\\\\"","\"",s)
        s = json.loads(s)
        img1 = base64.b64decode(s['img1'][23:])
        img2 = base64.b64decode(s['img2'][23:])
        with open("img1.jpeg", 'wb') as f:
            f.write(img1)
        with open("img2.jpeg", 'wb') as f:
            f.write(img2)
        resp = morph("img1.jpeg", "img2.jpeg", float(s['l'])/100, np.array(s['t1']), np.array(s['t2']))
        return jsonify({"img":resp})
    except:
        return "Some error ocuurred"

