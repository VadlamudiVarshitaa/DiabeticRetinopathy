import base64
import os
import flask
import shutil
from PIL import Image
import numpy as np
import torch
import pandas as pd
from flask import request

from prediction import model_prediction
from visualization_attribution import create_vis
from display_image import show_img
from uncertainty import uncertainty_measure
from preprocessing import preprocess

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST', ''])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html', result={'original': '', 'processed': ''}, std_dict={}, prob_mean = [],
                                     dr_prediction='')

    if flask.request.method == 'POST':

        file = request.files['file']
        print(type(file))

        def create_dir(path):
            if (os.path.isdir(path)):
                shutil.rmtree(path)
            else:
                os.makedirs(path)

        path = os.path.join(os.getcwd(), 'Result')
        create_dir(path)
        datapath = os.path.join(path, 'data')
        create_dir(datapath)

        file.save(os.path.join(datapath, 'image.png'))
        imagefile = Image.open(os.path.join(datapath, 'image.png'))
        imagefile_resized = imagefile.resize((224, 224), Image.BILINEAR)
        imagefile_resized.save(os.path.join(datapath, 'image.png'))

        with open(os.path.join(datapath, 'image.png'), 'rb') as f1:
            image_string = base64.b64encode(f1.read())
            s_image_string = image_string.decode("utf-8", "ignore")

        preprocess() 
        prob_mean = model_prediction()
        torch.cuda.empty_cache()
        f = create_vis() 
        show_img() 
        std = uncertainty_measure() 

        outputpath = os.path.join(path, 'output/Integrated_Gradient.png')

        with open(outputpath, 'rb') as f1:
            o_image_string = base64.b64encode(f1.read())
            oo_image_string = o_image_string.decode("utf-8", "ignore")

        std_dict = std.to_dict()
        dr_prediction = np.argmax(prob_mean)

        return flask.render_template('main.html', original_input={'Image': file},
                                     result={'original': s_image_string, 'processed': oo_image_string, 'std': std},
                                     std_dict=std_dict, prob_mean=prob_mean, dr_prediction=dr_prediction
                                     )

if __name__ == '__main__':
    app.run(debug=True)
