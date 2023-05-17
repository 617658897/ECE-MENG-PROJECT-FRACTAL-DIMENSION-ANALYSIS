from flask import Flask, request, jsonify, make_response
# from sklearn.externals import joblib
# import numpy as np
# import sys

import base64
from PIL import Image
from io import BytesIO
from sklearn.linear_model import LinearRegression
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import cv2
import sys
from collections import Counter
import math
from PIL import Image
from scipy.stats import linregress
from sklearn import metrics
import pyomo
from pyomo.environ import ConcreteModel, Var, Binary, Objective, ConstraintList, minimize, SolverFactory
import fd
import traceback

app = Flask(__name__)


# cbc_path = r"D:\编程学习\ML-React-App-Template\example\model\FD_calculate\service\cbc.exe"
# solver = SolverFactory('cbc', executable=cbc_path)


@app.route('/prediction/', methods=['OPTIONS', 'POST'])
def prediction():
    if request.method == 'OPTIONS':
        print("In Options")
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    if request.method == 'POST':
        print("In Post")
        try:
            img = request.json
            prefix = "data:image/png;base64,"
            url = img[len(prefix):]

            # decode base64 string
            image_bytes = base64.b64decode(url)

            # create Image object from bytes
            a = Image.open(BytesIO(image_bytes))

            # save Image object as JPG file
            a.save("output.png")

            print("output.png SAVED")

            fd.out()
            print("fd.out DONE")

            with open("D:\编程学习\ML-React-App-Template\example\model\FD_calculate\service\ConfM1.png", "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')
                encoded_string = "data:image/png;base64," + encoded_string
                print(encoded_string)

            with open("output.txt", "w") as file:
                file.write(encoded_string)

            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": encoded_string
            })

            # with open("cal.py") as file:
            #     code = file.read()
            #     exec(code)

            response.headers.add('Access-Control-Allow-Origin', '*')
            return response

        except Exception as e:
            traceback.print_exc()
            print(e)
            return jsonify({'error': 'Invalid JSON data'}), 400

        finally:
            print('1')

        # Do something with selectedImg
        return jsonify({'result': 'success'})
    else:
        selectedImg = request.json
        return selectedImg
