from __future__ import division, print_function
# coding=utf-8
import re, glob, os
from flask_cors import CORS, cross_origin
import time
import cv2
from datetime import timezone
import datetime
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from pymongo import MongoClient
from flask_pymongo import PyMongo
from mongopass import mongopass
import subprocess as sp

# Define a flask app
app = Flask(__name__)
CORS(app)

client = MongoClient(mongopass)
db = client.timeseries
numberplateCol = db.timeseries

print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/predict', methods=['POST'])
@cross_origin(origin='*')
def upload():
    files = request.files
    file = files.get('file')
    print(file)
    ts = time.strftime("%Y%m%d%H%M%S")
    filename = ts + '.jpeg'
    file.save(os.path.join('images', filename))
    message = 'done'

    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()

    img = cv2.imread(f'images/{filename}', cv2.IMREAD_COLOR)
    # img = cv2.imread(file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # edge detection

    # cv2.imshow(edged)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    print(location)
    os.remove(os.path.join('images', filename))

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    if result:
        print(result)
        text = result[0][-2]
        print(text)
        if ((len(text) < 9) and (len(text) > 5)):
            timeseries_data = {"timestamp": datetime.datetime.now(), "number_plate": text}
            numberplateCol.insert_one(timeseries_data)

        return text
    # return text


if __name__ == '__main__':
    app.run(debug=True)
