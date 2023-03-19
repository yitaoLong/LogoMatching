import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
import uuid
import numpy
import cv2
import h5py
import numpy as np
import tempfile

# Please don't run this on the cloud, it costs so much.

from methods import small_neural_method, neural_method, orb_method, zernike_method, contour_method
from icon_util import web_plot,  test_combined, load_databases, gray
methods = load_databases([zernike_method, orb_method, neural_method, contour_method, small_neural_method], "icon1k")
zernike, orb, neural, contour, small_neural = methods

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

hdf5_file = h5py.File('LLD-icon.hdf5', 'r')
images, _ = (hdf5_file['data'], hdf5_file['labels/resnet/rc_64'])
images = [np.transpose(i,(1,2,0)) if i.shape[0] == 3 else i for i in images[:1000]]
print(len(images))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            method_choice = request.form.getlist('methods')
            res = run_image(file, method_choice)
            return send_file(res, mimetype='image/png')

    return '''
    <!doctype html>
    <title>TEST YOUR IMAGE FOR PLAGIARISM!</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      Please choose which methods you would like to run from the checkboxes. <br>
      <input type=file name=file> <br>
      <input type="checkbox" name="methods" value="smallNeural" id="smallNeural" checked> <label for="smallNeural"> Small Neural </label><br>
      <input type="checkbox" name="methods" value="BIGNeural" id="BIGNeural" checked> <label for="BIGNeural"> Big Neural </label><br>
      <input type="checkbox" name="methods" value="ORB" id="ORB" checked> <label for="ORB"> ORB </label><br>
      <input type="checkbox" name="methods" value="Zernike" id="Zernike" checked> <label for="Zernike"> Zernike </label><br>
      <input type="checkbox" name="methods" value="Contour" id="Contour" checked> <label for="Contour"> Contour </label><br>
      <input type=submit value=Upload>
    </form>
    '''

def run_image(filename, method_choice):
    img  = numpy.frombuffer(filename.read(), dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    print("starting")
    method_names = ["Zernike", "ORB", "BIGNeural", "Contour", "smallNeural"]
    weights=[
        0.1240114,
        0.00359439,
        0.63103744,
        0.001776404,
        0.20621383]
    res_list = []
    res_weights = []
    for i in range(len(method_names)):
        if method_names[i] in method_choice:
            res_list = [methods[i].run_query(img)]
            res_weights += [weights[i]]
    print(res_weights)
    res = test_combined(res_list, res_weights)
    return web_plot(img, res, images)

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
