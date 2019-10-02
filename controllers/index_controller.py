from flask import render_template, request, redirect, url_for, session, jsonify, abort
import services.voxelizer as voxelizer
import services.generator as generator
from werkzeug.utils import secure_filename
import os
import numpy as np

ALLOWED_EXTENSIONS = set(['dae', 'off', 'ply', 'obj', 'stl'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def index():

    return render_template('index.html')

def playful_ai():

    return render_template('playful_ai.html')

def predefined(model_id, model_type):

    path = './meshes/model_0%d.obj' % model_id

    voxels = voxelizer.voxelize(path=path)

    if not isinstance(voxels, (np.ndarray, np.generic)):
        return jsonify({'error': 'The file might have several meshes in one file.'})

    voxels = voxelizer.voxelize(path=path)
    configuration = generator.generate(voxels=voxels, model_type=model_type)
    return jsonify({'result': configuration, 'model_type': model_type, 'error': None})

def generate():

    if not 'model' in request.files:
        abort(403)
        return

    model      = request.files['model']
    model_type = request.form['model_type']

    if model and allowed_file(model.filename):
        filename = secure_filename(model.filename)
        path = os.path.join('/tmp', filename)
        model.save(path)

        voxels = voxelizer.voxelize(path=path)

        if not isinstance(voxels, (np.ndarray, np.generic)):
            return jsonify({'error': 'The file might have several meshes in one file.'})

        voxels = voxelizer.voxelize(path=path)
        configuration = generator.generate(voxels=voxels, model_type=model_type)
        return jsonify({'result': configuration, 'model_type': model_type, 'error': None})

    else:

        abort(403)
        return