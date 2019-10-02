from flask import Flask, render_template, request, redirect, url_for, session
import controllers.index_controller as index_controller
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={"/generate": {"origins": "*"}})

@app.route('/')
def index():
    return index_controller.index()

@app.route('/tmp')
def tmp():
    return index_controller.tmp()

@app.route('/predefined/<int:model_id>/<string:model_type>')
def predefined(model_id, model_type):
    return index_controller.predefined(model_id, model_type)

@app.route('/generate', methods=['POST'])
def generate():
    return index_controller.generate()

if __name__ == '__main__':
    app.run(host= '127.0.0.1', port=5008, debug=False)
