import os
import engine

from flask import Flask, url_for, render_template, request


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        myurl = url_for('static', filename='style.css')
        return myurl+'   index'

    @app.route('/run')
    def run():
        name = 'Luciabella'
        return render_template('main.html')

    @app.route('/action')
    def action():
        first = request.args.get('firstname')
        last  = request.args.get('lastname')
        return first + " " + last    

    return app
