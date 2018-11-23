import os, sys
import engine

from flask import Flask, url_for, render_template, request, flash


def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return render_template('main.html')

    @app.route('/action')
    def action():
        flash('Engine is doing a lot of work, this will take a while.')

        first = request.args.get('firstname')
        last  = request.args.get('lastname')
        res   = engine.runsearch.runSearch(first,last)

        fullname = "{0} {1} {2}".format(first,res['midname'],last) 
        return render_template('main_res.html',fullname=fullname,**res)

    return app
