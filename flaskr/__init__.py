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
        #flash('Engine is doing a lot of work, this will take a while.')

        first   = str(request.args.get('firstname'))
        last    = str(request.args.get('lastname'))
        mid     = str(request.args.get('midname'))
        country = str(request.args.get('countryselect'))
        #request.form.get('countryselect')
        print first, last, mid, country
        res   = engine.runsearch.runSearch(first,last,mid)#,country)

        return render_template('main_res.html',**res)

    return app
