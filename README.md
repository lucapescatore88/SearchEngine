### Data Science position at Pictet: Search Engine

Assignment for Data Scientist position at Pictet. A search engine that will tell you if somebody is rich and famous or poor and miserable.

A presentation about how all this works is under development [at this link](https://cernbox.cern.ch/index.php/s/1gHOr0MYm3FGNFL) (low resolution).


## Installation 

In order to use this repository you need many libraries.

You can install them in your local environment using

```
pip install RUN pip install pandas sklearn 'matplotlib==1.4.3' seaborn pyspark \
       wikipedia Flask babel pyyaml forex-python unidecode tweepy xgboost nltk
python -c 'import nltk; nltk.download("all")'
```

Then run ```source startup.sh``` to set the environment.

## Running

You can run the application from command line as

```python engine/runsearch.py Donald Trump```

Or you can access it from the browser too by typing `flask run` to start the server 
and then going to `http://127.0.0.1/5000/` from any browser on your local machine.

To make it easier you can use directly the startup script:

```./startup.sh test```

runs a test from the command line,

```./startup.sh web```

starts the web interface.

## Docker deploiment

If you don't want to bother with all that you can build a Docker environment typing the following from the main folder:

```
docker build -t {some name} .
docker run -p 5000:5000 -i -t {some name}
[DockerConsole]>>> python engine/runsearch.py Donald Trump
```

N.B.: The web interface works from Docker too but there is some problem with port forwarding out of the container. 
On linux you can then go to a brower and use the app. On my Mac so the website will not appear... but this is a detail, 
the app actually works in the background! This is just to tell you that depending on what machine you are this may not work.

## Structure

- ```engine``` : Contains main pathon scripts to run the engines
    * runsearch.py -> Main script to launch search
    * engineutils.py -> Conteins generic utilities e.g. clean html pages and load countries DB
    * googleutils.py -> Functions to get data drom Google
    * wikiutils.py -> Functions to get data drom Wikipedia
    * twitterutils.py -> Functions to get data drom Twitter
    * checkPolitician.py -> Function to train and score a isPolitician classifier
    * checkFamous.py -> Function to train and score a isFamous classifier

- ```flaskr``` : Contains the python scripts and html files to run the web interface.

- ```resources``` : Contains data sources, csv files, pickle files with trained models.

- ```scripts``` : Standalone scripts that used to make some plots.


 
