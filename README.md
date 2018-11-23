### Data Science position at Pictet

In order to use this repository you need many libraries.

You can install them in your local environment using

```pip install matpotlib, seaborn, pandas, sklearn, Flask, wikipedia```

Then you can run the application from command line as

```python engine/runsearch.py Donald Trump```

Or you can access it from the browser too by typing `flask run` to start the server and then going to `http://127.0.0.1/5000/` from any browser on your local machine.

## Docker deploiment

If you don't want to bother with all that you can build a Docker environment typing the following from the main folder:

```docker build -t pictet-image .
docker run -p 5000:5000 -i -t pictet-image
[DockerConsole]>>> python engine/runsearch.py Donald Trump
```

N.B.: If before building the Docker image you uncomment "run flask" in startup.sh,
the docker will connect the app to the server. It works! On linux you can then go to a brower and use the app.
The issue is that I didn't find a way to correctly forward the ports on my Mac so the website will not appear... but this is a detail, the app actually works in the background! This is just to tell you that depending on what machine you are this may not work.

