docker run -d -P -v `pwd`:/test --name myserver pictet-image:latest
docker exec myserver python server.py

