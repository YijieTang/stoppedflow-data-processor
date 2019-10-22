# Stopped-flow data processor
## Build docker image
docker build --rm --tag stoppedflow/jupyternotebook .
## Run a container with the image on port 8888
docker run -p 8888:8000 stoppedflow/jupyternotebook
## Visit the notebook in browser
copy the token in the terminal and paste into browser
