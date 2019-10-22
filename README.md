# Stopped-flow data processor
## Data structure and functions
most functions are in sf_utils.py and uv_utils.py
## Preview the examples in the notebook
https://nbviewer.jupyter.org/github/YijieTang/stoppedflow-data-processor/blob/master/stoppedflow-analysis.ipynb
## Run the notebook in a container
1. Build docker image
```sh
docker build --rm --tag stoppedflow/jupyternotebook .
```
2. Run a container with the image on port 8888
```sh
docker run -p 8888:8000 stoppedflow/jupyternotebook
```
3. Visit the notebook in a web browser with the printed token
