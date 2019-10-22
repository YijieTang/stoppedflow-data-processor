FROM ubuntu:18.04

LABEL maintainer="Yijie Tang <yijietang@cmu.edu>"

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends python-autopep8 python3-pip python3 python3-setuptools vim git && \
  pip3 install --upgrade pip && \
  python3 -m pip install wheel jupyter mpld3 matplotlib numpy pandas jupyter_contrib_nbextensions jupyter_nbextensions_configurator && \
  python3 -m jupyter contrib nbextension install --system && \
  python3 -m jupyter nbextensions_configurator enable --system && \
  python3 -m jupyter nbextension enable codefolding/main --system && \
  python3 -m jupyter nbextension enable snippets/main --system && \
  python3 -m jupyter nbextension enable toc2/main --system && \
  python3 -m jupyter nbextension enable varInspector/main --system && \
  python3 -m jupyter nbextension enable snippets_menu/main --system && \
  python3 -m jupyter nbextension enable hinterland/hinterland --system && \
  python3 -m jupyter nbextension enable livemdpreview/livemdpreview --system && \
  python3 -m jupyter nbextension enable code_font_size/code_font_size --system

RUN mkdir -p /opt/app/data

ENV TINI_VERSION v0.6.0
ADD sf_utils.py /usr/lib/python3.6/
ADD uv_utils.py /usr/lib/python3.6/
ADD . /opt/app/data/
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8000
CMD ["jupyter", "notebook", "--port=8000", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/opt/app/data"]
