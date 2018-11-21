FROM ubuntu:14.04

## Update operating system and install pip
RUN set -xe \
    && apt-get update \
    && apt-get install -y libffi-dev libssl-dev \
    && apt-get install -y python python-dev python-pip \ 
    && apt-get install -y libxft-dev libfreetype6 libfreetype6-dev
RUN pip install --upgrade pip
RUN pip install pyopenssl ndg-httpsclient pyasn1

# Install python libraries

RUN pip install pandas
RUN pip install sklearn
RUN pip install 'matplotlib==1.4.3'
RUN pip install seaborn

COPY *.py /

RUN python test.py

#ENTRYPOINT /bin/bash
CMD bash -C 'python $HOME/test.py';'bash'
#["python","test.py"]


