FROM ubuntu:18.04

## Update operating system and install pip
RUN set -xe \
    && apt-get update \
    && apt-get install -y libffi-dev libssl-dev \
    && apt-get install -y python python-dev python-pip \ 
    && apt-get install -y libxft-dev libfreetype6 libfreetype6-dev
#    && apt-get install -y openssh-server
RUN pip install --upgrade pip
RUN pip install pyopenssl ndg-httpsclient pyasn1

# Install python libraries

RUN pip install pandas
RUN pip install sklearn
RUN pip install 'matplotlib==1.4.3'
RUN pip install seaborn
RUN pip install wikipedia
RUN pip install Flask
RUN pip install babel
RUN pip install google-search-results
RUN pip install pyyaml
RUN pip install forex-python

## Stuff to try fix the port forwarding on Mac
#RUN mkdir /var/run/sshd
#RUN echo 'root:screencast' | chpasswd
#RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile
#EXPOSE 22
#CMD ["/usr/sbin/sshd", "-D"]


EXPOSE 5000

COPY engine engine
COPY flaskr flaskr
COPY resources resources

#RUN python test.py

#COPY test/server.py server.py
COPY startup.sh startup.sh
COPY title.txt title.txt
RUN chmod +x startup.sh
CMD bash -C 'startup.sh';'bash'


