# syntax=docker/dockerfile:experimental

FROM python:latest

RUN echo "PS1='\[\033[01;32m\]WordNetSim\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] \$ '" >> /root/.bashrc

ENV DEBIAN_FRONTEND=noninteractive 
WORKDIR /dependencies

# Install NLTK
RUN pip install -U nltk
RUN python -m nltk.downloader -q all
COPY mon.zip /root/nltk_data/corpora/omw/.
RUN unzip /root/nltk_data/corpora/omw/mon.zip -d /root/nltk_data/corpora/omw

# Install numpy
RUN pip install -U numpy

# Install FreeLing 4.2
RUN --mount=type=cache,id=apt-debian-stable,target=/var/cache/apt apt-get -y update
RUN apt autoremove && apt autoclean
RUN --mount=type=cache,id=apt-debian-stable,target=/var/cache/apt apt-get -y install build-essential locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
RUN dpkg-reconfigure --frontend=noninteractive locales
RUN update-locale LANG=en_US.UTF-8
RUN --mount=type=cache,id=apt-debian-stable,target=/var/cache/apt apt-get -y install cmake
RUN --mount=type=cache,id=apt-debian-stable,target=/var/cache/apt apt-get -y install libboost-dev libboost-regex-dev libicu-dev libboost-system-dev libboost-program-options-dev libboost-filesystem-dev libboost-thread-dev zlib1g-dev
RUN --mount=type=cache,id=apt-debian-stable,target=/var/cache/apt apt-get -y install python3-dev swig
ADD https://github.com/TALP-UPC/FreeLing/releases/download/4.2/FreeLing-src-4.2.tar.gz .
ADD https://github.com/TALP-UPC/FreeLing/releases/download/4.2/FreeLing-langs-src-4.2.tar.gz .
RUN tar zxvf FreeLing-src-4.2.tar.gz
RUN tar zxvf FreeLing-langs-src-4.2.tar.gz
RUN cd FreeLing-4.2 && mkdir build && cd build && cmake -DPYTHON3_API=ON .. && make install
ENV PYTHONPATH=/usr/local/share/freeling/APIs/python3:$PYTHONPATH

# Install FastText
RUN git clone https://github.com/facebookresearch/fastText.git
RUN cd fastText && pip install . 

WORKDIR /app
COPY . .
ADD https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin .

CMD ["bash"]