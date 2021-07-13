# syntax=docker/dockerfile:experimental

FROM python:latest

RUN echo "PS1='\[\033[01;32m\]WordNetSim\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\] \$ '" >> /root/.bashrc

WORKDIR /dependencies

# Install NLTK
RUN pip install -U nltk
RUN python -m nltk.downloader -q all

# Install numpy
RUN pip install -U numpy

# Install FreeLing 4.2
RUN apt-get -y update
ADD https://github.com/TALP-UPC/FreeLing/releases/download/4.2/freeling-4.2-buster-amd64.deb .
ADD https://github.com/TALP-UPC/FreeLing/releases/download/4.2/freeling-langs-4.2.deb .
RUN apt -y install ./freeling-4.2-buster-amd64.deb
RUN apt -y install ./freeling-langs-4.2.deb
RUN pip install -U lxml
RUN pip install -U pyfreeling

# Install FastText
RUN git clone https://github.com/facebookresearch/fastText.git
RUN cd fastText && pip install . 

WORKDIR /app
COPY . .
ADD https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin .

CMD ["bash"]