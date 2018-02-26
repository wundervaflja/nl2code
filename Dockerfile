# Start with cuDNN base image
FROM continuumio/anaconda

RUN conda install Theano nltk && pip install vprof astor && python -c "import nltk; nltk.download('punkt')"

ADD . /home/NL2code
WORKDIR /home/NL2code

ENV THEANO_FLAGS="mode=FAST_RUN,device=cpu,floatX=float32"

# ENTRYPOINT ["./code_gen.py"]

