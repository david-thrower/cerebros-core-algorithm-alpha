
FROM tensorflow/tensorflow:2.20.0-gpu

# Add a volume to collect artifacts and cp -r [.] [path-to-folder]

WORKDIR /opt
RUN apt update
RUN apt upgrade
RUN apt install git
RUN git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git
RUN git checkout 309-from-307-dockerize-the-best-run-from-hpo-study
RUN git fetch
RUN git checkout 309-from-307-dockerize-the-best-run-from-hpo-study
RUN git pull origin 309-from-307-dockerize-the-best-run-from-hpo-study
WORKDIR /opt/cerebros-core-algorithm-alpha
RUN pip install -r docker-requirements.txt
RUN python train_a_generative_llm.py
