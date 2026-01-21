
FROM tensorflow/tensorflow:2.20.0-gpu

# NVIDIA GPU Runtime (for docker run --gpus all)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Add a volume to collect artifacts and cp -r [.] [path-to-folder]

WORKDIR /opt
RUN apt update -y
RUN apt upgrade -y
RUN apt install git -y
RUN git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git
WORKDIR /opt/cerebros-core-algorithm-alpha
RUN git fetch
RUN git checkout 309-from-307-dockerize-the-best-run-from-hpo-study
RUN git pull origin 309-from-307-dockerize-the-best-run-from-hpo-study

# Copy Thunderline integration files (overrides any from git)
COPY thunderline_integration.py /opt/cerebros-core-algorithm-alpha/
COPY train_a_generative_llm_docker.py /opt/cerebros-core-algorithm-alpha/
COPY test_llm_serialization.py /opt/cerebros-core-algorithm-alpha/

## Debug
RUN echo "##### LIST OF EXISTING PYTHON PACKAGES #####"
RUN pip list

RUN echo "##### END OF LIST OF EXISTING PYTHON PACKAGES #####"

## / debug

RUN pip install --upgrade pip
RUN pip install --ignore-installed blinker --ignore-installed ml_dtypes -r docker-requirements.txt

EXPOSE 7777

ENTRYPOINT ["python", "train_a_generative_llm_docker.py"]
