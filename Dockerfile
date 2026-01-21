
FROM tensorflow/tensorflow:2.20.0-gpu

WORKDIR /opt
RUN apt update -y
RUN apt upgrade -y
RUN apt install git -y
WORKDIR /opt/cerebros-core-algorithm-alpha
COPY . .
RUN pip install --upgrade pip
RUN pip install --ignore-installed blinker --ignore-installed ml_dtypes -r docker-requirements.txt

EXPOSE 7777

ENTRYPOINT ["python", "train_a_generative_llm_docker.py"]
