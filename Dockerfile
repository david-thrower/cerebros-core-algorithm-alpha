FROM tensorflow/tensorflow:2.19.0

WORKDIR /app

RUN git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git

WORKDIR /app/cerebros-core-algorithm-alpha

RUN git checkout 247-2025-09-27--new-hpo-study-with-sampling-plus-penalties

RUN pip3 install -r requirements.txt
RUN pip3 install -r cicd-requirements.txt

ENTRYPOINT ["python3", "generative-proof-of-concept-CPU-preprocessing-in-memory.py" ]
