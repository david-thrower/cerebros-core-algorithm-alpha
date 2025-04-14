
FROM tensorflow/tensorflow:2.19.0-jupyter

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN pip install -r cicd-requirements.txt
RUN pip install optuna==4.3.0 

ENTRYPOINT [ "python3", "optimize-nlp-standalone-mv-tpe.py" ]
