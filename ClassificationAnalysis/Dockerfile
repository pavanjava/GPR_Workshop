FROM python:3-slim-buster

RUN mkdir /src

WORKDIR /src

COPY requirements.txt .
COPY api.py .
COPY KNeighborsClassifier.pkl .
COPY ./payload_models ./payload_models

RUN pip install -r requirements.txt

CMD ["uvicorn", "api:app", "--host=0.0.0.0", "--port=80"]