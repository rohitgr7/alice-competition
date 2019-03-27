FROM python:3

WORKDIR /usr/src/app

RUN pip install pandas==0.23.4 scikit-learn==0.20.2 numpy==1.16.0 pathlib==1.0.1

COPY . .

CMD [ "python", "./script.py" ]
