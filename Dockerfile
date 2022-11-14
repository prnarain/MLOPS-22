FROM python:3.8.1
COPY ./*.py /exp/
COPY ./svm_gamma=0.001_C=0.5.joblib /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
COPY ./api/app.py /exp/
EXPOSE 5000
ENV FLASK_APP app.py
ENTRYPOINT ["python", "-m", "flask", "run", "--host=0.0.0.0"]

