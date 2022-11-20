FROM python:3.10.6
COPY ./*.py /exp/
COPY ./svm_gamma=0.0005_C=5.joblib /exp/
COPY ./requirement.txt /exp/requirement.txt
RUN pip3 install --no-cache-dir -r /exp/requirement.txt
WORKDIR /exp
EXPOSE 5000
ENV FLASK_APP api.py
ENTRYPOINT ["flask", "run"]

