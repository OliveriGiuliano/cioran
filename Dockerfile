FROM python:3.7.3-stretch

RUN mkdir /Hot_dog_or_not
WORKDIR /Hot_dog_or_not
COPY requirements.txt /Hot_dog_or_not/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /Hot_dog_or_not/
EXPOSE $PORT

CMD python app.py