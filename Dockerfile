FROM python:3.10-slim-bullseye

WORKDIR /usr/src/app

COPY . .

RUN apt-get update && apt-get --no-install-recommends install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6  -y \
&& pip install --no-cache-dir -r requirements.txt

# tell the port number the container should expose
EXPOSE 6000

CMD python API_main.py