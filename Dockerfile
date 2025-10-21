FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

WORKDIR /

RUN apt update 
RUN apt install -y python3 python3-pip python3.12-venv

COPY requirements.txt /requirements.txt
RUN python3 -m venv .venv
RUN . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY main.py /
COPY test_input.json /

CMD [".venv/bin/python", "-u", "main.py"]