FROM runpod/base:1.0.2-cuda1281-ubuntu2404

WORKDIR /


COPY requirements.txt /requirements.txt
RUN python3 -m venv .venv
RUN . .venv/bin/activate && pip install --no-cache-dir -r requirements.txt

COPY main.py /
COPY five/ /five
COPY test_input.json /

CMD [".venv/bin/python", "-u", "main.py"]