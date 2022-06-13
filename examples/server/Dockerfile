FROM ealgoo/aiortc:drowsy

# ignore dialogue
ENV DEBIAN_FRONTEND=noninteractive

RUN pip install scipy
# upgrade env
EXPOSE 8080
COPY . /workspace
WORKDIR /workspace

CMD python3 ./server.py