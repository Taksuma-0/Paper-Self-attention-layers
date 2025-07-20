<<<<<<< HEAD
FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt ./


RUN conda create -n atencion_env python=3.12 -y

RUN conda run -n atencion_env conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

RUN conda run -n atencion_env pip install --upgrade tabulate boto3 urllib3
RUN conda run -n atencion_env pip install "protobuf==3.20.3" "Pillow<10.0.0"


RUN conda run -n atencion_env pip install -r requirements.txt


RUN conda run -n atencion_env pip install scikit-learn seaborn matplotlib pyyaml tensorboardX tqdm

COPY . .

ENTRYPOINT ["conda", "run", "-n", "atencion_env", "python"]

=======
FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt ./


RUN conda create -n atencion_env python=3.12 -y

RUN conda run -n atencion_env conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

RUN conda run -n atencion_env pip install --upgrade tabulate boto3 urllib3
RUN conda run -n atencion_env pip install "protobuf==3.20.3" "Pillow<10.0.0"


RUN conda run -n atencion_env pip install -r requirements.txt


RUN conda run -n atencion_env pip install scikit-learn seaborn matplotlib pyyaml tensorboardX tqdm

COPY . .

ENTRYPOINT ["conda", "run", "-n", "atencion_env", "python"]

>>>>>>> 7c45fa74a46655569e67fca750aa3cd7ab9f63f2
CMD ["benchmark_final.py"] #importante para la ejecución