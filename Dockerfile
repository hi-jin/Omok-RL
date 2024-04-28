# 베이스 이미지 설정
FROM python:3.10.14

# 작업 디렉토리 설정
WORKDIR /workspace

# copy Dockerfile to /workspace
COPY Dockerfile /workspace

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y git vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pytorch essentials
RUN pip3 install torch torchmetrics lightning numpy pandas matplotlib transformers datasets ipykernel einops wandb

# for RL
RUN pip3 install gym stable-baselines3 sb3-contrib pettingzoo

# 컨테이너 시작 시 실행될 명령
CMD ["bash"]
