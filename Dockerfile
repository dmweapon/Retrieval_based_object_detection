#FROM ubuntu:22.04
#
#RUN apt update && apt install -y \
#    build-essential \
#    curl \
#    vim

FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

RUN apt update && apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# 파이썬 패키지 종속성 설치
COPY requirements-ubuntu.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-ubuntu.txt

# 컨테이너 시작 시 실행할 기본 명령어
#CMD ["python", "filename.py"]
#CMD ["tail", "-f", "/dev/null"]
CMD ["/bin/bash"]