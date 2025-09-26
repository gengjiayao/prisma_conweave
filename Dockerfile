FROM tensorflow/tensorflow:2.8.3-gpu

# 1. 安装PRISMA和conweave所需全部依赖
RUN apt-get update && \
    apt-get install -y --fix-missing\
        gcc g++ bc rsync \
        libzmq5 libzmq5-dev libprotobuf-dev protobuf-compiler \
        gnuplot python2 python2-dev python2-minimal \
        python3 python3-pip build-essential libgtk-3-0 bzip2 wget git \
        && rm -rf /var/lib/apt/lists/*

# 2. python3软链接，兼容Python3调用
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 3. 软链接python2（可选，方便手动调试）
RUN [ ! -e /usr/bin/python2.7 ] && ln -s /usr/bin/python2 /usr/bin/python2.7 || true


# 4. 拷贝项目文件
#COPY . /app/.

WORKDIR /app

# 5. 安装Python3依赖
RUN python3 -m pip install --no-cache-dir \
	pandas==1.4.0 \
    graphviz==0.20.1 \
    gym==0.26.1 \
    matplotlib \
    networkx==2.8.7 \
    numpy \
    paramiko==2.6.0 \
    protobuf==3.19.6 \
    pyzmq==24.0.1 \
    requests==2.22.0 \
    tensorflow==2.8.3 \
    tensorboard==2.8 \
    zmq \
    viztracer

# 6. 可选：安装常用Python2包（如有需要）
RUN python2 -m pip install --upgrade pip || true
# 如果有python2 requirements.txt可以加上：
# RUN python2 -m pip install --no-cache-dir -r requirements_py2.txt || true

# 7. 编译protobuf
#RUN cd prisma/ns3_model/ && ./compile_proto.sh && cd ../..

# 8. 拷贝/编译ns3文件（保留原PRISMA部分）
#COPY prisma/ns3 conweave-ns3-main/scratch/prisma
#COPY prisma/ns3_model/ipv4-interface.cc conweave-ns3-main/src/internet/model/.
#COPY prisma/ns3_model/ipv4-interface.h conweave-ns3-main/src/internet/model/.


# 10. 默认工作目录/入口（可按需改main.py/autorun.sh/run.py等）
WORKDIR /app/prisma
# ENTRYPOINT [ "python3", "main.py" ]

