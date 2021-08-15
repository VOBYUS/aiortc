FROM ubuntu:18.04
# FROM python:3.6-buster
# ADD file:e05689b5b0d51a2316f8a87b1a9d6cbf90d98b19a424dbb924ee3d0b1cc17bfc in /
RUN set -xe
RUN echo '#!/bin/sh' > /usr/sbin/policy-rc.d
RUN echo 'exit 101' >> /usr/sbin/policy-rc.d
RUN chmod +x /usr/sbin/policy-rc.d
RUN dpkg-divert --local --rename --add /sbin/initctl
RUN cp -a /usr/sbin/policy-rc.d /sbin/initctl
RUN sed -i 's/^exit.*/exit 0/' /sbin/initctl
RUN echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/docker-apt-speedup
RUN echo 'DPkg::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' > /etc/apt/apt.conf.d/docker-clean
RUN echo 'APT::Update::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' >> /etc/apt/apt.conf.d/docker-clean
RUN echo 'Dir::Cache::pkgcache ""; Dir::Cache::srcpkgcache "";' >> /etc/apt/apt.conf.d/docker-clean
RUN echo 'Acquire::Languages "none";' > /etc/apt/apt.conf.d/docker-no-languages
RUN echo 'Acquire::GzipIndexes "true"; Acquire::CompressionTypes::Order:: "gz";' > /etc/apt/apt.conf.d/docker-gzip-indexes
RUN echo 'Apt::AutoRemove::SuggestsImportant "false";' > /etc/apt/apt.conf.d/docker-autoremove-suggests
RUN [ -z "$(apt-get indextargets)" ]
RUN mkdir -p /run/systemd
RUN echo 'docker' > /run/systemd/container
# CMD ["/bin/bash"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y 
RUN apt install python3 -y 
RUN apt install python3-pip -y 
RUN apt install python3-dev -y 
RUN pip3 install --upgrade pip 
RUN apt install libavdevice-dev -y 
RUN apt install libavfilter-dev -y 
RUN apt install libopus-dev -y 
RUN apt install libvpx-dev -y 
RUN apt install pkg-config -y 
RUN apt install libopencv-dev -y 
RUN apt-get update && apt-get install -y cmake
RUN pip install aiortc 
RUN pip install aiohttp 
RUN pip install opencv-python 
RUN pip install scipy imutils
RUN pip install dlib
RUN pip install matplotlib
RUN pip install tensorflow@1.13
EXPOSE 8080
COPY ./examples/server /workspace 
WORKDIR /workspace 
# CMD [python3 ./server.py"] 
RUN python3 ./server.py