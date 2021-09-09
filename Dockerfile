FROM nvidia/cuda:11.4-devel

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y build-essential cmake
# cppgl deps
RUN apt-get install -y -qq --no-install-recommends build-essential libx11-dev xorg-dev libopengl-dev freeglut3-dev cmake
# OpenVDB deps
RUN apt-get install -y libboost-iostreams-dev libboost-system-dev libtbb-dev libilmbase-dev libopenexr-dev
# TODO verify if this works with OpenVDB
RUN apt-get install -y libblosc-dev
# python debs
RUN apt-get install -y python3-dev python3-pip

RUN rm -rf /code
WORKDIR /code
COPY setup.py CMakeLists.txt ./
COPY src/ src/
COPY submodules/ submodules/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev && cmake --build build --parallel

# TODO: python package setup
#COPY setup.py ./
#RUN pip3 install . -v
