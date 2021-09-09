FROM nvidia/cuda:11.1-devel

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y -qq --no-install-recommends build-essential cmake
# cppgl deps
RUN apt-get install -y -qq --no-install-recommends libx11-dev xorg-dev libopengl-dev freeglut3-dev
# OpenVDB deps
RUN apt-get install -y -qq --no-install-recommends libboost-iostreams-dev libboost-system-dev libtbb-dev libilmbase-dev libopenexr-dev
# TODO verify if this works with OpenVDB
RUN apt-get install -y -qq --no-install-recommends libblosc-dev
# python debs
RUN apt-get install -y -qq --no-install-recommends python3-dev python3-pip

RUN rm -rf /code
WORKDIR /code
COPY CMakeLists.txt ./
COPY src/ src/
COPY ptx/ ptx/
COPY shader/ shader/
COPY scripts/ scripts/
COPY submodules/ submodules/

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev && cmake --build build --parallel
