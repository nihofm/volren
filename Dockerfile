FROM ubuntu

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# update system
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y

# install deps
RUN apt-get install -y build-essential cmake
RUN apt-get install -y libx11-dev xorg-dev libopengl-dev freeglut3-dev
RUN apt-get install -y libglm-dev libassimp-dev libopenvdb-dev # reduce compile times
RUN apt-get install -y python3-dev

# copy code
WORKDIR /workspace
COPY CMakeLists.txt ./
COPY src/ src/
COPY shader/ shader/
COPY scripts/ scripts/
COPY submodules/ submodules/

# build
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev && cmake --build build --parallel
