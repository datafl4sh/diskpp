FROM debian:trixie-slim

RUN sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list.d/debian.sources
RUN apt update

RUN apt install -y apt-utils ca-certificates debian-keyring wget


RUN sh -c 'wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null'
RUN sh -c 'echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list'

RUN apt update

RUN apt install -y build-essential cmake git
RUN apt install -y libeigen3-dev libmumps-dev libmumps-seq-dev liblua5.4-0 liblua5.4-dev
RUN apt install -y libsiloh5-0 libsilo-dev gmsh libgmsh-dev libhdf5-dev
RUN apt install -y intel-oneapi-mkl-devel

WORKDIR /diskpp
RUN git clone --recursive https://github.com/datafl4sh/diskpp.git
WORKDIR /diskpp/diskpp/build
RUN cmake -DMKL_DIR=/opt/intel/oneapi/mkl/latest/lib/cmake/mkl -DCMAKE_BUILD_TYPE=Release ..
RUN make -j $(nproc)

