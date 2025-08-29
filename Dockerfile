FROM ubuntu:24.04 as compiler

#RUN echo y | unminimize
RUN apt update -y
RUN apt full-upgrade -y
RUN apt install -y  software-properties-common sudo wget gpg unminimize apt-transport-https
RUN apt install -y ubuntu-drivers-common nvidia-driver-535 nvidia-cuda-toolkit nvtop locales-all git git-lfs parallel psmisc bc rsync 
RUN apt install -y python3-full python3-venv ipython3 python3-matplotlib-inline python3-ipykernel 
RUN apt install -y cmake cmake-extras extra-cmake-modules libzip-dev libbzip3-dev xxd
RUN apt install -y libfftw3-dev libboost-all-dev libclfft-dev libtiff-dev libpng-dev libgsl-dev

RUN mkdir -p /usr/local/src

WORKDIR /usr/local/src
RUN git clone https://github.com/HDFGroup/hdf5.git
RUN mkdir -p /usr/local/src/hdf5/build/
WORKDIR /usr/local/src/hdf5/build/ 
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release   -DHDF5_ENABLE_THREADSAFE:BOOL=ON  -DHDF5_BUILD_HL_LIB:BOOL=0 -DH5EX_BUILD_HL:BOOL=0 ..
RUN make && make install

WORKDIR /usr/local/src
RUN git clone https://codeberg.org/antonmx/poptmx.git
RUN mkdir -p /usr/local/src/poptmx/build/
WORKDIR /usr/local/src/poptmx/build/
RUN cmake .. && make && make install

WORKDIR /usr/local/src
RUN git clone https://github.com/clMathLibraries/clFFT.git 
RUN mkdir -p /usr/local/src/clFFT/build/
WORKDIR /usr/local/src/clFFT/build/
RUN cmake ../src/ && make && make install

WORKDIR /usr/local/src
RUN git clone https://github.com/ImageMagick/ImageMagick.git
WORKDIR /usr/local/src/ImageMagick
RUN ./configure --prefix=/usr/local --with-modules --with-quantum-depth=32 --enable-hdri
RUN make && make install
RUN ldconfig /usr/local/lib 

WORKDIR /usr/local/src
RUN git clone https://codeberg.org/antonmx/ctas.git
RUN mkdir -p /usr/local/src/ctas/build/
WORKDIR /usr/local/src/ctas/build/
RUN cmake .. && make && make install

WORKDIR /usr/local/src
RUN git clone https://codeberg.org/antonmx/imblproc.git
WORKDIR /usr/local/src/imblproc 
RUN cp -r bin share /usr/local/ 

WORKDIR /usr/local/src
RUN git clone https://codeberg.org/antonmx/bctppl.git 
RUN ln -s /usr/local/src/bctppl/ /opt/
RUN ln -s /opt/bctppl/bctppl.sh /usr/local/bin/bctppl

WORKDIR /
RUN python3 -m venv /opt/torchEnv
ENV PATH="/opt/torchEnv/bin:$PATH"
#RUN source /opt/torchEnv/bin/activate
RUN pip install ipykernel ipython torch torchinfo torchvision matplotlib numpy h5py tifffile tqdm tensorboard tensorboard-plugin-profile scipy opencv-python 

FROM ubuntu:24.04 as runner
COPY --from=compiler /opt/torchEnv /opt/torchEnv

