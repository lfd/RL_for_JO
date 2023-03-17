FROM ubuntu:22.04

LABEL author="Maja Franz <maja.franz@othr.de>"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"

# Install required packages
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa # For Python 3.9
RUN apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-distutils \
    python3.9-dev \
    wget \
    git \
    r-base \
    libv8-dev \
    libreadline-dev \
    zlib1g-dev \
    texlive-latex-base \
    texlive-science \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-publishers \
    texlive-bibtex-extra \
    texlive-luatex \
    biber \
    sudo \
    latexmk

# Install R packages for plotting
RUN R -e "install.packages('tidyverse',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('ggh4x',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('patchwork',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('tikzDevice',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('scales',dependencies=TRUE, repos='http://cran.rstudio.com/')"

# Let Python 3.9 be global python version
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Add user
RUN useradd -m -G sudo -s /bin/bash repro && echo "repro:repro" | chpasswd
USER repro

WORKDIR /home/repro/postgres

# setup PostgreSQL
ENV RLJO_PSQL_BASE=/home/repro/postgres

## PostgreSQL
ENV RLJO_PSQL_DATA_DIRECTORY=$RLJO_PSQL_BASE/database
ENV RLJO_PSQL_SRC_DIRECTORY="$RLJO_PSQL_BASE/postgresql-16.0"
ENV RLJO_PSQL_INSTALL_DIRECTORY="$RLJO_PSQL_BASE/install"
ENV RLJO_PG_HINT_PLAN_BASE="$RLJO_PSQL_BASE/pg_hint_plan"
ENV RLJO_PG_HINT_PLAN_SRC_DIRECTORY="$RLJO_PG_HINT_PLAN_BASE/pg_hint_plan-REL16_1_6_0"

WORKDIR $RLJO_PSQL_BASE
RUN wget https://ftp.postgresql.org/pub/source/v16.0/postgresql-16.0.tar.gz
RUN tar xvfz postgresql-16.0.tar.gz
RUN mkdir $RLJO_PSQL_INSTALL_DIRECTORY
RUN mkdir $RLJO_PSQL_DATA_DIRECTORY
WORKDIR $RLJO_PSQL_SRC_DIRECTORY
RUN ./configure --prefix=$RLJO_PSQL_INSTALL_DIRECTORY --enable-debug
RUN make -j $(nproc)
RUN make install
WORKDIR $RLJO_PSQL_BASE

ENV PATH=$RLJO_PSQL_BASE/install/bin:$PATH
ENV LD_LIBRARY_PATH=$RLJO_PSQL_BASE/install/lib/

RUN initdb -D $RLJO_PSQL_DATA_DIRECTORY --user=postgres
RUN pg_ctl -D $RLJO_PSQL_DATA_DIRECTORY -l $RLJO_PSQL_BASE/logfile start

## PG hint plugin
RUN mkdir $RLJO_PG_HINT_PLAN_BASE
WORKDIR $RLJO_PG_HINT_PLAN_BASE
RUN wget https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL16_1_6_0.tar.gz
RUN tar xvfz REL16_1_6_0.tar.gz
WORKDIR $RLJO_PG_HINT_PLAN_SRC_DIRECTORY
RUN make -j $(nproc)
RUN make install

ENV PATH=$PATH:/home/repro/postgres/install/bin


# setup join order benchmark
WORKDIR /home/repro

ENV RLJO_JOB_BASE=/home/repro/JOB

RUN mkdir $RLJO_JOB_BASE
RUN git clone -n https://github.com/danolivo/jo-bench $RLJO_JOB_BASE/jo-bench

WORKDIR $RLJO_JOB_BASE/jo-bench
RUN git checkout a2019f9

# Add artifacts (from host) to home directory
ADD --chown=repro:repro . /home/repro/qce24_repro

WORKDIR /home/repro/qce24_repro

# install python packages
ENV PATH=$PATH:/home/repro/.local/bin
# set default python version to 3.9
RUN echo 'alias python="python3.9"' >> /home/repro/.bashrc

RUN python3.9 -m pip install -r experimental_analysis/requirements.txt

# Experiments can be run, plots can be generated or paper can be built when
# container is started, see options in README or run script
ENTRYPOINT ["./scripts/run.sh"]
CMD ["bash"]
