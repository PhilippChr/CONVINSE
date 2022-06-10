#!/usr/bin/bash 

# initialize root dir
CONVINSE_ROOT=$(pwd)

# create directories
mkdir -p _benchmarks
mkdir -p _data
mkdir -p _intermediate_representations
mkdir -p _results
mkdir -p _results/convmix
mkdir -p out
mkdir -p out/convmix
mkdir -p out/slurm

# download convmix
mkdir -p _benchmarks/convmix
cd _benchmarks/convmix
wget http://qa.mpi-inf.mpg.de/convinse/train_set.zip
unzip train_set.zip
rm train_set.zip
wget http://qa.mpi-inf.mpg.de/convinse/dev_set.zip
unzip dev_set.zip
rm dev_set.zip
wget http://qa.mpi-inf.mpg.de/convinse/test_set.zip
unzip test_set.zip
rm test_set.zip

# download data
cd $CONVINSE_ROOT
wget http://qa.mpi-inf.mpg.de/convinse/data.zip
unzip data.zip -d _data
rm data.zip

# download 
bash scripts/download.sh wikipedia
bash scripts/download.sh convinse
bash scripts/download.sh annotated