#!/usr/bin/bash 

# initialize root dir
CONVINSE_ROOT=$(pwd)

## check argument length
if [[ $# -lt 1 ]]
then
	echo "Error: Invalid number of options: Please specify the data which should be downloaded."
	echo "Usage: bash scripts/download.sh <DATA_FOR_DOWNLOAD>"
	exit 0
fi

case "$1" in
"convinse")
    echo "Downloading CONVINSE data..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/convinse.zip
    mkdir -p _data/convmix/
    unzip convinse.zip -d _data/convmix/
    rm convinse.zip
    echo "Successfully downloaded CONVINSE data!"
    ;;
"convmix")
    echo "Downloading ConvMix dataset..."
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
    echo "Successfully downloaded ConvMix dataset!"
    ;;
"wikipedia")
    echo "Downloading Wikipedia dump..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/wikipedia.zip
    mkdir -p _data/convmix/
    unzip wikipedia.zip -d _data/convmix/
    rm wikipedia.zip
    echo "Successfully downloaded Wikipedia dump!"
    ;;
"annotated")
    echo "Downloading annotated ConvMix data..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/annotated.zip
    mkdir -p _intermediate_representations/convmix/
    unzip annotated.zip -d _intermediate_representations/convmix/
    rm annotated.zip
    echo "Successfully downloaded annotated ConvMix data!"
    ;;
"data")
    echo "Downloading general repo data..."
    wget http://qa.mpi-inf.mpg.de/convinse/data.zip
    unzip data.zip -d _data
    rm data.zip
    echo "Successfully downloaded general repo data!"
    ;;
*)
    echo "Error: Invalid specification of the data. Data $1 could not be found."
	exit 0
    ;;
esac
