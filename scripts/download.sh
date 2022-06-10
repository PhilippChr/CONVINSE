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
    echo "Downloading data for '$1'..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/convinse.zip
    mkdir -p _data/convmix/
    unzip convinse.zip -d _data/convmix/
    rm convinse.zip
    echo "Successfully downloaded data for '$1'!"
    ;;
"nc_all-clocq_bm25-fid")
    echo "Data for '$1' not (yet) available!"
    ;;
"nc_init-clocq_bm25-fid")
    echo "Data for '$1' not (yet) available!"
    ;;
"nc_init_prev-clocq_bm25-fid")
    echo "Data for '$1' not (yet) available!"
    ;;
"nc_prev-clocq_bm25-fid")
    echo "Data for '$1' not (yet) available!"
    ;;
"wikipedia")
    echo "Downloading data for '$1'..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/wikipedia.zip
    mkdir -p _data/convmix/
    unzip wikipedia.zip -d _data/convmix/
    rm wikipedia.zip
    echo "Successfully downloaded data for '$1'!"
    ;;
"annotated")
    echo "Downloading data for '$1'..."
    wget http://qa.mpi-inf.mpg.de/convinse/convmix_data/annotated.zip
    mkdir -p _intermediate_representations/convmix/
    unzip annotated.zip -d _intermediate_representations/convmix/
    rm annotated.zip
    echo "Successfully downloaded data for '$1'!"
    ;;
*)
    echo "Error: Invalid specification of the data. Data $1 could not be found."
	exit 0
    ;;
esac