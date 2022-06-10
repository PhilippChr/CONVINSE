#!/usr/bin/bash 
# read config parameter: if no present, stick to convinse.yaml
CONFIG=${1:-config/convmix/convinse.yml}

# adjust name to log
IFS='/' read -ra NAME <<< "$CONFIG"
DATA=${NAME[1]}
IFS='.' read -ra NAME <<< "${NAME[2]}"
NAME=${NAME[0]}
OUT=out/${DATA}/silver_annotation_${NAME}.out
mkdir -p out/${DATA}

# start script
nohup python -u convinse/distant_supervision/silver_annotation.py --inference $CONFIG  > $OUT 2>&1 &