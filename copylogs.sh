#!/bin/bash
#--start config
LD=~/Desktop/Robotics/deeprl/avfone/
RD=/home/mlab-train/Desktop/deeprl/avfone/__runs
ID=~/.ssh/id_rsa
USER=mlab-train
HOST=158.130.228.202

BD="$LD/'date +%F'"
mkdir $BD
scp -ri $ID $USER@$HOST:$RD/. $BD
