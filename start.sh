#!/bin/bash

cd /home/uiu/MyTrolley

source venv/bin/activate

sudo pkill -9 -f python

python main.py &
python following.py &

wait
