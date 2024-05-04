#!/bin/bash

# Use random file name cuz it doesn't really matter
filename=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 8)

# Start packet capture
timeout 10 tcpdump -i any -n dst port 20000 -w "$filename" &
 
# Run attack scripts
python trip_IED4_SS1.py
python close_IED4_SS1.py

# Upload packet capture to AWS
# scp -i ec2_login_key.pem "$filename" ec2-user@3.149.237.185:~/tripclose/
