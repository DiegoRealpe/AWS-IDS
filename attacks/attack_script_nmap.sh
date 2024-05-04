#!/bin/bash

# Use random file name cuz it doesn't really matter
filename=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 8)

# Start packet capture
timeout 10 tcpdump -i any -n dst port 20000 -w "$filename" &
 
# Run attack scripts
nmap -sV -p 20000 6.87.151.210
nmap -sV -p 20000 6.87.152.210

# Upload packet capture to AWS
scp -i ec2_login_key.pem "$filename" ec2-user@3.149.237.185:/home/ec2-user/nmap/

# Cleanup
rm "$filename"
