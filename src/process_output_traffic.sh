#!/bin/bash

DIRECTORY=/home/ec2-user/raw_traffic
AGGREGATED_CSV=/home/ec2-user/assets/aggregated.csv

# Function to run Python script and delete file
process_file() {
    filename="$1"
    filepath="$DIRECTORY/$filename"
    cicflowmeter -f "${filepath}.pcap" -c "${filepath}.csv"
    python map_output_traffic.py "${filepath}.csv" "$AGGREGATED_CSV"
    rm "${filepath}.csv"
    # rm "${filepath}.pcap" # Delete .pcap file
    # Temporarily just move the .pcap file out of the way
    mv "${filepath}.pcap" "/home/ec2-user/discarded_traffic"
}

# Monitor directory for new files
inotifywait -m -e create -e moved_to --format '%f' "$DIRECTORY" | while read file
do
    echo "New file detected: $file"
    process_file "${file%.*}" # Passing file without extension
done
