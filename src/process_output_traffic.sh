#!/bin/bash

DIRECTORY=~/raw_traffic

# Function to run Python script and delete file
process_file() {
    filename="$1"
    filepath="$DIRECTORY/$filename"
    cicflowmeter -f "${filepath}.pcap" -c "${filepath}.csv"
    python map_output_traffic.py "${filepath}.csv" "aggregated.csv"
    rm "${filepath}.csv"
    # rm "${filepath}.pcap" # Delete .pcap file
    # Temporarily just move the .pcap file out of the way
    mv "${filepath}.pcap" "~/discarded_traffic/${filename}.pcap"
}

# Monitor directory for new files
inotifywait -m -e create --format '%f' "$DIRECTORY" | while read file
do
    echo "New file detected: $file"
    process_file "${file%.*}" # Passing file without extension
done
