#!/bin/bash

RAW_PCAP_DIR=/home/ec2-user/raw_traffic
DISCARDED_PCAP_DIR=/home/ec2-user/discarded_traffic
ASSETS_CSV_DIR=/home/ec2-user/assets

# Function to run Python script and delete file
process_file() {
    filename="$1"
    pcap_path="${RAW_PCAP_DIR}/${filename}.pcap"
    csv_path="${ASSETS_CSV_DIR}/${filename}.csv"
    cicflowmeter -f "${pcap_path}" -c "${csv_path}"
    mv "${pcap_path}" "$DISCARDED_PCAP_DIR"
    python map_output_traffic.py "${csv_path}" "${ASSETS_CSV_DIR}/aggregated.csv"
    rm "${csv_path}"
}

# Monitor RAW_PCAP_DIR for new files
inotifywait -m -e create -e moved_to --format '%f' "${RAW_PCAP_DIR}" | while read file
do
    echo "New file detected: ${file}"
    process_file "${file%.*}" & # Passing file without extension
done
