# import numpy as np
import pandas as pd
import sys

def main():
    create_aggregate_file = True
    aggregated_csv_filename = "./assets/aggregated.csv"
    if (len(sys.argv) < 2) or (len(sys.argv) > 3):
        print("Usage: python map_output_traffic.py <raw csv> <aggregated csv>")
        return 
    elif len(sys.argv) == 2:
        print(f"No aggregate file defined, creating one at {aggregated_csv_filename}")
    else:
        try:
            aggregated_traffic_df = pd.read_csv(sys.argv[2])
            aggregated_csv_filename = sys.argv[2]
            create_aggregate_file = False
        except FileNotFoundError:
            print(f"File {aggregated_csv_filename} not found, creating new aggregated file")
            pass

    raw_csv_filename = sys.argv[1]
    try:
        raw_traffic_df = pd.read_csv(raw_csv_filename)
    except FileNotFoundError:
        print(f"File {raw_csv_filename} not found")
        return

    column_mapping = {
        'src_ip': None,
        'dst_ip': None,
        'src_port': None,
        'dst_port': 'Dst Port',
        'protocol': 'Protocol',
        'timestamp': None,
        'flow_duration': 'Flow Duration',
        'flow_byts_s': 'Flow Bytes/s',
        'flow_pkts_s': 'Flow Packets/s',
        'fwd_pkts_s': 'Fwd Packets/s',
        'bwd_pkts_s': 'Bwd Packets/s',
        'tot_fwd_pkts': 'Total Fwd Packet',
        'tot_bwd_pkts': 'Total Bwd packets',
        'totlen_fwd_pkts': 'Total Length of Fwd Packet',
        'totlen_bwd_pkts': 'Total Length of Bwd Packet',
        'fwd_pkt_len_max': 'Fwd Packet Length Max',
        'fwd_pkt_len_min': 'Fwd Packet Length Min',
        'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
        'fwd_pkt_len_std': 'Fwd Packet Length Std',
        'bwd_pkt_len_max': 'Bwd Packet Length Max',
        'bwd_pkt_len_min': 'Bwd Packet Length Min',
        'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
        'bwd_pkt_len_std': 'Bwd Packet Length Std',
        'pkt_len_max': 'Packet Length Max',
        'pkt_len_min': 'Packet Length Min',
        'pkt_len_mean': 'Packet Length Mean',
        'pkt_len_std': 'Packet Length Std',
        'pkt_len_var': 'Packet Length Variance',
        'fwd_header_len': 'Fwd Header Length',
        'bwd_header_len': 'Bwd Header Length',
        'fwd_seg_size_min': 'Fwd Seg Size Min',
        'fwd_act_data_pkts': 'Fwd Act Data Pkts',
        'flow_iat_mean': 'Flow IAT Mean',
        'flow_iat_max': 'Flow IAT Max',
        'flow_iat_min': 'Flow IAT Min',
        'flow_iat_std': 'Flow IAT Std',
        'fwd_iat_tot': 'Fwd IAT Total',
        'fwd_iat_max': 'Fwd IAT Max',
        'fwd_iat_min': 'Fwd IAT Min',
        'fwd_iat_mean': 'Fwd IAT Mean',
        'fwd_iat_std': 'Fwd IAT Std',
        'bwd_iat_tot': 'Bwd IAT Total',
        'bwd_iat_max': 'Bwd IAT Max',
        'bwd_iat_min': 'Bwd IAT Min',
        'bwd_iat_mean': 'Bwd IAT Mean',
        'bwd_iat_std': 'Bwd IAT Std',
        'fwd_psh_flags': 'Fwd PSH Flags',
        'bwd_psh_flags': 'Bwd PSH Flags',
        'fwd_urg_flags': 'Fwd URG Flags',
        'bwd_urg_flags': 'Bwd URG Flags',
        'fin_flag_cnt': 'FIN Flag Count',
        'syn_flag_cnt': 'SYN Flag Count',
        'rst_flag_cnt': 'RST Flag Count',
        'psh_flag_cnt': 'PSH Flag Count',
        'ack_flag_cnt': 'ACK Flag Count',
        'urg_flag_cnt': 'URG Flag Count',
        'ece_flag_cnt': 'ECE Flag Count',
        'down_up_ratio': 'Down/Up Ratio',
        'pkt_size_avg': 'Average Packet Size',
        'init_fwd_win_byts': 'FWD Init Win Bytes',
        'init_bwd_win_byts': 'Bwd Init Win Bytes',
        'active_max': 'Active Max',
        'active_min': 'Active Min',
        'active_mean': 'Active Mean',
        'active_std': 'Active Std',
        'idle_max': 'Idle Max',
        'idle_min': 'Idle Min',
        'idle_mean': 'Idle Mean',
        'idle_std': 'Idle Std',
        'fwd_byts_b_avg': 'Fwd Bytes/Bulk Avg',
        'fwd_pkts_b_avg': 'Fwd Packet/Bulk Avg',
        'bwd_byts_b_avg': 'Bwd Bytes/Bulk Avg',
        'bwd_pkts_b_avg': 'Bwd Packet/Bulk Avg',
        'fwd_blk_rate_avg': 'Fwd Bulk Rate Avg',
        'bwd_blk_rate_avg': 'Bwd Bulk Rate Avg',
        'fwd_seg_size_avg': 'Fwd Segment Size Avg',
        'bwd_seg_size_avg': 'Bwd Segment Size Avg',
        'cwr_flag_count': 'CWR Flag Count',
        'subflow_fwd_pkts': 'Subflow Fwd Packets',
        'subflow_bwd_pkts': 'Subflow Bwd Packets',
        'subflow_fwd_byts': 'Subflow Fwd Bytes',
        'subflow_bwd_byts': 'Subflow Bwd Bytes'
    }

    # Dropping irrelevant features and map column names to dataset
    mapped_traffic_df = raw_traffic_df \
        .drop(columns=[
            'src_ip', 
            'dst_ip', 
            'src_port', 
            'timestamp',
            # Dropping empty columns 
            'fwd_urg_flags',
            'fwd_pkts_b_avg',
            'fwd_byts_b_avg',
            'fwd_blk_rate_avg',
            'cwr_flag_count',
            'bwd_urg_flags',
            'bwd_psh_flags'
        ]) \
        .rename(columns=column_mapping)

    if create_aggregate_file:
        result_df = mapped_traffic_df
    else:
        # Union the rows of both DataFrames into a single DataFrame
        result_df = pd.concat([aggregated_traffic_df, mapped_traffic_df], ignore_index=True)
        print(f"aggregated_df->{aggregated_traffic_df.shape}")

    # Write the result back
    result_df.to_csv(aggregated_csv_filename, index=False)
    print(f"""Merged data written to {aggregated_csv_filename}\nShapes: 
        raw_df->{raw_traffic_df.shape}
        mapped_df->{mapped_traffic_df.shape}
        result_df->{result_df.shape}
    """)


if __name__ == "__main__":
    main()

## TODO
# Set up listener netcat to output .pcap files to a dir
# Set up script to look for new .pcap files and run cicflowmeter
# Collect raw csv files, rename/drop and join them into one curated csv file
