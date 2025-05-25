import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def calculate_statistics(csv_dir, output_dir):
    """
    Calculate statistics from the YOLO benchmark CSV file.
    Args:
        csv_dir (str): Path to the directory containing the CSV files.
        output_dir (str): Directory to save the output statistics.
    """

    # Load CSV
    # csv files are expected to be named like '<videoname>_result.csv'
    csv_paths = [f'{csv_dir}/{file}' for file in os.listdir(csv_dir) if file.endswith('_result.csv')]
    if not csv_paths:
        print(f'No CSV files found in {csv_dir}.')
        return
    df = pd.DataFrame()
    for csv_path in csv_paths:
        print(f'Processing {csv_path}...')
        temp_df = pd.read_csv(csv_path)
        # Ensure numeric frame
        temp_df['frame'] = pd.to_numeric(temp_df['frame'], errors='coerce')
        temp_df = temp_df.dropna(subset=['frame'])
        temp_df['frame'] = temp_df['frame'].astype(int)
        # remove first 10 frames since they are usually warmup frames
        temp_df = temp_df[temp_df['frame'] >= 10]
        
        # get the last frame number
        first_readFrame_time = temp_df[temp_df['func'] == 'readFrame']['start'].min()
        max_frame = temp_df['frame'].max()
        last_display_time = temp_df[temp_df['func'] == 'displayFrame']['start'].max()

        average_fps = max_frame / (last_display_time - first_readFrame_time) * 1e6  # Convert to FPS
        print(f'Average FPS for {csv_path}: {average_fps:.2f}')
        with open(f'{output_dir}/meta.txt', 'a') as f:
            f.write(f'{csv_path}: {average_fps:.2f} FPS\n')
        df = pd.concat([df, temp_df], ignore_index=True)
    
    # Ensure numeric frame
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['frame'] = df['frame'].astype(int)

    # Calculate statistics
    stats = {
        'mean': df.groupby('func')['latency'].mean(),
        'std': df.groupby('func')['latency'].std(),
        'min': df.groupby('func')['latency'].min(),
        'max': df.groupby('func')['latency'].max()
    }

    # Save statistics to CSV
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f'{output_dir}/statistics.csv')
    print(f'Statistics saved to {output_dir}/statistics.csv')

def main():
    parser = argparse.ArgumentParser(description='Boxplot from YOLO benchmark CSV.')
    parser.add_argument('csv_file_dir', help='Path to the CSV files directory')
    args = parser.parse_args()
    calculate_statistics(args.csv_file_dir, args.csv_file_dir)

if __name__ == '__main__':
    main()
