import pandas as pd
import matplotlib.pyplot as plt
import argparse

EX_FRAMES = 40  # Number of frames to exclude for warmup
def plot_boxplot(csv_path):
    """
    Create a boxplot from the YOLO benchmark CSV file.
    Args:
        csv_path (str): Path to the CSV file containing benchmark data.
    """
    output_dir = csv_path.rsplit('/', 1)[0]  # Use the directory of the CSV file for output
    if not output_dir:
        output_dir = '.'

    # Load CSV
    df = pd.read_csv(csv_path)
    # Ensure numeric frame
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['frame'] = df['frame'].astype(int)
    # remove first 10 frames since they are usually warmup frames
    df = df[df['frame'] >= EX_FRAMES]

    funcs = df['func'].unique() # 0_readFrame, 1_preprocess, ...
    if len(funcs) == 0:
        print("No functions found in the CSV file.")
        return
    # sort functions by prefix number
    funcs = sorted(funcs, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf'))
    colors = plt.cm.tab20.colors

    # aggregate by function and frame
    data = {func: [] for func in funcs}
    for func in funcs:
        func_data = df[df['func'] == func]
        for frame in range(func_data['frame'].min(), func_data['frame'].max() + 1):
            frame_data = func_data[func_data['frame'] == frame]
            if not frame_data.empty:
                data[func].append(frame_data['latency'].values[0])
            else:
                data[func].append(0)  # No data for this frame
    # Prepare data for boxplot
    box_data = [data[func] for func in funcs]
    # Create boxplot
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_prop_cycle(color=colors[:len(funcs)])  # Set colors for boxplot
    bp = ax.boxplot(
        box_data, 
        labels=funcs,
        patch_artist=True,  # Fill boxes with color
    )
    for b, c in zip(bp['boxes'], colors[:len(funcs)]):
        b.set(color=c, linewidth=1.5)
        b.set_facecolor(c)
    
    plt.title('Boxplot of Latencies by Function')
    plt.xlabel('Function')
    plt.ylabel('Latency (µs)')
    plt.ylim(0, 20000)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig( f'{output_dir}/box_plot.png', dpi=300)

def main():
    parser = argparse.ArgumentParser(description='Boxplot from YOLO benchmark CSV.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    args = parser.parse_args()
    plot_boxplot(args.csv_file)

if __name__ == '__main__':
    main()
