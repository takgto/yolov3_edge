import pandas as pd
import matplotlib.pyplot as plt
import argparse

EX_FRAMES = 40  # Number of frames to exclude for warmup
def plot_gantt(csv_path, max_frame=15):
    """
    Create Gantt charts from the YOLO benchmark CSV file.
    Args:
        csv_path (str): Path to the CSV file containing benchmark data.
        max_frame (int): Number of initial frames to plot.
    """

    output_dir = csv_path.rsplit('/', 1)[0]  # Get directory from CSV path
    if not output_dir:
        output_dir = '.'
    output_file_base = f'{output_dir}/gantt_charts'

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure numeric frame
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df = df.dropna(subset=['frame'])
    df['frame'] = df['frame'].astype(int)
    # remove first EX_FRAMES frames since they are usually warmup frames
    df = df[df['frame'] >= EX_FRAMES]

    # Filter frames
    df = df[df['frame'] < max_frame + EX_FRAMES]  # Include frames up to max_frame + EX_FRAMES for better visualization
    df['start'] = df['start'] - df['start'].min()  # Normalize start times
    
    funcs = df['func'].unique() # 0_readFrame, 1_preprocess, ...
    if len(funcs) == 0:
        print("No functions found in the CSV file.")
        return
    # sort functions by prefix number
    funcs = sorted(funcs, key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf'))
    colors = plt.cm.tab20.colors
    
    legend_handles = [
        plt.Rectangle((0,0),1,1, color=colors[i % len(colors)])
        for i in range(len(funcs))
    ]

    # Gantt chart by thread
    # Unique threads
    threads = sorted(df['tid'].unique())
    n_threads = len(threads)
    
    # Create subplots per thread
    fig, axs = plt.subplots(n_threads, 1, figsize=(12, 4*n_threads), sharex=True)
    if n_threads == 1:
        axs = [axs]
    
    for ax, tid in zip(axs, threads):
        df_thread = df[df['tid'] == tid]
        for frame in sorted(df_thread['frame'].unique()):
            sub = df_thread[df_thread['frame'] == frame]
            for idx, func in enumerate(funcs):
                row = sub[sub['func'] == func]
                if row.empty:
                    continue
                start = row['start'].values[0]
                duration = row['latency'].values[0]
                ax.barh(y=frame, left=start, width=duration, height=0.8,
                        color=colors[idx % len(colors)], 
                        label=func if (frame == 0 and ax == axs[0]) else "")
        ax.set_ylabel(f'Thread {tid}')
        ax.set_yticks(range(EX_FRAMES, max_frame + EX_FRAMES))
        ax.set_yticklabels([f'Frame {i}' for i in range(EX_FRAMES, max_frame+EX_FRAMES)])
        ax.set_ylim(EX_FRAMES-0.5, max_frame + EX_FRAMES - 0.5)
        ax.legend(legend_handles, funcs, loc='lower right', fontsize='small')
    
    axs[-1].set_xlabel('Time since start (µs)')
    fig.suptitle(f'Gantt Chart by Thread (first {max_frame} frames, excluding warmup frames)', fontsize=16)

    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_file_base}_thread.png', dpi=300)

    # Gantt chart by frame
    fig, ax = plt.subplots(figsize=(12, 6))
    for frame in sorted(df['frame'].unique()):
        sub = df[df['frame'] == frame]
        for idx, func in enumerate(funcs):
            row = sub[sub['func'] == func]
            if row.empty:
                continue
            start = row['start'].values[0]
            dur   = row['latency'].values[0]
            ax.barh(
                y=frame,
                width=dur,
                left=start,
                height=0.8,
                color=colors[idx % len(colors)],
                label=func if frame == 0 else ""
            )

    ax.set_yticks(range(EX_FRAMES, max_frame+EX_FRAMES))
    ax.set_yticklabels([f"Frame {i}" for i in range(EX_FRAMES, max_frame+EX_FRAMES)])
    ax.set_xlabel("Time since start (µs)")
    ax.set_ylabel("Frame")

    ax.set_title(f"Gantt Chart by Frame (first {max_frame} frames, excluding warmup frames)")
    ax.legend(legend_handles, funcs, loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(f'{output_file_base}_frame.png', dpi=300)


def main():
    parser = argparse.ArgumentParser(description='Plot Gantt chart from YOLO benchmark CSV.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--frames', type=int, default=15, help='Number of initial frames to plot')
    args = parser.parse_args()
    plot_gantt(args.csv_file, max_frame=args.frames)

if __name__ == '__main__':
    main()
