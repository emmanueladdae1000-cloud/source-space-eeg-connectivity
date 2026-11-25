import numpy as np
import matplotlib
# Use the 'Agg' backend so Python saves the image cleanly on a remote server
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_clean_brainwaves(csv_file="Y_clean.csv", n_points=500, sfreq=250.0):
    """
    Loads Y_clean.csv and plots all 10 cortical source signals stacked vertically
    over a 2-second window (500 samples at 250 Hz).
    """
    # 1. Load the 10 x 15000 signal matrix from Step 2
    Y_clean = np.loadtxt(csv_file, delimiter=",")
    n_nodes = Y_clean.shape[0]
    
    # Create a time axis in seconds for the first 500 samples (0 to 2.0 seconds)
    time_sec = np.arange(n_points) / sfreq

    # 2. Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    
    # Vertical spacing between the 10 stacked channels
    spacing = 8.0
    yticks = []
    yticklabels = []

    # Color-code channels by their Step 1 Ground Truth motif
    for i in range(n_nodes):
        node_num = i + 1
        offset = (n_nodes - 1 - i) * spacing  # Put Node 1 at the top, Node 10 at the bottom
        yticks.append(offset)
        
        if node_num in [1, 2, 3]:
            color = "#1f77b4"  # Blue: Indirect Cascade (Node 1 -> 2 -> 3)
            label_tag = f"Node {node_num} (Cascade)"
        elif node_num in [4, 5, 6]:
            color = "#d62728"  # Red: Common Source (Node 4 -> 5, 6)
            label_tag = f"Node {node_num} (Common Src)"
        elif node_num in [7, 8]:
            color = "#2ca02c"  # Green: Direct Connection (Node 7 -> 8)
            label_tag = f"Node {node_num} (Direct)"
        else:
            color = "#7f7f7f"  # Gray: Unconnected background nodes (Nodes 9, 10)
            label_tag = f"Node {node_num} (Isolated)"
            
        yticklabels.append(label_tag)
        
        # Plot the wave shifted to its vertical row
        ax.plot(time_sec, Y_clean[i, :n_points] + offset, color=color, linewidth=1.2)
        # Draw a faint dashed zero-line for each channel
        ax.axhline(offset, color="black", linestyle="--", linewidth=0.4, alpha=0.4)

    # 3. Format labels and title
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Phase 1, Step 2: Clean Simulated Cortical Brainwaves (Y_clean - First 2 Seconds)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, time_sec[-1])
    
    plt.tight_layout()
    plt.savefig("step2_brainwaves.png")
    plt.close()
    print("Saved brainwave plot to: 'step2_brainwaves.png'")

if __name__ == "__main__":
    plot_clean_brainwaves()