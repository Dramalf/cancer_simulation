import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_simulation_data(file_path=None, columns_to_plot=None, title=None, log_scale=False):
    """
    Plot simulation data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file. If None, uses the most recent file.
        columns_to_plot (list): List of column names to plot. If None, plots all percentage columns.
        title (str): Title for the plot. If None, uses a default title.
        log_scale (bool): Whether to use logarithmic scale for y-axis.
    
    Returns:
        None
    """
    # If no file specified, find the most recent one
    if file_path is None:
        csv_files = glob.glob('data/simulation_data_*.csv')
        if not csv_files:
            print("No simulation data CSV files found in the data directory")
            return
        file_path = max(csv_files)
    
    print(f"Loading data from: {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic info about the data
    print(f"Data contains {len(df)} time points")
    print(df.head())
    
    # If no columns specified, use all percentage columns
    if columns_to_plot is None:
        columns_to_plot = [col for col in df.columns if 'percent' in col]
    
    # Define colors for different cell types
    colors = {
        'cancer_percent': 'r',
        'wbc_percent': 'g',
        'dead_percent': 'k',
        'normal_percent': 'b',
        'regen_percent': 'm'
    }
    
    # Create a figure with appropriate size
    plt.figure(figsize=(12, 8))
    
    # Plot each specified column against timestamp
    for column in columns_to_plot:
        if column in df.columns:
            color = colors.get(column, 'gray')  # Default to gray if column not in colors dict
            label = column.replace('_percent', '').capitalize() + ' %'
            plt.plot(df['timestamp'], df[column], f'{color}-', label=label)
    
    # Add labels and title
    plt.xlabel('Timestamp', fontsize=12)
    
    if log_scale:
        plt.ylabel('Percentage (log scale)', fontsize=12)
        plt.yscale('log')
    else:
        plt.ylabel('Percentage', fontsize=12)
        plt.ylim(0, 100)
    
    # Set default title if none provided
    if title is None:
        title = 'Cell Population Dynamics Over Time'
        if log_scale:
            title += ' (Log Scale)'
    
    plt.title(title, fontsize=14)
    
    # Add legend and grid
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Plot minor cell populations with log scale
plot_simulation_data(
    columns_to_plot=['cancer_percent', 'wbc_percent', 'dead_percent', 'regen_percent'],
    title='Minor Cell Population Dynamics Over Time',
    log_scale=True
)
