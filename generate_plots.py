import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Set the style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def set_common_style(ax, title, xlabel, ylabel):
    """Apply common styling to the plot"""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
def create_performance_plot():
    """Create plot for performance.csv data"""
    # Read the CSV file
    df = pd.read_csv('performance.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Model sammenligning med Coral datasæt', fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for models (consistent across plots)
    model_colors = {
        'hviske-v2': sns.color_palette("colorblind")[0],
        'hviske-tiske': sns.color_palette("colorblind")[1],
        'faster-whisper-large-v3-turbo-ct2': sns.color_palette("colorblind")[2]
    }
    
    # Plot 1: Error Rates (WER & CER)
    ax1 = axes[0]
    
    # Set width of bars
    barWidth = 0.35
    positions = np.arange(len(df['model']))
    
    # Create bars for WER and CER
    bars1 = ax1.bar(positions - barWidth/2, df['wer (mean)'] * 100, barWidth, 
                   color=[model_colors[m] for m in df['model']], alpha=0.8, label='WER')
    
    bars2 = ax1.bar(positions + barWidth/2, df['cer (mean)'] * 100, barWidth,
                   color=[model_colors[m] for m in df['model']], alpha=0.5, hatch='//', label='CER')
    
    # Add error bars
    ax1.errorbar(positions - barWidth/2, df['wer (mean)'] * 100, yerr=df['wer (std)'] * 100, 
                fmt='none', color='black', capsize=5)
    ax1.errorbar(positions + barWidth/2, df['cer (mean)'] * 100, yerr=df['cer (std)'] * 100, 
                fmt='none', color='black', capsize=5)
    
    # Customize the plot
    ax1.set_xticks(positions)
    ax1.set_xticklabels(df['model'], rotation=45, ha='right')
    set_common_style(ax1, 'Word and Character Error Rates', 'Model', 'Error Rate (%)')
    ax1.yaxis.set_major_formatter(PercentFormatter())
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.legend(loc='upper right')
    
    # Plot 2: Processing Time
    ax2 = axes[1]
    
    # Create bars for processing time
    bars3 = ax2.bar(positions, df['time (mean)'], width=0.6, 
                   color=[model_colors[m] for m in df['model']], alpha=0.8)
    
    # Add error bars
    ax2.errorbar(positions, df['time (mean)'], yerr=df['time (std)'], 
                fmt='none', color='black', capsize=5)
    
    # Customize the plot
    ax2.set_xticks(positions)
    ax2.set_xticklabels(df['model'], rotation=45, ha='right')
    set_common_style(ax2, 'Processing Time Comparison', 'Model', 'Processing Time (seconds)')
    
    # Add value labels on top of bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Add a legend
    ax2.legend(['Processing Time'], loc='upper right')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('performance_plot.png', dpi=300, bbox_inches='tight')
    print("Performance plot saved as 'performance_plot.png'")

def create_duration_plot():
    """Create plot for duration.csv data focusing only on processing time"""
    # Read the CSV file
    df = pd.read_csv('duration.csv', skiprows=2)
    
    # Clean up the dataframe
    df.columns = ['duration', 'model', 'wer_mean', 'cer_mean', 'time_mean']
    
    # Sort durations in a logical order
    duration_order = ['5s', '10s', '30s', '1min', '10min', '30min']
    df['duration'] = pd.Categorical(df['duration'], categories=duration_order, ordered=True)
    df = df.sort_values('duration')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create line plot for processing time by duration
    g = sns.lineplot(
        data=df,
        x='duration',
        y='time_mean',
        hue='model',
        style='model',
        markers=True,
        markersize=10,
        linewidth=3,
        dashes=False
    )
    
    # Use log scale for better visualization of wide time differences
    plt.yscale('log')
    
    # Set plot title and labels
    set_common_style(g, 'Tid på at transskribere vs. længden på lyd', 'Længden på lyd', 'Tid på at transskribere')
    
    # Add values as text next to markers
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        for i, row in model_data.iterrows():
            plt.text(
                duration_order.index(row['duration']), 
                row['time_mean'] * 1.1, 
                f"{row['time_mean']:.2f}s",
                horizontalalignment='center',
                size='small',
                color='black',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
    
    # Move legend to the top right outside the plot
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a note about the scale
    plt.annotate('Note: Y-axis uses logarithmic scale to better display time differences', 
                xy=(0.5, -0.1), 
                xycoords='axes fraction',
                ha='center',
                fontsize=10,
                fontstyle='italic')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('duration_plot.png', dpi=300, bbox_inches='tight')
    print("Duration plot saved as 'duration_plot.png'")

if __name__ == "__main__":
    print("Generating performance plot...")
    create_performance_plot()
    
    print("Generating duration plot...")
    create_duration_plot()
    
    print("Done! Both plots have been saved as PNG files.")