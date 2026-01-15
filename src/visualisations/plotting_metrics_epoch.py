import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_metrics_epoch(metrics_df: pd.DataFrame, metrics: list, save_path: str = None):
    """
    Plots training and validation metrics over epochs on a single plot.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics.
        metrics (list): List of metric names to plot.
        save_path (str, optional): Path to save the plot.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set(style="whitegrid")
    
    # Create a melted dataframe for seaborn plotting
    plot_data = []
    
    for metric in metrics:
        val_col = f'val_{metric}'
        # Process validation data if column exists
        if val_col in metrics_df.columns:
            val_rows = metrics_df[['epoch', val_col]].dropna()
            for _, row in val_rows.iterrows():
                plot_data.append({
                    'epoch': row['epoch'],
                    'value': row[val_col],
                    'metric': metric.capitalize(),
                    'type': 'Validation'
                })
    
    if not plot_data:
        print("No data to plot!")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create line plot using seaborn
    ax = sns.lineplot(data=plot_df, x='epoch', y='value', 
                      hue='metric', style='type',
                      markers=True, dashes=True, linewidth=2.5,
                      markersize=10, palette='husl')
    
    # Customize the plot
    ax.set_title('Validation Metrics over Epochs', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    
    # Customize legend
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc='best')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()