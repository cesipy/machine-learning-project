import pandas as pd
import matplotlib.pyplot as plt
import math

# Function to read CSV file and create grouped bar plots
def create_grouped_bar_plots_from_csv(csv_file, output_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Group by 'criterion' and 'splitter'
    grouped = data.groupby(['criterion', 'splitter', 'test_size'])
    
    # Number of unique groups
    num_groups = len(grouped)
    
    # Determine the number of rows and columns for the plot grid
    num_cols = 3  # You can change this based on your preference
    num_plots_per_row = num_cols
    num_rows_in_grid = math.ceil(num_groups / num_plots_per_row)
    
    # Create a figure and axes for the grid of plots
    fig, axes = plt.subplots(num_rows_in_grid, num_cols, figsize=(15, 5 * num_rows_in_grid))
    
    # Flatten the axes array if there's more than one row
    if num_rows_in_grid > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Iterate over each group and create a bar plot for each
    for group_index, ((criterion, splitter, test_size), group_data) in enumerate(grouped):
        # Extract train and test accuracies for the current group
        train_acc = group_data['train_acc']
        test_acc = group_data['test_acc']
        
        # Create the bar plot for the current group
        bar_width = 0.35
        r1 = range(len(train_acc))
        r2 = [x + bar_width for x in r1]
        
        axes[group_index].bar(r1, train_acc, color='blue', width=bar_width, edgecolor='grey', label='Train Accuracy')
        axes[group_index].bar(r2, test_acc, color='green', width=bar_width, edgecolor='grey', label='Test Accuracy')
        
        # Set y-axis limit
        axes[group_index].set_ylim(0.99875, 1)
        
        # Add labels and title
        axes[group_index].set_ylabel('Accuracy', fontweight='bold')
        axes[group_index].set_title(f'Criterion: {criterion}\nSplitter: {splitter}')
        
        # Add x-tick labels with the values of the given columns
        x_labels = [f'{"None" if math.isnan(row["max_depth"]) else int(row["max_depth"])}, {"None" if math.isnan(row["max_features"]) else int(row["max_features"])}, {row["ccp_alpha"]}, {row["min_impurity_decrease"]}' for _, row in group_data.iterrows()]
        axes[group_index].set_xticks([r + bar_width / 2 for r in range(len(train_acc))])
        axes[group_index].set_xticklabels(x_labels, rotation=45, ha='right')
        
        # Add legend
        axes[group_index].legend(loc='lower left')

    # Remove any unused subplots
    for i in range(num_groups, len(axes)):
        fig.delaxes(axes[i])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

csv_file = 'search_results.csv'
output_file = 'accuracy_plot.png'
create_grouped_bar_plots_from_csv(csv_file, output_file)
