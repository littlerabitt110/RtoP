# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 12:08:53 2025

@author: chris
"""
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# Update the data with training time
training_time_data = {
    'Model': [
        'Linear Regression', 'Ridge', 'Lasso', 'MLP', 'Gradient Boosting', 
        'Random Forest', 'Decision Tree', 'XGBoost', 'KNN', 'Stacking', 
        'Cat Boost', 'Light GBM'
    ],
    'v3o': [0.01, 0.00, 0.01, 0.36, 0.10, 0.24, 0.00, 0.07, 0.00, 1.77, 1.32, 0.05],
    'v4u': [0.01, 0.01, 0.01, 0.13, 0.08, 0.00, 0.22, 0.06, 0.00, 1.25, 0.03, 0.06]
}

# Create DataFrame
df_time = pd.DataFrame(training_time_data)

# Define professional colors (e.g., Color Universal Design palette)
colors = {
    'v3o': '#E69F00',  # orange
    'v4u': '#56B4E9'    # blue
}

# Replot with adjustments for better clarity in downsized image

plt.figure(figsize=(10, 6))  # Slightly smaller figure to preserve clarity at small sizes
bar_width = 0.35
x = range(len(df_time['Model']))

# Clearer fonts and gridlines for better readability when downsized
plt.bar(x, df_time['v3o'], width=bar_width, label='v3o', color=colors['v3o'])
plt.bar([i + bar_width for i in x], df_time['v4u'], width=bar_width, label='v4u', color=colors['v4u'])

plt.xlabel('Model', fontsize=12)
plt.ylabel('Training Time (s)', fontsize=12)
plt.title('Training Time Comparison: v3o vs v4u', fontsize=14)
plt.xticks([i + bar_width / 2 for i in x], df_time['Model'], rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the clearer version with 900 DPI
clear_training_time_path = 'training_time_comparison_clear.png'
plt.savefig(clear_training_time_path, dpi=1500)

clear_training_time_path
