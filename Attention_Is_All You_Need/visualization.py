"""
Visualization utilities for transformer models.

This module contains functions for visualizing various aspects of transformer
models, such as attention masks and attention weights.
"""

import pandas as pd
import altair as alt
import torch
from typing import Optional

def subsequent_mask(size: int) -> torch.Tensor:
    """Create a mask to hide future positions in the sequence.
    
    Args:
        size: The sequence length
        
    Returns:
        A boolean mask tensor of shape (1, size, size) where True values
        allow attention and False values prevent attention
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def visualize_mask(mask_size: int = 20) -> alt.Chart:
    """Create a visualization of the subsequent mask.
    
    Args:
        mask_size: Size of the mask to visualize
        
    Returns:
        An Altair chart object representing the mask
    """
    LS_data = pd.concat(
        [pd.DataFrame(
            {"Subsequent Mask": subsequent_mask(mask_size)[0][x, y].flatten(),
             "Window": y,
             "Masking": x,
            }
            )
            for y in range(mask_size)
            for x in range(mask_size)
            ]
    )
    return (alt.Chart(LS_data).mark_rect().properties(height=250, width=250).encode(
        x='Window:O',
        y='Masking:O',
        color=alt.Color('Subsequent Mask:Q', scale=alt.Scale(scheme='viridis'))
    ).interactive())

def display_chart(chart: alt.Chart, filename: Optional[str] = None) -> Optional[alt.Chart]:
    """Display an Altair chart or save it to a file.
    
    Args:
        chart: The Altair chart to display
        filename: Optional filename to save the chart to
        
    Returns:
        The chart object if successful, None otherwise
    """
    try:
        # Try to display the chart interactively
        return chart.display()
    except Exception as e:
        print(f"Warning: Could not display chart interactively: {e}")
        try:
            # Fall back to saving as HTML
            if filename is None:
                filename = "visualization.html"
            chart.save(filename)
            print(f"Visualization saved to {filename}")
            return chart
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    mask_chart = visualize_mask()
    display_chart(mask_chart, "mask_visualization.html")