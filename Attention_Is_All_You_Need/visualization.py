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

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


example_learning_schedule()

if __name__ == "__main__":
    # Example usage
    mask_chart = visualize_mask()
    display_chart(mask_chart, "mask_visualization.html")