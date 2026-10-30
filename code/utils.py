from __future__ import annotations
import math

import pandas as pd
import numpy as np

DISPLAY_WIDTH_PIXELS = 3840
DISPLAY_HEIGHT_PIXELS = 2160

DISPLAY_WIDTH_CM = 60
DISPLAY_HEIGHT_CM = 34

VIEWING_DISTANCE_CM = 80

color_map = {
    "Novices": "tab:blue",
    "Experienced": "tab:orange",
    "Experts": "tab:green",
    "Students": "tab:blue",
    "ATCs": "tab:orange",}

# 2. Define the specific order you want on the X-axis
# This ensures the categories appear in the sequence you prefer
group_order = ["Novices", "Experienced", "Experts", "Students", "ATCs"]

def pixels_to_degrees(pixel_length):
    # 1. Calculate the total visual angle of the screen's width in radians, then convert to degrees
    screen_width_rad = 2 * math.atan(DISPLAY_WIDTH_CM / (2 * VIEWING_DISTANCE_CM))
    screen_width_deg = math.degrees(screen_width_rad)
    
    # 2. Calculate the average Pixels Per Degree (PPD)
    ppd = DISPLAY_WIDTH_PIXELS / screen_width_deg
    
    # 3. Convert the input pixel length to degrees
    return pixel_length / ppd


def aggregate_global_feature(df, feature_col, agg_func, participant_col='participant_id', group_col='group'):
    """
    Aggregates a feature per participant, ignoring the condition variable.
    """
    # Group by participant and group to keep the label, then apply the function
    agg_df = df.groupby([participant_col, group_col], as_index=False)[feature_col].agg(agg_func)
    
    return agg_df
