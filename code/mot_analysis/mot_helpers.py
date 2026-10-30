import os
import sys
from pathlib import Path
# add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import math
import yaml
from utils import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
from scipy.spatial.distance import pdist

# Load or initialize config
CONFIG_PATH = Path(__file__).resolve().with_name("mot_config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
WIDTH, HEIGHT = DISPLAY_WIDTH_PIXELS, DISPLAY_HEIGHT_PIXELS
BALL_RADIUS  = int(20 * 1.75)
xmin, xmax = BALL_RADIUS, WIDTH - BALL_RADIUS
ymin, ymax = BALL_RADIUS, HEIGHT - BALL_RADIUS
Lx, Ly = xmax - xmin, ymax - ymin


def _reflect(s, L):
    """Triangle-wave reflection onto [0, L]. Handles any real s, L>0."""
    if L <= 0:
        return 0.0
    r = s % (2.0 * L)
    return r if r <= L else (2.0 * L - r)

def ball_positions_at_times(trial_index, fps=30, mode="discrete"):
    output = []
    trial = config["trials"][trial_index]
    num_objects, num_targets, trial_duration, speed= trial["params"]
    speed = int(speed * 1.75)

    # Normalize directions
    dirs = []
    for d in trial["directions"]:
        v = np.array(d)
        v = v / np.linalg.norm(v) * speed
        dirs.append(v.tolist())

    objects = [{"pos": list(pos), "dir": dir[:]} for pos, dir in zip(trial["locations"], dirs)]
    target_indices = trial["targets"]

    target_indices = trial.get("targets", [])

    times = trial_duration * fps

    for t in range(times):
        for obj in objects:
            obj["pos"][0] += int(obj["dir"][0])
            obj["pos"][1] += int(obj["dir"][1])
            if obj["pos"][0] <= BALL_RADIUS or obj["pos"][0] >= WIDTH - BALL_RADIUS:
                obj["dir"][0] *= -1
            if obj["pos"][1] <= BALL_RADIUS or obj["pos"][1] >= HEIGHT - BALL_RADIUS:
                obj["dir"][1] *= -1

        output.append(([obj["pos"][:] for obj in objects], target_indices))
    
    return output

   
def target_ball_positions_at_times(trial_index, fps=30):
    targets_at_times =  [[positions[i] for i in target_indices] for positions, target_indices in ball_positions_at_times(trial_index, fps)]
    return [(targets, safe_centroid(targets)) for targets in targets_at_times]
# --- helpers ---
def safe_centroid(points):
    # print(f"input = {points}")
    pts = np.asarray(points, float)
    if len(pts) < 3:
        # fallback: just mean of points
        return pts.mean(axis=0)
    
    # Step 1: make convex hull to ensure proper order and no self-intersections
    hull = MultiPoint(pts).convex_hull

    # Step 2: if hull is a Polygon, take its centroid; else fallback
    if hull.geom_type == "Polygon":
        output = np.array(hull.centroid.coords[0])
        # print(f"output = {output}")
        return output
    else:
        return pts.mean(axis=0)


def point_in_convex_polygon(point, poly):
    return Polygon(poly).covers(Point(point))

def get_fixation_distance_from_centroid(centroid_positions_list, fixation_start, fixation_end, fixation_centroid, fps=30):
    # Calculate middle time of the fixation
    mid_time = (fixation_start + fixation_end) / 2.0
    
    # Convert time to frame index. 
    # Note: Assuming fixation times are in milliseconds (standard for eye tracking). 
    # If your times are in seconds, remove the `/ 1000.0`.
    frame_index = int((mid_time) * fps) 
    
    # Clamp index to prevent out-of-bounds errors if the fixation outlasts the trial duration
    frame_index = min(frame_index, len(centroid_positions_list) - 1)
    
    # Retrieve the targets' centroid at the calculated frame 
    # (Your previous function returns a list of tuples: (targets, safe_centroid))
    _, targets_centroid = centroid_positions_list[frame_index]
    
    # Calculate Euclidean distance between the fixation and the targets' centroid
    return math.dist(fixation_centroid, targets_centroid)    