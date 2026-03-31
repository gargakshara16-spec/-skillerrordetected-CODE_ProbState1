import os
import pickle
import numpy as np
from collections import defaultdict
from nuscenes.nuscenes import NuScenes

# ======================
# CONFIG & PATHS
# ======================
# Ensure these paths are correct for your MacBook
DATAROOT = "/Users/aksharagarg/Downloads/nuscenes"
VERSION = "v1.0-trainval"
SAVE_PATH = "trajectory_dataset.pkl"

# nuScenes keyframes are 2Hz (annotations every 0.5s)
PAST_STEPS = 4   # 2 seconds of history
FUTURE_STEPS = 6 # 3 seconds of prediction
TOTAL_STEPS = PAST_STEPS + FUTURE_STEPS

TARGET_CLASSES = ["pedestrian", "bicycle"]
MIN_MOVE_DISTANCE = 1.0 # Filter out static objects

def run_extraction():
    # 1. Check if we already did the work
    if os.path.exists(SAVE_PATH):
        print(f"⏩ {SAVE_PATH} already exists. Skipping extraction step.")
        return

    # 2. Initialize nuScenes
    print("📂 Initializing nuScenes and starting extraction...")
    try:
        nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    except Exception as e:
        print(f"❌ Error loading nuScenes: {e}")
        return

    dataset = []

    # 3. Process Scenes
    for scene in nusc.scene:
        scene_tracks = defaultdict(list)
        sample_token = scene['first_sample_token']
        
        while sample_token != "":
            sample = nusc.get('sample', sample_token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat = ann['category_name'].lower()
                
                # Only grab the classes we care about
                if any(cls in cat for cls in TARGET_CLASSES):
                    scene_tracks[ann['instance_token']].append(ann['translation'][:2])
            
            sample_token = sample['next']

        # 4. Sliding Window & Normalization
        for inst_id, traj in scene_tracks.items():
            traj = np.array(traj)
            if len(traj) < TOTAL_STEPS:
                continue
                
            # Filter: Object must move at least 1 meter to be "interesting"
            if np.linalg.norm(traj[-1] - traj[0]) < MIN_MOVE_DISTANCE:
                continue

            for i in range(len(traj) - TOTAL_STEPS + 1):
                window = traj[i : i + TOTAL_STEPS]
                
                # NORMALIZATION: Shift coordinates so the sequence starts at (0,0)
                # This helps the GRU learn "how people move" regardless of map location
                origin = window[0] 
                normalized_window = window - origin
                
                dataset.append({
                    "past": normalized_window[:PAST_STEPS],
                    "future": normalized_window[PAST_STEPS:]
                })

    # 5. Secure Save
    print(f"💾 Saving {len(dataset)} samples to {SAVE_PATH}...")
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Data Prepared and Saved!")

if __name__ == "__main__":
    run_extraction()