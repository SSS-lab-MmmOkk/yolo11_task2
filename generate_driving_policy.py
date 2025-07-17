import pandas as pd
import numpy as np
import cv2
import os

# --- Constants ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
TARGET_CLASSES = ['cyclist', 'person', 'car']
DISTANCE_THRESHOLDS = {
    'NEAR': FRAME_HEIGHT * 0.6,
    'MID': FRAME_HEIGHT * 0.3
}

def get_distance_to_ego(obj_bbox):
    """A simple calculation of the distance between an object and the ego vehicle."""
    x1, y1, x2, y2 = obj_bbox
    bottom_center_y = y2
    return FRAME_HEIGHT - bottom_center_y

def analyze_frame_for_policy_en(frame_df):
    """Determines the driving policy for a single frame in English."""
    target_objects = frame_df[frame_df['class_name'].isin(TARGET_CLASSES)].copy()
    if target_objects.empty:
        return "No target objects detected. Driving safely."

    target_objects['bbox'] = target_objects.apply(lambda row: (row['x1'], row['y1'], row['x2'], row['y2']), axis=1)
    target_objects['distance'] = target_objects['bbox'].apply(get_distance_to_ego)
    most_critical_object = target_objects.loc[target_objects['distance'].idxmin()]

    obj_class = most_critical_object['class_name']
    obj_bbox = most_critical_object['bbox']
    obj_bottom_y = obj_bbox[3]

    obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
    if obj_center_x < FRAME_WIDTH / 3:
        horizontal_pos = "ahead on the left"
    elif obj_center_x > FRAME_WIDTH * 2 / 3:
        horizontal_pos = "ahead on the right"
    else:
        horizontal_pos = "ahead"

    if obj_bottom_y > DISTANCE_THRESHOLDS['NEAR']:
        policy = f"A {obj_class} is close {horizontal_pos}. Slow down and keep safe distance."
    elif obj_bottom_y > DISTANCE_THRESHOLDS['MID']:
        policy = f"A {obj_class} is detected {horizontal_pos}. Keep distance and follow."
    else:
        policy = f"A {obj_class} is in the distance {horizontal_pos}. Proceed with caution."

    vx = most_critical_object['velocity_x']
    if vx > 5:
        policy += " (Moving to the right)"
    elif vx < -5:
        policy += " (Moving to the left)"

    return policy

def process_and_visualize(video_path, tracking_csv_path, output_dir):
    """
    Processes tracking data to generate driving policies, creates a visualization video,
    and outputs a CSV file with timestamps and policies.
    """
    try:
        df = pd.read_csv(tracking_csv_path)
        print(f"Successfully loaded {tracking_csv_path}")
    except FileNotFoundError:
        print(f"Error: {tracking_csv_path} not found. Please run task1_detection_tracking_simple.py first.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    # Prepare for video output
    output_video_path = os.path.join(output_dir, "policy_visualization.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare for CSV output
    policy_records = []

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_df = df[df['frame_id'] == frame_id]
        policy_text = "No target objects detected. Driving safely."
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if not frame_df.empty:
            policy_text = analyze_frame_for_policy_en(frame_df)

            # Draw bounding boxes
            for _, row in frame_df.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{row['class_name']} ID:{row['track_id']}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw policy text
        cv2.putText(frame, policy_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        policy_records.append({'time': timestamp, 'driving_policy': policy_text})

        frame_id += 1
        if frame_id % 30 == 0:
            print(f"Processing... Frame {frame_id}")

    cap.release()
    out.release()
    print(f"Visualization video saved to {output_video_path}")

    # Save policy records to CSV
    output_csv_path = os.path.join(output_dir, "driving_policy.csv")
    policy_df = pd.DataFrame(policy_records)
    policy_df.to_csv(output_csv_path, index=False, float_format='%.3f')
    print(f"Driving policy CSV saved to {output_csv_path}")


if __name__ == "__main__":
    VIDEO_PATH = "Solving the Long-Tail_cyclist.mp4"
    TRACKING_RESULTS_PATH = "task1_simple_output/tracking_results.csv"
    OUTPUT_DIR = "task1_simple_output"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_and_visualize(VIDEO_PATH, TRACKING_RESULTS_PATH, OUTPUT_DIR)
