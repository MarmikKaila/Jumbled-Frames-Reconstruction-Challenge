import cv2
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
import os

# --- AI & ML Imports ---
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

# --- TSP Solver Import ---
import elkai  # This is the LKH-3 solver

# --- Global lists ---
frames = []
feature_vectors = []

# --- 1. Load the AI Model (ResNet50) ---
print("Loading pre-trained ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
print("Model loaded successfully.")

# --- 2. The "AI Feature Extractor" Function ---
def extract_features(frame):
    """
    Takes a single frame, processes it, and runs it through
    the ResNet50 model to get a 2048-dimension feature vector.
    """
    img_resized = cv2.resize(frame, (224, 224))
    img_expanded = np.expand_dims(img_resized, axis=0)
    img_processed = preprocess_input(img_expanded)
    features = model.predict(img_processed, verbose=0)
    return features.flatten()

# --- 3. The "AI Comparison" Function ---
def calculate_ai_difference(vectors):
    """
    Takes a pair of indices (i, j), gets their feature vectors,
    and calculates the cosine difference.
    """
    i, j = vectors
    vec1 = feature_vectors[i]
    vec2 = feature_vectors[j]
    
    # Calculate cosine similarity (ranges from -1 to 1, typically 0 to 1 for similar vectors)
    sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    
    # Return the "distance" (0.0 is identical, 1.0 is very different)
    return 1.0 - sim

# --- 4. The Main Execution ---
def main():
    
    # --- Fix file paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"=" * 60)
    print(f"Running in directory: {script_dir}")
    print(f"=" * 60)
    
    # Check if filename provided as command line argument
    import sys
    if len(sys.argv) > 1:
        video_input_path = sys.argv[1]
    else:
        video_input_path = os.path.join(script_dir, 'jumbled_video.mp4')
    
    video_output_path = os.path.join(script_dir, 'reconstructed_video.mp4')
    log_output_path = os.path.join(script_dir, 'execution_log.txt')
    
    # === PHASE 1: INGESTION ===
    print("\n[PHASE 1] Reading all frames into memory...")
    start_total_time = time.time()
    
    global frames
    cap = cv2.VideoCapture(video_input_path) 
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file at: {video_input_path}")
        print("[ERROR] Please make sure 'jumbled_video.mp4' is in the same folder.")
        return 

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # CORRECT SPECIFICATION: 10 seconds at 30 FPS = 300 frames
    FPS = 30
    EXPECTED_FRAMES = 300
    
    print(f"Video properties:")
    print(f"  - Detected FPS: {detected_fps}")
    print(f"  - Frame count: {frame_count}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Target output FPS: {FPS}")
    
    if frame_count != EXPECTED_FRAMES:
        print(f"WARNING: Expected {EXPECTED_FRAMES} frames but found {frame_count}")
    
    for _ in tqdm(range(frame_count), desc="Reading Frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[SUCCESS] Successfully read {len(frames)} frames.")
    
    # === PHASE 2: AI FEATURE EXTRACTION ===
    print("\n[PHASE 2] Extracting features with ResNet50 (AI)...")
    start_feature_time = time.time()
    
    global feature_vectors
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(extract_features, frames), 
                           total=len(frames), desc="Extracting Features"))
    
    feature_vectors = results 
    feature_duration = time.time() - start_feature_time
    print(f"[SUCCESS] AI Feature extraction took {feature_duration:.2f} seconds.")

    # === PHASE 3: BUILD DIFFERENCE MATRIX ===
    print("\n[PHASE 3] Building difference matrix from AI vectors...")
    start_matrix_time = time.time()
    
    num_frames = len(frames)
    tasks = []
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            tasks.append((i, j))
            
    print(f"Total comparisons to calculate: {len(tasks):,}")
    diff_matrix = np.zeros((num_frames, num_frames))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(calculate_ai_difference, tasks), 
                           total=len(tasks), desc="Comparing Vectors"))
    
    for (i, j), diff in zip(tasks, results):
        diff_matrix[i, j] = diff
        diff_matrix[j, i] = diff 
        
    matrix_duration = time.time() - start_matrix_time
    print(f"[SUCCESS] Matrix calculation took {matrix_duration:.2f} seconds.")

    # === PHASE 4: SORTING (TSP SOLVER) ===
    print("\n[PHASE 4] Solving sequence as a Traveling Salesperson Problem (TSP)...")
    start_tsp_time = time.time()
    
    # 1. Convert float matrix to integer matrix for the LKH-3 solver
    # Scale up to preserve precision (elkai works with integers)
    int_matrix = (diff_matrix * 1000000).astype(int)
    
    # 2. Create the distance matrix and solve TSP using elkai
    print("Creating distance matrix for TSP solver...")
    print("Running LKH-3 solver... (This may take a few seconds)")
    
    try:
        # CORRECT SYNTAX: Use solve_int_matrix for integer matrices
        solution_cycle = elkai.solve_int_matrix(int_matrix, runs=10)
        print("[SUCCESS] TSP solution found.")
    except Exception as e:
        print(f"[ERROR] TSP solver failed: {e}")
        print("Falling back to greedy nearest neighbor approach...")
        # Fallback: simple greedy approach
        solution_cycle = greedy_nearest_neighbor(diff_matrix)
    
    tsp_duration = time.time() - start_tsp_time
    print(f"TSP solving took {tsp_duration:.2f} seconds.")
    
    # 4. Linearize the path (Cut the "longest link" to break the cycle)
    print("Linearizing the cyclic TSP solution...")
    longest_link_dist = -1
    cut_point = -1
    
    for i in range(num_frames):
        idx1 = solution_cycle[i]
        idx2 = solution_cycle[(i + 1) % num_frames] 
        dist = diff_matrix[idx1, idx2]
        
        if dist > longest_link_dist:
            longest_link_dist = dist
            cut_point = i
    
    # Cut at the longest link to create a linear sequence        
    reconstructed_order = solution_cycle[cut_point + 1:] + solution_cycle[:cut_point + 1]
    
    print(f"[SUCCESS] Sequence reconstruction complete.")
    print(f"  - Cut point: {cut_point}")
    print(f"  - Longest link distance: {longest_link_dist:.6f}")

    # === PHASE 5: OUTPUT (Write the Reconstructed Video) ===
    print("\n[PHASE 5] Writing reconstructed video file...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, FPS, (width, height))
    
    if not out.isOpened():
        print("[ERROR] Could not open video writer")
        return

    for frame_index in tqdm(reconstructed_order, desc="Writing Video"):
        out.write(frames[frame_index])

    out.release()
    print(f"[SUCCESS] Successfully saved {video_output_path}")
    
    end_total_time = time.time()
    total_duration = end_total_time - start_total_time
    
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total execution time:        {total_duration:.2f} seconds")
    print(f"  - Feature extraction:      {feature_duration:.2f} seconds")
    print(f"  - Matrix calculation:      {matrix_duration:.2f} seconds")
    print(f"  - TSP solving:             {tsp_duration:.2f} seconds")
    print(f"Frames processed:            {num_frames}")
    print(f"Algorithm:                   AI (ResNet50) + TSP (LKH-3)")
    print(f"Output specifications:       {FPS} FPS, {width}x{height}")
    print("=" * 60)
    
    # Save execution log
    with open(log_output_path, "w", encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("VIDEO FRAME RECONSTRUCTION - EXECUTION LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Algorithm: AI (ResNet50) + TSP (LKH-3 Solver)\n\n")
        f.write(f"Total execution time:        {total_duration:.2f} seconds\n")
        f.write(f"  - Feature extraction:      {feature_duration:.2f} seconds\n")
        f.write(f"  - Matrix calculation:      {matrix_duration:.2f} seconds\n")
        f.write(f"  - TSP solving:             {tsp_duration:.2f} seconds\n\n")
        f.write(f"Frames processed:            {num_frames}\n")
        f.write(f"Comparisons performed:       {len(tasks):,}\n")
        f.write(f"Output FPS:                  {FPS}\n")
        f.write(f"Output resolution:           {width}x{height}\n")
    
    print(f"\n[SUCCESS] Execution log saved to '{log_output_path}'")

def greedy_nearest_neighbor(diff_matrix):
    """
    Fallback greedy algorithm if TSP solver fails
    """
    num_frames = len(diff_matrix)
    visited = [False] * num_frames
    path = [0]  # Start from frame 0
    visited[0] = True
    
    for _ in range(num_frames - 1):
        current = path[-1]
        best_next = -1
        best_dist = float('inf')
        
        for next_frame in range(num_frames):
            if not visited[next_frame]:
                dist = diff_matrix[current][next_frame]
                if dist < best_dist:
                    best_dist = dist
                    best_next = next_frame
        
        if best_next != -1:
            path.append(best_next)
            visited[best_next] = True
    
    return path

if __name__ == "__main__":
    main()