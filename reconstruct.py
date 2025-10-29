import cv2
import numpy as np
import time
import concurrent.futures
from tqdm import tqdm
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

print("Loading pre-trained ResNet50 model...")
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)
print("Model loaded successfully.")

frames = []
feature_vectors = []

# ---------------- FEATURE EXTRACTION ---------------- #
def extract_features(frame):
    """Extract combined AI + perceptual features"""
    img_resized = cv2.resize(frame, (224, 224))
    img_expanded = np.expand_dims(img_resized, axis=0)
    img_processed = preprocess_input(img_expanded)
    features = model.predict(img_processed, verbose=0).flatten()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_small = cv2.resize(gray, (96, 96)).flatten() / 255.0

    combined = np.concatenate([features / np.linalg.norm(features), gray_small])
    return combined


def calculate_ai_difference(pair):
    """Calculate combined similarity (ResNet + SSIM-based)"""
    i, j = pair
    f1, f2 = frames[i], frames[j]

    vec1, vec2 = feature_vectors[i], feature_vectors[j]
    cosine_dist = 1.0 - cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (96, 96))
    gray2 = cv2.resize(gray2, (96, 96))
    ssim_diff = 1 - ssim(gray1, gray2)

    return 0.3 * cosine_dist + 0.7 * ssim_diff


# ---------------- HYBRID GREEDY + LOCAL REFINEMENT ---------------- #
def order_frames_hybrid(diff_matrix):
    """Hybrid frame ordering using greedy + local optimization"""
    num_frames = len(diff_matrix)
    ordered = [0]
    remaining = set(range(1, num_frames))

    print("\nOrdering frames (hybrid greedy)...")
    for _ in tqdm(range(num_frames - 1)):
        last = ordered[-1]
        next_idx = min(remaining, key=lambda x: diff_matrix[last, x])
        ordered.append(next_idx)
        remaining.remove(next_idx)

    # --- Safe local refinement (2-opt) ---
    improved = True
    while improved:
        improved = False
        for i in range(1, num_frames - 3):
            a, b = ordered[i], ordered[i + 1]
            c, d = ordered[i + 2], ordered[i + 3]
            if diff_matrix[a, b] + diff_matrix[c, d] > diff_matrix[a, c] + diff_matrix[b, d]:
                ordered[i + 1], ordered[i + 2] = ordered[i + 2], ordered[i + 1]
                improved = True
    return ordered


def auto_correct_direction(frames, ordered_indices, feature_vectors):
    """Detects if video is reversed and fixes it"""
    # Compare similarity of first few frames vs last few
    first_vec = feature_vectors[ordered_indices[0]]
    last_vec = feature_vectors[ordered_indices[-1]]
    mid_vec = feature_vectors[ordered_indices[len(ordered_indices)//2]]

    start_to_mid = cosine_similarity(first_vec.reshape(1, -1), mid_vec.reshape(1, -1))[0][0]
    end_to_mid = cosine_similarity(last_vec.reshape(1, -1), mid_vec.reshape(1, -1))[0][0]

    if end_to_mid > start_to_mid:
        print("\n[INFO] Sequence appears reversed — flipping order.")
        ordered_indices = ordered_indices[::-1]

    return ordered_indices


# ---------------- MAIN PIPELINE ---------------- #
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_input_path = os.path.join(script_dir, "jumbled_video.mp4")
    video_output_path = os.path.join(script_dir, "reconstructed_video.mp4")

    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {total_frames} frames @ {fps} FPS, {width}x{height}")

    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print("\nExtracting features (ResNet + grayscale)...")
    global feature_vectors
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        feature_vectors = list(tqdm(executor.map(extract_features, frames),
                                    total=len(frames), desc="Extracting"))

    print("\nBuilding difference matrix...")
    n = len(frames)
    diff_matrix = np.zeros((n, n))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(calculate_ai_difference, pairs),
                           total=len(pairs), desc="Comparing"))

    for (i, j), val in zip(pairs, results):
        diff_matrix[i, j] = val
        diff_matrix[j, i] = val

    ordered_indices = order_frames_hybrid(diff_matrix)

    # --- Detect and fix reversed sequence ---
    ordered_indices = auto_correct_direction(frames, ordered_indices, feature_vectors)

    print("\nWriting reconstructed video...")
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for idx in tqdm(ordered_indices, desc="Writing"):
        out.write(frames[idx])
    out.release()

    print(f"\n✅ Reconstructed video saved at: {video_output_path}")


if __name__ == "__main__":
    main()
