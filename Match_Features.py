import numpy as np

def match_features(im1_features, im2_features, threshold=0.8):
    """
    Matches feature descriptors between two images using the Nearest Neighbor Distance Ratio (NNDR) test.

    Args:
        im1_features (np.ndarray): Feature descriptors from image 1 (n1 x d).
        im2_features (np.ndarray): Feature descriptors from image 2 (n2 x d).
        threshold (float, optional): Threshold for the Nearest Neighbor Distance Ratio test. Defaults to 0.8.

    Returns:
        tuple: 
            - matches (np.ndarray): Array of matched feature indices (k x 2)
            - confidences (np.ndarray): Confidence scores for each match (k)
    """
    matches = []
    confidences = []

    # Compute distances and perform NNDR test
    for i in range(im1_features.shape[0]):
        distances = np.linalg.norm(im1_features[i] - im2_features, axis=1)
        sorted_indices = np.argsort(distances)

        # NNDR ratio
        if len(sorted_indices) > 1:
            ratio = distances[sorted_indices[0]] / distances[sorted_indices[1]]
            if ratio < threshold:
                matches.append([i, sorted_indices[0]])
                confidences.append(1.0 - ratio)

    matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    confidences = np.array(confidences) if confidences else np.empty(0)

    return matches, confidences
