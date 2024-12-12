import numpy as np
from skimage.filters import sobel_h, sobel_v

def normalize_descriptor(descriptor, clip_threshold=0.2):
    """
    Normalize feature descriptor to improve matching robustness.
    
    Args:
        descriptor (np.ndarray): Input descriptor vector
        clip_threshold (float): Threshold for descriptor value clipping
    
    Returns:
        np.ndarray: Normalized descriptor
    """
    # Normalize to unit length, avoiding division by zero
    norm = np.linalg.norm(descriptor)
    if norm == 0:
        return np.zeros_like(descriptor)
    
    # Initial normalization
    normalized = descriptor / norm
    
    # Clip values to reduce impact of large gradients
    normalized = np.clip(normalized, 0, clip_threshold)
    
    # Re-normalize after clipping
    return normalized / (np.linalg.norm(normalized) + 1e-10)
def get_features(image, x, y, scales, feature_width):
    """
    Computes multi-scale SIFT-like feature descriptors.
    
    Args:
        image (np.ndarray): Grayscale input image
        x (np.ndarray): X-coordinates of interest points
        y (np.ndarray): Y-coordinates of interest points
        scales (np.ndarray): Scale of each interest point
        feature_width (int): Base feature width
    
    Returns:
        np.ndarray: Computed feature descriptors
    """
    # Compute image gradients
    gradient_x = sobel_v(image)
    gradient_y = sobel_h(image)
    
    # Compute gradient magnitude and orientation
    magnitude = np.hypot(gradient_x, gradient_y)
    direction = np.mod(np.arctan2(gradient_y, gradient_x), 2 * np.pi)
    
    features = []
    
    for x_, y_, scale in zip(x, y, scales):
        # Scale-aware feature window
        current_feature_width = int(feature_width * scale)
        half_width = current_feature_width // 2
        cell_width = current_feature_width // 4
        
        # Compute safe image window boundaries
        row_min = max(0, int(y_ - half_width))
        row_max = min(image.shape[0], int(y_ + half_width))
        col_min = max(0, int(x_ - half_width))
        col_max = min(image.shape[1], int(x_ + half_width))
        
        # Extract local image window
        window_magnitude = magnitude[row_min:row_max, col_min:col_max]
        window_direction = direction[row_min:row_max, col_min:col_max]
        
        # Compute descriptor
        descriptor = []
        for cell_y in range(4):
            for cell_x in range(4):
                # Define cell boundaries
                r_start = cell_y * cell_width
                r_end = (cell_y + 1) * cell_width
                c_start = cell_x * cell_width
                c_end = (cell_x + 1) * cell_width
                
                # Extract cell gradients
                cell_magnitude = window_magnitude[r_start:r_end, c_start:c_end]
                cell_direction = window_direction[r_start:r_end, c_start:c_end]
                
                # Compute gradient histogram
                hist, _ = np.histogram(
                    cell_direction, 
                    bins=8, 
                    range=(0, 2 * np.pi), 
                    weights=cell_magnitude
                )
                
                descriptor.extend(hist)
        
        # Normalize descriptor
        descriptor = np.array(descriptor)
        descriptor = normalize_descriptor(descriptor)
        
        features.append(descriptor)
    
    return np.array(features)

