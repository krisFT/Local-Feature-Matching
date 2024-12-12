import numpy as np
from skimage.filters import gaussian, sobel_v, sobel_h
from skimage.feature import peak_local_max

def get_interest_points(image, feature_width, num_scales=8, scale_factor=1.5):
    """
    Detect interest points across multiple scales using Harris corner detector.
    
    Args:
        image (np.ndarray): Input image
        feature_width (int): Base feature width
        num_scales (int): Number of scales to investigate
        scale_factor (float): Multiplicative factor between scales
    
    Returns:
        tuple: 
            - x coordinates of interest points
            - y coordinates of interest points
            - scales of detected interest points
    """
    # Detector parameters
    ALPHA = 0.06  # Harris corner detector sensitivity
    THRESHOLD = 0.01  # Minimum corner response threshold
    STRIDE = 2  # Step size for sliding window
    MIN_DISTANCE = 3  # Minimum distance between interest points

    # Collect interest points across scales
    all_x, all_y, all_scales = [], [], []

    for scale_idx in range(num_scales):
        # Compute current scale
        current_scale = scale_factor ** scale_idx
        current_feature_width = int(feature_width * current_scale)
        
        # Gaussian pyramid: scale image
        scaled_image = gaussian(image, sigma=current_scale)

        # Calculate image gradients
        I_x = sobel_v(scaled_image)
        I_y = sobel_h(scaled_image)

        # Compute gradient products
        I_xx = gaussian(I_x**2, sigma=current_scale)
        I_xy = gaussian(I_x * I_y, sigma=current_scale)
        I_yy = gaussian(I_y**2, sigma=current_scale)

        # Initialize corner response matrix
        corner_response = np.zeros_like(scaled_image, dtype=float)

        # Compute Harris corner response
        for y in range(0, scaled_image.shape[0] - current_feature_width, STRIDE):
            for x in range(0, scaled_image.shape[1] - current_feature_width, STRIDE):
                # Compute local gradient statistics
                Sxx = np.sum(I_xx[y:y+current_feature_width+1, x:x+current_feature_width+1])
                Syy = np.sum(I_yy[y:y+current_feature_width+1, x:x+current_feature_width+1])
                Sxy = np.sum(I_xy[y:y+current_feature_width+1, x:x+current_feature_width+1])

                # Compute Harris corner response
                det_M = (Sxx * Syy) - (Sxy**2)
                trace_M = Sxx + Syy
                response = det_M - ALPHA * (trace_M**2)
                
                # Mark strong corner responses
                if response > THRESHOLD:
                    corner_response[y+current_feature_width//2, x+current_feature_width//2] = response

        # Find local maxima
        interest_points = peak_local_max(
            corner_response, 
            min_distance=MIN_DISTANCE, 
            threshold_abs=THRESHOLD
        )

        # Store interest points with their scale
        x_points = interest_points[:, 1]
        y_points = interest_points[:, 0]
        scales = np.full_like(x_points, current_scale)

        all_x.extend(x_points)
        all_y.extend(y_points)
        all_scales.extend(scales)

    return (
        np.array(all_x), 
        np.array(all_y), 
        np.array(all_scales)
    )
