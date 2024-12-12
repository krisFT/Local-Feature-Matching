import numpy as np
import scipy.io as scio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import img_as_float32
from skimage.io import imread
from skimage.feature import plot_matches

def load_data(file_name):
    """
    Load image data and evaluation file for a given dataset.

    Parameters:
        file_name (str): One of 'notre_dame', 'mt_rushmore', or 'e_gaudi'.

    Returns:
        image1 (ndarray): First image as a NumPy array.
        image2 (ndarray): Second image as a NumPy array.
        eval_file (str): Path to the evaluation file.
    """

    paths = {
        "notre_dame": (
            "./data/NotreDame/NotreDame1.jpg",
            "./data/NotreDame/NotreDame2.jpg",
            "./data/NotreDame/NotreDameEval.mat",
        ),
        "mt_rushmore": (
            "./data/MountRushmore/Mount_Rushmore1.jpg",
            "./data/MountRushmore/Mount_Rushmore2.jpg",
            "./data/MountRushmore/MountRushmoreEval.mat",
        ),
        "e_gaudi": (
            "./data/EpiscopalGaudi/EGaudi_1.jpg",
            "./data/EpiscopalGaudi/EGaudi_2.jpg",
            "./data/EpiscopalGaudi/EGaudiEval.mat",
        ),
    }

    if file_name not in paths:
        raise ValueError(f"Invalid file name '{file_name}'. Must be one of {list(paths.keys())}.")

    image1_file, image2_file, eval_file = paths[file_name]
    image1 = img_as_float32(imread(image1_file))
    image2 = img_as_float32(imread(image2_file))
    return image1, image2, eval_file

def visualize_interest_points(image, x, y):
    """Utility function to visualize interest points on an image."""
    plt.imshow(image, cmap="gray")
    plt.scatter(x, y, alpha=0.9, s=3)
    plt.show()

def show_correspondences(imgA, imgB, X1, Y1, X2, Y2, matches, mode='arrows', filename=None):
    """
    Visualizes corresponding points between two images as arrows or dots.
    """
    fig, ax = plt.subplots()

    if mode == 'dots':
        raise NotImplementedError("Dot visualization not implemented.")
    elif mode == 'arrows':
        kp1 = np.column_stack((Y1, X1))
        kp2 = np.column_stack((Y2, X2))
        plot_matches(ax, imgA, imgB, kp1, kp2, matches.astype(int))

    if filename:
        plt.savefig(filename)
    plt.show()

def evaluate_correspondence(ground_truth_correspondence_file, scale_factor, x1_est, y1_est, x2_est, y2_est, matches, confidences):
    """
    Evaluates the quality of correspondence by comparing estimated matches against ground truth.
    
    Parameters:
        ground_truth_correspondence_file (str): Path to the ground truth correspondence file.
        scale_factor (float): Scale factor applied to the images.
        x1_est, y1_est, x2_est, y2_est (ndarray): Estimated interest point coordinates.
        matches (ndarray): Matched indices between two images.
        confidences (ndarray): Confidence values for each match.

    Returns:
        accuracy100 (float): Accuracy based on the top 100 matches.
    """
    # Sort matches by confidence in descending order
    sorted_indices = np.argsort(-confidences, kind='mergesort')
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]

    # Unscale estimated points
    x1_est_scaled, y1_est_scaled = x1_est / scale_factor, y1_est / scale_factor
    x2_est_scaled, y2_est_scaled = x2_est / scale_factor, y2_est / scale_factor

    # Extract matched coordinates
    x1_matches = x1_est_scaled[matches[:, 0].astype(int)]
    y1_matches = y1_est_scaled[matches[:, 0].astype(int)]
    x2_matches = x2_est_scaled[matches[:, 1].astype(int)]
    y2_matches = y2_est_scaled[matches[:, 1].astype(int)]

    # Load ground truth points
    gt_data = scio.loadmat(ground_truth_correspondence_file)
    x1, y1, x2, y2 = gt_data['x1'], gt_data['y1'], gt_data['x2'], gt_data['y2']

    # Parameters
    uniqueness_dist = 150
    good_match_dist = 150

    correct_matches = np.zeros(x2.shape[0])
    top_100_counter = 0

    # Evaluate matches against ground truth
    for i in range(x1.shape[0]):
        dists = np.sqrt((x1_matches - x1[i])**2 + (y1_matches - y1[i])**2)
        close_to_truth = dists < uniqueness_dist

        image2_x = x2_matches[close_to_truth]
        image2_y = y2_matches[close_to_truth]

        dists_2 = np.sqrt((image2_x - x2[i])**2 + (image2_y - y2[i])**2)
        good = np.any(dists_2 < good_match_dist)

        if good:
            correct_matches[i] = 1
            if i < 100:
                top_100_counter += 1

    # Calculate precision and top-100 accuracy
    total_good_matches = np.sum(correct_matches)
    precision = (total_good_matches / x2.shape[0]) * 100.0
    accuracy100 = min(top_100_counter, 100)

    # Print evaluation results
    print(f"{total_good_matches} total good matches.")
    print(f"{x2.shape[0] - total_good_matches} total bad matches.")
    print(f"{precision:.2f}% precision.")
    print(f"{accuracy100}% accuracy (top 100).")

    return accuracy100
