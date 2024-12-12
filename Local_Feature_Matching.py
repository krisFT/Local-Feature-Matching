import argparse
from skimage.transform import rescale
from skimage.color import rgb2gray

from Get_Interest_Points import get_interest_points
from Get_Features import get_features
from Match_Features import match_features
from utils import evaluate_correspondence, load_data, visualize_interest_points, show_correspondences

def parse_args(): 
    parser = argparse.ArgumentParser(description="Feature Matching Pipeline")
    parser.add_argument(
        "-p", "--pair",
        type=str,
        choices=["notre_dame", "mt_rushmore", "e_gaudi"],
        default="mt_rushmore",
        help="Specifies which image pair to match. Options: 'notre_dame', 'mt_rushmore', or 'e_gaudi'. Default is 'notre_dame'."
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    pair = args.pair
    print(f"Selected pair: {pair}")

    # Load data
    image1, image2, eval_file = load_data(pair)

    # Convert to grayscale and resize 
    image1 = rescale(rgb2gray(image1), 0.5)
    image2 = rescale(rgb2gray(image2), 0.5)

    # Parameters
    feature_width = 16

    # Detect interest points
    print("Detecting interest points...")
    x1, y1, scales1 = get_interest_points(image1, feature_width)
    x2, y2, scales2 = get_interest_points(image2, feature_width)

    # Visualize interest points
    visualize_interest_points(image1, x1, y1)
    visualize_interest_points(image2, x2, y2)
    print("Interest points detected!")

    # Extract feature descriptors
    print("Extracting features...")
    image1_features = get_features(image1, x1, y1, scales1, feature_width)
    image2_features = get_features(image2, x2, y2, scales2, feature_width)
    print("Features extracted!")

    # Match features
    print("Matching features...")
    matches, confidences = match_features(image1_features, image2_features, threshold=0.8)
    if matches.size == 0:
        print("No matches found!")
    print(f"Found {matches.shape[0]} matches!")

    # Evaluate matches
    evaluate_correspondence(eval_file, 0.5, x1, y1, x2, y2, matches, confidences)

    # Visualize correspondences
    filename = f"{pair}_matches.png"
    show_correspondences(image1, image2, x1, y1, x2, y2, matches, filename=filename)

if __name__ == "__main__":
    main()

