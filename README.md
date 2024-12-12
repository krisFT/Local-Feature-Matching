# **Local Feature Matching**

This project focuses on local feature matching in images. It involves detecting interest points across multiple scales using the Harris corner detector, computing multi-scale SIFT-like feature descriptors, and matching these descriptors between two images using the Nearest Neighbor Distance Ratio (NNDR) test.

---

## **Features**
- **Interest Point Detection**:
  - Detects interest points across multiple scales using the Harris corner detector.
  - Applies non-maximum suppression to refine the detected points.
- **Feature Description**:
  - Computes multi-scale SIFT-like feature descriptors using a sliding window approach.
- **Feature Matching**:
  - Matches feature descriptors between two images using the Nearest Neighbor Distance Ratio (NNDR) test.

---

## **Accuracy Results**

| Image Pair       | Accuracy (Top 100 Features) |
|------------------|-----------------------------|
| Episcopal Gaudi  | 51%                         |
| Mount Rushmore   | 95%                         |
| Notre Dame       | 92%                         |

---

## **Usage**

### Command Line Example
To run the feature matching pipeline on the Episcopal Gaudi dataset, use the following command:
```sh
python Local_Feature_Matching.py -p e_gaudi
```
### **Required Libraries**

Ensure the following Python libraries are installed before running the pipeline:

- **numpy**
- **scikit-image**
- **opencv-python**
- **matplotlib**

## **Technical Details**

### **Harris Corner Response Calculation**

1. **Compute the determinant of the matrix \( M \)**:
   ```math
   \text{det}(M) = (S_{xx} \cdot S_{yy}) - (S_{xy}^2)
   ```

2. **Compute the trace of the matrix \( M \)**:
   ```math
   \text{trace}(M) = S_{xx} + S_{yy}
   ```

3. **Compute the Harris response**:
   ```math
   \text{response} = \text{det}(M) - \alpha \cdot (\text{trace}(M)^2)
   ```

---

### **Multi-Scale SIFT-like Feature Descriptors**

Multi-scale SIFT-like feature descriptors are computed by extracting features at multiple scales to ensure robustness to scale variations. The process involves:

1. **Detection of Interest Points**:
   - Interest points are detected at multiple scales using the Harris corner detector.

2. **Descriptor Calculation**:
   - For each interest point, a SIFT-like descriptor is computed using a sliding window approach.
   - This involves creating a histogram of gradient orientations within a local region around the interest point.

3. **Normalization**:
   - The descriptors are normalized to achieve invariance to changes in illumination and contrast.

---

### **Nearest Neighbor Distance Ratio (NNDR) Calculation**

1. **Find Nearest Neighbors**:
   - For each feature descriptor in the first image, find the two nearest feature descriptors in the second image.

2. **Compute the Ratio**:
   - Compute the ratio of the distance to the closest neighbor to the distance of the second closest neighbor:
     ```math
     \text{ratio} = \frac{\text{distance to closest neighbor}}{\text{distance to second closest neighbor}}
     ```

3. **Thresholding**:
   - A match is considered valid if this ratio is below a certain threshold, indicating that the closest match is significantly better than the second closest match.

