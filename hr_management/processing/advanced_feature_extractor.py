"""
Advanced Feature Extraction for Person Recognition

This module provides enhanced feature extraction methods beyond simple color histograms
and HOG features, including deep learning features and advanced statistical descriptors.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.image import extract_patches_2d
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """Advanced feature extraction methods for person recognition"""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 256)):
        self.target_size = target_size
        
    def extract_color_moments(self, image: np.ndarray) -> np.ndarray:
        """Extract color moments (mean, std, skewness) for each channel"""
        features = []
        
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for img, name in [(image, 'bgr'), (lab, 'lab'), (hsv, 'hsv')]:
            for channel in range(3):
                chan = img[:, :, channel].flatten()
                
                # Mean
                mean = np.mean(chan)
                # Standard deviation
                std = np.std(chan)
                # Skewness
                skewness = np.mean(((chan - mean) / (std + 1e-7)) ** 3)
                
                features.extend([mean, std, skewness])
        
        return np.array(features)
    
    def extract_lbp_features(self, image: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP implementation
        rows, cols = gray.shape
        lbp = np.zeros_like(gray)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = gray[i, j]
                binary_string = ""
                
                # 8 neighbors
                neighbors = [
                    gray[i-radius, j-radius], gray[i-radius, j], gray[i-radius, j+radius],
                    gray[i, j+radius], gray[i+radius, j+radius], gray[i+radius, j],
                    gray[i+radius, j-radius], gray[i, j-radius]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=[0, 256])
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def extract_gabor_features(self, image: np.ndarray, num_scales: int = 4, num_orientations: int = 6) -> np.ndarray:
        """Extract Gabor filter responses"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        for scale in range(num_scales):
            for orientation in range(num_orientations):
                # Gabor kernel parameters
                ksize = 15 + scale * 10
                sigma = 3.0 + scale
                theta = orientation * np.pi / num_orientations
                lambda_val = 10.0 + scale * 5
                gamma = 0.5
                
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambda_val, gamma, 0, ktype=cv2.CV_32F
                )
                
                # Apply filter
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                
                # Extract statistics
                features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.max(filtered) - np.min(filtered)
                ])
        
        return np.array(features)
    
    def extract_shape_context(self, image: np.ndarray, num_points: int = 50) -> np.ndarray:
        """Extract shape context features from edge points"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(num_points * 2)
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Sample points uniformly
        if len(largest_contour) > num_points:
            indices = np.linspace(0, len(largest_contour) - 1, num_points, dtype=int)
            sampled_points = largest_contour[indices].reshape(-1, 2)
        else:
            sampled_points = largest_contour.reshape(-1, 2)
            # Pad with zeros if needed
            if len(sampled_points) < num_points:
                padding = np.zeros((num_points - len(sampled_points), 2))
                sampled_points = np.vstack([sampled_points, padding])
        
        # Normalize coordinates
        if sampled_points.size > 0:
            sampled_points = sampled_points.astype(np.float32)
            sampled_points[:, 0] /= (image.shape[1] + 1e-7)
            sampled_points[:, 1] /= (image.shape[0] + 1e-7)
        
        return sampled_points.flatten()
    
    def extract_spatial_pyramid_features(self, image: np.ndarray, levels: int = 3) -> np.ndarray:
        """Extract spatial pyramid features"""
        features = []
        
        for level in range(levels):
            # Number of cells at this level
            num_cells = 2 ** level
            
            # Divide image into grid
            h, w = image.shape[:2]
            cell_h = h // num_cells
            cell_w = w // num_cells
            
            for i in range(num_cells):
                for j in range(num_cells):
                    # Extract cell
                    y1 = i * cell_h
                    y2 = (i + 1) * cell_h if i < num_cells - 1 else h
                    x1 = j * cell_w
                    x2 = (j + 1) * cell_w if j < num_cells - 1 else w
                    
                    cell = image[y1:y2, x1:x2]
                    
                    # Compute color histogram for cell
                    hist_b = cv2.calcHist([cell], [0], None, [16], [0, 256])
                    hist_g = cv2.calcHist([cell], [1], None, [16], [0, 256])
                    hist_r = cv2.calcHist([cell], [2], None, [16], [0, 256])
                    
                    # Normalize
                    hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
                    hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
                    hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)
                    
                    features.extend(hist_b)
                    features.extend(hist_g)
                    features.extend(hist_r)
        
        return np.array(features)
    
    def extract_covariance_features(self, image: np.ndarray) -> np.ndarray:
        """Extract covariance matrix features"""
        # Create feature vectors for each pixel
        h, w = image.shape[:2]
        
        # Spatial coordinates
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # Color values
        b, g, r = cv2.split(image)
        
        # Gradients
        dx = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 0, 1, ksize=3)
        
        # Feature vector: [x, y, R, G, B, |dx|, |dy|]
        features_per_pixel = np.stack([
            xx.flatten() / w,
            yy.flatten() / h,
            r.flatten() / 255.0,
            g.flatten() / 255.0,
            b.flatten() / 255.0,
            np.abs(dx).flatten() / 255.0,
            np.abs(dy).flatten() / 255.0
        ], axis=1)
        
        # Compute covariance matrix
        cov_matrix = np.cov(features_per_pixel.T)
        
        # Extract upper triangular part (since symmetric)
        upper_tri_indices = np.triu_indices(cov_matrix.shape[0])
        cov_features = cov_matrix[upper_tri_indices]
        
        return cov_features
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """Extract all advanced features and concatenate them"""
        # Resize image
        if image.shape[:2] != self.target_size[::-1]:
            image = cv2.resize(image, self.target_size)
        
        all_features = []
        
        # Color moments
        color_moments = self.extract_color_moments(image)
        all_features.append(color_moments)
        
        # LBP
        lbp = self.extract_lbp_features(image)
        all_features.append(lbp)
        
        # Gabor
        gabor = self.extract_gabor_features(image)
        all_features.append(gabor)
        
        # Shape context
        shape = self.extract_shape_context(image)
        all_features.append(shape)
        
        # Spatial pyramid
        spatial = self.extract_spatial_pyramid_features(image)
        all_features.append(spatial)
        
        # Covariance
        covariance = self.extract_covariance_features(image)
        all_features.append(covariance)
        
        # Concatenate all features
        features = np.concatenate(all_features)
        
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def extract_patch_statistics(self, image: np.ndarray, patch_size: Tuple[int, int] = (32, 32), 
                               max_patches: int = 10) -> np.ndarray:
        """Extract statistics from random patches"""
        patches = extract_patches_2d(image, patch_size, max_patches=max_patches, random_state=42)
        
        features = []
        for patch in patches:
            # Color statistics per patch
            for channel in range(3):
                chan = patch[:, :, channel].flatten()
                features.extend([
                    np.mean(chan),
                    np.std(chan),
                    np.median(chan),
                    np.percentile(chan, 25),
                    np.percentile(chan, 75)
                ])
        
        return np.array(features)