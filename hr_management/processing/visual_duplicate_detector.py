"""
Visual duplicate detection using lightweight methods
Doesn't require TensorFlow, uses OpenCV and NumPy only
"""
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

class VisualDuplicateDetector:
    def __init__(self, similarity_threshold=0.90):
        self.similarity_threshold = similarity_threshold
    
    def calculate_image_hash(self, img_path, hash_size=16):
        """Calculate perceptual hash of an image"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to hash_size + 1 (we need one extra for gradient calculation)
            resized = cv2.resize(gray, (hash_size + 1, hash_size))
            
            # Calculate horizontal gradient
            diff = resized[:, 1:] > resized[:, :-1]
            
            # Convert to hash
            return diff.flatten()
            
        except Exception as e:
            print(f"Error calculating hash for {img_path}: {e}")
            return None
    
    def hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        if hash1 is None or hash2 is None:
            return float('inf')
        return np.sum(hash1 != hash2)
    
    def calculate_structural_similarity(self, img1_path, img2_path):
        """Calculate structural similarity between two images"""
        try:
            # Read images
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Resize to same size
            size = (128, 128)
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Calculate SSIM-like metric
            # Mean
            mu1 = np.mean(gray1)
            mu2 = np.mean(gray2)
            
            # Variance
            var1 = np.var(gray1)
            var2 = np.var(gray2)
            
            # Covariance
            cov = np.mean((gray1 - mu1) * (gray2 - mu2))
            
            # SSIM formula constants
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            # Luminance comparison
            luminance = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
            
            # Contrast comparison
            contrast = (2 * np.sqrt(var1) * np.sqrt(var2) + c2) / (var1 + var2 + c2)
            
            # Structure comparison
            structure = (cov + c2/2) / (np.sqrt(var1) * np.sqrt(var2) + c2/2)
            
            # Combined SSIM
            ssim = luminance * contrast * structure
            
            return float(max(0, min(1, ssim)))
            
        except Exception as e:
            print(f"Error calculating structural similarity: {e}")
            return 0.0
    
    def calculate_color_similarity(self, img1_path, img2_path):
        """Calculate color histogram similarity"""
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Calculate histograms for each channel
            hist_similarity = []
            
            for i in range(3):  # BGR channels
                hist1 = cv2.calcHist([img1], [i], None, [64], [0, 256])
                hist2 = cv2.calcHist([img2], [i], None, [64], [0, 256])
                
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()
                
                # Use correlation coefficient
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                hist_similarity.append(correlation)
            
            # Average similarity across channels
            return float(np.mean(hist_similarity))
            
        except Exception as e:
            print(f"Error calculating color similarity: {e}")
            return 0.0
    
    def calculate_overall_similarity(self, img1_path, img2_path):
        """Calculate overall similarity combining multiple metrics"""
        # Perceptual hash
        hash1 = self.calculate_image_hash(img1_path)
        hash2 = self.calculate_image_hash(img2_path)
        
        if hash1 is not None and hash2 is not None:
            # Normalize Hamming distance to similarity score
            max_distance = len(hash1)
            hamming_dist = self.hamming_distance(hash1, hash2)
            hash_similarity = 1.0 - (hamming_dist / max_distance)
        else:
            hash_similarity = 0.0
        
        # Structural similarity
        structural_sim = self.calculate_structural_similarity(img1_path, img2_path)
        
        # Color similarity
        color_sim = self.calculate_color_similarity(img1_path, img2_path)
        
        # Weighted combination
        overall_similarity = (
            hash_similarity * 0.4 +
            structural_sim * 0.3 +
            color_sim * 0.3
        )
        
        return overall_similarity
    
    def find_duplicates(self, image_files):
        """Find duplicate images in a list of files"""
        if len(image_files) < 2:
            return []
        
        # First pass: Calculate perceptual hashes
        hashes = {}
        for img_file in image_files:
            hash_val = self.calculate_image_hash(img_file)
            if hash_val is not None:
                hashes[img_file] = hash_val
        
        # Group by similar hashes (allows some differences)
        hash_groups = defaultdict(list)
        processed = set()
        
        image_list = list(hashes.keys())
        for i in range(len(image_list)):
            if image_list[i] in processed:
                continue
            
            current_group = [image_list[i]]
            processed.add(image_list[i])
            
            for j in range(i + 1, len(image_list)):
                if image_list[j] in processed:
                    continue
                
                # Quick hash check first
                hamming_dist = self.hamming_distance(
                    hashes[image_list[i]], 
                    hashes[image_list[j]]
                )
                
                # If hashes are very similar, do detailed check
                if hamming_dist <= len(hashes[image_list[i]]) * 0.2:  # 20% difference allowed
                    similarity = self.calculate_overall_similarity(
                        image_list[i], 
                        image_list[j]
                    )
                    
                    if similarity >= self.similarity_threshold:
                        current_group.append(image_list[j])
                        processed.add(image_list[j])
            
            if len(current_group) > 1:
                hash_groups[f"group_{len(hash_groups)}"] = current_group
        
        return dict(hash_groups)