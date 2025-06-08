#!/usr/bin/env python3
"""
Test if images in each person folder actually belong to the same person
by checking visual similarity
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import random

def calculate_image_similarity(img1_path, img2_path):
    """Calculate similarity between two face images"""
    try:
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize to same size for comparison
        size = (100, 100)
        img1_resized = cv2.resize(img1, size)
        img2_resized = cv2.resize(img2, size)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation
        
    except Exception as e:
        print(f"Error comparing images: {e}")
        return 0.0

def test_person_folder_consistency():
    """Test if images in each person folder are consistent"""
    
    # Get untrained persons
    trained_persons = ['PERSON-0001', 'PERSON-0002', 'PERSON-0007', 'PERSON-0008', 
                      'PERSON-0010', 'PERSON-0017', 'PERSON-0019', 'PERSON-0020', 'PERSON-0021']
    
    persons_dir = Path("processing/outputs/persons")
    
    # Test some untrained persons with many images
    test_persons = [
        ('PERSON-0011', 2073),  # Has most images
        ('PERSON-0123', 378),
        ('PERSON-0098', 333),
        ('PERSON-0056', 159),
        ('PERSON-0082', 120)
    ]
    
    print("Testing image consistency in NEW person folders (untrained persons)")
    print("=" * 80)
    
    for person_id, expected_count in test_persons:
        person_folder = persons_dir / person_id
        if not person_folder.exists():
            continue
            
        print(f"\nTesting {person_id}:")
        
        # Get all images
        images = list(person_folder.glob("*.jpg"))
        print(f"  Found {len(images)} images")
        
        if len(images) < 2:
            print("  Not enough images to test")
            continue
        
        # Sample random images for testing
        sample_size = min(10, len(images))
        sampled_images = random.sample(images, sample_size)
        
        # Calculate similarities between sampled images
        similarities = []
        
        # Compare first image with others
        base_img = sampled_images[0]
        print(f"  Comparing {sample_size-1} images with base image: {base_img.name}")
        
        for i, test_img in enumerate(sampled_images[1:], 1):
            similarity = calculate_image_similarity(base_img, test_img)
            similarities.append(similarity)
            
            if i <= 5:  # Show first 5 comparisons
                status = "✓" if similarity > 0.6 else "✗"
                print(f"    {status} {test_img.name}: similarity = {similarity:.3f}")
        
        # Calculate statistics
        if similarities:
            avg_similarity = np.mean(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            print(f"  Summary:")
            print(f"    Average similarity: {avg_similarity:.3f}")
            print(f"    Min similarity: {min_similarity:.3f}")
            print(f"    Max similarity: {max_similarity:.3f}")
            
            # Check for potential issues
            low_similarity_count = sum(1 for s in similarities if s < 0.5)
            if low_similarity_count > 0:
                print(f"  ⚠️  Warning: {low_similarity_count}/{len(similarities)} images have low similarity")
                print(f"     This might indicate different persons mixed in the same folder")
    
    # Now let's check what happens during detection
    print("\n" + "=" * 80)
    print("ANALYSIS: Why are these persons getting new codes?")
    print("=" * 80)
    
    print("\n1. PERSON-0011 (2073 images):")
    print("   - This person appears very frequently")
    print("   - NOT in the trained model")
    print("   - Each time they appear: Model returns 'unknown' → New PERSON code assigned")
    
    print("\n2. The cycle continues:")
    print("   - Same person detected → 'unknown' → PERSON-0235")
    print("   - Same person detected again → 'unknown' → PERSON-0236")
    print("   - And so on...")
    
    print("\n3. Why UI test works:")
    print("   - When you test an image from PERSON-0011 folder")
    print("   - The model correctly says 'unknown' (because not trained)")
    print("   - This is the expected behavior!")
    
    print("\n" + "=" * 80)
    print("SOLUTION:")
    print("=" * 80)
    print("To recognize PERSON-0011 and other frequent persons:")
    print("1. Retrain the model including these persons")
    print("2. Then PERSON-0011 will be recognized instead of getting new codes")
    print("3. Use the refinement scripts to update the model")

if __name__ == "__main__":
    test_person_folder_consistency()