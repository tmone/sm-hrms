# Batch Testing and Auto Split/Merge Feature

## Overview
The batch testing feature allows users to test multiple person folders for misidentification and automatically move misidentified images to their correct persons.

## How It Works

### 1. Select Persons for Testing
- Click "Select Mode" on the persons list page
- Check the boxes for persons you want to test
- Click "Test Recognition" button

### 2. Recognition Testing
The system will:
- Load the default SVM model (person_model_svm_20250607_181818)
- Test each image in the selected person folders
- Use the `process_cropped_image` method since person images are already cropped faces
- Compare results against the 9 trained persons

### 3. Test Results Display
Shows:
- Summary of tested persons and misidentifications found
- Detailed results for each person including:
  - Prediction breakdown (which person IDs were predicted and how many times)
  - Average confidence scores
  - Specific images that should be moved

### 4. Detailed Confirmation Modal
When clicking "Auto Split & Merge Misidentified", a detailed modal shows:

#### Operation Summary
- Total operations and images to move

#### Grouped by Destination
Images are grouped by where they will be moved to, showing:
- Target person ID (e.g., "Move to PERSON-0019")
- Total images going to that person
- Source breakdown showing:
  - Which person folder each image comes from
  - Image thumbnails with filenames
  - Confidence scores for each image
  - "And X more images..." if more than 6

#### Visual Features
- Image thumbnails for preview
- Truncated filenames with full name on hover
- Confidence percentages
- Warning about permanent operation

### 5. Processing
After confirmation:
- Images are physically moved between person folders
- Metadata is updated for both source and target persons
- Success message with refresh option

## Technical Implementation

### Fixed Issues
1. **Method Name**: Changed from `recognize_person` to `process_cropped_image`
2. **Variable Fix**: Fixed undefined `sampled_images` to `test_images`
3. **Enhanced UI**: Added detailed confirmation modal with image previews

### Key Files
- `/mnt/d/sm-hrm/hr_management/blueprints/persons.py` - Backend API endpoints
- `/mnt/d/sm-hrm/templates/persons/index.html` - Frontend UI and JavaScript
- `/mnt/d/sm-hrm/processing/simple_person_recognition_inference.py` - Recognition wrapper

## Example Use Case
If PERSON-0022 contains misidentified images:
- 3 images actually belonging to PERSON-0019
- 1 image actually belonging to PERSON-0021
- 1 image actually belonging to PERSON-0002

The batch test will detect these and allow automatic movement to the correct persons with a single click.

## Benefits
1. **Efficiency**: Process multiple persons at once instead of manual review
2. **Accuracy**: Uses the trained model to identify correct persons
3. **Transparency**: Shows exactly what will be moved before confirmation
4. **Safety**: Requires explicit confirmation before moving images