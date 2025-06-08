# Recognition Issue Summary

## Root Cause
The person recognition is creating new PERSON IDs for already trained persons because **the recognition model cannot be loaded** due to compatibility issues.

## Debug Evidence
From the debug logs:
```
2025-06-07 14:19:11,883 - __main__ - DEBUG - Frame 42: No recognition model available
ERROR:processing.simple_recognition_fix:Failed to load model: UnpicklingError: invalid load key, '\x08'.
```

Every frame shows "No recognition model available" - this means recognition is completely disabled.

## Why This Happens

1. **Model Incompatibility**: The model was trained with a different version of scikit-learn/numpy than what's currently installed
2. **Missing Files**: The model was missing `label_encoder.pkl` and `persons.json` (we created these, but the main model.pkl is still incompatible)
3. **Import Issues**: The system cannot import the recognition modules properly

## Current Behavior

When processing videos:
- YOLO detects persons ✓
- Tracking assigns temporary IDs ✓  
- Recognition attempts to load model ✗
- Since recognition fails, EVERY person gets a new PERSON-XXXX ID ✗

## Solutions

### Option 1: Retrain the Model (Recommended)
```bash
# Retrain with current environment
python scripts/retrain_person_model.py
```
This will create a compatible model with all required files.

### Option 2: Fix the Current Model
The model file appears corrupted or was saved with incompatible versions. You need to:
1. Check what version of scikit-learn was used to train the model
2. Install that specific version
3. Or convert the model to a compatible format

### Option 3: Use a Different Model
```bash
# List available models
ls models/person_recognition/

# Update config to use a different model
# Edit models/person_recognition/config.json
```

## How to Verify When Fixed

1. **During Processing**: You should see logs like:
   ```
   ✅ Recognition model loaded successfully
   Frame 100: Recognized PERSON-0001 (confidence: 0.95)
   Reusing existing PERSON ID for recognized person PERSON-0001
   ```

2. **After Processing**: Check person folders:
   ```bash
   # Before fix: New folders created for known persons
   PERSON-0001/  # Original
   PERSON-0022/  # Duplicate of PERSON-0001
   PERSON-0023/  # Another duplicate
   
   # After fix: Reuses existing folders
   PERSON-0001/  # All detections go here
   PERSON-0022/  # Only truly new persons
   ```

## Immediate Workaround

While the recognition model is broken, you can:
1. Manually merge duplicate person folders
2. Use the person merge feature in the UI
3. Train a new model with the current environment

## Technical Details

The error `UnpicklingError: invalid load key, '\x08'` typically means:
- The pickle file is corrupted
- Version mismatch between Python/scikit-learn versions
- The file was not saved properly

The model needs to be retrained or the environment needs to match the original training environment.