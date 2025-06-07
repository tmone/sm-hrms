# Person Recognition Status and Debugging Guide

## Current Status

### ✅ What's Fixed
1. **Recognition Model Files**: Created missing files for the model:
   - `label_encoder.pkl` - Maps prediction indices to person IDs
   - `persons.json` - Contains person ID mappings
   - `person_id_mapping.json` - Additional mapping file

2. **Improved Components Created**:
   - `ImprovedSharedStateManagerV3` - Prioritizes recognition before creating new IDs
   - `PersonIDManager` - Centralized management of person ID assignments
   - Both components are integrated into `chunked_video_processor.py`

### 🔍 How Recognition Should Work

1. **During Video Processing**:
   ```
   Video Frame → YOLO Detection → Person Crop → Recognition Model
                                                    ↓
                                             Recognized ID or None
                                                    ↓
                                             PersonIDManager
                                                    ↓
                                        Existing ID or New PERSON-XXXX
   ```

2. **Key Decision Point**:
   - If person is recognized (e.g., as PERSON-0001), that ID should be reused
   - Only create new PERSON-XXXX if person is NOT recognized

### 🐛 Known Issues

1. **NumPy Version Incompatibility**: 
   - Error: `No module named 'numpy._core'`
   - This might prevent the recognition model from loading properly

2. **Import Path Issues**:
   - Some scripts have trouble importing from `hr_management` module
   - This affects the recognition inference module

### 📋 How to Check if Recognition is Working

1. **Check Recognition During Processing**:
   Look in the processing logs for messages like:
   ```
   ✅ Using recognized person ID: PERSON-0001
   🆕 Creating new person ID: PERSON-0022
   ```

2. **Check Person Folders**:
   ```bash
   ls -la processing/outputs/persons/
   ```
   - If same person gets multiple folders (e.g., PERSON-0001 and PERSON-0022 for same person), recognition is NOT working
   - If recognized persons reuse their existing IDs, recognition IS working

3. **Check Debug Logs**:
   ```bash
   ls -la processing/debug_logs/
   cat processing/debug_logs/recognition_debug_*.log | grep "Recognition"
   ```

4. **Check Chunked Processor Logs**:
   During video processing, look for:
   ```
   PersonIDManager assigned PERSON-0001 for recognized person PERSON-0001
   Reusing existing PERSON ID for recognized person PERSON-0001
   ```

### 🔧 Troubleshooting

1. **If Recognition Not Working**:
   - Check if model files exist: `ls models/person_recognition/refined_quick_20250606_054446/`
   - Should have: model.pkl, scaler.pkl, label_encoder.pkl, persons.json

2. **If Creating Duplicate IDs**:
   - Check PersonIDManager is loading: Look for "Loaded X person mappings from model"
   - Check recognition confidence threshold (default 0.85 might be too high)

3. **To Enable More Logging**:
   - Set logging level to DEBUG in chunked_video_processor.py
   - Add print statements in SharedStateManager.resolve_person_ids()

### 💡 Quick Test

1. Upload a video that contains persons already in the model (PERSON-0001, PERSON-0002, etc.)
2. After processing, check:
   ```bash
   # Count person folders
   ls -d processing/outputs/persons/PERSON-* | wc -l
   
   # Check for duplicates
   ls -la processing/outputs/persons/ | grep PERSON
   ```
3. If count increases significantly, recognition is not working properly

### 🎯 Expected Behavior

When uploading a video with trained persons:
- Trained persons should keep their existing IDs (PERSON-0001, etc.)
- Only new/unknown persons should get new IDs (PERSON-0022+)
- Console/logs should show "Reusing existing PERSON ID" messages

### 📝 Model Information

Current default model: `refined_quick_20250606_054446`
Contains 9 trained persons:
- PERSON-0001
- PERSON-0002
- PERSON-0007
- PERSON-0008
- PERSON-0010
- PERSON-0017
- PERSON-0019
- PERSON-0020
- PERSON-0021

These persons should be recognized and reuse their IDs when detected in new videos.