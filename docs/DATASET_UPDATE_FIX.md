# Dataset Update Fix - KeyError Resolution

## Problem
When removing multiple persons through the UI, the system was throwing a `KeyError: 'total_embeddings'` error. This occurred in the `update_datasets_after_person_deletion` function when trying to update dataset totals after person removal.

## Root Cause
The code assumed that all dataset_info.json files would have certain fields like:
- `total_embeddings`
- `total_images`
- `total_faces`
- etc.

However, some datasets might not have all these fields, or they might be organized differently.

## Solution
Updated the `update_datasets_after_person_deletion` function in `/mnt/d/sm-hrm/hr_management/blueprints/persons.py` to:

1. Check if fields exist before trying to update them
2. Handle all possible total fields that might be present
3. Gracefully skip fields that don't exist

### Code Changes
```python
# Before (would throw KeyError if field didn't exist):
dataset_info['total_embeddings'] -= person_data.get('embeddings_count', 0)

# After (checks field existence first):
if 'total_embeddings' in dataset_info:
    dataset_info['total_embeddings'] -= person_data.get('embeddings_count', 0)
```

### Fields Now Handled
- `total_images`
- `total_faces`
- `total_embeddings`
- `total_train_images`
- `total_val_images`
- `total_train_features`
- `total_val_features`
- `total_features`

## Testing
Created test scripts to verify the fix:
- `/mnt/d/sm-hrm/scripts/test_dataset_update_fix.py`
- `/mnt/d/sm-hrm/scripts/test_remove_multiple_fix.py`

Both tests pass successfully, confirming that:
- Empty person ID lists are handled correctly
- Non-existent person IDs don't cause errors
- Datasets with missing fields are handled gracefully

## Impact
Users can now safely remove multiple persons through the UI without encountering the KeyError. The system will:
- Update only the fields that exist in each dataset
- Skip missing fields without errors
- Continue processing other datasets even if one has issues