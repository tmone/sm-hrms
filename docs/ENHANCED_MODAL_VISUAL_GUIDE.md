# Enhanced Confirmation Modal - Visual Guide

## Layout Structure

The enhanced confirmation modal now shows preview images of the target persons at the header of each group:

```
+------------------------------------------------------------------+
|  Confirm Image Movements                                      X  |
+------------------------------------------------------------------+
|                                                                  |
|  [i] Operation Summary                                          |
|  Found 3 operations to move 8 images total.                    |
|                                                                  |
| +--------------------------------------------------------------+ |
| | [IMG][IMG][IMG] Move to PERSON-0019                         | |
| |                 5 images will be moved here                  | |
| |                                                              | |
| |   > From PERSON-0022 (3 images)                             | |
| |     - Image1.jpg  Confidence: 93.4%                         | |
| |     - Image2.jpg  Confidence: 97.6%                         | |
| |     - Image3.jpg  Confidence: 99.0%                         | |
| |                                                              | |
| |   > From PERSON-0045 (2 images)                             | |
| |     - Image4.jpg  Confidence: 89.0%                         | |
| |     - Image5.jpg  Confidence: 80.0%                         | |
| +--------------------------------------------------------------+ |
|                                                                  |
| +--------------------------------------------------------------+ |
| | [IMG][IMG][+3] Move to PERSON-0021                          | |
| |                1 image will be moved here                    | |
| |                                                              | |
| |   > From PERSON-0022 (1 image)                              | |
| |     - Image6.jpg  Confidence: 96.8%                         | |
| +--------------------------------------------------------------+ |
|                                                                  |
|  [!] Note: This operation will permanently move these images.   |
|                                                                  |
|                         [Confirm & Process]  [Cancel]            |
+------------------------------------------------------------------+
```

## Key Features

### 1. Target Person Preview Images
- Shows up to 3 circular preview images from the target person's folder
- Images are overlapped for a professional stacked look
- Each image has a white border and subtle shadow

### 2. Visual Hierarchy
- **[IMG][IMG][IMG]** - Preview images of the target person
- **Move to PERSON-0019** - Clear destination identifier
- **5 images will be moved here** - Total count for this destination

### 3. Source Organization
- Images grouped by their source person
- Each image shows filename and confidence score
- Clear indentation shows the relationship

### 4. Additional Indicators
- **[+3]** - Shows when target person has more than 3 images
- Loading animation while fetching preview images
- Fallback icon if no images available

## Implementation Details

### CSS Classes Used:
- `rounded-full` - Circular images
- `border-2 border-white` - White border around images
- `shadow-sm` - Subtle shadow for depth
- `-ml-3` - Negative margin for overlap effect
- `z-index` - Proper stacking order

### Dynamic Loading:
- Preview images load asynchronously after modal opens
- Shows loading placeholders initially
- Graceful fallback for missing images

## Benefits

1. **Visual Confirmation**: Users can see who they're moving images to
2. **Prevents Errors**: Visual preview helps avoid moving to wrong person
3. **Professional Look**: Stacked avatars common in modern UIs
4. **Quick Recognition**: Faster than reading person IDs alone
5. **Confidence Building**: Users feel more confident about the operation