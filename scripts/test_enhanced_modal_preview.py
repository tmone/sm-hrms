#!/usr/bin/env python3
"""
Visual preview of how the enhanced confirmation modal will look
"""

print("""
ENHANCED CONFIRMATION MODAL PREVIEW
===================================

┌─────────────────────────────────────────────────────────────────────┐
│  ✕  Confirm Image Movements                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ℹ️ Operation Summary                                               │
│  Found 3 operations to move 8 images total.                        │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  [👤][👤][👤] Move to PERSON-0019                              │ │
│ │               5 images will be moved here                       │ │
│ │                                                                 │ │
│ │   📁 From PERSON-0022 (3 images)                               │ │
│ │      [🖼] 75d7a48d-3c16-4949...  Confidence: 93.4%            │ │
│ │      [🖼] 9930be65-42ef-4eda...  Confidence: 97.6%            │ │
│ │      [🖼] 0d490aea-d24c-4d95...  Confidence: 99.0%            │ │
│ │                                                                 │ │
│ │   📁 From PERSON-0045 (2 images)                               │ │
│ │      [🖼] abc123.jpg              Confidence: 89.0%            │ │
│ │      [🖼] def456.jpg              Confidence: 80.0%            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  [👤][👤][+3] Move to PERSON-0021                              │ │
│ │               1 image will be moved here                        │ │
│ │                                                                 │ │
│ │   📁 From PERSON-0022 (1 image)                                │ │
│ │      [🖼] 4fc7a323-efdf-47c4...  Confidence: 96.8%            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │  [👤][👤] Move to PERSON-0002                                  │ │
│ │           2 images will be moved here                           │ │
│ │                                                                 │ │
│ │   📁 From PERSON-0022 (1 image)                                │ │
│ │      [🖼] 750adbbe-efdb-4575...  Confidence: 95.5%            │ │
│ │                                                                 │ │
│ │   📁 From PERSON-0088 (1 image)                                │ │
│ │      [🖼] xyz789.jpg              Confidence: 92.0%            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ⚠️  Note: This operation will permanently move these images.       │
│     The source persons will have fewer images after this operation. │
│                                                                     │
│                              [✓ Confirm & Process]  [Cancel]        │
└─────────────────────────────────────────────────────────────────────┘

KEY FEATURES:
1. Target person preview images shown as overlapping circles (like [👤][👤][👤])
2. Images are pulled from the target person's existing folder
3. Shows up to 3 preview images with overlap effect
4. If more than 3 images exist, shows "+N" indicator
5. Preview images help user visually confirm they're moving to the right person
6. All preview images have rounded borders and slight shadow for depth

VISUAL BENEFITS:
- Immediate visual confirmation of target person identity
- Professional stacked avatar design
- Helps prevent accidental moves to wrong person
- Clean, organized layout with clear hierarchy
""")