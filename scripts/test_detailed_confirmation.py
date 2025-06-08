#!/usr/bin/env python3
"""
Test script to generate sample data for testing the detailed confirmation modal
"""
import json

# Sample test results that would be returned by batch testing
sample_test_results = {
    "success": True,
    "tested_persons": 3,
    "misidentified_count": 3,
    "total_images_to_move": 8,
    "model_info": {
        "name": "person_model_svm_20250607_181818",
        "trained_persons": ["PERSON-0001", "PERSON-0002", "PERSON-0007", "PERSON-0008", 
                           "PERSON-0010", "PERSON-0017", "PERSON-0019", "PERSON-0020", "PERSON-0021"]
    },
    "results": [
        {
            "person_id": "PERSON-0022",
            "total_images": 6,
            "tested_images": 6,
            "misidentified": True,
            "issue": "untrained_recognized_as_trained",
            "predictions": {
                "PERSON-0019": {
                    "count": 3,
                    "percentage": 50.0,
                    "avg_confidence": 0.967
                },
                "PERSON-0021": {
                    "count": 1,
                    "percentage": 16.7,
                    "avg_confidence": 0.968
                },
                "PERSON-0002": {
                    "count": 1,
                    "percentage": 16.7,
                    "avg_confidence": 0.955
                },
                "unknown": {
                    "count": 1,
                    "percentage": 16.7,
                    "avg_confidence": 0.101
                }
            },
            "split_suggestions": [
                {
                    "split_to": "PERSON-0019",
                    "images": [
                        {"image": "75d7a48d-3c16-4949-80bd-6ae052f35fee.jpg", "confidence": 0.934},
                        {"image": "9930be65-42ef-4eda-8fa2-4308eba0ad0c.jpg", "confidence": 0.976},
                        {"image": "0d490aea-d24c-4d95-afa0-098d56ef116a.jpg", "confidence": 0.990}
                    ],
                    "count": 3,
                    "confidence": 0.967
                },
                {
                    "split_to": "PERSON-0021",
                    "images": [
                        {"image": "4fc7a323-efdf-47c4-861d-747606793185.jpg", "confidence": 0.968}
                    ],
                    "count": 1,
                    "confidence": 0.968
                },
                {
                    "split_to": "PERSON-0002",
                    "images": [
                        {"image": "750adbbe-efdb-4575-8cc1-3585228b0b79.jpg", "confidence": 0.955}
                    ],
                    "count": 1,
                    "confidence": 0.955
                }
            ],
            "images_to_move": {
                "PERSON-0019": [
                    {"image": "75d7a48d-3c16-4949-80bd-6ae052f35fee.jpg", "confidence": 0.934},
                    {"image": "9930be65-42ef-4eda-8fa2-4308eba0ad0c.jpg", "confidence": 0.976},
                    {"image": "0d490aea-d24c-4d95-afa0-098d56ef116a.jpg", "confidence": 0.990}
                ],
                "PERSON-0021": [
                    {"image": "4fc7a323-efdf-47c4-861d-747606793185.jpg", "confidence": 0.968}
                ],
                "PERSON-0002": [
                    {"image": "750adbbe-efdb-4575-8cc1-3585228b0b79.jpg", "confidence": 0.955}
                ]
            }
        },
        {
            "person_id": "PERSON-0045",
            "total_images": 4,
            "tested_images": 4,
            "misidentified": True,
            "issue": "untrained_recognized_as_trained",
            "predictions": {
                "PERSON-0019": {
                    "count": 2,
                    "percentage": 50.0,
                    "avg_confidence": 0.845
                },
                "unknown": {
                    "count": 2,
                    "percentage": 50.0,
                    "avg_confidence": 0.150
                }
            },
            "split_suggestions": [
                {
                    "split_to": "PERSON-0019",
                    "images": [
                        {"image": "abc123.jpg", "confidence": 0.890},
                        {"image": "def456.jpg", "confidence": 0.800}
                    ],
                    "count": 2,
                    "confidence": 0.845
                }
            ],
            "images_to_move": {
                "PERSON-0019": [
                    {"image": "abc123.jpg", "confidence": 0.890},
                    {"image": "def456.jpg", "confidence": 0.800}
                ]
            }
        },
        {
            "person_id": "PERSON-0088",
            "total_images": 1,
            "tested_images": 1,
            "misidentified": True,
            "issue": "untrained_recognized_as_trained",
            "predictions": {
                "PERSON-0002": {
                    "count": 1,
                    "percentage": 100.0,
                    "avg_confidence": 0.920
                }
            },
            "split_suggestions": [
                {
                    "split_to": "PERSON-0002",
                    "images": [
                        {"image": "xyz789.jpg", "confidence": 0.920}
                    ],
                    "count": 1,
                    "confidence": 0.920
                }
            ],
            "images_to_move": {
                "PERSON-0002": [
                    {"image": "xyz789.jpg", "confidence": 0.920}
                ]
            }
        }
    ],
    "move_summary": {
        "PERSON-0019": [
            {"source_person": "PERSON-0022", "image": "75d7a48d-3c16-4949-80bd-6ae052f35fee.jpg", "confidence": 0.934},
            {"source_person": "PERSON-0022", "image": "9930be65-42ef-4eda-8fa2-4308eba0ad0c.jpg", "confidence": 0.976},
            {"source_person": "PERSON-0022", "image": "0d490aea-d24c-4d95-afa0-098d56ef116a.jpg", "confidence": 0.990},
            {"source_person": "PERSON-0045", "image": "abc123.jpg", "confidence": 0.890},
            {"source_person": "PERSON-0045", "image": "def456.jpg", "confidence": 0.800}
        ],
        "PERSON-0021": [
            {"source_person": "PERSON-0022", "image": "4fc7a323-efdf-47c4-861d-747606793185.jpg", "confidence": 0.968}
        ],
        "PERSON-0002": [
            {"source_person": "PERSON-0022", "image": "750adbbe-efdb-4575-8cc1-3585228b0b79.jpg", "confidence": 0.955},
            {"source_person": "PERSON-0088", "image": "xyz789.jpg", "confidence": 0.920}
        ]
    }
}

print("Sample test results for testing detailed confirmation modal:")
print(json.dumps(sample_test_results, indent=2))

print("\n\nTo test the modal:")
print("1. Copy the JSON above")
print("2. Open browser console on the persons page")
print("3. Run: testResults = <paste JSON here>")
print("4. Run: showDetailedConfirmation()")
print("\nThe modal should show:")
print("- 8 total images to move")
print("- Images grouped by destination:")
print("  - PERSON-0019: 5 images (3 from PERSON-0022, 2 from PERSON-0045)")
print("  - PERSON-0021: 1 image (from PERSON-0022)")
print("  - PERSON-0002: 2 images (1 from PERSON-0022, 1 from PERSON-0088)")
print("- Image thumbnails with names and confidence scores")
print("- Warning message about permanent move")