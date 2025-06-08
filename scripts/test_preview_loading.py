#!/usr/bin/env python3
"""
Test script to debug preview loading issue
"""

test_js = """
// Test function to debug preview loading
function testPreviewLoading() {
    // Simulate test results with multiple target persons
    const testResults = {
        results: [
            {
                person_id: "PERSON-0022",
                misidentified: true,
                split_suggestions: [
                    {
                        split_to: "PERSON-0019",
                        images: [
                            {image: "img1.jpg", confidence: 0.9},
                            {image: "img2.jpg", confidence: 0.95}
                        ]
                    },
                    {
                        split_to: "PERSON-0021",
                        images: [
                            {image: "img3.jpg", confidence: 0.92}
                        ]
                    }
                ]
            },
            {
                person_id: "PERSON-0045",
                misidentified: true,
                split_suggestions: [
                    {
                        split_to: "PERSON-0002",
                        images: [
                            {image: "img4.jpg", confidence: 0.88}
                        ]
                    }
                ]
            }
        ]
    };
    
    // Build operations
    const operations = [];
    testResults.results.forEach(result => {
        if (result.misidentified && result.split_suggestions) {
            result.split_suggestions.forEach(suggestion => {
                operations.push({
                    type: 'split_merge',
                    source_person: result.person_id,
                    target_person: suggestion.split_to,
                    images: suggestion.images
                });
            });
        }
    });
    
    // Group by target
    const groupedByTarget = {};
    operations.forEach(op => {
        if (!groupedByTarget[op.target_person]) {
            groupedByTarget[op.target_person] = [];
        }
        groupedByTarget[op.target_person].push(op);
    });
    
    console.log("Grouped by target:", groupedByTarget);
    console.log("Target persons:", Object.keys(groupedByTarget));
    
    // Check if containers exist
    setTimeout(() => {
        Object.keys(groupedByTarget).forEach(targetPerson => {
            const container = document.getElementById(`targetPreview-${targetPerson}`);
            console.log(`Container for ${targetPerson}:`, container ? "FOUND" : "NOT FOUND");
        });
    }, 200);
}

// Run the test
testPreviewLoading();
"""

print("JavaScript test code to debug preview loading:")
print("=" * 60)
print(test_js)
print("=" * 60)
print("\nTo test:")
print("1. Open the persons page in browser")
print("2. Open browser console (F12)")
print("3. Copy and paste the JavaScript code above")
print("4. Check console output for:")
print("   - Which target persons are found")
print("   - Whether preview containers exist for each")
print("\nExpected output:")
print("   Container for PERSON-0019: FOUND")
print("   Container for PERSON-0021: FOUND")
print("   Container for PERSON-0002: FOUND")