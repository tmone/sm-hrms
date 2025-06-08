#!/usr/bin/env python3
"""
Test the batch recognition API directly
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import json
from collections import defaultdict
import numpy as np

# Import Flask app
from app import app
from flask_login import login_user
from models.employee import Employee

def test_batch_api():
    """Test batch recognition API without HTTP"""
    
    with app.app_context():
        # Create test client
        client = app.test_client()
        
        # Login first (create a test session)
        with client:
            # Mock login by setting session
            with client.session_transaction() as sess:
                sess['_user_id'] = '1'  # Mock user ID
            
            print("Testing batch recognition API...")
            print("=" * 80)
            
            # Test with PERSON-0022
            test_data = {
                'person_ids': ['PERSON-0022']
            }
            
            print(f"Testing persons: {test_data['person_ids']}")
            
            # Call the batch test endpoint
            response = client.post('/persons/api/batch-test',
                                 json=test_data,
                                 content_type='application/json')
            
            if response.status_code == 200:
                data = response.get_json()
                
                if data.get('success'):
                    print("\n✓ Batch test successful!")
                    print(f"  Tested persons: {data.get('tested_persons', 0)}")
                    print(f"  Misidentified: {data.get('misidentified_count', 0)}")
                    print(f"  Images to move: {data.get('total_images_to_move', 0)}")
                    
                    # Show model info
                    model_info = data.get('model_info', {})
                    print(f"\n  Model: {model_info.get('name', 'unknown')}")
                    print(f"  Trained persons: {', '.join(model_info.get('trained_persons', []))}")
                    
                    # Show results for each person
                    results = data.get('results', [])
                    for result in results:
                        print(f"\n  {result['person_id']}:")
                        print(f"    - Tested: {result['tested_images']}/{result['total_images']} images")
                        print(f"    - Misidentified: {'YES' if result['misidentified'] else 'NO'}")
                        
                        if result.get('predictions'):
                            print("    - Predictions:")
                            for pred_id, pred_data in result['predictions'].items():
                                print(f"      {pred_id}: {pred_data['count']} images ({pred_data['percentage']:.1f}%)")
                        
                        if result.get('split_suggestions'):
                            print("    - Suggested moves:")
                            for suggestion in result['split_suggestions']:
                                print(f"      → Move {suggestion['count']} images to {suggestion['split_to']}")
                                print(f"        (avg confidence: {suggestion['confidence']:.3f})")
                else:
                    print(f"✗ Batch test failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"✗ HTTP error: {response.status_code}")
                print(f"Response: {response.data}")

# Mock login_required decorator
import functools
def mock_login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function

# Patch login_required
import hr_management.blueprints.persons as persons_module
persons_module.login_required = mock_login_required

if __name__ == "__main__":
    test_batch_api()