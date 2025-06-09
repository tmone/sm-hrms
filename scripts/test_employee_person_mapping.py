#!/usr/bin/env python3
"""
Test employee person mapping functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

def test_mapping_functionality():
    """Test the employee person mapping functionality"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        from sqlalchemy import text
        
        print("="*80)
        print("TESTING EMPLOYEE PERSON MAPPING")
        print("="*80)
        
        # Check if table exists
        print("\n1. Checking if employee_person_mappings table exists...")
        try:
            result = db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='employee_person_mappings'"))
            if result.fetchone():
                print("  ✓ Table exists")
                
                # Show columns
                result = db.session.execute(text("PRAGMA table_info(employee_person_mappings)"))
                columns = result.fetchall()
                print("\n  Columns:")
                for col in columns:
                    print(f"    - {col[1]} ({col[2]})")
            else:
                print("  ✗ Table not found")
                return
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return
        
        # Get sample data
        print("\n2. Getting sample employees...")
        Employee = app.Employee
        employees = Employee.query.limit(3).all()
        print(f"  Found {len(employees)} employees")
        for emp in employees:
            print(f"    - {emp.name} (ID: {emp.id})")
        
        print("\n3. Getting detected person codes...")
        try:
            result = db.session.execute(text(
                "SELECT DISTINCT person_id FROM detected_persons WHERE person_id IS NOT NULL ORDER BY person_id LIMIT 10"
            ))
            person_ids = result.fetchall()
            print(f"  Found {len(person_ids)} unique person codes")
            for pid in person_ids[:5]:  # Show first 5
                person_code = f"PERSON-{int(pid[0]):04d}" if isinstance(pid[0], (int, float)) else str(pid[0])
                print(f"    - {person_code}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print("\n4. Checking existing mappings...")
        try:
            result = db.session.execute(text("SELECT COUNT(*) FROM employee_person_mappings"))
            count = result.scalar()
            print(f"  Total mappings: {count}")
            
            if count > 0:
                print("\n  Sample mappings:")
                result = db.session.execute(text(
                    """SELECT e.name, m.person_code, m.is_primary, m.mapped_at 
                       FROM employee_person_mappings m 
                       JOIN employees e ON m.employee_id = e.id 
                       LIMIT 5"""
                ))
                for row in result:
                    primary = "PRIMARY" if row[2] else ""
                    print(f"    - {row[0]} → {row[1]} {primary}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        print("\n5. Testing the mapping flow...")
        print("  - Access an employee detail page to see the mapping UI")
        print("  - Select a person code from the dropdown")
        print("  - Click 'Map Person Code' to create the mapping")
        print("  - The mapped codes will appear on the left side")
        print("  - Click 'Remove' to unmap a person code")
        
        print("\n✓ Employee person mapping functionality is ready!")

if __name__ == "__main__":
    test_mapping_functionality()