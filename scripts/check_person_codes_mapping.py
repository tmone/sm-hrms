#!/usr/bin/env python3
"""
Check person codes and mapping status
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from sqlalchemy import text

def check_person_codes():
    """Check all person codes and their mapping status"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        
        print("="*80)
        print("PERSON CODES AND MAPPING STATUS")
        print("="*80)
        
        # 1. Check total detected persons
        print("\n1. Detected Persons Summary:")
        try:
            result = db.session.execute(text(
                "SELECT COUNT(DISTINCT person_id) FROM detected_persons WHERE person_id IS NOT NULL"
            ))
            total_persons = result.scalar()
            print(f"   Total unique person codes detected: {total_persons}")
            
            # Show sample person IDs
            result = db.session.execute(text(
                "SELECT DISTINCT person_id FROM detected_persons WHERE person_id IS NOT NULL ORDER BY person_id LIMIT 20"
            ))
            person_ids = result.fetchall()
            print(f"\n   Sample person IDs from database:")
            for pid in person_ids:
                print(f"   - Raw ID: {pid[0]} (Type: {type(pid[0]).__name__})")
                # Show how it converts to PERSON code
                if isinstance(pid[0], (int, float)):
                    person_code = f"PERSON-{int(pid[0]):04d}"
                else:
                    person_code = str(pid[0])
                print(f"     → Converts to: {person_code}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Check employee mappings
        print("\n2. Employee Person Mappings:")
        try:
            result = db.session.execute(text(
                "SELECT COUNT(*) FROM employee_person_mappings"
            ))
            total_mappings = result.scalar()
            print(f"   Total mappings: {total_mappings}")
            
            if total_mappings > 0:
                print("\n   Current mappings:")
                result = db.session.execute(text(
                    """SELECT e.name, m.person_code, m.is_primary 
                       FROM employee_person_mappings m 
                       JOIN employees e ON m.employee_id = e.id 
                       ORDER BY e.name, m.person_code"""
                ))
                for row in result:
                    primary = " (PRIMARY)" if row[2] else ""
                    print(f"   - {row[0]}: {row[1]}{primary}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 3. Check unmapped person codes
        print("\n3. Unmapped Person Codes:")
        try:
            # Get all person IDs
            all_persons = db.session.execute(text(
                "SELECT DISTINCT person_id FROM detected_persons WHERE person_id IS NOT NULL ORDER BY person_id"
            )).fetchall()
            
            # Get mapped codes
            mapped_codes = db.session.execute(text(
                "SELECT DISTINCT person_code FROM employee_person_mappings"
            )).fetchall()
            mapped_codes_set = {row[0] for row in mapped_codes}
            
            unmapped_count = 0
            unmapped_list = []
            
            for row in all_persons:
                person_id = row[0]
                if isinstance(person_id, (int, float)):
                    person_code = f"PERSON-{int(person_id):04d}"
                else:
                    person_code = str(person_id)
                
                if person_code not in mapped_codes_set:
                    unmapped_count += 1
                    unmapped_list.append(person_code)
            
            print(f"   Total unmapped person codes: {unmapped_count}")
            if unmapped_list:
                print(f"\n   First 20 unmapped codes:")
                for code in unmapped_list[:20]:
                    print(f"   - {code}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 4. Check for data type issues
        print("\n4. Data Type Analysis:")
        try:
            result = db.session.execute(text(
                """SELECT person_id, COUNT(*) as count 
                   FROM detected_persons 
                   WHERE person_id IS NOT NULL 
                   GROUP BY person_id 
                   ORDER BY count DESC 
                   LIMIT 5"""
            ))
            print("   Most frequent person IDs:")
            for row in result:
                print(f"   - ID: {row[0]} (appears {row[1]} times)")
        except Exception as e:
            print(f"   Error: {e}")
        
        # 5. Check specific employee
        print("\n5. Check Specific Employee (ID=1):")
        try:
            # Assigned to employee 1
            result = db.session.execute(text(
                "SELECT person_code FROM employee_person_mappings WHERE employee_id = 1"
            ))
            assigned = [row[0] for row in result]
            print(f"   Person codes assigned to Admin User: {assigned if assigned else 'None'}")
            
            # Available for employee 1
            available_count = 0
            for row in all_persons:
                person_id = row[0]
                if isinstance(person_id, (int, float)):
                    person_code = f"PERSON-{int(person_id):04d}"
                else:
                    person_code = str(person_id)
                
                if person_code not in mapped_codes_set:
                    available_count += 1
            
            print(f"   Person codes available for mapping: {available_count}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    check_person_codes()