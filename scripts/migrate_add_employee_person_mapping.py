#!/usr/bin/env python3
"""
Add employee to person code mapping table
This allows linking employees to detected PERSON codes for attendance tracking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from sqlalchemy import text

def add_employee_person_mapping():
    """Add employee person mapping table and fields"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        
        print("Adding employee person mapping functionality...")
        
        try:
            # Create employee_person_mappings table
            create_mapping_table = """
            CREATE TABLE IF NOT EXISTS employee_person_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                person_code VARCHAR(20) NOT NULL,
                is_primary BOOLEAN DEFAULT TRUE,
                confidence FLOAT DEFAULT 1.0,
                mapped_by VARCHAR(100),
                mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE,
                UNIQUE(employee_id, person_code)
            )
            """
            
            db.session.execute(text(create_mapping_table))
            print("✓ Created employee_person_mappings table")
            
            # Add indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_emp_person_employee ON employee_person_mappings(employee_id)",
                "CREATE INDEX IF NOT EXISTS idx_emp_person_code ON employee_person_mappings(person_code)",
                "CREATE INDEX IF NOT EXISTS idx_emp_person_primary ON employee_person_mappings(is_primary)"
            ]
            
            for sql in indexes:
                db.session.execute(text(sql))
                print(f"✓ {sql}")
            
            # Add assigned_person_codes column to employees table for quick lookup
            try:
                db.session.execute(text(
                    "ALTER TABLE employees ADD COLUMN IF NOT EXISTS assigned_person_codes TEXT"
                ))
                print("✓ Added assigned_person_codes column to employees table")
            except Exception as e:
                print(f"Note: assigned_person_codes column may already exist: {e}")
            
            # Update DetectedPerson to link with employee through mapping
            try:
                # Add employee_id to detected_persons if not exists
                db.session.execute(text(
                    "ALTER TABLE detected_persons ADD COLUMN IF NOT EXISTS employee_id INTEGER REFERENCES employees(id)"
                ))
                print("✓ Added employee_id to detected_persons table")
            except Exception as e:
                print(f"Note: employee_id column may already exist: {e}")
            
            db.session.commit()
            print("\n✓ Employee person mapping tables created successfully!")
            
            # Show summary
            print("\nNew functionality added:")
            print("1. employee_person_mappings table - Links employees to PERSON codes")
            print("2. assigned_person_codes field - Quick lookup of assigned codes")
            print("3. employee_id in detections - Direct link for identified persons")
            
        except Exception as e:
            print(f"\nError adding employee person mapping: {e}")
            db.session.rollback()
            raise

def verify_mapping_tables():
    """Verify the mapping tables were created correctly"""
    app = create_app()
    
    with app.app_context():
        db = app.db
        
        print("\nVerifying mapping tables...")
        
        # Check employee_person_mappings
        result = db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='employee_person_mappings'"))
        if result.fetchone():
            print("✓ employee_person_mappings table exists")
            
            # Show columns
            result = db.session.execute(text("PRAGMA table_info(employee_person_mappings)"))
            columns = result.fetchall()
            print("\n  Columns:")
            for col in columns:
                print(f"    - {col[1]} ({col[2]})")
        else:
            print("✗ employee_person_mappings table not found")

if __name__ == '__main__':
    add_employee_person_mapping()
    # verify_mapping_tables() # Skip verification since it creates app twice