# Employee Person Mapping Feature

## Overview
This feature allows mapping employees to PERSON codes detected in videos for automatic attendance tracking.

## Important Concepts

### PERSON Codes vs Employee IDs
- **PERSON-XXXX**: These are codes assigned to detected persons in videos (e.g., PERSON-0001, PERSON-0002)
- **Employee ID**: The actual employee identifier in the HR system

### Mapping Rules
- **One Employee → Many PERSON codes**: YES ✓
  - The same employee can appear as different persons in different videos
  - Different clothing, lighting, angles can cause the system to assign different PERSON codes
  - Example: John Doe might be detected as PERSON-0001 in Monday's video and PERSON-0047 in Tuesday's video
  
- **One PERSON code → Many Employees**: NO ✗
  - Each PERSON code should map to only ONE real employee
  - Once PERSON-0001 is mapped to John Doe, it cannot be mapped to Jane Smith
  - This ensures accurate attendance tracking

## What Was Implemented

### 1. Database Changes
- Created `employee_person_mappings` table with fields:
  - `employee_id` - Links to employees table
  - `person_code` - The PERSON code (e.g., PERSON-0001)
  - `is_primary` - Whether this is the primary person code for the employee
  - `confidence` - Confidence level of the mapping
  - `mapped_by` - Who created the mapping
  - `mapped_at` - When the mapping was created
  - `notes` - Optional notes about the mapping
- Added `assigned_person_codes` field to employees table for quick lookup
- Added `employee_id` field to detected_persons table for direct linking

### 2. Backend Routes
Added two new routes in `/blueprints/employees.py`:
- `POST /employees/<id>/map-person` - Maps a person code to an employee
- `POST /employees/<id>/unmap-person` - Removes a person code mapping

### 3. Frontend UI
Updated employee detail page (`/templates/employees/detail.html`):
- Added "Person Code Mapping" section
- Shows currently assigned person codes with primary indicator
- Provides dropdown to select available unmapped person codes
- Allows adding notes and setting primary status
- Includes remove button for each mapped code

Updated employee list page (`/templates/employees/index.html`):
- Added visual indicator showing if employee has person mappings
- Shows mapped person codes directly on employee cards

### 4. Migration Script
Created `/scripts/migrate_add_employee_person_mapping.py`:
- Creates the employee_person_mappings table
- Adds necessary indexes for performance
- Adds fields to existing tables

## How It Works

### Mapping Process
1. Go to an employee's detail page
2. In the "Person Code Mapping" section, select a person code from dropdown
3. Optionally check "Set as primary" and add notes
4. Click "Map Person Code" to create the mapping
5. The person code appears in the assigned list

### Unmapping Process
1. Click "Remove" next to any assigned person code
2. Confirm the removal
3. The person code becomes available for other employees

### Primary Person Code
- Each employee can have multiple person codes mapped
- One can be marked as "primary" (shown with blue highlight)
- Primary mappings are useful for priority matching in attendance

## Benefits
1. **Automatic Attendance**: Once mapped, attendance is automatically tracked when the person is detected in videos
2. **Multiple Codes**: Supports employees who may appear as different person codes in different videos
3. **Confidence Tracking**: Each mapping has a confidence score for quality control
4. **Audit Trail**: Tracks who mapped codes and when

## Next Steps
To use this feature:
1. Process videos to detect persons
2. Review detected person codes in the Persons section
3. Map employees to their corresponding person codes
4. Attendance will be automatically tracked for mapped employees

## Technical Notes
- Uses many-to-many relationship between employees and person codes
- Enforces unique constraint to prevent duplicate mappings
- Updates are atomic to maintain data consistency
- Compatible with existing attendance tracking system