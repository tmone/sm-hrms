#!/usr/bin/env python3
"""Check HTML div structure balance"""

import re

def check_div_balance(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Track div depth
    depth = 0
    line_num = 0
    max_depth = 0
    
    for line in content.split('\n'):
        line_num += 1
        
        # Count opening divs
        opening_divs = len(re.findall(r'<div[^>]*>', line))
        # Count closing divs
        closing_divs = len(re.findall(r'</div>', line))
        
        depth += opening_divs
        depth -= closing_divs
        
        if depth > max_depth:
            max_depth = depth
            
        if depth < 0:
            print(f"❌ Error at line {line_num}: More closing divs than opening divs!")
            print(f"   Line content: {line.strip()}")
            return False
    
    print(f"Max div depth: {max_depth}")
    print(f"Final div depth: {depth}")
    
    if depth == 0:
        print("✅ All divs are properly balanced!")
        return True
    else:
        print(f"❌ Unbalanced divs! {depth} unclosed divs remain.")
        return False

if __name__ == '__main__':
    check_div_balance('templates/videos/detail.html')