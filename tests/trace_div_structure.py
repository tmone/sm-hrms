#!/usr/bin/env python3
"""Trace div structure with details"""

import re

def trace_div_structure(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    depth = 0
    div_stack = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Find opening divs with their classes
        opening_matches = re.finditer(r'<div([^>]*)>', line)
        for match in opening_matches:
            class_match = re.search(r'class="([^"]*)"', match.group(1))
            class_name = class_match.group(1) if class_match else "(no class)"
            div_stack.append((line_num, class_name))
            depth += 1
        
        # Find closing divs
        closing_count = len(re.findall(r'</div>', line))
        for _ in range(closing_count):
            if div_stack:
                opened_at, class_name = div_stack.pop()
                depth -= 1
            else:
                print(f"❌ Extra closing div at line {line_num}")
                print(f"   Line: {line.strip()}")
                print(f"\nCurrent div stack:")
                for ln, cn in div_stack[-5:]:
                    print(f"   Line {ln}: <div class='{cn}'>")
                return
    
    if div_stack:
        print(f"❌ {len(div_stack)} unclosed divs:")
        for line_num, class_name in div_stack:
            print(f"   Line {line_num}: <div class='{class_name}'>")
    else:
        print("✅ All divs are properly closed!")

if __name__ == '__main__':
    trace_div_structure('templates/videos/detail.html')