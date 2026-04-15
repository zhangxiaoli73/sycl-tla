#!/usr/bin/env python3
"""
Update perf_results.md with projection results for each M/N/K from the tables.
"""

import re
import sys
from projection import proj_allgather_gemm, proj_gemm_reducescatter, parse_data_type_size


def parse_markdown_table(table_text):
    """Parse markdown table into list of dicts."""
    lines = table_text.strip().split('\n')
    if len(lines) < 3:
        return []
    
    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split('|')[1:-1]]
    
    # Skip separator line (lines[1])
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split('|')[1:-1]]
        if len(cols) != len(headers):
            continue
        row = dict(zip(headers, cols))
        rows.append(row)
    
    return headers, rows


def format_markdown_table(headers, rows):
    """Format headers and rows back to markdown table."""
    # Header row
    result = "| " + " | ".join(headers) + " |\n"
    
    # Separator
    result += "|" + "|".join(["---" for _ in headers]) + "|\n"
    
    # Data rows
    for row in rows:
        values = [row.get(h, "") for h in headers]
        result += "| " + " | ".join(values) + " |\n"
    
    return result


def main():
    # Read perf_results.md
    with open('perf_results.md', 'r') as f:
        content = f.read()
    
    # Data type and TP from config
    data_type_size = parse_data_type_size("bf16")
    intra_node_tp = 4
    
    # Split by sections
    sections = content.split('## ')
    
    updated_content = ""
    section_idx = 0
    
    for section in sections:
        if section_idx == 0:
            # Keep the header as is
            updated_content += "## " + section
            section_idx += 1
            continue
        
        lines = section.split('\n')
        section_title = lines[0]
        
        # Determine pattern
        if "Allgather" in section_title:
            pattern_name = "allgather_gemm"
        elif "Reduce-Scatter" in section_title:
            pattern_name = "gemm_reducescatter"
        else:
            updated_content += "## " + section
            section_idx += 1
            continue
        
        # Find table start and end
        table_start = None
        table_end = None
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                if table_start is None:
                    table_start = i
                table_end = i
        
        if table_start is None:
            updated_content += "## " + section
            section_idx += 1
            continue
        
        # Extract table
        table_lines = lines[table_start:table_end + 1]
        table_text = '\n'.join(table_lines)
        
        headers, rows = parse_markdown_table(table_text)
        
        # Add projection column if not present
        if "projection" not in headers:
            headers.append("projection")
        
        # Process each row: call projection function with M, N, K
        print(f"Processing {pattern_name} pattern...")
        for row in rows:
            try:
                m = int(row['M'])
                n = int(row['N'])
                k = int(row['K'])
                
                if pattern_name == "allgather_gemm":
                    proj_result = proj_allgather_gemm(m, n, k, data_type_size, intra_node_tp)
                else:  # gemm_reducescatter
                    proj_result = proj_gemm_reducescatter(m, n, k, data_type_size, intra_node_tp)
                
                # Format result in ms with 3 decimal places
                row["projection"] = f"{proj_result * 1000:.3f}"
                print(f"  M={m}, N={n}, K={k} -> projection={proj_result * 1000:.3f}ms")
                
            except Exception as e:
                print(f"  Error processing row {row}: {e}")
                row["projection"] = "N/A"
        
        # Reconstruct section
        updated_content += "## " + section_title + "\n"
        updated_content += "\n".join(lines[1:table_start]) + "\n"
        updated_content += format_markdown_table(headers, rows)
        if table_end + 1 < len(lines):
            updated_content += "\n".join(lines[table_end + 1:])
        
        section_idx += 1
    
    # Write back
    with open('perf_results.md', 'w') as f:
        f.write(updated_content)
    
    print("Done! Updated perf_results.md with projection results.")


if __name__ == '__main__':
    main()
