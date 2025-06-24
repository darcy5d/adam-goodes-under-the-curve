#!/usr/bin/env python3
"""
AFL Prediction Model - Output Navigation Script

This script helps navigate the organized outputs directory structure
and provides quick access to generated files.
"""

import os
import json
from pathlib import Path

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Print a formatted directory tree."""
    if current_depth > max_depth:
        return
    
    items = sorted(os.listdir(path))
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        
        if os.path.isdir(item_path):
            print(f"{prefix}{current_prefix}{item}/")
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_directory_tree(item_path, new_prefix, max_depth, current_depth + 1)
        else:
            size = os.path.getsize(item_path)
            size_str = f" ({size:,} bytes)" if size > 1024 else ""
            print(f"{prefix}{current_prefix}{item}{size_str}")

def list_outputs():
    """List all outputs organized by category."""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("❌ Outputs directory not found. Run the analysis scripts first.")
        return
    
    print("📁 AFL Prediction Model - Outputs Directory")
    print("=" * 50)
    
    # Visualizations
    print("\n📊 VISUALIZATIONS")
    print("-" * 20)
    viz_dir = outputs_dir / "visualizations"
    if viz_dir.exists():
        print_directory_tree(viz_dir, max_depth=2)
    else:
        print("No visualizations found")
    
    # Reports
    print("\n📄 REPORTS")
    print("-" * 20)
    reports_dir = outputs_dir / "reports"
    if reports_dir.exists():
        print_directory_tree(reports_dir, max_depth=2)
    else:
        print("No reports found")
    
    # Data
    print("\n💾 DATA FILES")
    print("-" * 20)
    data_dir = outputs_dir / "data"
    if data_dir.exists():
        print_directory_tree(data_dir, max_depth=2)
    else:
        print("No data files found")

def show_file_info(file_path):
    """Show information about a specific file."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n📄 File: {path}")
    print(f"📏 Size: {path.stat().st_size:,} bytes")
    print(f"📅 Modified: {path.stat().st_mtime}")
    
    # Show file type specific info
    if path.suffix == '.json':
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"📊 JSON keys: {list(data.keys())}")
        except:
            print("📊 JSON file (could not parse)")
    elif path.suffix == '.csv':
        print("📊 CSV file")
    elif path.suffix == '.png':
        print("🖼️ PNG image")
    elif path.suffix == '.md':
        print("📝 Markdown report")

def main():
    """Main navigation interface."""
    print("🔍 AFL Prediction Model - Output Navigator")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. List all outputs")
        print("2. Show file info")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            list_outputs()
        elif choice == "2":
            file_path = input("Enter file path: ").strip()
            show_file_info(file_path)
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 