import sys
import json
import os
from pathlib import Path

# Usage: python coco_prep.py <directory_with_txt_files> [output_name]
# Example: python coco_prep.py ./labels coco_labels

if len(sys.argv) < 2:
    print("Usage: python coco_prep.py <directory_with_txt_files> [output_name]")
    print("Example: python coco_prep.py ./labels coco_labels")
    sys.exit(1)

input_dir = sys.argv[1]
output_name = sys.argv[2] if len(sys.argv) > 2 else "predictions"

# Check if directory exists
if not os.path.isdir(input_dir):
    print(f"Error: Directory '{input_dir}' does not exist!")
    sys.exit(1)

predictions = []
merged_dict = {}

# Get all .txt files in the directory
txt_files = list(Path(input_dir).glob("*.txt"))

if not txt_files:
    print(f"Warning: No .txt files found in '{input_dir}'")
    sys.exit(1)

print(f"Found {len(txt_files)} .txt files to process...")

# Process each .txt file
for txt_file in txt_files:
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        # Skip empty files
        if len(lines) < 2:
            print(f"Warning: Skipping {txt_file.name} - insufficient data")
            continue
        
        # Parse file structure
        image_filename = lines[0].strip()
        num_objects = int(lines[1].strip())
        
        # Initialize list for this image
        if image_filename not in merged_dict:
            merged_dict[image_filename] = []
        
        # Parse bounding boxes (format: x y w h confidence)
        for i in range(2, min(2 + num_objects, len(lines))):
            bbox_line = lines[i].strip()
            if not bbox_line:  # Skip empty lines
                continue
            
            values = bbox_line.split()
            if len(values) >= 5:
                # Parse: x, y, w, h, confidence
                x = round(float(values[0]), 5)
                y = round(float(values[1]), 5)
                w = round(float(values[2]), 5)
                h = round(float(values[3]), 5)
                confidence = round(float(values[4]), 5)
                
                bbox = [x, y, w, h, confidence]
                
                # Add to predictions list
                predictions.append({
                    "image_id": image_filename, 
                    "category_id": 1, 
                    "bbox": bbox
                })
                
                # Add to merged dictionary
                merged_dict[image_filename].append(bbox)
        
        if len(merged_dict[image_filename]) != num_objects:
            print(f"Warning: {txt_file.name} - expected {num_objects} objects, found {len(merged_dict[image_filename])}")
    
    except Exception as e:
        print(f"Error processing {txt_file.name}: {str(e)}")
        continue

# Save results
output_file = f"coco-preped-{output_name}.json"
with open(output_file, 'w') as f:
    json.dump(merged_dict, f, indent=2)

print(f"\n✓ Successfully processed {len(merged_dict)} images")
print(f"✓ Total predictions: {len(predictions)}")
print(f"✓ Output saved to: {output_file}")
