#!/bin/bash
# Script to organize preprocessed patient data for training
# Usage: ./organize_for_training.sh <processed_data_dir> <output_training_dir>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <processed_data_dir> <output_training_dir>"
    echo "Example: $0 ./processed_data ./training_data"
    exit 1
fi

PROCESSED_DIR="$1"
OUTPUT_DIR="$2"

# Check if processed directory exists
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Error: Processed data directory not found: $PROCESSED_DIR"
    exit 1
fi

# Create output directory structure
echo "Creating output directory structure..."
mkdir -p "$OUTPUT_DIR/t2_images"
mkdir -p "$OUTPUT_DIR/fat_fraction_maps"
mkdir -p "$OUTPUT_DIR/liver_masks"

# Counter
count=0

# Process each patient directory
for patient_dir in "$PROCESSED_DIR"/*/; do
    if [ ! -d "$patient_dir" ]; then
        continue
    fi

    patient_id=$(basename "$patient_dir")
    echo "Processing $patient_id..."

    # Find and copy T2 image
    t2_file="$patient_dir/${patient_id}_t2_aligned.nii.gz"
    if [ -f "$t2_file" ]; then
        cp "$t2_file" "$OUTPUT_DIR/t2_images/${patient_id}.nii.gz"
    else
        echo "  Warning: T2 file not found: $t2_file"
        continue
    fi

    # Find and copy fat fraction
    ff_file="$patient_dir/${patient_id}_ff.nii.gz"
    if [ -f "$ff_file" ]; then
        cp "$ff_file" "$OUTPUT_DIR/fat_fraction_maps/${patient_id}.nii.gz"
    else
        echo "  Warning: Fat fraction file not found: $ff_file"
        continue
    fi

    # Find and copy liver mask
    mask_file="$patient_dir/${patient_id}_mask_aligned.nii.gz"
    if [ -f "$mask_file" ]; then
        cp "$mask_file" "$OUTPUT_DIR/liver_masks/${patient_id}.nii.gz"
    else
        echo "  Warning: Mask file not found: $mask_file"
        continue
    fi

    ((count++))
done

echo ""
echo "======================================"
echo "Organization complete!"
echo "======================================"
echo "Total patients processed: $count"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "File counts:"
echo "  T2 images: $(ls -1 $OUTPUT_DIR/t2_images/*.nii.gz 2>/dev/null | wc -l)"
echo "  Fat fraction maps: $(ls -1 $OUTPUT_DIR/fat_fraction_maps/*.nii.gz 2>/dev/null | wc -l)"
echo "  Liver masks: $(ls -1 $OUTPUT_DIR/liver_masks/*.nii.gz 2>/dev/null | wc -l)"
echo ""
echo "Next steps:"
echo "1. Update config.yaml with data_dir: \"$OUTPUT_DIR\""
echo "2. Run: python train.py --config config.yaml"
