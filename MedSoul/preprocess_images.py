"""
Preprocess MIMIC-CXR images: resize to 224x224 and save to disk
This eliminates the resize bottleneck during training
"""
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import argparse


def process_single_image(row, image_root, output_root, target_size=224):
    """Process a single image: resize and save"""
    subject_id = int(row['subject_id'])
    study_id = int(row['study_id'])
    dicom_id = row['dicom_id']
    
    # Construct paths
    subject_prefix = str(subject_id)[:2]
    
    # Source path
    src_path = (image_root / 
                f'p{subject_prefix}' / 
                f'p{subject_id}' / 
                f's{study_id}' / 
                f'{dicom_id}.jpg')
    
    # Destination path (same structure)
    dst_dir = (output_root / 
               f'p{subject_prefix}' / 
               f'p{subject_id}' / 
               f's{study_id}')
    dst_path = dst_dir / f'{dicom_id}.jpg'
    
    # Skip if already processed
    if dst_path.exists():
        return True, None
    
    # Check if source exists
    if not src_path.exists():
        return False, f"Missing: {src_path}"
    
    try:
        # Load, resize, and save
        img = Image.open(src_path).convert('RGB')
        img_resized = img.resize((target_size, target_size), Image.LANCZOS)
        
        # Create output directory
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with high quality
        img_resized.save(dst_path, 'JPEG', quality=95)
        
        return True, None
    except Exception as e:
        return False, f"Error processing {src_path}: {str(e)}"


def preprocess_split(split_name, image_root, output_root, split_csv, target_size=224, num_workers=8):
    """Preprocess all images in a split"""
    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*70}")
    
    # Load split data
    split_df = pd.read_csv(split_csv)
    split_data = split_df[split_df['split'] == split_name].reset_index(drop=True)
    
    print(f"Total images in {split_name}: {len(split_data)}")
    
    # Check how many already processed
    existing_count = 0
    for idx, row in split_data.iterrows():
        subject_id = int(row['subject_id'])
        study_id = int(row['study_id'])
        dicom_id = row['dicom_id']
        subject_prefix = str(subject_id)[:2]
        
        dst_path = (output_root / 
                   f'p{subject_prefix}' / 
                   f'p{subject_id}' / 
                   f's{study_id}' / 
                   f'{dicom_id}.jpg')
        
        if dst_path.exists():
            existing_count += 1
    
    print(f"Already processed: {existing_count}/{len(split_data)}")
    
    if existing_count == len(split_data):
        print(f"[SKIP] All images already processed for {split_name}")
        return
    
    # Process images in parallel
    print(f"Processing remaining {len(split_data) - existing_count} images...")
    
    process_func = partial(
        process_single_image,
        image_root=image_root,
        output_root=output_root,
        target_size=target_size
    )
    
    success_count = 0
    error_count = 0
    errors = []
    
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, [row for _, row in split_data.iterrows()]),
            total=len(split_data),
            desc=f"Processing {split_name}"
        ))
    
    for success, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            if error_msg and len(errors) < 10:
                errors.append(error_msg)
    
    print(f"\n[RESULTS]")
    print(f"  Success: {success_count}/{len(split_data)}")
    print(f"  Errors: {error_count}/{len(split_data)}")
    
    if errors:
        print(f"\n[ERRORS] First {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-CXR images')
    parser.add_argument('--image_root', type=str, 
                       default='datasets/mimic-cxr-jpg-2.1.0/files',
                       help='Source image root directory')
    parser.add_argument('--output_root', type=str,
                       default='datasets/mimic-cxr-jpg-224',
                       help='Output directory for preprocessed images')
    parser.add_argument('--split_csv', type=str,
                       default='datasets/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz',
                       help='Path to split CSV file')
    parser.add_argument('--target_size', type=int, default=224,
                       help='Target image size (default: 224)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--splits', type=str, nargs='+', 
                       default=['train', 'validate', 'test'],
                       help='Splits to process (default: train validate test)')
    
    args = parser.parse_args()
    
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)
    
    print("="*70)
    print("MIMIC-CXR Image Preprocessing")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Source: {image_root}")
    print(f"  Output: {output_root}")
    print(f"  Target size: {args.target_size}x{args.target_size}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Splits: {args.splits}")
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name in args.splits:
        preprocess_split(
            split_name=split_name,
            image_root=image_root,
            output_root=output_root,
            split_csv=args.split_csv,
            target_size=args.target_size,
            num_workers=args.num_workers
        )
    
    print("\n" + "="*70)
    print("[SUCCESS] Preprocessing completed!")
    print("="*70)
    print(f"\nPreprocessed images saved to: {output_root}")
    print(f"\nNext steps:")
    print(f"  1. Update config.yaml:")
    print(f"     image_root: '{output_root}'")
    print(f"  2. Set transform to skip resize (only normalize)")
    print(f"  3. Start training with much faster data loading!")


if __name__ == '__main__':
    main()

