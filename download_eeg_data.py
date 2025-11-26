"""
Download real EEG data from OpenNeuro dataset ds004504
Uses direct HTTP downloads from OpenNeuro's file API
"""

import os
import requests
from pathlib import Path
import time

# Configuration
DATASET_ID = "ds004504"
VERSION = "1.0.7"
BASE_URL = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"
DATA_DIR = Path(r"c:\Users\Govin\Desktop\ML_dataset\data\ds004504")

# All 88 subjects
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 89)]

def download_file(url, output_path, max_retries=3):
    """Download a file with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {output_path.name}... (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file in chunks
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file size
                file_size = output_path.stat().st_size
                if file_size > 1000:  # Real EEG files should be > 1KB
                    print(f"✅ Downloaded {output_path.name} ({file_size / 1024 / 1024:.2f} MB)")
                    return True
                else:
                    print(f"❌ File too small ({file_size} bytes), retrying...")
                    output_path.unlink()
            else:
                print(f"❌ HTTP {response.status_code}: {response.reason}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)  # Wait before retry
    
    return False

def main():
    print("=" * 80)
    print(f"📥 Downloading EEG data from OpenNeuro dataset {DATASET_ID}")
    print("=" * 80)
    
    success_count = 0
    failed_subjects = []
    
    for subject in SUBJECTS:
        print(f"\n📂 Processing {subject}...")
        
        # Download .set file (contains both header and data)
        set_file = f"{subject}_task-eyesclosed_eeg.set"
        set_url = f"{BASE_URL}/derivatives/{subject}/eeg/{set_file}"
        set_path = DATA_DIR / "derivatives" / subject / "eeg" / set_file
        
        # Skip if already downloaded
        if set_path.exists() and set_path.stat().st_size > 1000:
            print(f"✅ {set_file} already exists ({set_path.stat().st_size / 1024 / 1024:.2f} MB), skipping...")
            success_count += 1
            continue
        
        # Download .set file
        set_success = download_file(set_url, set_path)
        
        if set_success:
            success_count += 1
        else:
            failed_subjects.append(subject)
            print(f"⚠️ Failed to download {subject}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully downloaded: {success_count}/{len(SUBJECTS)} subjects")
    
    if failed_subjects:
        print(f"\n❌ Failed subjects ({len(failed_subjects)}):")
        for subj in failed_subjects:
            print(f"   - {subj}")
    else:
        print("\n🎉 All files downloaded successfully!")
    
    print("\n💡 TIP: If downloads failed, try:")
    print("   1. Check internet connection")
    print("   2. Run script again (will skip existing files)")
    print("   3. Download manually from: https://openneuro.org/datasets/ds004504")

if __name__ == "__main__":
    main()
