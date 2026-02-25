import os
import glob
from PIL import Image

def test_davis(base_dir):
    print("--- Testing DAVIS Dataset ---")
    davis_dir = os.path.join(base_dir, "DAVIS", "DAVIS")
    
    if not os.path.exists(davis_dir):
        print("❌ Error: DAVIS directory not found.")
        return False
        
    images_dir = os.path.join(davis_dir, "JPEGImages", "480p")
    annotations_dir = os.path.join(davis_dir, "Annotations", "480p")
    
    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        print("❌ Error: JPEGImages or Annotations missing.")
        return False
        
    videos = os.listdir(images_dir)
    print(f"✅ Found {len(videos)} video sequences in DAVIS.")
    
    if len(videos) > 0:
        first_video = os.path.join(images_dir, videos[0])
        first_image = glob.glob(os.path.join(first_video, "*.jpg"))[0]
        
        try:
            with Image.open(first_image) as img:
                print(f"✅ Successfully loaded image {os.path.basename(first_image)}: size={img.size}, mode={img.mode}")
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return False
            
    print("✔️ DAVIS dataset passed smoke test.\n")
    return True

def test_hmdb51(base_dir):
    print("--- Testing HMDB51 Dataset ---")
    hmdb51_dir = os.path.join(base_dir, "HMDB51")
    zip_path = os.path.join(hmdb51_dir, "hmdb51.zip")
    extracted_dir = os.path.join(hmdb51_dir, "extracted")
    
    if not os.path.exists(zip_path):
        print(f"❌ Error: hmdb51.zip not found at {zip_path}.")
        return False
    
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"✅ hmdb51.zip found, size: {size_mb:.2f} MB")
    
    if size_mb < 2000:
        print("⚠️ Warning: hmdb51.zip size seems too small.")
    
    if not os.path.exists(extracted_dir):
        print(f"Extracting hmdb51.zip to {extracted_dir}...")
        os.makedirs(extracted_dir, exist_ok=True)
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            print("✅ Extracted hmdb51.zip successfully.")
        except Exception as e:
            print(f"❌ Failed to extract hmdb51.zip: {e}")
            return False
            
    # Check what is inside
    files = os.listdir(extracted_dir)
    print(f"✅ Found {len(files)} items extracted in HMDB51.")
    
    # Let's see if there are RAR files or avi files directly
    rar_files = glob.glob(os.path.join(extracted_dir, "*.rar"))
    if len(rar_files) > 0:
        print(f"✅ Found {len(rar_files)} RAR files representing categories.")
        print("Note: To use the dataset, you might need to extract these RAR files (e.g., using unrar).")
        
    print("✔️ HMDB51 dataset passed smoke test.\n")
    return True

if __name__ == "__main__":
    base_data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
    
    davis_ok = test_davis(base_data_dir)
    hmdb51_ok = test_hmdb51(base_data_dir)
    
    if davis_ok and hmdb51_ok:
        print("🎉 ALL DATASETS PASSED SMOKE TEST 🎉")
    else:
        print("❌ SOME DATASETS FAILED SMOKE TEST")
