import os
import urllib.request
import zipfile

def download_davis(base_dir):
    print("Downloading DAVIS 2017 Dataset (480p trainval)...")
    davis_dir = os.path.join(base_dir, "DAVIS")
    os.makedirs(davis_dir, exist_ok=True)
    
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    zip_path = os.path.join(davis_dir, "DAVIS-2017-trainval-480p.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading from {url} to {zip_path}...")
        # Add a simple progress hook
        def hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rDownloading: {percent}%", end="")
            
        urllib.request.urlretrieve(url, zip_path, reporthook=hook)
        print("\nDownload complete.")
    else:
        print(f"File {zip_path} already exists. Skipping download.")
        
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(davis_dir)
    print("Extraction complete.")

def download_hmdb51(base_dir):
    print("\nDownloading HMDB51 Dataset from HuggingFace...")
    hmdb51_dir = os.path.join(base_dir, "HMDB51")
    os.makedirs(hmdb51_dir, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading to {hmdb51_dir}...")
        snapshot_download(repo_id="jili5044/hmdb51", repo_type="dataset", local_dir=hmdb51_dir)
        print("HMDB51 Download complete.")
        
        # Add extraction logic
        zip_path = os.path.join(hmdb51_dir, "hmdb51.zip")
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(hmdb51_dir)
            print("HMDB51 Extraction complete.")
        else:
            print(f"Warning: {zip_path} not found for extraction.")
            
    except ImportError:
        print("\nError: The 'huggingface_hub' package is required to download HMDB51.")
        print("Please install it by running: pip install huggingface_hub")
        print("Alternatively, you can manually clone the repository:")
        print(f"git clone https://huggingface.co/datasets/jili5044/hmdb51 {hmdb51_dir}")

if __name__ == "__main__":
    # Define the base data directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Datasets will be saved to: {data_dir}\n")
    
    # 1. Download DAVIS
    download_davis(data_dir)
    
    # 2. Download HMDB51
    download_hmdb51(data_dir)
    
    print("\nAll tasks finished.")
