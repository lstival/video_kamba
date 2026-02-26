import os
import urllib.request
import zipfile

def download_davis_test(base_dir):
    print("Downloading DAVIS 2017 Dataset (480p test-dev)...")
    davis_dir = os.path.join(base_dir, "DAVIS")
    os.makedirs(davis_dir, exist_ok=True)
    
    # Official URL for 480p test-dev 2017
    url = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip"
    zip_path = os.path.join(davis_dir, "DAVIS-2017-test-dev-480p.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading from {url} to {zip_path}...")
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

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    download_davis_test(data_dir)
