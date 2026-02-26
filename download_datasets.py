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

def download_youtubevos(base_dir: str) -> None:
    """Print manual download instructions for YouTube-VOS 2019.

    YouTube-VOS requires an account on the official challenge server.
    Automated bulk download is not provided through a public static URL.

    Expected layout after extraction::

        data/YouTubeVOS/
            train/
                JPEGImages/<video_id>/<frame>.jpg
                Annotations/<video_id>/<frame>.png  (integer object-ID masks)
                meta.json                           (category & object metadata)
            valid/
                JPEGImages/<video_id>/<frame>.jpg
                Annotations/<video_id>/<frame>.png
                meta.json

    Note:
        Annotations are 6 FPS by default.  Request 30 FPS variant from the
        official site for highest accuracy.

    Steps:
        1. Register at https://youtube-vos.org/dataset/vos/
        2. Accept the license agreement.
        3. Download ``train.zip`` (~26 GB) and ``valid.zip`` (~6 GB).
        4. Extract both archives so they produce the layout shown above, e.g.::

               unzip train.zip -d data/YouTubeVOS/
               unzip valid.zip -d data/YouTubeVOS/

        5. verify the path: ``data/YouTubeVOS/train/meta.json`` must exist.

    Args:
        base_dir: Root of the ``data/`` directory (absolute path).
    """
    target = os.path.join(base_dir, "YouTubeVOS")
    if os.path.isfile(os.path.join(target, "train", "meta.json")):
        print("YouTube-VOS already present at:", target)
        return

    print(
        "\n──────────────────────────────────────────────────────────────────\n"
        "  YouTube-VOS 2019 — MANUAL DOWNLOAD REQUIRED\n"
        "──────────────────────────────────────────────────────────────────\n"
        "  1. Register at https://youtube-vos.org/dataset/vos/\n"
        "  2. Download train.zip (~26 GB) and valid.zip (~6 GB).\n"
        "  3. Extract to:\n"
        f"     {target}/train/   (JPEGImages/ + Annotations/ + meta.json)\n"
        f"     {target}/valid/   (JPEGImages/ + Annotations/ + meta.json)\n"
        "──────────────────────────────────────────────────────────────────\n"
    )


def download_mose(base_dir: str) -> None:
    """Print manual download instructions for MOSE 2023.

    MOSE is distributed via the MOSE Challenge GitHub repository and
    associated Google Drive links.

    Expected layout after extraction::

        data/MOSE/
            train/
                JPEGImages/<video_id>/<frame>.jpg
                Annotations/<video_id>/<frame>.png  (integer object-ID masks)
            valid/
                JPEGImages/<video_id>/<frame>.jpg
                Annotations/<video_id>/<frame>.png

    Note:
        MOSE has a 41.5% object-disappearance rate.  Frames where an object
        is absent simply have no pixels of that object's ID in the annotation.

    Steps:
        1. Visit https://github.com/henghuiding/MOSE-api
        2. Follow the instructions for downloading the full dataset (~24 GB).
        3. Extract so that the layout shown above is produced, e.g.::

               unzip mose_train.zip -d data/MOSE/
               unzip mose_valid.zip -d data/MOSE/

        4. Verify: ``data/MOSE/train/JPEGImages/`` must be a non-empty directory.

    Args:
        base_dir: Root of the ``data/`` directory (absolute path).
    """
    target = os.path.join(base_dir, "MOSE")
    train_images = os.path.join(target, "train", "JPEGImages")
    if os.path.isdir(train_images) and len(os.listdir(train_images)) > 0:
        print("MOSE already present at:", target)
        return

    print(
        "\n──────────────────────────────────────────────────────────────────\n"
        "  MOSE 2023 — MANUAL DOWNLOAD REQUIRED\n"
        "──────────────────────────────────────────────────────────────────\n"
        "  1. Visit https://github.com/henghuiding/MOSE-api\n"
        "  2. Download mose_train.zip (~24 GB) and mose_valid.zip (~6 GB).\n"
        "  3. Extract to:\n"
        f"     {target}/train/   (JPEGImages/ + Annotations/)\n"
        f"     {target}/valid/   (JPEGImages/ + Annotations/)\n"
        "──────────────────────────────────────────────────────────────────\n"
    )


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

    # 3. YouTube-VOS 2019 (manual)
    download_youtubevos(data_dir)

    # 4. MOSE 2023 (manual)
    download_mose(data_dir)
    
    print("\nAll tasks finished.")
