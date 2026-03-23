import os
import json
import torch
from data.multi_object_vos import MultiObjectVOSDataset

def test_yt_unseen_split():
    root_dir = "data/YouTubeVOS"
    
    print("Checking YouTube-VOS Dataset Initialization...")
    try:
        # Initialize validation dataset
        dataset = MultiObjectVOSDataset(
            root_dir=root_dir,
            dataset_type="youtubevos",
            split="valid",
            seq_len=2,
            augment=False
        )
        
        print(f"Total videos in validation set: {len(dataset._video_ids)}")
        
        total_seen = 0
        total_unseen = 0
        videos_with_unseen = 0
        
        # Check a few videos
        sample_vids = dataset._video_ids[:10]
        for vid in sample_vids:
            meta = dataset._meta.get(vid, {})
            seen = meta.get("seen_obj_ids", [])
            unseen = meta.get("unseen_obj_ids", [])
            
            print(f"Video {vid}: Seen={seen}, Unseen={unseen}")
            
            total_seen += len(seen)
            total_unseen += len(unseen)
            if len(unseen) > 0:
                videos_with_unseen += 1
                
        print("\nSummary (First 10 videos):")
        print(f"Total seen objects: {total_seen}")
        print(f"Total unseen objects: {total_unseen}")
        print(f"Videos with at least one unseen object: {videos_with_unseen}/10")
        
        if total_unseen > 0:
            print("\nSUCCESS: Unseen objects correctly identified via category comparison!")
        else:
            print("\nFAILURE: Still no unseen objects found. Check category comparison logic.")
            
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yt_unseen_split()
