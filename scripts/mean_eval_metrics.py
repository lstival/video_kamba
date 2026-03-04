import json
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate mean values from eval manifest JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file.")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    if not data:
        print("JSON file is empty.")
        return

    # Extract all metrics keys that are numbers
    first_item = data[0]
    metrics = [k for k, v in first_item.items() if isinstance(v, (int, float))]
    
    if not metrics:
        print("No numeric metrics found in the JSON file.")
        return

    means = {m: [] for m in metrics}
    for item in data:
        for m in metrics:
            if m in item:
                means[m].append(item[m])

    print(f"--- Mean values over {len(data)} items ---")
    for m in metrics:
        if means[m]:
            mean_val = np.mean(means[m])
            print(f"Mean {m}: {mean_val:.4f}")
        else:
            print(f"Mean {m}: N/A")

if __name__ == "__main__":
    main()
