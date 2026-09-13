"""
extract_vad.py — Run once to generate train/val/test_vad.json from CVPR17_Annotations.mat
Usage: python tools/extract_vad.py --mat emotic_dataset/CVPR17_Annotations.mat --out emotic_dataset/
"""
import scipy.io
import json
import os
import argparse
import numpy as np


def extract_vad(mat_path, out_dir):
    print(f"Loading {mat_path} ...")
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)

    for split in ['train', 'val', 'test']:
        data = mat[split]
        vad_dict = {}  # key: "folder/filename" -> [V, A, D] normalized [0,1]

        for item in data:
            filename = item['filename']   # e.g. "007eear5kx5qhbzewz.jpg"
            folder   = item['folder']     # e.g. "emodb_small/images"
            key = f"{folder}/{filename}"

            persons = item['person']
            if not isinstance(persons, list):
                persons = [persons]

            vad_values = []
            for person in persons:
                if not isinstance(person, dict):
                    continue
                # Use combined_continuous (consensus) for val/test if available
                cont_key = 'combined_continuous' if 'combined_continuous' in person else 'annotations_continuous'
                cont = person.get(cont_key, None)
                if cont is None:
                    continue
                if isinstance(cont, dict):
                    v = float(cont.get('valence', 5))
                    a = float(cont.get('arousal', 5))
                    d = float(cont.get('dominance', 5))
                elif hasattr(cont, '__len__') and len(cont) == 3:
                    v, a, d = float(cont[0]), float(cont[1]), float(cont[2])
                else:
                    continue
                # Normalize from [1,10] to [0,1]
                vad_values.append([(v - 1) / 9.0, (a - 1) / 9.0, (d - 1) / 9.0])

            if vad_values:
                # Average across all annotators/persons in image
                avg = np.mean(vad_values, axis=0).tolist()
                vad_dict[key] = avg

        out_path = os.path.join(out_dir, f"{split}_vad.json")
        with open(out_path, 'w') as f:
            json.dump(vad_dict, f)
        print(f"  [{split}] {len(vad_dict)} entries → {out_path}")
        sample_key = next(iter(vad_dict))
        print(f"    sample: {sample_key!r} → {vad_dict[sample_key]}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', default='emotic_dataset/CVPR17_Annotations.mat')
    parser.add_argument('--out', default='emotic_dataset/')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    extract_vad(args.mat, args.out)
