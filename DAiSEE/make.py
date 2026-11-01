import csv
from pathlib import Path

ROOT_DATASET = Path("DataSet")   # Train/Validation/Test
ROOT_LABELS  = Path("Labels")
TARGET_DIM = "Engagement"        # Boredom | Engagement | Confusion | Frustration

# ưu tiên nếu có folder frames/; nếu không có thì lấy ảnh ngay trong video_dir
FRAMES_DIRNAME = "frames"

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")
IMG_EXTS = (".jpg", ".jpeg", ".png")


def map_level_to_label(level_0_to_3: int) -> int:
    # CSV 0..3 -> txt 1..4 (vì dataloader return label-1)
    if level_0_to_3 not in (0, 1, 2, 3):
        raise ValueError(f"Invalid level: {level_0_to_3} (expected 0..3)")
    return level_0_to_3 + 1


def load_labels_csv(csv_path: Path, target_dim: str) -> dict:
    """
    Return dict keyed by clip STEM (no extension): 
      "1100011002" -> level 0..3
    CSV ClipID thường là "1100011002.avi"
    """
    mapping = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if "ClipID" not in cols:
            raise RuntimeError(f"{csv_path} missing ClipID. Found: {cols}")
        if target_dim not in cols:
            raise RuntimeError(f"{csv_path} missing {target_dim}. Found: {cols}")

        for row in reader:
            clipid = (row.get("ClipID") or "").strip()
            if not clipid:
                continue
            stem = Path(clipid).stem  # bỏ .avi
            val = (row.get(target_dim) or "").strip()
            if val == "":
                continue
            try:
                level = int(float(val))
            except ValueError:
                continue
            mapping[stem] = level
    return mapping


def find_video_file(video_dir: Path) -> Path | None:
    vids = []
    for ext in VIDEO_EXTS:
        vids.extend(video_dir.glob(f"*{ext}"))
        vids.extend(video_dir.glob(f"*{ext.upper()}"))
    if not vids:
        return None
    # chọn file đầu tiên (thường chỉ có 1)
    return sorted(vids)[0]


def count_frames_in_dir(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(dir_path.glob(f"*{ext}"))
        imgs.extend(dir_path.glob(f"*{ext.upper()}"))
    return len(imgs)


def get_frames_folder(video_dir: Path) -> Path | None:
    """
    Ưu tiên frames/ nếu tồn tại và có ảnh.
    Nếu không, xem ảnh nằm trực tiếp trong video_dir.
    """
    frames_dir = video_dir / FRAMES_DIRNAME
    n1 = count_frames_in_dir(frames_dir)
    if n1 > 0:
        return frames_dir

    n2 = count_frames_in_dir(video_dir)
    if n2 > 0:
        return video_dir

    return None


def iter_video_folders(split_dir: Path):
    for subject in sorted(split_dir.iterdir()):
        if not subject.is_dir() or subject.name.startswith("."):
            continue
        for video_dir in sorted(subject.iterdir()):
            if not video_dir.is_dir() or video_dir.name.startswith("."):
                continue
            yield subject, video_dir


def make_split_txt(split_name: str, labels_map: dict, out_path: Path) -> None:
    split_dir = ROOT_DATASET / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    lines = []
    total = 0
    missing_label = 0
    missing_video = 0
    missing_frames = 0

    for _, video_dir in iter_video_folders(split_dir):
        total += 1

        video_file = find_video_file(video_dir)
        if video_file is None:
            missing_video += 1
            continue

        stem = video_file.stem  # bỏ .mp4/.avi...
        if stem not in labels_map:
            missing_label += 1
            continue

        frames_folder = get_frames_folder(video_dir)
        if frames_folder is None:
            missing_frames += 1
            continue

        n_frames = count_frames_in_dir(frames_folder)
        label_txt = map_level_to_label(labels_map[stem])

        # record.path cần là path tương đối từ ROOT_DATASET
        rel_path = frames_folder.relative_to(ROOT_DATASET).as_posix()
        # Fix path to be relative to project root
        full_rel_path = f"DAiSEE_data/DataSet/{rel_path}"
        lines.append(f"{full_rel_path} {n_frames} {label_txt}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[{split_name}] wrote: {out_path}")
    print(f"  total video folders scanned: {total}")
    print(f"  kept (have label + frames): {len(lines)}")
    print(f"  missing video file: {missing_video}")
    print(f"  missing label: {missing_label}")
    print(f"  missing frames: {missing_frames}")


def main():
    csv_files = {
        "Train": ROOT_LABELS / "TrainLabels.csv",
        "Validation": ROOT_LABELS / "ValidationLabels.csv",
        "Test": ROOT_LABELS / "TestLabels.csv",
    }

    label_maps = {}
    for split, csv_path in csv_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing label CSV: {csv_path}")
        label_maps[split] = load_labels_csv(csv_path, TARGET_DIM)

    make_split_txt("Train",      label_maps["Train"],      Path("daisee_train.txt"))
    make_split_txt("Validation", label_maps["Validation"], Path("daisee_val.txt"))
    make_split_txt("Test",       label_maps["Test"],       Path("daisee_test.txt"))

    print("\nDone.")
    print("NOTE: txt labels are 1..4, dataloader returns label-1 -> final classes 0..3.")


if __name__ == "__main__":
    main()
