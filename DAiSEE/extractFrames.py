import subprocess
from pathlib import Path

ROOT = Path("DataSet")
SPLITS = ["Train", "Validation", "Test"]
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")

def is_hidden(p: Path) -> bool:
    return p.name.startswith(".")

def find_video(video_dir: Path) -> Path | None:
    for ext in VIDEO_EXTS:
        vids = list(video_dir.glob(f"*{ext}")) + list(video_dir.glob(f"*{ext.upper()}"))
        if vids:
            return sorted(vids)[0]
    return None

def extract(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / f"{video_path.stem}_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        out_pattern
    ]
    subprocess.check_call(cmd)

def main():
    if not ROOT.exists():
        raise FileNotFoundError("❌ Không thấy DataSet/. Hãy chạy script ở đúng thư mục.")

    # ===== PASS 1: đếm tổng số video cần extract =====
    jobs = []
    for split in SPLITS:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue

        for subject in split_dir.iterdir():
            if not subject.is_dir() or is_hidden(subject):
                continue

            for video_dir in subject.iterdir():
                if not video_dir.is_dir() or is_hidden(video_dir):
                    continue

                video = find_video(video_dir)
                if video is None:
                    continue

                frames_dir = video_dir / "frames"
                if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                    continue  # đã chạy rồi → skip

                jobs.append((video, frames_dir))

    total = len(jobs)
    if total == 0:
        print("✅ Không có video nào cần extract (tất cả đã xong).")
        return

    print(f"🔢 Total videos to extract: {total}")

    # ===== PASS 2: chạy extract + progress =====
    done = 0
    for video, frames_dir in jobs:
        done += 1
        percent = (done / total) * 100

        print(f"[{done:4d}/{total}] ({percent:6.2f}%) RUN  → {video}")
        try:
            extract(video, frames_dir)
            print(f"[{done:4d}/{total}] ({percent:6.2f}%) OK   → {frames_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed extracting {video}: {e}")

    print("\n✅ Frame extraction DONE.")

if __name__ == "__main__":
    main()