from collections import defaultdict

TXT_FILE = "daisee_train.txt"   # đổi sang daisee_val.txt / daisee_test.txt nếu muốn

# label -> {videos, frames}
stats = defaultdict(lambda: {"videos": 0, "frames": 0})

with open(TXT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        path, num_frames, label = line.strip().split()
        num_frames = int(num_frames)
        label = int(label)   # label trong txt là 1..4

        stats[label]["videos"] += 1
        stats[label]["frames"] += num_frames

print(f"\n📊 Frame statistics for {TXT_FILE}\n")
print(f"{'Label':<8}{'Videos':<10}{'Frames':<12}{'Avg frames/video'}")
print("-" * 45)

total_videos = 0
total_frames = 0

for label in sorted(stats.keys()):
    v = stats[label]["videos"]
    f = stats[label]["frames"]
    avg = f / v if v > 0 else 0
    print(f"{label:<8}{v:<10}{f:<12}{avg:.1f}")
    total_videos += v
    total_frames += f

print("-" * 45)
print(f"{'Total':<8}{total_videos:<10}{total_frames:<12}")