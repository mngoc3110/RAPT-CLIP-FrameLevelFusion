import os, cv2, numpy as np

# Load the face crop
fo = cv2.imread('/tmp/face_orig.png')
# Take the center of the face (nose/mouth region, completely clean from any text!)
face_core = fo[40:140, 30:120] # (100, 90, 3)
face_core_small = cv2.resize(face_core, (32, 32))

print("Searching across all RAER video directories...")
min_diff = 1e9
best_match = None

for root, dirs, files in os.walk('RAER/images'):
    jpgs = [f for f in files if f.endswith('.jpg')]
    if jpgs:
        # Check frame 1 or 15
        cand = '15.jpg' if '15.jpg' in jpgs else jpgs[0]
        p = os.path.join(root, cand)
        img = cv2.imread(p)
        if img is not None:
            # RAER frames are typically 1080x1920 or 720x1280
            # Let's search if face_core matches any region or check color histogram
            # Fast template matching on downsampled image
            s_img = cv2.resize(img, (160, 90))
            # Just compare color and scene
            pass

