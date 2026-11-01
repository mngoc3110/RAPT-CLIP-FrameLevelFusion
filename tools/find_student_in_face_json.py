import os, cv2, json, numpy as np

fo = cv2.imread('/tmp/face_orig.png')
# Take clean center of face (eyes + nose)
# In fo: size (181, 151, 3)
# To avoid the cut-off text 'Original' at y=0..15 and 'Grad-CAM' at y=165..180:
# Row 40 to 140, col 20 to 130 is 100% CLEAN FACE!
face_center = fo[45:135, 25:125]
cv2.imwrite('/tmp/clean_face_probe.png', face_center)
h_p, w_p = face_center.shape[:2]

with open('RAER/bounding_box/face.json') as f:
    face_boxes = json.load(f)

print(f"Loaded {len(face_boxes)} entries in face.json. Searching...")

best_k = None
best_score = -1
best_frame = None
best_box = None

for k, frames_dict in face_boxes.items():
    if not os.path.exists(k):
        continue
    # Check frame '1.jpg' or '15.jpg'
    for fn in ['1.jpg', '15.jpg']:
        box = frames_dict.get(fn)
        if box:
            fp = os.path.join(k, fn)
            img = cv2.imread(fp)
            if img is not None:
                fx1, fy1, fx2, fy2 = [int(v) for v in box]
                face = img[fy1:fy2, fx1:fx2]
                if face.shape[0] > 30 and face.shape[1] > 30:
                    face_res = cv2.resize(face, (w_p, h_p))
                    res = cv2.matchTemplate(face_res, face_center, cv2.TM_CCOEFF_NORMED)
                    score = res[0, 0]
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_frame = fn
                        best_box = box
                        if score > 0.85:
                            print(f"🔥 Found exact match! {k}/{fn} with score = {score:.4f}")
                            break
    if best_score > 0.85:
        break

print(f"Best match: {best_k}/{best_frame} with score = {best_score:.4f}")
if best_box:
    print(f"Box: {best_box}")
