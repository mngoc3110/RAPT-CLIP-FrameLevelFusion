import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *

# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            # Convert to Grayscale (L) then back to RGB to keep 3 channels
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        return int(self._data[1])

    @property       # 标签
    def label(self):
        # Support multi-label (e.g. '1,4,12')
        lbl_str = str(self._data[2])
        if ',' in lbl_str:
            return [int(x.strip()) for x in lbl_str.split(',') if x.strip()]
        return int(lbl_str)
        
    @property
    def inline_bbox(self):
        # Returns [x1, y1, x2, y2] if available
        if len(self._data) >= 7:
            return [int(self._data[3]), int(self._data[4]), int(self._data[5]), int(self._data[6])]
        return None

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,bounding_box_face,bounding_box_body, crop_body=False, root_dir="", num_classes=8, dataset_name=""):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.crop_body = crop_body
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        
        # Debugging: Initialize for saving sample images
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body: # Only read body boxes if cropping is enabled
            self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        if self.bounding_box_body:
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            if mode == 'face':
                if self.dataset_name == 'EMOTIC':
                    # Fallback for EMOTIC: Center crop 60% to avoid copying full context
                    w, h = img.size
                    return img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
                # Default FALLBACK: Return original image instead of black image if no face detected
                return img
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        self.sample_list = []
        with open(self.list_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    # Case: [path, label] (e.g., EMOTIC images train.txt)
                    # We inject num_frames = 1
                    self.sample_list.append([parts[0], 1, parts[1]])
                elif len(parts) == 6 and ',' in parts[1]:
                    # Case: [path, label, x1, y1, x2, y2] (e.g., EMOTIC train_bbox.txt)
                    # We inject num_frames = 1 and keep the inline bbox coordinates
                    self.sample_list.append([parts[0], 1, parts[1], parts[2], parts[3], parts[4], parts[5]])
                elif len(parts) > 3:
                    # Path contains spaces, join all parts except the last two
                    path = ' '.join(parts[:-2])
                    num_frames = parts[-2]
                    label = parts[-1]
                    self.sample_list.append([path, num_frames, label])
                else:
                    # Case: [path, num_frames, label] (e.g., standard RAER)
                    if self.dataset_name == 'DAiSEE' and parts[0].startswith('DAiSEE_data/'):
                        parts[0] = parts[0].replace('DAiSEE_data/', '')
                    self.sample_list.append(parts)


    def _parse_list(self):
        # 
        # Data Form: [video_id, num_frames, class_idx]
        # 
        self.video_list = [VideoRecord([os.path.join(self.root_dir, item[0])] + item[1:]) for item in self.sample_list]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        # 
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        # 
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Check if record.path is a directory (frames) or a file (video/image)
        IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        is_image_file = record.path.lower().endswith(IMAGE_EXTS)
        
        if os.path.isdir(record.path):
            video_frames_path = glob.glob(os.path.join(record.path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
            is_video_file = False
            is_valid = num_real_frames > 0
        elif self.dataset_name == 'DAiSEE' and record.path.endswith('frames') and not os.path.exists(record.path):
            clip_id = os.path.basename(os.path.dirname(record.path))
            fallback_video = os.path.join(os.path.dirname(record.path), f"{clip_id}.avi")
            if not os.path.exists(fallback_video):
                fallback_video = os.path.join(os.path.dirname(record.path), f"{clip_id}.mp4")
            record._data[0] = fallback_video  # Update path
            is_video_file = True
            cap = cv2.VideoCapture(fallback_video)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_valid = cap.isOpened() and num_real_frames > 0
            if not is_valid:
                print(f"Warning: DAiSEE video fallback failed for {fallback_video}")
        elif is_image_file:
            # Static image (EMOTIC): read directly with PIL, treat as single frame
            is_video_file = False
            num_real_frames = 1
            is_valid = os.path.exists(record.path)
            if not is_valid:
                print(f"Warning: Image not found: {record.path}, returning zeros.")
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_valid = cap.isOpened() and num_real_frames > 0
            if not is_valid:
                print(f"Warning: Could not open video file {record.path}, returning zeros.")

        # For frame-directory datasets, is_valid is based on whether frames exist
        if not is_video_file and not is_image_file:
            is_valid = num_real_frames > 0

        if not is_valid:
            # print(f"Warning: No frames found for video {record.path}, returning zeros.")
            dummy_shape = (self.num_segments, 3, self.image_size, self.image_size)
            if self.dataset_name == "EMOTIC":
                multi_hot = torch.zeros(26)
                if isinstance(record.label, list):
                    for l in record.label:
                        multi_hot[l] = 1.0
                else:
                    multi_hot[record.label] = 1.0
                return torch.zeros(dummy_shape), torch.zeros(dummy_shape), torch.zeros(dummy_shape), multi_hot
            else:
                return torch.zeros(dummy_shape), torch.zeros(dummy_shape), torch.zeros(dummy_shape), record.label - 1

        # Clamp indices to be valid
        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = list()
        images_face = list()
        images_context = list()
        
        current_frame_idx = -1
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                box = None
                
                # 1. Read Image — smart sequential seek to avoid expensive cap.set() on every frame
                if is_image_file:
                    # Static image file (EMOTIC): just open with PIL directly
                    try:
                        img_pil = Image.open(record.path).convert('RGB')
                    except Exception as e:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                elif is_video_file:
                    gap = p - current_frame_idx
                    if gap == 1:
                        # Next consecutive frame: just read directly
                        ret, frame = cap.read()
                    elif 1 < gap <= 5:
                        # Small gap: use grab() to skip without decoding (much faster than seek)
                        for _ in range(gap - 1):
                            cap.grab()
                        ret, frame = cap.read()
                    else:
                        # Large gap or backward jump: forced seek
                        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                        ret, frame = cap.read()
                    current_frame_idx = p

                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                    else:
                        # Failed to read frame, use black image
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                else:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                    except:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))

                # 2. Key Lookup Strategy for Bounding Box
                # Normalize path separators to forward slash
                rel_path = record.path.replace('./', '').replace('\\', '/')
                # Remove root_dir prefix to get relative key
                if self.root_dir and rel_path.startswith(self.root_dir.replace('\\', '/')):
                    rel_key = rel_path[len(self.root_dir.rstrip('/')) + 1:]
                else:
                    rel_key = rel_path
                    # Also try stripping any common leading path
                    parts_full = rel_key.split('/')
                    for start in range(1, len(parts_full)):
                        candidate = '/'.join(parts_full[start:])
                        if candidate in self.boxs:
                            rel_key = candidate
                            break

                box = None
                if is_image_file:
                    # EMOTIC flat JSON: {"mscoco/images/xxx.jpg": [x1,y1,x2,y2]}
                    if rel_key in self.boxs:
                        raw = self.boxs[rel_key]
                        # Value can be [x1,y1,x2,y2] directly or a list of lists
                        if isinstance(raw[0], (int, float)):
                            box = raw
                        else:
                            box = raw[0]  # take first bbox
                    else:
                        # Try suffix-match fallback
                        rel_key_parts = rel_key.split('/')
                        for start in range(1, len(rel_key_parts)):
                            candidate = '/'.join(rel_key_parts[start:])
                            if candidate in self.boxs:
                                raw = self.boxs[candidate]
                                box = raw if isinstance(raw[0], (int, float)) else raw[0]
                                break
                else:
                    # RAER nested JSON: {"video_id": {"0.jpg": [x1,y1,x2,y2]}}
                    video_key_full = os.path.splitext(rel_key)[0]
                    frame_key = f"{p}.jpg"
                    matched_video_key = None
                    if video_key_full in self.boxs:
                        matched_video_key = video_key_full
                    if matched_video_key is None:
                        parts = video_key_full.split('/')
                        for idx in range(1, len(parts)):
                            sub_key = '/'.join(parts[idx:])
                            if sub_key in self.boxs:
                                matched_video_key = sub_key
                                break
                    if matched_video_key and frame_key in self.boxs[matched_video_key]:
                        box = self.boxs[matched_video_key][frame_key]

                # 4. Face Detection (Crop)
                # Reduce margin to 10 (Tight Crop) to zoom in on micro-expressions (eyebrows/eyes)
                # This helps separate Neutral vs Confusion
                # IMPORTANT FIX: Return full image if no box found (Fallback)
                img_pil_face = self._face_detect(img_pil, box, margin=10, mode='face')

                # 5. Body Crop (Optional)
                img_pil_body = img_pil # Default to full image
                if self.crop_body:
                    body_box = None
                    if is_image_file:
                        # EMOTIC flat JSON: {"mscoco/images/xxx.jpg": [x1,y1,x2,y2]}
                        if rel_key in self.body_boxes:
                            raw = self.body_boxes[rel_key]
                            body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                        else:
                            # suffix-match fallback
                            rel_key_parts = rel_key.split('/')
                            for start in range(1, len(rel_key_parts)):
                                candidate = '/'.join(rel_key_parts[start:])
                                if candidate in self.body_boxes:
                                    raw = self.body_boxes[candidate]
                                    body_box = raw if isinstance(raw[0], (int, float)) else raw[0]
                                    break
                    else:
                        # RAER nested JSON lookup (uses matched_video_key/frame_key from above)
                        if 'matched_video_key' in dir() and matched_video_key and matched_video_key in self.body_boxes:
                            if 'frame_key' in dir() and frame_key in self.body_boxes[matched_video_key]:
                                body_box = self.body_boxes[matched_video_key][frame_key]
                    
                    # Fallback to inline bbox from train_bbox.txt if JSON is missing
                    if body_box is None and record.inline_bbox is not None:
                        body_box = record.inline_bbox
                    
                    if body_box is not None:
                        left, upper, right, lower = body_box
                        # Ensure coordinates are within image bounds
                        left = max(0, left); upper = max(0, upper)
                        right = min(img_pil.width, right); lower = min(img_pil.height, lower)
                        if right > left and lower > upper:
                            img_pil_body = img_pil.crop((left, upper, right, lower))

                # 6. Resize and Stack
                # Resize Body
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                # Resize Face (face_detect returns cropped image, we need to resize it too?)
                # Wait, _face_detect returns cropped image but doesn't resize it to self.image_size?
                # The original code didn't resize face explicitly here, but transform does GroupResize.
                # However, for consistency, let's trust the transform pipeline which includes GroupResize.
                # But wait, `img_pil_face` might be tiny.
                
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                images_context.append(img_pil)
                
                if p < num_real_frames - 1:
                    p += 1
        
        if is_video_file and 'cap' in dir() and cap is not None:
            cap.release()

        # Transforms take a list of PIL images
        # images_face: List[PIL.Image]
        # images: List[PIL.Image]
        
        # Apply transforms
        # Note: self.transform usually expects a list and returns a Tensor of shape (T*C, H, W)
        # ToTorchFormatTensor returns (C*T, H, W) because Stack concatenates on axis 2.
        
        process_data = self.transform(images) # (C*T, H, W)
        process_data_face = self.transform(images_face) # (C*T, H, W)
        process_data_context = self.transform(images_context) # (C*T, H, W)

        # Reshape to (T, C, H, W)
        # Since Stack concatenated [img1, img2, ...] -> [R1,G1,B1, R2,G2,B2, ...]
        # Reshaping to (-1, 3, H, W) correctly separates frames.
        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        process_data_context = process_data_context.view(-1, 3, self.image_size, self.image_size)
        
        # Process target to multi-hot if it's a list (EMOTIC)
        if isinstance(record.label, list):
            # record.label is already a list of 0s and 1s representing the multi-hot vector
            target_tensor = torch.tensor(record.label, dtype=torch.float32)
            return process_data_face, process_data, process_data_context, target_tensor
        else:
            return process_data_face, process_data, process_data_context, record.label - 1

    def __len__(self):
        return len(self.video_list)


def train_data_loader(root_dir, list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    if dataset_name == "RAER" or dataset_name == "CAER":
         train_transforms = torchvision.transforms.Compose([
            # Apply ColorJitter from video_transform (works on list of images)
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), 
            GroupRandomGrayscale(p=0.2), # Custom transform for list of images
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    elif dataset_name == "EMOTIC":
         train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
         ])
    else:
         # Default transforms for other datasets like CK+
         train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              crop_body=crop_body,
                              num_classes=num_classes,
                              dataset_name=dataset_name
                              )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8, dataset_name=""):
    if dataset_name == "EMOTIC":
        test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                         Stack(),
                                                         ToTorchFormatTensor(),
                                                         GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    else:
        test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                         Stack(),
                                                         ToTorchFormatTensor()])
    
    test_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             crop_body=crop_body,
                             num_classes=num_classes,
                             dataset_name=dataset_name
                             )
    return test_data