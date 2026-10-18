# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from models.clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader, EmoticNpyDataset
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    # Pass drop_path_rate to clip.load so it can be used during model construction
    CLIP_model, _ = clip.load(args.clip_path, device='cpu', drop_path=args.drop_path_rate)

    print("\nInput Text Prompts:")
    # Handle the case where input_text is a list of lists for prompt ensembling
    if any(isinstance(i, list) for i in input_text):
        for class_prompts in input_text:
            print(f"- Class: {class_prompts}")
    else:
        for text in input_text:
            print(text)


    print("\nInstantiating GenerateModel...")
    if hasattr(args, 'use_v2') and args.use_v2:
        from models.Generate_Model_v2 import GenerateModel_v2
        model = GenerateModel_v2(input_text=input_text, clip_model=CLIP_model, args=args)
    elif hasattr(args, 'ablation_no_text') and args.ablation_no_text:
        from models.Generate_Model_NoText import GenerateModel_NoText
        model = GenerateModel_NoText(clip_model=CLIP_model, args=args)
    else:
        model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0 and not args.freeze_image_encoder:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    trainable_params_keywords = ["temporal_net", "temporal_net_body", "project_fc", "face_adapter", "classifier", "cross_attn", "cmaf", "gate_fc"]
    tune_prompt = str(getattr(args, 'load_and_tune_prompt_learner', True)).lower() in ['true', '1'] and getattr(args, 'lr_prompt_learner', 1e-4) > 0
    if tune_prompt:
        trainable_params_keywords.append("prompt_learner")
    
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    dataset_name = args.dataset.strip()
    print(f"DEBUG: get_class_info processing dataset '{dataset_name}'")

    if dataset_name == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    elif dataset_name == "CK+":
        class_names = class_names_ckplus
        class_names_with_context = class_names_with_context_ckplus
        class_descriptor = class_descriptor_ckplus
        ensemble_prompts = prompt_ensemble_ckplus
    elif dataset_name == "SFER":
        class_names = class_names_sfer
        class_names_with_context = class_names_with_context_sfer
        class_descriptor = class_descriptor_sfer
        ensemble_prompts = prompt_ensemble_sfer
    elif dataset_name == "DAiSEE":
        if getattr(args, 'daisee_merge_low', False):
            print("=> Using DAiSEE 3-Class Merged Setup (Low, High, Very High)")
            class_names = class_names_daisee_3class
            class_names_with_context = class_names_with_context_daisee_3class
            class_descriptor = class_descriptor_daisee_3class
            ensemble_prompts = prompt_ensemble_daisee_3class
        else:
            class_names = class_names_daisee
            class_names_with_context = class_names_with_context_daisee
            class_descriptor = class_descriptor_daisee
            ensemble_prompts = prompt_ensemble_daisee
    elif dataset_name in ["CAER", "CAER_S", "CAER-S"]:
        class_names = class_names_caer
        class_names_with_context = class_names_with_context_caer
        class_descriptor = class_descriptor_caer
        ensemble_prompts = prompt_ensemble_caer
    elif dataset_name == "EMOTIC":
        class_names = class_names_emotic
        class_names_with_context = class_names_with_context_emotic
        class_descriptor = class_descriptor_emotic_26
        ensemble_prompts = prompt_ensemble_emotic_26
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    elif args.text_type == "au_guided_prompts":
        if dataset_name == "RAER":
            from models.Text import class_descriptor_5_au
            input_text = class_descriptor_5_au
        else:
            raise ValueError("AU-guided prompts are only defined for RAER dataset currently.")
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    # Debug print
    print(f"DEBUG: args.dataset = '{args.dataset}'")

    if hasattr(args, 'use_npy') and args.use_npy and args.dataset == 'EMOTIC':
        import torchvision
        from dataloader.video_transform import GroupResize, GroupRandomHorizontalFlip, Stack, ToTorchFormatTensor, GroupNormalize, ColorJitter, RandomRotation, GroupRandomErasing
        from dataloader.video_dataloader import GroupRandomGrayscale
        print(f"Bypassing standard video dataloader. Using EmoticNpyDataset from {args.npy_dir}")
        train_transforms = torchvision.transforms.Compose([
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            GroupRandomGrayscale(p=0.1),
            GroupResize(args.image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            GroupRandomErasing(p=0.4, scale=(0.02, 0.2))
        ])
        test_transform = torchvision.transforms.Compose([
            GroupResize(args.image_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        train_data = EmoticNpyDataset(npy_dir=args.npy_dir, mode='train', transform=train_transforms, num_classes=num_classes)
        val_data = EmoticNpyDataset(npy_dir=args.npy_dir, mode='val', transform=test_transform, num_classes=num_classes)
        test_data = EmoticNpyDataset(npy_dir=args.npy_dir, mode='test', transform=test_transform, num_classes=num_classes)
        print(f"NPY Dataset Sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader

    def normalize_colab(p):
        if not p or not isinstance(p, str):
            return p
        if os.path.exists(p):
            return p
        if p.startswith('content/'):
            alt = '/' + p
            if os.path.exists(alt):
                return alt
        elif p.startswith('drive/'):
            alt = '/content/' + p
            if os.path.exists(alt):
                return alt
        return p

    train_annotation_file_path = normalize_colab(getattr(args, 'train_annotation', None))
    val_annotation_file_path = normalize_colab(getattr(args, 'val_annotation', None))
    test_annotation_file_path = normalize_colab(getattr(args, 'test_annotation', None))
    args.train_annotation = train_annotation_file_path
    args.val_annotation = val_annotation_file_path
    args.test_annotation = test_annotation_file_path
    if hasattr(args, 'root_dir') and args.root_dir:
        args.root_dir = normalize_colab(args.root_dir)

    # Auto-recovery if annotation file does not exist (e.g. Kaggle path casing or custom slug)
    if train_annotation_file_path and not os.path.exists(train_annotation_file_path):
        target_name = os.path.basename(train_annotation_file_path).lower()
        print(f"Warning: Annotation file {train_annotation_file_path} not found. Searching for {target_name}...")
        found_path = None
        for base in ['/kaggle/input', '/content/drive/MyDrive', '/content', '.', '..']:
            if os.path.exists(base):
                for root, dirs, files in os.walk(base):
                    for f in files:
                        if f.lower() == target_name:
                            found_path = os.path.join(root, f)
                            break
                    if found_path:
                        break
            if found_path:
                break
        
        if found_path:
            detected_dir = os.path.dirname(found_path)
            print(f"=> [Auto-Recovered] Found matching annotation at: {found_path}")
            train_annotation_file_path = found_path
            args.train_annotation = found_path
            
            if val_annotation_file_path:
                val_target = os.path.basename(val_annotation_file_path).lower()
                for f in os.listdir(detected_dir):
                    if f.lower() == val_target:
                        val_annotation_file_path = os.path.join(detected_dir, f)
                        args.val_annotation = val_annotation_file_path
                        break
            if test_annotation_file_path:
                test_target = os.path.basename(test_annotation_file_path).lower()
                for f in os.listdir(detected_dir):
                    if f.lower() == test_target:
                        test_annotation_file_path = os.path.join(detected_dir, f)
                        args.test_annotation = test_annotation_file_path
                        break
            
            if not os.path.exists(args.root_dir):
                args.root_dir = detected_dir
                print(f"=> [Auto-Recovered] Updated root_dir to: {args.root_dir}")

    # Auto-resolve bounding box paths if specified relative or not directly found
    for bbox_attr in ['bounding_box_face', 'bounding_box_body']:
        bbox_path = normalize_colab(getattr(args, bbox_attr, None))
        setattr(args, bbox_attr, bbox_path)
        if bbox_path and not os.path.exists(bbox_path):
            candidates = [
                os.path.join(args.root_dir, bbox_path),
                os.path.join(args.root_dir, os.path.basename(bbox_path)),
                os.path.join(os.path.dirname(train_annotation_file_path or ''), os.path.basename(bbox_path)),
                os.path.join(args.root_dir, 'RAER', bbox_path),
                os.path.join(args.root_dir, 'bounding_box', os.path.basename(bbox_path)),
                os.path.join(args.root_dir, 'RAER', 'bounding_box', os.path.basename(bbox_path)),
                os.path.join('.', bbox_path),
                os.path.join('.', 'RAER', bbox_path),
                os.path.join('/kaggle/input/datasets/bearmn/raer-video-emotion-dataset', bbox_path),
                os.path.join('/kaggle/input/datasets/bearmn/raer-video-emotion-dataset/RAER', bbox_path),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    setattr(args, bbox_attr, cand)
                    print(f"=> [Auto-Recovered] Found {bbox_attr} at: {cand}")
                    break

    mask_context_face = getattr(args, 'mask_context_face', False)
    daisee_merge_low = getattr(args, 'daisee_merge_low', False)
    if mask_context_face:
        print("=> Option enabled: Masking face region in Context Stream to prevent leakage.")
    if daisee_merge_low:
        print("=> Option enabled: Merging DAiSEE Low engagement classes.")

    print(f"Loading train data (Standard) for {args.dataset}...")
    train_data = train_data_loader(
        root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size, dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes,
        mask_context_face=mask_context_face,
        daisee_merge_low=daisee_merge_low
    )
    
    print(f"Loading validation data (Standard) for {args.dataset}...")
    val_data = test_data_loader(
        root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes,
        dataset_name=args.dataset,
        mask_context_face=mask_context_face,
        daisee_merge_low=daisee_merge_low
    )

    print(f"Loading test data (Standard) for {args.dataset}...")
    test_data = test_data_loader(
        root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        num_classes=num_classes,
        dataset_name=args.dataset,
        mask_context_face=mask_context_face,
        daisee_merge_low=daisee_merge_low
    )

    print(f"Total number of training images: {len(train_data)}")
    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        print("=> Using WeightedRandomSampler.")
        daisee_merge_low = getattr(args, 'daisee_merge_low', False) and args.dataset == 'DAiSEE'
        class_counts = get_class_counts(train_annotation_file_path, daisee_merge_low=daisee_merge_low)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        
        # Create a weight for each sample
        sample_weights = []
        offset = getattr(train_data, 'label_offset', 0 if args.dataset in ['CAER', 'CAER-S', 'CAER_S'] else 1)
        with open(train_annotation_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                lbl_str = parts[-1] if not (len(parts) == 7 and ',' in parts[2]) else parts[2]
                label = int(lbl_str) - offset
                if daisee_merge_low:
                    label = 0 if label <= 1 else (1 if label == 2 else 2)
                if 0 <= label < len(class_weights):
                    sample_weights.append(class_weights[label])
                else:
                    sample_weights.append(torch.tensor(1.0))
        
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False # Sampler and shuffle are mutually exclusive

    extra_loader_kwargs = {}
    if args.workers > 0:
        extra_loader_kwargs['persistent_workers'] = True
        extra_loader_kwargs['prefetch_factor'] = 2

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True, **extra_loader_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, **extra_loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, **extra_loader_kwargs
    )
    
    return train_loader, val_loader, test_loader