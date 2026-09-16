import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import clip
from models.Text import prompt_ensemble_emotic_26, class_names_emotic
import os

# 1. Compute Data Co-occurrence Matrix
print("Computing data co-occurrence...")
train_annot = "/Users/macbook/Downloads/RAPT-CLIP/emotic_dataset/train_bbox.txt"
labels_list = []
with open(train_annot, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                labels = np.array([int(x) for x in parts[1].split(',')])
                if len(labels) == 26:
                    labels_list.append(labels)
            except:
                pass

labels_mat = np.stack(labels_list) # (N, 26)
# Co-occurrence matrix: C[i, j] = how many times label i and j appear together
co_occ = labels_mat.T @ labels_mat # (26, 26)

# Normalize by the frequency of the individual labels to get conditional probability P(j | i)
freq = np.diag(co_occ)
cond_prob = co_occ.astype(float) / freq[:, None]

# 2. Compute CLIP Text Similarity Matrix
print("Computing CLIP text similarity...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

model, _ = clip.load("ViT-B/16", device=device)
model.eval()

text_features_list = []
with torch.no_grad():
    for class_prompts in prompt_ensemble_emotic_26:
        # Tokenize the ensemble of prompts for this class
        text_tokens = clip.tokenize(class_prompts).to(device)
        # Get features
        features = model.encode_text(text_tokens) # (8, 512)
        features = features / features.norm(dim=-1, keepdim=True)
        # Average features for the ensemble
        mean_feature = features.mean(dim=0)
        mean_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
        text_features_list.append(mean_feature.cpu().numpy())

text_features_mat = np.stack(text_features_list) # (26, 512)
# Cosine similarity matrix
sim_mat = text_features_mat @ text_features_mat.T # (26, 26)

# 3. Plot and save
os.makedirs("scratch", exist_ok=True)

plt.figure(figsize=(12, 10))
sns.heatmap(cond_prob, xticklabels=class_names_emotic, yticklabels=class_names_emotic, cmap="YlGnBu")
plt.title("Label Co-occurrence in EMOTIC (Conditional Probability P(x-axis | y-axis))")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("scratch/emotic_cooccurrence.png")
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(sim_mat, xticklabels=class_names_emotic, yticklabels=class_names_emotic, cmap="YlOrRd")
plt.title("CLIP Text Embedding Cosine Similarity")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("scratch/clip_similarity.png")
plt.close()

print("Done. Saved images to scratch/")
