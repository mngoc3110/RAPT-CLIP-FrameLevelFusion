from torch import nn
from models.Temporal_Model import *
from models.Prompt_Learner import *
from models.Text import class_descriptor_5_only_face
from models.Adapter import Adapter
from models.CrossModalAttentionFusion import CrossModalAttentionFusion
from models.clip import clip
import copy
import itertools

class GenerateModel(nn.Module):
    """
    RAPT-CLIP Unified Multimodal Emotion & Engagement Architecture.
    
    Dataset Mapping & Workflow Guide:
    ---------------------------------
    1. RAER (Video Classroom Emotion - 5 classes):
       - Stream: Dual-stream (Face + Body) -> use_context=False
       - Fusion: CMAF (CrossModalAttentionFusion) -> Unified Temporal Transformer (AttnPool)
       - Loss: LDAM Loss + MI/DC Loss
       - Script: train_sh/raer_cmaf.sh

    2. DAiSEE (Video Engagement - 4 dimensions / multi-class):
       - Stream: Dual-stream (Face + Body / Webcam) -> use_context=False
       - Fusion: CMAF -> Unified Temporal Transformer (AttnPool)
       - Loss: LDAM / CrossEntropy + MI Loss
       - Script: train_sh/daisee_run.sh

    3. CAER / CAER-S (Context-Aware Emotion - 7 classes):
       - Stream: Triple-stream (Face + Body + Context) -> use_context=True
       - Fusion: Tridirectional CMAF -> Unified Temporal Net (T=1 for CAER-S)
       - Loss: CrossEntropy / LDAM
       - Script: train_sh/caer_s_run.sh

    4. EMOTIC (Multi-label Context Emotion - 26 classes):
       - Stream: Triple-stream (Face + Body + Context) -> use_context=True
       - Fusion: CMAF / Q2L (Query2Label) / CMAF-Decoupled Router
       - Loss: Asymmetric Loss (ASL) / Dynamic ASL
       - Script: train_sh/emotic_cmaf_npy.sh
    """
    def __init__(self, input_text, clip_model, args):
        super().__init__()
        self.args = args
        
        self.is_ensemble = any(isinstance(i, list) for i in input_text)
        
        if self.is_ensemble:
            self.num_classes = len(input_text)
            self.num_prompts_per_class = len(input_text[0])
            self.input_text = list(itertools.chain.from_iterable(input_text))
            print(f"=> Using Prompt Ensembling with {self.num_prompts_per_class} prompts per class.")
        else:
            self.input_text = input_text

        self.prompt_learner = PromptLearner(self.input_text, clip_model, args)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual

        # Freeze Image Encoder if requested
        if hasattr(args, 'freeze_image_encoder') and args.freeze_image_encoder:
            print("=> Freezing Image Encoder")
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # For EAA
        self.face_adapter = Adapter(c_in=512, reduction=4)

        # For MI Loss (Supported datasets: RAER, CAER/CAER-S, DAiSEE, EMOTIC)
        if args.dataset == "RAER":
            hand_crafted_prompts = class_descriptor_5_only_face
        elif args.dataset == "DAiSEE":
            if getattr(args, 'daisee_merge_low', False):
                from models.Text import class_descriptor_daisee_3class
                hand_crafted_prompts = class_descriptor_daisee_3class
            else:
                from models.Text import class_descriptor_daisee
                hand_crafted_prompts = class_descriptor_daisee
        elif args.dataset == "EMOTIC":
            from models.Text import class_descriptor_emotic_26
            hand_crafted_prompts = class_descriptor_emotic_26
        elif args.dataset in ["CAER", "CAER-S", "CAER_S"]:
            from models.Text import class_descriptor_caer
            hand_crafted_prompts = class_descriptor_caer
        else:
            from models.Text import class_descriptor_7_only_face
            hand_crafted_prompts = class_descriptor_7_only_face
            
        self.tokenized_hand_crafted_prompts = torch.cat([clip.tokenize(p) for p in hand_crafted_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_hand_crafted_prompts.to(clip_model.token_embedding.weight.device)).type(self.dtype)
        self.register_buffer("hand_crafted_prompt_embeddings", embedding)


        # Context Stream Configuration
        self.use_context = getattr(args, 'use_context', False)
        print(f"=> Using Context Stream: {self.use_context}")

        self.fusion_type = getattr(args, 'fusion_type', 'cmaf')
        print(f"=> Using Fusion Type: {self.fusion_type}")

        if self.fusion_type == 'q2l':
            self.q2l_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
            self.q2l_norm = nn.LayerNorm(512)
            in_dim = 512 # Not actually used in Q2L but keep to avoid undefined errors
        elif self.fusion_type == 'cmaf_decoupled':
            # Decoupled CMAF with Learnable Modality Router.
            # num_modalities = 3 (face, body, context) or 2 (face, body)
            # The router learns per-class soft attention weights: which modality
            # to trust most for each emotion class, simultaneously.
            # Gradient flows to ALL modalities (unlike hard max which blocks 2/3).
            num_mod = 3 if self.use_context else 2
            num_classes = getattr(args, 'num_classes', 26)
            # Router: takes stack of (B, num_mod) logits per class → weights
            # Implemented as a small class-aware linear layer over modality dim
            self.decoupled_vote_router = nn.Sequential(
                nn.Linear(num_mod, num_mod * 4),
                nn.GELU(),
                nn.Linear(num_mod * 4, num_mod),
            )  # Input: (B*C, num_mod) → Output: (B*C, num_mod) softmax weights
            # Initialize near-uniform to start
            nn.init.zeros_(self.decoupled_vote_router[0].weight)
            nn.init.zeros_(self.decoupled_vote_router[2].weight)
            nn.init.zeros_(self.decoupled_vote_router[0].bias)
            nn.init.zeros_(self.decoupled_vote_router[2].bias)
            in_dim = 512
        else:
            in_dim = 1536 if self.use_context else 1024

        # Determine whether temporal modeling is needed:
        # Bypassed for static image datasets (e.g. EMOTIC, CAER-S) or when --no-temporal is set or temporal_layers == 0.
        # Explicit --use-temporal forces temporal modeling (required for legacy checkpoints trained with unified_temporal_net).
        if getattr(args, 'use_temporal', False):
            self.use_temporal = True
        else:
            self.use_temporal = not (getattr(args, 'dataset', '') in ['EMOTIC', 'CAER-S', 'CAER_S'] or 
                                     getattr(args, 'no_temporal', False) or 
                                     getattr(args, 'temporal_layers', 1) == 0)

        # Fusion Stage: 'frame' (CMAF before Temporal, Default SOTA) vs 'video' (Temporal before CMAF)
        self.fusion_stage = getattr(args, 'fusion_stage', 'frame')
        print(f"=> Fusion Stage: {self.fusion_stage}")

        temporal_pooling = getattr(args, 'temporal_pooling', 'attn')
        print(f"=> Temporal Pooling: {temporal_pooling}")

        if not self.use_temporal:
            print("=> Static Image Dataset (EMOTIC / no-temporal): Bypassing Temporal Transformer.")
            self.temporal_net_face = None
            self.temporal_net_body = None
            self.temporal_net_context = None
            self.unified_temporal_net = None
        elif self.fusion_stage == 'video':
            # Video-level fusion: Separate Temporal Transformers operate on individual streams first (512-d)
            self.temporal_net_face = Temporal_Transformer_AttnPool(
                num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                pooling=temporal_pooling
            )
            self.temporal_net_body = Temporal_Transformer_AttnPool(
                num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                pooling=temporal_pooling
            )
            if self.use_context:
                self.temporal_net_context = Temporal_Transformer_AttnPool(
                    num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                    pooling=temporal_pooling
                )
        elif self.fusion_stage == 'parallel':
            # [Ablation: parallel] CMAF per-frame (decoupled, no concat) → 3 stream-specific Temporal → video-level concat
            self.temporal_net_face = Temporal_Transformer_AttnPool(
                num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                pooling=temporal_pooling
            )
            self.temporal_net_body = Temporal_Transformer_AttnPool(
                num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                pooling=temporal_pooling
            )
            if self.use_context:
                self.temporal_net_context = Temporal_Transformer_AttnPool(
                    num_patches=16, input_dim=512, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                    pooling=temporal_pooling
                )
        else:
            # Frame-level fusion: Unified Temporal Transformer operates on fused multi-modal frames (1536-d or 1024-d)
            self.unified_temporal_net = Temporal_Transformer_AttnPool(
                num_patches=16, input_dim=in_dim, depth=args.temporal_layers, heads=8, mlp_dim=1024, dim_head=64,
                pooling=temporal_pooling
            )

        # Store clip_model_ as a plain Python attribute (NOT an nn.Module submodule).
        object.__setattr__(self, 'clip_model_', clip_model)
        
        self.project_fc = nn.Linear(in_dim, 512)

        # Fusion Selection: gfi (Gated Feature Integration) or cmaf (Cross-Modal Attention Fusion)
        if self.fusion_type == 'gfi':
            self.gate_fc = nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.ReLU(),
                nn.Linear(in_dim // 4, in_dim),
                nn.Sigmoid()
            )
        else:
            self.cmaf = CrossModalAttentionFusion(dim=512, num_heads=4, dropout=0.1, use_context=self.use_context)

        
    def forward(self, image_face, image_body, image_context=None):
        ################# Text Part ###################
        # Compute text features first since Q2L needs them as queries
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        
        with torch.cuda.amp.autocast(enabled=False):
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features = text_features.float()
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)

        hand_crafted_prompts = self.hand_crafted_prompt_embeddings
        tokenized_hand_crafted_prompts = self.tokenized_hand_crafted_prompts.to(hand_crafted_prompts.device)
        
        with torch.cuda.amp.autocast(enabled=False):
            hand_crafted_text_features = self.text_encoder(hand_crafted_prompts, tokenized_hand_crafted_prompts)
            hand_crafted_text_features = hand_crafted_text_features.float()
            hand_crafted_text_features = hand_crafted_text_features / (hand_crafted_text_features.norm(dim=-1, keepdim=True) + 1e-6)

        ################# Visual Part #################
        n, t, c, h, w = image_face.shape

        # Face Part
        image_face_reshaped = image_face.contiguous().view(-1, c, h, w)
        if self.fusion_type == 'q2l':
            image_face_features = self.image_encoder.forward_features(image_face_reshaped.type(self.dtype)) # (B, 197, 512)
            image_face_features = self.face_adapter(image_face_features) # Apply EAA to patches
        else:
            image_face_features = self.image_encoder(image_face_reshaped.type(self.dtype))
            image_face_features = self.face_adapter(image_face_features) # Apply EAA
        
        # Body Part
        image_body_reshaped = image_body.contiguous().view(-1, c, h, w)
        if self.fusion_type == 'q2l':
            image_body_features = self.image_encoder.forward_features(image_body_reshaped.type(self.dtype))
        else:
            image_body_features = self.image_encoder(image_body_reshaped.type(self.dtype))

        # Context Part
        if self.use_context:
            assert image_context is not None, "image_context must be provided when use_context=True"
            image_context_reshaped = image_context.contiguous().view(-1, c, h, w)
            if self.fusion_type == 'q2l':
                image_context_features = self.image_encoder.forward_features(image_context_reshaped.type(self.dtype))
            else:
                image_context_features = self.image_encoder(image_context_reshaped.type(self.dtype))

        # Frame-Level Fusion (CMAF, GFI, or Q2L)
        if self.fusion_type == 'q2l':
            # 1. Combine patches (exclude CLS token if preferred, but we can keep it)
            # image_face_features: (B, 197, 512)
            if self.use_context:
                all_patches = torch.cat([image_face_features, image_body_features, image_context_features], dim=1)
            else:
                all_patches = torch.cat([image_face_features, image_body_features], dim=1)
            
            # 2. Setup queries from text_features
            # text_features: (C, 512) -> queries: (B, C, 512)
            if self.is_ensemble:
                q2l_text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1).mean(dim=1)
                q2l_text_features = q2l_text_features / (q2l_text_features.norm(dim=-1, keepdim=True) + 1e-6)
            else:
                q2l_text_features = text_features
            
            queries = q2l_text_features.unsqueeze(0).expand(n, -1, -1)
            
            # 3. Cross Attention
            class_features, _ = self.q2l_attention(query=queries, key=all_patches, value=all_patches)
            class_features = self.q2l_norm(class_features + queries) # Residual + Norm
            
            # 4. Normalize the class-specific features (Crucial for Cosine Similarity to prevent exploding logits)
            class_features = class_features / (class_features.norm(dim=-1, keepdim=True) + 1e-6)
            
            # We bypass the TemporalNet for now since EMOTIC is image-based (t=1).
            # Output Logits = Cosine Similarity scaled by temperature
            output = (class_features * queries).sum(dim=-1) / self.args.temperature
            
            # Pack a dummy video_features to not break MoCo or other references
            video_features = class_features.mean(dim=1)
            self.last_video_features = video_features
            
            # Return immediately for Q2L, skipping the old classification block
            moco_logits = None
            return output, text_features, hand_crafted_text_features, moco_logits

        elif self.fusion_type == 'gfi':
            features_to_concat = [image_face_features, image_body_features]
            if self.use_context:
                features_to_concat.append(image_context_features)
            fused_frame_features = torch.cat(features_to_concat, dim=-1)
            gate = self.gate_fc(fused_frame_features)
            fused_frame_features = fused_frame_features * gate
            
            if self.use_temporal:
                fused_frame_features = fused_frame_features.contiguous().view(n, t, -1)
                video_features = self.unified_temporal_net(fused_frame_features)
            else:
                video_features = fused_frame_features.contiguous().view(n, -1)
            video_features = self.project_fc(video_features)
            video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
            
        elif self.fusion_type == 'cmaf_decoupled':
            if self.use_context:
                face_out, body_out, ctx_out = self.cmaf(image_face_features, image_body_features, image_context_features, return_decoupled=True)
                outs = [face_out, body_out, ctx_out]
            else:
                face_out, body_out = self.cmaf(image_face_features, image_body_features, return_decoupled=True)
                outs = [face_out, body_out]
            
            video_features_list = []
            for out in outs:
                if self.use_temporal:
                    out = out.contiguous().view(n, t, -1)
                    vid_feat = self.unified_temporal_net(out)
                else:
                    vid_feat = out.contiguous().view(n, -1)
                vid_feat = self.project_fc(vid_feat)
                vid_feat = vid_feat / (vid_feat.norm(dim=-1, keepdim=True) + 1e-6)
                video_features_list.append(vid_feat)
            
            # Pack the decoupled features into a tuple to be handled by the trainer
            video_features = tuple(video_features_list)
        else:
            if self.fusion_stage == 'video':
                # Video-Level Fusion: Temporal Transformers operate on each stream first, then CMAF
                video_face = self.temporal_net_face(image_face_features.contiguous().view(n, t, -1))
                video_body = self.temporal_net_body(image_body_features.contiguous().view(n, t, -1))
                if self.use_context:
                    video_context = self.temporal_net_context(image_context_features.contiguous().view(n, t, -1))
                    fused_video = self.cmaf(video_face, video_body, video_context)
                else:
                    fused_video = self.cmaf(video_face, video_body)
                
                video_features = self.project_fc(fused_video)
                video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
            elif self.fusion_stage == 'parallel':
                # [Ablation: parallel] ─────────────────────────────────────────────────────────────
                # Step 1: Frame-level Cross-Attention (decoupled — NO concat yet)
                #   Each modality cross-attends to others per frame to absorb cross-modal context,
                #   but keeps its own 512-d representation (return_decoupled=True).
                if self.use_context:
                    face_out, body_out, ctx_out = self.cmaf(
                        image_face_features, image_body_features, image_context_features,
                        return_decoupled=True
                    )  # each: (B*T, 512)
                else:
                    face_out, body_out = self.cmaf(
                        image_face_features, image_body_features,
                        return_decoupled=True
                    )  # each: (B*T, 512)

                # Step 2: Stream-specific Temporal Transformers (learn independent temporal dynamics)
                #   face: focuses attention on micro-expression keyframes
                #   body: focuses attention on posture/gesture trends across frames
                #   context: focuses attention on scene-change moments
                face_vid = self.temporal_net_face(face_out.contiguous().view(n, t, -1))   # (B, 512)
                body_vid = self.temporal_net_body(body_out.contiguous().view(n, t, -1))   # (B, 512)

                if self.use_context:
                    ctx_vid = self.temporal_net_context(ctx_out.contiguous().view(n, t, -1))  # (B, 512)
                    # Step 3: Video-level concat: 3 × 512 → 1536 → project_fc → 512
                    fused = torch.cat([face_vid, body_vid, ctx_vid], dim=-1)
                else:
                    # Step 3: Video-level concat: 2 × 512 → 1024 → project_fc → 512
                    fused = torch.cat([face_vid, body_vid], dim=-1)

                video_features = self.project_fc(fused)
                video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
                # ─────────────────────────────────────────────────────────────────────────────────
            else:
                # Frame-Level Fusion (Default SOTA): CMAF on each frame
                if self.use_context:
                    fused_frame_features = self.cmaf(image_face_features, image_body_features, image_context_features)
                else:
                    fused_frame_features = self.cmaf(image_face_features, image_body_features)
                
                if self.use_temporal:
                    # Unified Temporal Transformer (Videos)
                    fused_frame_features = fused_frame_features.contiguous().view(n, t, -1)
                    video_features = self.unified_temporal_net(fused_frame_features)
                else:
                    # Image Datasets (e.g. EMOTIC): Direct projection without temporal net
                    video_features = fused_frame_features.contiguous().view(n, -1)

                video_features = self.project_fc(video_features)
                # Robust normalization to avoid NaN on MPS
                video_features = video_features / (video_features.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Save video features for feature-level knowledge distillation
        self.last_video_features = video_features

        # Text features are already computed above.

        moco_logits = None

        ################# Classification ###################
        # Calculate logits
        if self.is_ensemble:
            # Reshape text features for ensembling: (C*P, D) -> (C, P, D)
            text_features = text_features.view(self.num_classes, self.num_prompts_per_class, -1)
            # Normalize again just in case (optional but safe) - Robust version
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            
            if isinstance(video_features, tuple):
                outputs = []
                for vid_feat in video_features:
                    # Compute logits per prompt: (B, D) @ (D, P, C) -> (B, P, C)
                    logits = torch.einsum('bd,cpd->bcp', vid_feat, text_features)
                    # Average the logits across the prompts for each class
                    out = torch.mean(logits, dim=2) / self.args.temperature
                    outputs.append(out)
                # ── Soft Modality Router ──────────────────────────────────────
                # outputs: list of (B, C) tensors, one per modality
                # Stack → (B, C, num_mod), route per-class
                stacked = torch.stack(outputs, dim=-1)  # (B, C, M)
                if hasattr(self, 'decoupled_vote_router'):
                    B, C, M = stacked.shape
                    # Run router over stacked logits: (B*C, M) → (B*C, M)
                    router_in = stacked.view(B * C, M)
                    router_weights = torch.softmax(
                        self.decoupled_vote_router(router_in), dim=-1
                    )  # (B*C, M)
                    output = (stacked.view(B * C, M) * router_weights).sum(dim=-1).view(B, C)
                else:
                    # Fallback: uniform average (safe default)
                    output = stacked.mean(dim=-1)  # (B, C)
            else:
                logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
                output = torch.mean(logits, dim=2) / self.args.temperature

        else:
            if isinstance(video_features, tuple):
                outputs = []
                for vid_feat in video_features:
                    out = vid_feat @ text_features.t() / self.args.temperature
                    outputs.append(out)
                # ── Soft Modality Router (non-ensemble path) ─────────────────
                stacked = torch.stack(outputs, dim=-1)  # (B, C, M)
                if hasattr(self, 'decoupled_vote_router'):
                    B, C, M = stacked.shape
                    router_in = stacked.view(B * C, M)
                    router_weights = torch.softmax(
                        self.decoupled_vote_router(router_in), dim=-1
                    )  # (B*C, M)
                    output = (stacked.view(B * C, M) * router_weights).sum(dim=-1).view(B, C)
                else:
                    output = stacked.mean(dim=-1)  # fallback: uniform average
            else:
                output = video_features @ text_features.t() / self.args.temperature

        return output, text_features, hand_crafted_text_features, moco_logits