"""
File containing the main model.

- This model uses a X3D network architecture for video classification.
- It includes augmentations and normalization specific to video data.
"""

#Standard imports
import torch
import math
from torch import nn
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from fvcore.nn import FlopCountAnalysis, flop_count_table


from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


#Local imports
from model.modules import BaseRGBModel, FCLayers, step


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models to provide temporal information
    """
    def __init__(self, d_model, max_len=64, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # Replace 2D CNN with X3D (new code)
            if self._feature_arch.startswith('x3d_s'):
                self._features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
                feature_dim = 192 # TODO: check this value
            elif self._feature_arch.startswith('x3d_m'):
                print("Using X3D (M version)")
                self._features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
                feature_dim = 192 # TODO: check this value
            
            # Get max sequence length from args or use default
            self.max_seq_length = getattr(args, 'clip_len', 50)
            print("Max sequence length:", self.max_seq_length)
            self.transformer_layers = args.transformer_layers if "transformer_layers" in args else 2
            print("Transformers layers:", self.transformer_layers)
            self.transformer_dims = args.transformer_dims if "transformer_dims" in args else 2048
            print("Transformers FF dims:", self.transformer_dims)
            self.transformer_heads = args.transformer_heads if "transformer_heads" in args else 8
            print("Transformers FF dims:", self.transformer_heads)
            self.use_learnable_pe = args.use_learnable_pe if "use_learnable_pe" in args else False
            print("Enhanced positional encoding:", self.use_learnable_pe)
            self.dropout = args.dropout if "dropout" in args else 0.1
            print("Dropout:", self.dropout)
            
            # Positional encoding
            self.positional_encoding = PositionalEncoding(
                d_model=feature_dim,
                max_len=self.max_seq_length,
                dropout=self.dropout
            )
            if self.use_learnable_pe:
                self.positional_encoding = nn.Embedding(self.max_seq_length, feature_dim)
            
            # Transformer encoder layer for temporal modeling (replacing LSTM)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=self.transformer_heads, # TODO: check this value
                dim_feedforward=self.transformer_dims, # TODO: check this value
                dropout=self.dropout,
                activation=F.gelu,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=self.transformer_layers # TODO: check this value
            )
            
            # Final classification layer
            self._fc = FCLayers(feature_dim, args.num_classes+1)

            # Update normalization for video models (critical change)
            self.standarization = T.Compose([
                T.Normalize(mean = (0.45, 0.45, 0.45), 
                            std = (0.225, 0.225, 0.225)) # Kinetics-400 stats
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape # B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch
            x = self.standarize(x) # Standarization Kinetics-400 stats
            
            # Reformat input for 3D CNN: (B,T,C,H,W) -> (B,C,T,H,W)
            x = x.permute(0, 2, 1, 3, 4)
            
            # Pass through X3D model
            # The X3D model expects input of shape (B, C, T, H, W)
            for i, block in enumerate(self._features.blocks):
                if i == 5:
                    break # Skip ResNet projection head
                x = block(x)
            
            # Now x has shape (B, C, T', H', W')
            # Pooling spatial dimensions only, keeping temporal dimension
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, C, T', 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (B, C, T')
            
            # Rearrange to (B, T', C) for Transformer
            x = x.permute(0, 2, 1)  # (B, T', C)
            
            # Apply positional encoding
            if self.use_learnable_pe:
                # Generate position indices for current sequence length
                positions = torch.arange(
                    x.size(1), 
                    dtype=torch.long, 
                    device=x.device
                ).unsqueeze(0).expand(x.size(0), -1)  # Shape: (B, T')

                # Positional encoding
                x = x + self.positional_encoding(positions)
            else:
                x = self.positional_encoding(x)  # (B, T', C)
            
            # Apply transformer encoder
            x = self.transformer_encoder(x)  # (B, T', C)

            # Final classification
            x = self._fc(x)  # (B, T', num_classes+1)
            return x
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            # Original augmentations but applied consistently across frames
            # x shape: (B, T, C, H, W)
            
            # Generate random parameters ONCE per clip
            flip_prob = torch.rand(x.size(0)) < 0.5  # (B,) boolean tensor
            jitter_params = torch.rand(x.size(0), 4) # (B, 4) for brightness, contrast, saturation, hue
            
            # Apply same augmentation to all frames in a clip
            for b in range(x.size(0)):
                # Horizontal flip (same for all frames)
                if flip_prob[b]:
                    x[b] = T.functional.hflip(x[b])
                
                # Color jitter (same parameters for all frames)
                x[b] = T.functional.adjust_brightness(x[b], jitter_params[b,0] * 0.2 + 0.9)  # 0.9-1.1
                x[b] = T.functional.adjust_contrast(x[b], jitter_params[b,1] * 0.2 + 0.9)
                x[b] = T.functional.adjust_saturation(x[b], jitter_params[b,2] * 0.2 + 0.9)
                x[b] = T.functional.adjust_hue(x[b], jitter_params[b,3] * 0.1 - 0.05)  # -0.05-0.05
                
                # Gaussian blur (same for all frames)
                if torch.rand(1) < 0.25:
                    x[b] = T.functional.gaussian_blur(x[b], kernel_size=5)
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)

        # Compute FLOPs of the model
        B, T, C, H, W = 1, 50, 3, 224, 224  # Example: 1 batch, 16 frames, 3 channels, 224x224 resolution
        x = torch.randn(B, T, C, H, W)

        # Count FLOPs
        flops = FlopCountAnalysis(self._model, x)
        print(flop_count_table(flops))
        print(f"Total FLOPs: {flops.total()}")
        
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()
        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()

