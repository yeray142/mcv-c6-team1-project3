"""
File containing the main model.

- This model uses a X3D network architecture for video classification.
- It includes augmentations and normalization specific to video data.
"""

#Standard imports
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss



from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # Replace 2D CNN with X3D (new code)
            if self._feature_arch.startswith('x3d_s'):
                self._features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
            elif self._feature_arch.startswith('x3d_m'):
                print("Using X3D (M version)")
                self._features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
                

            # Add LSTM for temporal processing (similar to the second model)
            self._lstm = nn.LSTM(input_size=192, hidden_size=self._features.blocks[5].proj.in_features, 
                                batch_first=True, num_layers=1, bidirectional=True)
            lstm_out_dim = self._features.blocks[5].proj.in_features * 2  # because of bidirectionality
            
            # Add attention layer for better temporal modeling
            self.attention_heads = args.attention_heads if "attention_heads" in args else 4
            print("Using attention heads:", self.attention_heads)
            
            # Attention layer
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,  # Matches LSTM's output dimension
                num_heads=self.attention_heads,  # 4 attention heads
                batch_first=True
            )
            
            self._fc = FCLayers(lstm_out_dim, args.num_classes+1)

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
            
            # Rearrange to (B, T', C) for LSTM
            x = x.permute(0, 2, 1)  # (B, T', C)
            
            # Pass through LSTM
            x, _ = self._lstm(x)  # output shape: (B, T', 2*_d)
            
            # Apply attention
            attn_out, _ = self.attention_layer(x, x, x)

            # Final classification
            x = self._fc(attn_out)  # (B, T', num_classes+1)
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

