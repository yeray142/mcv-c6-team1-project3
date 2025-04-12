"""
File containing the main model.

- This model uses a 3D ResNet architecture for video classification. (3D-backbone)
- It includes augmentations and normalization specific to video data.
- Includes LSTM and attention layers for temporal processing. (LSTM + attention)
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


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            # Replace 2D CNN with 3D ResNet (new code)
            if self._feature_arch.startswith('3dresnet') or self._feature_arch.startswith('r3d_18'):
                print('Using 3D ResNet-18')
                self._features = torchvision.models.video.r3d_18(pretrained=True)
            elif self._feature_arch.startswith('mc3_18'):
                print('Using MC3-18')
                self._features = torchvision.models.video.mc3_18(pretrained=True)
            elif self._feature_arch.startswith('r2plus1d_18'):
                print('Using R2Plus1D-18')
                self._features = torchvision.models.video.r2plus1d_18(pretrained=True)
            self._d = 512
            
            # Important: Replace the FC layer with Identity to get features
            # self._features.fc = nn.Identity()
            
            # Add LSTM for temporal processing (similar to the second model)
            self._lstm = nn.LSTM(input_size=self._d, hidden_size=self._d, 
                                batch_first=True, num_layers=1, bidirectional=True)
            lstm_out_dim = self._d * 2  # because of bidirectionality
            
            # Add attention layer for better temporal modeling
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,  # Matches LSTM's output dimension
                num_heads=4,            # 4 attention heads
                batch_first=True
            )
            
            # Final classification layers
            self._fc = FCLayers(lstm_out_dim, args.num_classes+1)

            # Update normalization for video models (critical change)
            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), 
                            std = (0.22803, 0.22145, 0.216989)) # Kinetics-400 stats
            ])

        def forward(self, x):
            # Normalize input
            print("ResNet architecture configuration:")
            print(self._features)
            x = self.normalize(x)  # Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape  # B, T, C, H, W
            print(f"After normalization: {x.shape}")

            if self.training:
                x = self.augment(x)  # augmentation per-batch
            x = self.standarize(x)  # Standardization (Kinetics-400 stats)
            print(f"After augmentation and standardization: {x.shape}")
            
            # Reformat input for 3D CNN: (B, T, C, H, W) -> (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            print(f"After permute (B, C, T, H, W): {x.shape}")
            
            # Get 3D CNN features without pooling
            x = self._features.stem(x)
            print(f"After stem (B, C', T', H', W'): {x.shape}")
            
            x = self._features.layer1(x)
            print(f"After layer1 (B, C', T', H', W'): {x.shape}")
            
            x = self._features.layer2(x)
            print(f"After layer2 (B, C', T', H', W'): {x.shape}")
            
            x = self._features.layer3(x)
            print(f"After layer3 (B, C', T', H', W'): {x.shape}")
            
            x = self._features.layer4(x)
            print(f"After layer4 (B, C', T', H', W'): {x.shape}")
            print(f"After CNN layers (B, C', T', H', W'): {x.shape}")
            
            # Pooling spatial dimensions only, keeping temporal dimension
            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, C, T', 1, 1)
            print(f"After adaptive avg pool (B, C, T', 1, 1): {x.shape}")
            
            x = x.squeeze(-1).squeeze(-1)  # (B, C, T')
            print(f"After squeeze (B, C, T'): {x.shape}")
            
            # Rearrange to (B, T', C) for LSTM
            x = x.permute(0, 2, 1)  # (B, T', C)
            print(f"After permute for LSTM (B, T', C): {x.shape}")
            
            # Pass through LSTM
            x, _ = self._lstm(x)  # output shape: (B, T', 2*_d)
            print(f"After LSTM (B, T', 2*_d): {x.shape}")
            
            # Apply attention
            attn_out, _ = self.attention_layer(x, x, x)
            print(f"After attention layer (B, T', 2*_d): {attn_out.shape}")
            
            # Optional: Global temporal pooling (if used)
            # x = torch.mean(attn_out, dim=1)  # (B, 2*_d)
            # print(f"After temporal pooling (B, 2*_d): {x.shape}")
            
            # Final classification (pass through fully connected layer)
            x = self._fc(attn_out)  # (B, num_classes)
            print(f"After final FC layer (B, num_classes): {x.shape}")
            
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
                    print("pred", pred.shape)
                    print("label", label.shape)
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

