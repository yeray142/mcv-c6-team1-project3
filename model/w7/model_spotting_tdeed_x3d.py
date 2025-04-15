"""
File containing the main model.
"""

#Standard imports
import torch
import math
import random
import torchvision.transforms as T
import torch.nn.functional as F
from pytorch_tcn import TCN
from fvcore.nn import FlopCountAnalysis, flop_count_table
from contextlib import nullcontext
from tqdm import tqdm
from torch import nn


#Local imports
from model.modules import BaseRGBModel, FCLayers, step, EDSGPMIXERLayers, process_prediction
from model.w7.shift import make_temporal_shift


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
            self._double_head = False
            
            # Check if feature architecture is supported
            self._use_gray = args.use_gray
            assert not self._use_gray, 'Gray not supported for this model yet'
            self._temp_arch = args.temporal_arch
            self._radi_displacement = args.radi_displacement
            
            # Feature extractor
            if self._feature_arch.startswith('x3d_m'):
                features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
                feat_dim = 192
                self._d = feat_dim
            elif self._feature_arch.startswith('x3d_l'):
                features = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
                feat_dim = 192
                self._d = feat_dim
            else:
                raise NotImplementedError(args._feature_arch)

            # Model parameters
            self._features = features
            self._feat_dim = self._d
            feat_dim = self._d
            
            # Positional encoding
            self.temp_enc = PositionalEncoding(d_model=feat_dim, max_len=args.clip_len, dropout=0.1)
            
            # Temporal architecture
            if self._temp_arch == 'ed_sgp_mixer':
                self._temp_fine = EDSGPMIXERLayers(feat_dim, args.clip_len, num_layers=args.n_layers, ks = args.sgp_ks, k = args.sgp_r, concat = True)
                self._pred_fine = FCLayers(self._feat_dim, args.num_classes+1)
            elif self._temp_arch == 'tcn':
                # TODO: This must be tested properly.
                print(f"TCNN (kernel {args.kernel_size}, {args.num_channels} channels, {args.n_layers} layers)")

                num_channels = [args.num_channels for _ in range(args.n_layers)]
                self._temp_fine = TCN(
                    num_inputs=self._feat_dim,
                    num_channels=num_channels,
                    kernel_size=args.kernel_size,
                    dilations=None, # auto-computed
                    dilation_reset=None,
                    dropout=0.1,
                    causal=True,
                    use_skip_connections=True,
                    input_shape='NLC', # transformers shape [B, T, D]
                    output_projection=args.num_classes + 1 # directly project to classes
                )
            else:
                raise NotImplementedError(self._temp_arch)
            
            # Displacement prediction
            if self._radi_displacement > 0:
                self._pred_displ = FCLayers(self._feat_dim, 1)

            # Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.45, 0.45, 0.45), 
                            std = (0.225, 0.225, 0.225)) # Kinetics-400 stats
            ])
            
            # Croping in case of using it
            self.croping = args.crop_dim
            if self.croping != None and self.croping > 0:
                self.cropT = T.RandomCrop((self.croping, self.croping))
                self.cropI = T.CenterCrop((self.croping, self.croping))
            else:
                self.cropT = torch.nn.Identity()
                self.cropI = torch.nn.Identity()

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W
            
            # Reshape to (B*T, C, H, W)
            x.view(-1, channels, height, width)
            if self.croping != None and self.croping > 0:
                height = self.croping
                width = self.croping
            x = self.cropT(x) # same crop for all frames
            x = x.view(batch_size, clip_len, channels, height, width)

            # Apply augmentations
            if self.training:
                x = self.augment(x) #augmentation per-batch
            x = self.standarize(x) #standarization imagenet stats
            
            # Get image features from X3D
            x = x.permute(0, 2, 1, 3, 4)
            for i, block in enumerate(self._features.blocks):
                if i == 5:
                    break # Skip ResNet projection head
                x = block(x)

            x = F.adaptive_avg_pool3d(x, (x.size(2), 1, 1))  # (B, C, T', 1, 1)
            x = x.squeeze(-1).squeeze(-1)  # (B, C, T')
            
            # Rearrange to (B, T', C) for Transformer
            im_feat = x.permute(0, 2, 1)  # (B, T', C)
            im_feat = self.temp_enc(im_feat) # B, T, D

            if self._temp_arch == 'ed_sgp_mixer':
                im_feat = self._temp_fine(im_feat)
                
                # Displacement prediction
                if self._radi_displacement > 0:
                    displ_feat = self._pred_displ(im_feat).squeeze(-1)
                    im_feat = self._pred_fine(im_feat)
                    return {'im_feat': im_feat, 'displ_feat': displ_feat}
                
                im_feat = self._pred_fine(im_feat)
                return im_feat
            elif self._temp_arch == 'tcn':
                # TODO: Missing testing.
                im_feat = self._temp_fine(im_feat)
                return im_feat
            else:
                raise NotImplementedError(self._temp_arch)
        
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
        B, T, C, H, W = 1, args.clip_len, 3, 336, 597  # Example: 1 batch, 16 frames, 3 channels, 224x224 resolution
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
            for _, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()
                
                if 'labelD' in batch.keys():
                    labelD = batch['labelD'].to(self.device).float()
                
                # If mixup is used, we need to mix the frames and labels
                if 'frame2' in batch.keys():
                    frame2 = batch['frame2'].to(self.device).float()
                    label2 = batch['label2']
                    label2 = label2.to(self.device)

                    if 'labelD2' in batch.keys():
                        labelD2 = batch['labelD2'].to(self.device).float()
                        labelD_dist = torch.zeros((labelD.shape[0], label.shape[1])).to(self.device)

                    l = [random.betavariate(0.2, 0.2) for _ in range(frame2.shape[0])]

                    label_dist = torch.zeros((label.shape[0], label.shape[1], self._num_classes+1)).to(self.device)

                    for i in range(frame2.shape[0]):
                        frame[i] = l[i] * frame[i] + (1 - l[i]) * frame2[i]
                        lbl1 = label[i]
                        lbl2 = label2[i]

                        label_dist[i, range(label.shape[1]), lbl1] += l[i]
                        label_dist[i, range(label2.shape[1]), lbl2] += 1 - l[i]

                        if 'labelD2' in batch.keys():
                            labelD_dist[i] = l[i] * labelD[i] + (1 - l[i]) * labelD2[i]

                    label = label_dist
                    if 'labelD2' in batch.keys():
                        labelD = labelD_dist
                
                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    
                    # If Radi displacement is used, we need to get the displacement prediction
                    if 'labelD' in batch.keys():
                        predD = pred['displ_feat']
                        pred = pred['im_feat']

                    pred = pred.contiguous().view(-1, self._num_classes + 1) # B*T, num_classes
                    # label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)
                    
                    if 'labelD' in batch.keys():
                        lossD = F.mse_loss(predD, labelD, reduction = 'none')
                        lossD = (lossD).mean()
                        loss = loss + lossD

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
            if isinstance(pred, dict):
                predD = pred['displ_feat']
                pred = pred['im_feat']
                if isinstance(pred, list):
                    pred = pred[0]
                if isinstance(predD, list):
                    predD = predD[0]
                if self._model._double_head:
                    raise NotImplementedError('Double head not implemented')
                    # pred = process_double_head(pred, predD, num_classes = self._args.num_classes+1)
                else:
                    pred = process_prediction(pred, predD)
                return pred.cpu().numpy()
            else:
                pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
