import torch
from torch import nn
# Layer Norm 2D code directly taken from the SAM Repository
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
'''
The input of the SAM encoder is: 1024x1024x3
the output of the SAM encoder is: 256x64x64

Hence, having multuple conv2dTranspose to get an output shape of: 1x1024x1024
Note: The last layer of decoder is 1x1 layer such that: 16x1024x1024 -->  1x1024x1024
'''
class SAM_Decoder(nn.Module):
    def __init__(self, sam_encoder, sam_preprocess):
        super().__init__()
        self.sam_encoder = sam_encoder
        self.sam_preprocess = sam_preprocess
        for layer_no, param in enumerate(sam_encoder.parameters()):
            pass
        last_layer_no = layer_no
        print("Last layer No: ", last_layer_no)
        for layer_no, param in enumerate(self.sam_encoder.parameters()):
            if(layer_no > (last_layer_no - 6)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.nn_drop = nn.Dropout(p = 0.2)
        
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 0)
        self.norm1 = LayerNorm2d(128)
        
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 0)
        self.norm2 = LayerNorm2d(64)
        
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2, padding = 0)
        self.norm3 = LayerNorm2d(32)
        
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding = 0)
        self.norm4 = LayerNorm2d(16)
        
        self.conv5 = nn.ConvTranspose2d(16, 1, kernel_size = 1, stride = 1, padding = 0)
        
    def forward(self, x):
        x = self.sam_preprocess(x)
        x = self.sam_encoder(x)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.nn_drop(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = torch.nn.functional.relu(x)
        x = self.nn_drop(x)
        
        x = self.conv4(x)
        x = self.norm4(x)
        x = torch.nn.functional.relu(x)
        
        x = self.conv5(x)
        x = torch.nn.functional.sigmoid(x)
        return x