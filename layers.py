import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample=True):
        super(DownBlock, self).__init__()
        
        self.conv_1_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.conv_2_1 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
        self.conv_2_2 = nn.Conv2d(out_features, out_features, kernel_size=1, padding=0)

        self.norm1_1 = nn.BatchNorm2d(out_features)
        self.norm2_1 = nn.BatchNorm2d(out_features)
        self.norm1_2 = nn.BatchNorm2d(out_features)
        self.norm2_2 = nn.BatchNorm2d(out_features)
        
        self.act_fn = nn.ReLU()
        
        if downsample:
            self.downsample = nn.MaxPool2d(kernel_size=2)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x_out_1 = self.act_fn(self.norm1_1(self.conv_1_1(x)))
        x_out_1 = self.act_fn(self.norm2_1(self.conv_2_1(x_out_1)))

        x_out_2 = self.act_fn(self.norm1_2(self.conv_1_2(x)))
        x_out_2 = self.act_fn(self.norm2_2(self.conv_2_2(x_out_2)))

        x_out = self.downsample(x_out_1 + x_out_2)

        return x_out


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, upsample=True):
        super(UpBlock, self).__init__()
        
        self.conv_1_1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.conv_2_1 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0)
        self.conv_2_2 = nn.Conv2d(out_features, out_features, kernel_size=1, padding=0)

        self.norm1_1 = nn.BatchNorm2d(out_features)
        self.norm2_1 = nn.BatchNorm2d(out_features)
        self.norm1_2 = nn.BatchNorm2d(out_features)
        self.norm2_2 = nn.BatchNorm2d(out_features)
        
        self.act_fn = nn.ReLU()
        
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x):
        x = self.upsample(x)
        
        x_out_1 = self.act_fn(self.norm1_1(self.conv_1_1(x)))
        x_out_1 = self.act_fn(self.norm2_1(self.conv_2_1(x_out_1)))

        x_out_2 = self.act_fn(self.norm1_2(self.conv_1_2(x)))
        x_out_2 = self.act_fn(self.norm2_2(self.conv_2_2(x_out_2)))

        x_out = x_out_1 + x_out_2
        return x_out
        

class AdjustSizeDown(nn.Module):
    def __init__(self, num_features):
        super(AdjustSizeDown, self).__init__()
        self.num_features = num_features
    
    def forward(self, x):
        return x.view(-1, self.num_features*1*1)


class AdjustSizeUp(nn.Module):
    def __init__(self, num_features):
        super(AdjustSizeUp, self).__init__()
        self.num_features = num_features
    
    def forward(self, x):
        return x.view(-1, self.num_features, 1, 1)


class CategoricalLinear(nn.Module):
    def __init__(self, in_features, out_features, num_categories, enabled=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_categories = num_categories

        # for optionally disabling the categorical select
        self.enabled = enabled

        self.weight = nn.Parameter(torch.randn(num_categories, out_features, in_features))
    
    def forward(self, x, selected_ids):
        # x.shape: (batch_size, in_features)
        x = x.unsqueeze(2) # x.shape: (batch_size, in_features, 1)
        
        if self.enabled:
            # default operation
            selected_weights = self.weight[selected_ids]
        else:
            # disabled categorical operation
            # ignore the "selected_ids" and use first set of weights (batched)
            selected_weights = self.weight[torch.zeros(x.shape[0], dtype=torch.long)]
        
        # selected_weights.shape: (batch_size, out_features, in_features)
        # x.shape: (batch_size, in_features, 1)
        # out.shape: (batch_size, out_features, 1)
        out = torch.bmm(selected_weights, x)
        # out.shape: (batch_size, out_features)
        out = out.squeeze(2)
        return out
