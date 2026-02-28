import torch
import torch.nn as nn
import torch.nn.functional as F
from deepmreye.util.util import mish
from deepmreye.config import DeepMReyeConfig


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, activation_fn):
        super().__init__()
        self.strides = strides
        
        # PyTorch Conv3D uses padding='same' similarly to Keras if stride is 1
        # For stride > 1, we calculate padding manually or just use padding=kernel_size//2
        padding = kernel_size // 2 if strides == 1 else kernel_size // 2
        
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides if strides == 1 else 1, # Keras code did stride=1 then AvgPool if stride > 1
            padding='same' if strides == 1 else padding
        )
        
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2) if strides > 1 else nn.Identity()
        self.activation = activation_fn

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.pool(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, filters, groups, activation_fn):
        super().__init__()
        
        self.input_layer_res = Conv3DBlock(in_channels, filters, kernel_size=1, strides=1, activation_fn=activation_fn)
        
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.act1 = activation_fn
        self.conv1 = Conv3DBlock(in_channels, filters, kernel_size=3, strides=1, activation_fn=activation_fn)
        
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=filters)
        self.act2 = activation_fn
        self.conv2 = Conv3DBlock(filters, filters, kernel_size=3, strides=1, activation_fn=activation_fn)

    def forward(self, x):
        res = self.input_layer_res(x)
        
        x = self.gn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        
        x = self.gn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        
        return x + res


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, filters, depth, multiplier, groups, activation_fn):
        super().__init__()
        self.depth = depth
        
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.gns = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        current_channels = in_channels
        
        for level_number in range(depth):
            n_level_filters = int(multiplier**level_number) * filters
            
            level_blocks = nn.ModuleList()
            for _ in range(level_number):
                level_blocks.append(ResBlock(current_channels, n_level_filters, groups, activation_fn))
                current_channels = n_level_filters
            self.blocks.append(level_blocks)
            
            if level_number < (depth - 1):
                self.downsamples.append(Conv3DBlock(current_channels, n_level_filters, kernel_size=3, strides=2, activation_fn=activation_fn))
                current_channels = n_level_filters
            else:
                self.downsamples.append(None)
                
        self.final_gn = nn.GroupNorm(num_groups=groups, num_channels=current_channels)
        self.final_act = activation_fn

    def forward(self, x):
        skip_layers = []
        for level_number in range(self.depth):
            for block in self.blocks[level_number]:
                x = block(x)
            
            skip_layers.append(x)
            
            if self.downsamples[level_number] is not None:
                x = self.downsamples[level_number](x)
                
        x = self.final_gn(x)
        x = self.final_act(x)
        return x, skip_layers


class RegressionBlock(nn.Module):
    def __init__(self, in_features, num_dense, num_fc, activation_fn, dropout_rate, inner_timesteps, dense_out=2):
        super().__init__()
        self.inner_timesteps = inner_timesteps
        
        self.timestep_networks = nn.ModuleList()
        for _ in range(inner_timesteps):
            layers = []
            current_in = in_features
            for _ in range(num_dense):
                layers.append(nn.Linear(current_in, num_fc))
                layers.append(activation_fn)
                layers.append(nn.Dropout(dropout_rate))
                current_in = num_fc
            layers.append(nn.Linear(current_in, dense_out))
            self.timestep_networks.append(nn.Sequential(*layers))

    def forward(self, x):
        # x is the bottleneck flat representation
        outputs = []
        for i in range(self.inner_timesteps):
            # In Keras it splits timesteps but passes the whole x to each if x isn't sequence
            # Wait, the Keras code does: RepeatVector(inner_timesteps)(x) -> shape (batch, steps, features)
            # Then slices: x[:, i, :] which is exactly x natively repeated.
            # So passing x directly to each network is entirely equivalent.
            out = self.timestep_networks[i](x) # Shape: (batch, dense_out)
            outputs.append(out.unsqueeze(1)) # Shape: (batch, 1, dense_out)
            
        return torch.cat(outputs, dim=1) # Shape: (batch, inner_timesteps, dense_out)


class ConfidenceBlock(nn.Module):
    def __init__(self, in_features, num_fc, activation_fn, dropout_rate, inner_timesteps):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, num_fc),
            activation_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(num_fc, inner_timesteps),
            activation_fn
        )

    def forward(self, x):
        return self.net(x)


class DeepMReyeModel(nn.Module):
    def __init__(self, input_shape, config: DeepMReyeConfig):
        super().__init__()
        self.config = config
        
        # Determine activation
        if config.activation == mish:
             self.activation_fn = Mish()
        else:
             self.activation_fn = nn.ReLU() # Fallback
             
        # Add a dummy dimension for channels since input is (X,Y,Z)
        # Expected input to Conv3D is (Batch, Channels, D, H, W)
        in_channels = 1 
        
        self.noise = nn.Identity() # nn.GaussianNoise equivalent can be handled in forward or dataset
        
        self.initial_conv = Conv3DBlock(
            in_channels=in_channels, 
            out_channels=config.filters, 
            kernel_size=config.kernel, 
            strides=1, 
            activation_fn=self.activation_fn
        )
        self.initial_dropout = nn.Dropout(config.dropout_rate)
        
        self.downsample = DownsampleBlock(
            in_channels=config.filters,
            filters=config.filters,
            depth=config.depth,
            multiplier=config.multiplier,
            groups=config.groups,
            activation_fn=self.activation_fn
        )
        
        # Calculate bottleneck size
        # We need a dummy forward pass to get the flattened size
        dummy_input = torch.zeros(1, in_channels, *input_shape)
        with torch.no_grad():
            x = self.initial_conv(dummy_input)
            x, _ = self.downsample(x)
            bottleneck_features = x.view(1, -1).size(1)
            
        self.regression_block = RegressionBlock(
            in_features=bottleneck_features,
            num_dense=config.num_dense,
            num_fc=config.num_fc,
            activation_fn=self.activation_fn,
            dropout_rate=config.dropout_rate,
            inner_timesteps=config.inner_timesteps
        )
        
        self.confidence_block = ConfidenceBlock(
            in_features=bottleneck_features,
            num_fc=config.num_fc,
            activation_fn=self.activation_fn,
            dropout_rate=config.dropout_rate,
            inner_timesteps=config.inner_timesteps
        )

    def forward(self, x):
        # Add gaussian noise during training if needed (omitted here for simplicity, do in dataset)
        x = self.initial_conv(x)
        # Handle mc_dropout requirement by forcing train mode if needed, but normally use module state
        if self.config.mc_dropout:
            self.initial_dropout.train()
        x = self.initial_dropout(x)
        
        x, skip_layers = self.downsample(x)
        
        bottleneck_layer = torch.flatten(x, 1)
        
        if self.config.mc_dropout:
            self.regression_block.train()
            self.confidence_block.train()
            
        out_regression = self.regression_block(bottleneck_layer)
        out_confidence = self.confidence_block(bottleneck_layer)
        
        return out_regression, out_confidence


def compute_standard_loss(out_confidence, real_reg, pred_reg):
    """
    Computes Euclidean loss and Confidence loss.
    """
    # Euclidean distance along the last dimension (X, Y)
    loss_euclidean = torch.sqrt(torch.sum(torch.square(real_reg - pred_reg), dim=-1)) # Shape: (batch, timesteps)
    
    # Mean squared error between the empirical euclidean error and the predicted confidence
    loss_confidence = torch.square(loss_euclidean - out_confidence) # Shape: (batch, timesteps)
    
    return torch.mean(loss_euclidean), torch.mean(loss_confidence)

