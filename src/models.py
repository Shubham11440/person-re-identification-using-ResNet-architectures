# src/models.py

import torch
import torch.nn as nn
import torchvision.models as models
import math

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a tensor.
    Filtering is performed seperately for each channel in the input using a depthwise convolution.
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        # Create a gaussian kernel
        kernel = torch.arange(kernel_size, dtype=torch.float32)
        kernel -= kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Reshape to 2D kernel
        self.kernel = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        self.kernel_t = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        # Apply horizontal and vertical blur
        x = nn.functional.pad(x, (self.padding, self.padding, 0, 0), mode='replicate')
        x = nn.functional.conv2d(x, self.kernel, groups=self.groups)
        x = nn.functional.pad(x, (0, 0, self.padding, self.padding), mode='replicate')
        x = nn.functional.conv2d(x, self.kernel_t, groups=self.groups)
        return x

class CPR_Baseline(nn.Module):
    """
    Convolutional Part Refine (CPR) Baseline Model
    """
    def __init__(self, num_classes, num_parts=6):
        super(CPR_Baseline, self).__init__()
        self.num_parts = num_parts

        # Gaussian smoothing layer
        self.gaussian_smooth = GaussianSmoothing(channels=3, kernel_size=5, sigma=1.0)
        
        # --- Backbone B1: ResNet50 ---
        resnet50 = models.resnet50(pretrained=True)
        # Remove the final GAP and FC layers
        self.backbone_b1 = nn.Sequential(*list(resnet50.children())[:-2])
        
        # --- Backbone B2: ResNeXt50 ---
        resnext50 = models.resnext50_32x4d(pretrained=True)
        # Remove the final GAP and FC layers
        self.backbone_b2 = nn.Sequential(*list(resnext50.children())[:-2])
        
        # --- Part-level Feature Processing ---
        feature_dim = 2048 # Output channels from both ResNet50 and ResNeXt50
        
        # Adaptive pooling for partitioning the feature map
        self.part_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        
        # 1x1 Conv for dimension reduction on each part
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Shared classifier for all parts
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights for the reducer and classifier
        self._init_weights()

    def _init_weights(self):
        for m in self.feature_reducer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        # 1. Initial smoothing
        x = self.gaussian_smooth(x)
        
        # 2. Parallel feature extraction
        X = self.backbone_b1(x)  # Output from ResNet50 branch
        Y = self.backbone_b2(x)  # Output from ResNeXt50 branch
        
        # 3. Feature fusion
        Z = X + Y  # Element-wise addition
        
        # 4. Convolutional partition
        part_features = self.part_pool(Z)  # Shape: (batch, 2048, 6, 1)
        
        # 5. Part-level feature reduction
        reduced_features = self.feature_reducer(part_features) # Shape: (batch, 256, 6, 1)
        
        # Reshape for classification
        reshaped_features = reduced_features.view(reduced_features.size(0), reduced_features.size(1), -1) # Shape: (batch, 256, 6)
        reshaped_features = reshaped_features.permute(0, 2, 1) # Shape: (batch, 6, 256)
        
        if self.training:
            # During training, return the output of the classifier for each part
            part_logits = []
            for i in range(self.num_parts):
                part_logit = self.classifier(reshaped_features[:, i, :])
                part_logits.append(part_logit)
            return part_logits
        else:
            # During evaluation, return the concatenated feature vector
            # This is the final 1536-dim descriptor for the person
            final_descriptor = reshaped_features.contiguous().view(reshaped_features.size(0), -1)
            return final_descriptor


# Optional: Add a test block to verify the model architecture
if __name__ == '__main__':
    print("Testing CPR_Baseline model...")
    num_test_classes = 751 # Example: Market-1501 training set
    model = CPR_Baseline(num_classes=num_test_classes)
    
    # Create a dummy input tensor
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 3, 384, 128)
    
    # --- Test in training mode ---
    model.train()
    print("Testing in training mode...")
    train_outputs = model(dummy_input)
    print(f"Number of part outputs: {len(train_outputs)}")
    print(f"Shape of a single part output: {train_outputs[0].shape}")
    assert len(train_outputs) == 6
    assert train_outputs[0].shape == (4, num_test_classes)
    print("Training mode output is correct. ✅")
    
    # --- Test in evaluation mode ---
    model.eval()
    print("\nTesting in evaluation mode...")
    eval_output = model(dummy_input)
    print(f"Shape of evaluation descriptor: {eval_output.shape}")
    assert eval_output.shape == (4, 6 * 256) # 6 parts * 256 dims
    print("Evaluation mode output is correct. ✅")
    
    print("\nModel test complete.")
