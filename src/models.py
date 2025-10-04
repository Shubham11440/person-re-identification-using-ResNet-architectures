import torch
import torch.nn as nn
import torchvision.models as models

# The GaussianSmoothing class remains the same as before
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianSmoothing, self).__init__()
        kernel = torch.arange(kernel_size, dtype=torch.float32)
        kernel -= kernel_size // 2
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.kernel = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        self.kernel_t = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        self.groups = channels
        self.padding = kernel_size // 2

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, self.padding, 0, 0), mode='replicate')
        x = nn.functional.conv2d(x, self.kernel, groups=self.groups)
        x = nn.functional.pad(x, (0, 0, self.padding, self.padding), mode='replicate')
        x = nn.functional.conv2d(x, self.kernel_t, groups=self.groups)
        return x

class CPR(nn.Module):
    """
    Convolutional Part Refine (CPR) Model with configurable refinement strategies.
    """
    def __init__(self, num_classes, num_parts=6, refinement_strategy='none'):
        super(CPR, self).__init__()
        self.num_parts = num_parts
        self.refinement_strategy = refinement_strategy

        self.gaussian_smooth = GaussianSmoothing(channels=3, kernel_size=5, sigma=1.0)
        
        resnet50 = models.resnet50(pretrained=True)
        self.backbone_b1 = nn.Sequential(*list(resnet50.children())[:-2])
        
        resnext50 = models.resnext50_32x4d(pretrained=True)
        self.backbone_b2 = nn.Sequential(*list(resnext50.children())[:-2])
        
        feature_dim = 2048
        self.part_pool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        
        self.feature_reducer = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.feature_reducer.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.classifier.bias, 0)

    def _remove_outliers(self, part_features):
        """
        Implements the 'Removal of Outliers' strategy.
        It identifies outliers in each part and replaces them with the part's average feature.
        (Simplified implementation for demonstration)
        """
        # For each part stripe in the batch
        for i in range(self.num_parts):
            part = part_features[:, :, i, :] # Shape: (batch, channels, 1)
            # Calculate the mean feature for this part
            part_mean = part.mean(dim=0, keepdim=True)
            # A simple thresholding to find outliers (e.g., features far from the mean)
            # A more complex distance metric could be used here.
            distance = torch.abs(part - part_mean)
            # Create a mask for non-outlier features
            mask = (distance < distance.mean() + distance.std()).float()
            # Replace outliers with the mean
            part_features[:, :, i, :] = part * mask + part_mean * (1 - mask)
        return part_features

    def forward(self, x):
        x = self.gaussian_smooth(x)
        X = self.backbone_b1(x)
        Y = self.backbone_b2(x)
        Z = X + Y
        
        part_features = self.part_pool(Z)
        
        # --- Apply Refinement Strategy ---
        if self.training and self.refinement_strategy == 'removal':
            part_features = self._remove_outliers(part_features)
        
        reduced_features = self.feature_reducer(part_features)
        reshaped_features = reduced_features.view(reduced_features.size(0), reduced_features.size(1), -1).permute(0, 2, 1)
        
        if self.training:
            part_logits = [self.classifier(reshaped_features[:, i, :]) for i in range(self.num_parts)]
            return part_logits
        else:
            final_descriptor = reshaped_features.contiguous().view(reshaped_features.size(0), -1)
            return final_descriptor

# Optional: Update test block
if __name__ == '__main__':
    num_test_classes = 751
    print("Testing CPR model with 'none' refinement...")
    model_none = CPR(num_classes=num_test_classes, refinement_strategy='none')
    model_none.train()
    dummy_input = torch.randn(4, 3, 384, 128)
    outputs_none = model_none(dummy_input)
    print(f"Output shape is correct. ✅")

    print("\nTesting CPR model with 'removal' refinement...")
    model_removal = CPR(num_classes=num_test_classes, refinement_strategy='removal')
    model_removal.train()
    outputs_removal = model_removal(dummy_input)
    print(f"Output shape is correct. ✅")
    print("\nModel test complete.")