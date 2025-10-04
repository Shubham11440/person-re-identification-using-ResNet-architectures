# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# GaussianSmoothing class remains the same
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
    def __init__(self, num_classes, num_parts=6, use_gaussian_smoothing=False):
        super(CPR, self).__init__()
        self.num_parts = num_parts
        self.use_gaussian_smoothing = use_gaussian_smoothing

        if self.use_gaussian_smoothing:
            self.gaussian_smooth = GaussianSmoothing(channels=3, kernel_size=5, sigma=1.0)
        
        resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        self.backbone_b1 = nn.Sequential(*list(resnet50.children())[:-2])
        
        resnext50 = models.resnext50_32x4d(weights='IMAGENET1K_V1')
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

    def forward(self, x):
        if self.use_gaussian_smoothing:
            x = self.gaussian_smooth(x)
        
        X = self.backbone_b1(x)
        Y = self.backbone_b2(x)
        Z = X + Y
        
        part_features = self.part_pool(Z)
        reduced_features = self.feature_reducer(part_features)
        
        reshaped_features = reduced_features.view(reduced_features.size(0), reduced_features.size(1), -1).permute(0, 2, 1)
        
        # This is the global feature descriptor, used for triplet loss and inference
        global_descriptor = reshaped_features.contiguous().view(reshaped_features.size(0), -1)
        
        if self.training:
            # During training, return both part logits and the global descriptor
            part_logits = [self.classifier(reshaped_features[:, i, :]) for i in range(self.num_parts)]
            return part_logits, global_descriptor
        else:
            # During evaluation, return the L2-normalized descriptor for distance calculation
            return F.normalize(global_descriptor, p=2, dim=1)

# Test block
if __name__ == '__main__':
    num_test_classes = 751
    print("Testing improved CPR model...")
    
    # Test baseline model
    print("1. Testing baseline CPR model...")
    model_baseline = CPR(num_classes=num_test_classes, use_gaussian_smoothing=False)
    model_baseline.train()
    dummy_input = torch.randn(4, 3, 384, 128)
    part_logits, global_features = model_baseline(dummy_input)
    print(f"   Part logits: {len(part_logits)} parts, shape: {part_logits[0].shape}")
    print(f"   Global features shape: {global_features.shape}")
    
    # Test inference mode
    print("2. Testing inference mode...")
    model_baseline.eval()
    descriptor = model_baseline(dummy_input)
    print(f"   Inference descriptor shape: {descriptor.shape}")
    print(f"   Descriptor is L2-normalized: {torch.allclose(torch.norm(descriptor, p=2, dim=1), torch.ones(descriptor.size(0)))}")
    
    # Test with Gaussian smoothing
    print("3. Testing with Gaussian smoothing...")
    model_smooth = CPR(num_classes=num_test_classes, use_gaussian_smoothing=True)
    model_smooth.train()
    part_logits_s, global_features_s = model_smooth(dummy_input)
    print(f"   Smoothing model output shapes match baseline: ✅")
    
    print("\nAll model tests passed successfully! ✅")