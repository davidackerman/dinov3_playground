#!/usr/bin/env python
"""
Example: Integrating preprocessed DINOv3 features into your training loop.

This shows how to replace on-the-fly feature extraction with cached features.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from preprocessed_dataloader import create_preprocessed_dataloader


# ============================================================================
# OLD WAY (SLOW) - Extracting features during training
# ============================================================================
def train_with_feature_extraction_OLD(model, raw_dataloader, num_epochs=10):
    """
    OLD approach: Extract DINOv3 features during training.
    This is SLOW because feature extraction takes 2-5 seconds per volume.
    """
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch_idx, (raw, gt, mask) in enumerate(raw_dataloader):
            # BOTTLENECK: This takes 2-5 seconds per batch!
            features = extract_dinov3_features(raw)  # SLOW!
            
            # Forward pass
            predictions = model(features)
            loss = compute_loss(predictions, gt, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")


# ============================================================================
# NEW WAY (FAST) - Using preprocessed features
# ============================================================================
def train_with_preprocessed_features_NEW(
    model,
    preprocessed_dir,
    num_epochs=10,
    batch_size=4,
    num_threads=8,
):
    """
    NEW approach: Load pre-extracted DINOv3 features.
    This is FAST because features are loaded from cache in ~10-50ms.
    """
    # Create DataLoader with preprocessed features
    train_loader = create_preprocessed_dataloader(
        preprocessed_dir=preprocessed_dir,
        batch_size=batch_size,
        shuffle=True,
        num_threads=num_threads,
        num_workers=0,  # Use TensorStore's internal parallelism
    )
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch_idx, (features, gt, mask) in enumerate(train_loader):
            # Features are already extracted! Just use them directly.
            # This is ~50-200x faster than extracting during training.
            
            # Move to GPU if available
            if torch.cuda.is_available():
                features = features.cuda()
                gt = gt.cuda()
                if mask is not None:
                    mask = mask.cuda()
            
            # Forward pass
            predictions = model(features)
            loss = compute_loss(predictions, gt, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")


# ============================================================================
# Example with train/val split
# ============================================================================
def train_with_train_val_split(
    model,
    preprocessed_dir,
    train_indices,
    val_indices,
    num_epochs=10,
    batch_size=4,
):
    """
    Example with separate train and validation sets.
    """
    # Create train DataLoader
    train_loader = create_preprocessed_dataloader(
        preprocessed_dir=preprocessed_dir,
        volume_indices=train_indices,  # Only use these volumes for training
        batch_size=batch_size,
        shuffle=True,
        num_threads=8,
    )
    
    # Create validation DataLoader
    val_loader = create_preprocessed_dataloader(
        preprocessed_dir=preprocessed_dir,
        volume_indices=val_indices,  # Only use these volumes for validation
        batch_size=batch_size,
        shuffle=False,
        num_threads=8,
    )
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for features, gt, mask in train_loader:
            if torch.cuda.is_available():
                features, gt = features.cuda(), gt.cuda()
                if mask is not None:
                    mask = mask.cuda()
            
            predictions = model(features)
            loss = compute_loss(predictions, gt, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, gt, mask in val_loader:
                if torch.cuda.is_available():
                    features, gt = features.cuda(), gt.cuda()
                    if mask is not None:
                        mask = mask.cuda()
                
                predictions = model(features)
                loss = compute_loss(predictions, gt, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")


# ============================================================================
# Example with custom transforms
# ============================================================================
def train_with_custom_transforms(
    model,
    preprocessed_dir,
    num_epochs=10,
    batch_size=4,
):
    """
    Example with custom data augmentation/transforms on preprocessed features.
    """
    from preprocessed_dataloader import PreprocessedDINOv3Dataset
    from torch.utils.data import DataLoader
    
    def custom_transform(features, gt, mask):
        """Custom transform to apply to each sample."""
        # Example: Random flip
        if torch.rand(1) > 0.5:
            features = torch.flip(features, dims=[-1])  # Flip width
            gt = torch.flip(gt, dims=[-1])
            if mask is not None:
                mask = torch.flip(mask, dims=[-1])
        
        # Example: Normalize features
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features, gt, mask
    
    # Create dataset with transform
    dataset = PreprocessedDINOv3Dataset(
        preprocessed_dir=preprocessed_dir,
        num_threads=8,
        transform=custom_transform,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        for batch_idx, (features, gt, mask) in enumerate(train_loader):
            if torch.cuda.is_available():
                features, gt = features.cuda(), gt.cuda()
                if mask is not None:
                    mask = mask.cuda()
            
            predictions = model(features)
            loss = compute_loss(predictions, gt, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")


# ============================================================================
# Dummy functions (replace with your actual implementations)
# ============================================================================
def extract_dinov3_features(raw):
    """Placeholder for feature extraction (SLOW)."""
    # This would be your actual DINOv3 feature extraction
    pass


def compute_loss(predictions, gt, mask):
    """Placeholder for loss computation."""
    # This would be your actual loss function
    return torch.randn(1, requires_grad=True)


# ============================================================================
# Main example
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example training with preprocessed features")
    parser.add_argument("--preprocessed-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    
    # Dummy model (replace with your actual model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1024, 1, 1)  # 1024 input channels from DINOv3
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("Training with preprocessed features (FAST)...")
    train_with_preprocessed_features_NEW(
        model=model,
        preprocessed_dir=args.preprocessed_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )
    
    print("\nTraining complete!")
    print(f"Your training is now ~50-200x faster than extracting features on-the-fly!")


# ============================================================================
# Key takeaways:
# ============================================================================
"""
1. Replace your raw data DataLoader with create_preprocessed_dataloader()

2. Features are already extracted, so you skip the bottleneck:
   OLD: raw -> extract_features() -> model -> loss  (SLOW)
   NEW: features -> model -> loss                     (FAST)

3. Everything else stays the same:
   - Same model architecture
   - Same loss function
   - Same optimizer
   - Same training loop structure

4. You get a ~50-200x speedup on data loading, making training much faster!

5. Use volume_indices parameter to split train/val/test:
   - train_indices = [0, 1, 2, ..., 799]
   - val_indices = [800, 801, ..., 899]
   - test_indices = [900, 901, ..., 999]

6. Apply custom transforms if needed (augmentation, normalization, etc.)

7. Adjust num_threads based on your system (8-16 works well)
"""
