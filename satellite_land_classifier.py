"""
================================================================================
SATELLITE LAND CLASSIFIER - ML CAPSTONE PROJECT
================================================================================
Deep Learning pipeline for classifying satellite imagery into agricultural 
vs non-agricultural land using CNNs, Vision Transformers, and hybrid models.

Skills Demonstrated:
• CNN architectures in Keras and PyTorch
• Vision Transformer (ViT) implementation
• CNN-ViT hybrid model design
• Data augmentation and preprocessing
• Model evaluation with comprehensive metrics
• Framework comparison (TensorFlow/Keras vs PyTorch)

Author: Yoav Sela
================================================================================
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, roc_auc_score, roc_curve)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SEED = 42
IMG_SIZE = 64
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_CLASSES = 2
CLASS_NAMES = ['agricultural', 'non_agricultural']

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)

set_seed(SEED)

# ==============================================================================
# SECTION 1: KERAS/TENSORFLOW IMPLEMENTATION
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Dense, Dropout, GlobalAveragePooling2D, 
                                     Flatten, Reshape)
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform

# Set TensorFlow seed
tf.random.set_seed(SEED)


def build_keras_cnn(img_size: int = 64, num_classes: int = 2) -> Sequential:
    """
    Build a 6-layer CNN for binary image classification.
    
    Architecture: 6 Conv blocks with increasing filters (32→1024),
    followed by fully connected layers with dropout regularization.
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # Block 1: 32 filters
        Conv2D(32, (5,5), activation="relu", padding="same", 
               kernel_initializer=HeUniform(), input_shape=(img_size, img_size, 3)),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 2: 64 filters
        Conv2D(64, (5,5), activation="relu", padding="same", kernel_initializer=HeUniform()),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 3: 128 filters
        Conv2D(128, (5,5), activation="relu", padding="same", kernel_initializer=HeUniform()),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 4: 256 filters
        Conv2D(256, (5,5), activation="relu", padding="same", kernel_initializer=HeUniform()),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 5: 512 filters
        Conv2D(512, (5,5), activation="relu", padding="same", kernel_initializer=HeUniform()),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Block 6: 1024 filters
        Conv2D(1024, (5,5), activation="relu", padding="same", kernel_initializer=HeUniform()),
        MaxPooling2D(2, 2),
        BatchNormalization(),
        
        # Classifier head
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_initializer=HeUniform()),
        BatchNormalization(),
        Dropout(0.4),
        Dense(2048, activation="relu", kernel_initializer=HeUniform()),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1, activation="sigmoid")  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------------------------------------
# KERAS VISION TRANSFORMER COMPONENTS
# ------------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="Custom")
class AddPositionEmbedding(layers.Layer):
    """Adds learnable positional embeddings to token sequences."""
    
    def __init__(self, num_patches: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, tokens):
        return tokens + self.pos

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    """
    Standard Transformer encoder block with multi-head attention and MLP.
    
    Components:
    - Multi-head self-attention for global context
    - Feed-forward MLP with GELU activation
    - Layer normalization and residual connections
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, 
                 mlp_dim: int = 2048, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        
        self.mha = layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

    def call(self, x):
        # Self-attention with residual connection
        x = self.norm1(x + self.mha(x, x))
        # MLP with residual connection
        return self.norm2(x + self.mlp(x))

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        })
        return config


def build_keras_cnn_vit_hybrid(
    cnn_model: Model,
    feature_layer_name: str,
    num_transformer_layers: int = 4,
    num_heads: int = 8,
    mlp_dim: int = 2048,
    num_classes: int = 2
) -> Model:
    """
    Build hybrid CNN-ViT model combining CNN local features with Transformer 
    global attention.
    
    Architecture:
    1. CNN backbone extracts spatial features
    2. Feature maps reshaped into token sequence
    3. Positional embeddings added for spatial awareness
    4. Transformer blocks process tokens globally
    5. Classification head produces predictions
    
    Args:
        cnn_model: Pre-trained CNN for feature extraction
        feature_layer_name: Layer name to extract features from
        num_transformer_layers: Number of Transformer encoder blocks
        num_heads: Attention heads per Transformer block
        mlp_dim: MLP hidden dimension
        num_classes: Output classes
        
    Returns:
        Keras Model (CNN-ViT hybrid)
    """
    # Freeze CNN backbone
    cnn_model.trainable = False
    
    # Extract features from specified layer
    features = cnn_model.get_layer(feature_layer_name).output
    H, W, C = features.shape[1], features.shape[2], features.shape[3]
    
    # Reshape spatial grid to token sequence + add positional embeddings
    x = layers.Reshape((H * W, C))(features)
    x = AddPositionEmbedding(H * W, C)(x)

    # Stack Transformer encoder blocks
    for _ in range(num_transformer_layers):
        x = TransformerBlock(C, num_heads, mlp_dim)(x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(cnn_model.layers[0].input, outputs, name="CNN_ViT_hybrid")


# ==============================================================================
# SECTION 2: PYTORCH IMPLEMENTATION
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# Set PyTorch seeds
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PyTorchCNN(nn.Module):
    """
    6-layer CNN matching the Keras architecture for fair comparison.
    
    Architecture: Conv blocks with BatchNorm → Global average pooling → Classifier
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1-6: Progressive filter increase
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(1024),
        )
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 2048), nn.ReLU(), nn.BatchNorm1d(2048), nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps (for hybrid ViT integration)."""
        return self.features(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ------------------------------------------------------------------------------
# PYTORCH VISION TRANSFORMER COMPONENTS
# ------------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Project CNN features to ViT embedding dimension."""
    
    def __init__(self, input_channel: int = 1024, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(input_channel, embed_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, L, D) where L = H*W
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, N, self.heads, -1).transpose(1, 2)
        k = k.reshape(B, N, self.heads, -1).transpose(1, 2)
        v = v.reshape(B, N, self.heads, -1).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        
        # Combine heads
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(x))


class PyTorchTransformerBlock(nn.Module):
    """Transformer block with MHSA + MLP."""
    
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4., dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PyTorchViT(nn.Module):
    """Vision Transformer for processing CNN feature maps."""
    
    def __init__(self, in_ch: int = 1024, num_classes: int = 2, embed_dim: int = 768,
                 depth: int = 6, heads: int = 8, mlp_ratio: float = 4., 
                 dropout: float = 0.1, max_tokens: int = 50):
        super().__init__()
        self.patch = PatchEmbed(in_ch, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        
        self.blocks = nn.ModuleList([
            PyTorchTransformerBlock(embed_dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)  # (B, L, D)
        B, L, _ = x.shape
        
        # Add class token and positional embeddings
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat((cls, x), 1)
        x = x + self.pos[:, :L + 1]
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Classify using CLS token
        return self.head(self.norm(x)[:, 0])


class PyTorchCNNViTHybrid(nn.Module):
    """Hybrid CNN-ViT combining local CNN features with global Transformer attention."""
    
    def __init__(self, num_classes: int = 2, embed_dim: int = 768, 
                 depth: int = 6, heads: int = 8):
        super().__init__()
        self.cnn = PyTorchCNN(num_classes)
        self.vit = PyTorchViT(
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN extracts local features, ViT processes globally
        return self.vit(self.cnn.forward_features(x))


# ==============================================================================
# SECTION 3: DATA LOADING & AUGMENTATION
# ==============================================================================

def create_keras_data_generators(dataset_path: str, img_size: int = 64, 
                                  batch_size: int = 128, validation_split: float = 0.2):
    """
    Create Keras ImageDataGenerator with augmentation for training.
    
    Augmentations: rotation, shifts, shear, zoom, horizontal flip
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split
    )
    
    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )
    
    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )
    
    return train_gen, val_gen


def create_pytorch_data_loaders(dataset_path: str, img_size: int = 64,
                                 batch_size: int = 128, val_split: float = 0.2):
    """
    Create PyTorch DataLoaders with augmentation.
    
    Uses ImageFolder for automatic class labeling from directory structure.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


# ==============================================================================
# SECTION 4: TRAINING FUNCTIONS
# ==============================================================================

def train_keras_model(model, train_gen, val_gen, epochs: int = 20, 
                      model_save_path: str = "keras_model.keras"):
    """Train Keras model with model checkpointing."""
    checkpoint_cb = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint_cb],
        verbose=1
    )
    return history


def train_pytorch_model(model, train_loader, val_loader, epochs: int = 20,
                        lr: float = 0.001, model_save_path: str = "pytorch_model.pth"):
    """Train PyTorch model with validation and checkpointing."""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        
        # Save best model
        if history['val_loss'][-1] < best_loss:
            best_loss = history['val_loss'][-1]
            torch.save(model.state_dict(), model_save_path)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val Acc: {history['val_acc'][-1]:.4f}")
    
    return history


# ==============================================================================
# SECTION 5: EVALUATION METRICS
# ==============================================================================

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive classification metrics.
    
    Returns:
        Dictionary with accuracy, precision, recall, F1, and optionally ROC-AUC
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        # For binary classification
        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
    
    return metrics


def plot_training_history(history, title: str = "Training History"):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.get('accuracy', history.get('train_acc')), label='Train')
    axes[0].plot(history.get('val_accuracy', history.get('val_acc')), label='Validation')
    axes[0].set_title(f'{title} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.get('loss', history.get('train_loss')), label='Train')
    axes[1].plot(history.get('val_loss'), label='Validation')
    axes[1].set_title(f'{title} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def plot_confusion_matrix(cm, class_names, title: str = "Confusion Matrix"):
    """Visualize confusion matrix as heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()


def compare_model_performance(keras_metrics: dict, pytorch_metrics: dict):
    """Print side-by-side comparison of Keras vs PyTorch model performance."""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON: KERAS vs PYTORCH")
    print("="*60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    print(f"{'Metric':<15} {'Keras':>12} {'PyTorch':>12} {'Winner':>10}")
    print("-"*50)
    
    for metric in metrics:
        keras_val = keras_metrics.get(metric, 'N/A')
        pytorch_val = pytorch_metrics.get(metric, 'N/A')
        
        if isinstance(keras_val, float) and isinstance(pytorch_val, float):
            winner = "Keras" if keras_val > pytorch_val else "PyTorch" if pytorch_val > keras_val else "Tie"
            print(f"{metric:<15} {keras_val:>12.4f} {pytorch_val:>12.4f} {winner:>10}")
        else:
            print(f"{metric:<15} {str(keras_val):>12} {str(pytorch_val):>12}")
    
    print("="*60)


# ==============================================================================
# MAIN EXECUTION EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SATELLITE LAND CLASSIFIER - ML CAPSTONE PROJECT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  • Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  • Batch Size: {BATCH_SIZE}")
    print(f"  • Learning Rate: {LEARNING_RATE}")
    print(f"  • Classes: {CLASS_NAMES}")
    print(f"  • Device: {DEVICE}")
    
    # Example: Build models
    print("\n📦 Building Keras CNN...")
    keras_cnn = build_keras_cnn()
    print(f"   Parameters: {keras_cnn.count_params():,}")
    
    print("\n📦 Building PyTorch CNN...")
    pytorch_cnn = PyTorchCNN()
    pytorch_params = sum(p.numel() for p in pytorch_cnn.parameters())
    print(f"   Parameters: {pytorch_params:,}")
    
    print("\n📦 Building PyTorch CNN-ViT Hybrid...")
    hybrid_model = PyTorchCNNViTHybrid()
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"   Parameters: {hybrid_params:,}")
    
    print("\n✅ All models built successfully!")
    print("\nTo train models, provide a dataset path and call:")
    print("  • train_keras_model(model, train_gen, val_gen)")
    print("  • train_pytorch_model(model, train_loader, val_loader)")
