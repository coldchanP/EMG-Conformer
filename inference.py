"""
EMG Conformer Large Model - Inference Test

Test inference on trained EMG Conformer model with dummy data
- Manual model path specification (no automatic detection)
- Input data validation
- Performance metrics (inference time, FLOPs)
- Feature extraction capability test

Usage:
    python inference.py --model_path your_model.pth
    python inference.py  # uses default 'emg_conformer.pth'
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F
import time
import glob
import matplotlib.pyplot as plt
import argparse

try:
    from fvcore.nn import flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available, FLOPs calculation will be skipped")


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=128):
        super().__init__()
        
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 64, (1, 7), (1, 1)),
            nn.Conv2d(64, 64, (4, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 3), (1, 2)),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 96, (1, 3), (1, 1)),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d((1, 3), (1, 2)),
            nn.Dropout(0.4),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(96, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=16,
                 drop_p=0.3,
                 forward_expansion=6,
                 forward_drop_p=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # Combination of Global Average Pooling + Global Max Pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        x = avg_pool + max_pool  # Feature fusion
        
        out = self.fc(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, emb_size=128, depth=10, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class EMGConformerInference:
    """
    EMG Conformer Inference Class - Final Version
    
    Args:
        model_path: Path to saved model weight file (required)
        device: Device to use ('auto', 'cuda', 'cpu')
    """
    def __init__(self, model_path="emg_conformer.pth", device='auto'):
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Class name mapping
        self.class_names = {
            0: "Rest",
            1: "Pinch Grasp", 
            2: "Cylinder Grasp",
            3: "Extension"
        }
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        print(f"Model path: {self.model_path}")
        
        # Model loading
        self.model = self.load_model()
        print("Model loading completed!")

    def _determine_model_path(self, model_path, run_dir):
        """This method is no longer used - keeping for compatibility"""
        return model_path

    def load_model(self):
        """Load model weights"""
        print(f"Loading model: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        model_config = checkpoint.get('model_config', {
            'emb_size': 128,
            'depth': 10,
            'n_classes': 4
        })
        
        print(f"Model configuration: {model_config}")
        
        # Create model
        model = ViT(
            emb_size=model_config['emb_size'],
            depth=model_config['depth'],
            n_classes=model_config['n_classes']
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        # Display model information
        if 'accuracy' in checkpoint:
            print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
        if 'fold' in checkpoint:
            print(f"Model fold: {checkpoint['fold'] + 1}")
        if 'epoch' in checkpoint:
            print(f"Model epoch: {checkpoint['epoch']}")
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return model

    def preprocess_signal(self, emg_signal):
        """
        EMG signal preprocessing
        
        Args:
            emg_signal: (batch_size, channels, time_samples) or (channels, time_samples)
                       or numpy array
        
        Returns:
            torch.Tensor: preprocessed tensor (batch_size, 1, channels, time_samples)
        """
        # Convert numpy array to tensor
        if isinstance(emg_signal, np.ndarray):
            emg_signal = torch.FloatTensor(emg_signal)
        
        # Add and adjust dimensions
        if emg_signal.dim() == 2:  # (channels, time)
            emg_signal = emg_signal.unsqueeze(0).unsqueeze(0)  # (1, 1, channels, time)
        elif emg_signal.dim() == 3:  # (batch, channels, time)
            emg_signal = emg_signal.unsqueeze(1)  # (batch, 1, channels, time)
        elif emg_signal.dim() == 4:  # (batch, 1, channels, time)
            pass  # Already in correct format
        else:
            raise ValueError(f"Unsupported input dimension: {emg_signal.dim()}")
        
        # Move to device
        emg_signal = emg_signal.to(self.device)
        
        return emg_signal

    def predict(self, emg_signal, return_features=False):
        """
        EMG signal prediction
        
        Args:
            emg_signal: EMG signal (channels=4, time_samples=30)
            return_features: Whether to return feature vectors
        
        Returns:
            dict: prediction results
        """
        with torch.no_grad():
            # Preprocessing
            input_tensor = self.preprocess_signal(emg_signal)
            
            # Inference
            start_time = time.time()
            features, logits = self.model(input_tensor)
            inference_time = time.time() - start_time
            
            # Calculate prediction probabilities
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            # Organize results
            results = {
                'predicted_class': predicted_class.cpu().numpy(),
                'class_name': [self.class_names[cls.item()] for cls in predicted_class],
                'confidence': confidence.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'inference_time': inference_time
            }
            
            if return_features:
                results['features'] = features.cpu().numpy()
            
            return results

    def predict_batch(self, emg_signals):
        """
        Batch prediction
        
        Args:
            emg_signals: (batch_size, channels, time_samples)
        
        Returns:
            dict: batch prediction results
        """
        return self.predict(emg_signals)

    def print_prediction(self, prediction_result):
        """Print prediction results in a nice format"""
        print(f"\n{'='*50}")
        print("EMG Signal Classification Results")
        print(f"{'='*50}")
        
        classes = prediction_result['predicted_class']
        names = prediction_result['class_name']
        confidences = prediction_result['confidence']
        probabilities = prediction_result['probabilities']
        
        for i in range(len(classes)):
            print(f"\nSample {i+1}:")
            print(f"   Predicted Class: {classes[i]} - {names[i]}")
            print(f"   Confidence: {confidences[i]:.4f} ({confidences[i]*100:.2f}%)")
            print(f"   Class Probabilities:")
            for j, (class_idx, class_name) in enumerate(self.class_names.items()):
                prob = probabilities[i][j]
                print(f"    {class_idx} - {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"\nInference Time: {prediction_result['inference_time']*1000:.2f}ms")
        print(f"{'='*50}\n")

    def visualize_prediction(self, emg_signal, prediction_result, save_path=None):
        """Visualize EMG signal with prediction results"""
        if isinstance(emg_signal, torch.Tensor):
            emg_signal = emg_signal.cpu().numpy()
        
        # Handle single sample case
        if emg_signal.ndim == 4:
            emg_signal = emg_signal[0, 0]  # (1, 1, 4, 30) -> (4, 30)
        elif emg_signal.ndim == 3:
            emg_signal = emg_signal[0]     # (1, 4, 30) -> (4, 30)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'EMG Signal Prediction: {prediction_result["class_name"][0]} '
                    f'(Confidence: {prediction_result["confidence"][0]:.3f})', 
                    fontsize=14, fontweight='bold')
        
        # Plot EMG signal for each channel
        channel_names = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4']
        colors = ['red', 'blue', 'green', 'orange']
        
        for i in range(4):
            ax = axes[i//2, i%2]
            ax.plot(emg_signal[i], color=colors[i], linewidth=2)
            ax.set_title(f'{channel_names[i]}')
            ax.set_xlabel('Time Samples')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            
            # Display signal statistics
            mean_val = np.mean(emg_signal[i])
            std_val = np.std(emg_signal[i])
            ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


def generate_dummy_emg_signal(batch_size=1, channels=4, time_samples=30, signal_type='random'):
    """
    Generate dummy EMG signal
    
    Args:
        batch_size: batch size
        channels: number of channels (4 EMG channels)
        time_samples: number of time samples (30 = 100ms at 300Hz)
        signal_type: signal type ('random', 'sine', 'realistic')
    
    Returns:
        torch.Tensor: generated EMG signal
    """
    if signal_type == 'random':
        # Completely random signal
        signal = torch.randn(batch_size, channels, time_samples)
        
    elif signal_type == 'sine':
        # Sine wave based signal (different frequency for each channel)
        t = torch.linspace(0, 1, time_samples)
        signal = torch.zeros(batch_size, channels, time_samples)
        
        for b in range(batch_size):
            for c in range(channels):
                freq = 10 + c * 5  # 10, 15, 20, 25 Hz
                amplitude = 0.5 + c * 0.1
                signal[b, c] = amplitude * torch.sin(2 * np.pi * freq * t)
                # Add noise
                signal[b, c] += 0.1 * torch.randn(time_samples)
                
    elif signal_type == 'realistic':
        # More realistic EMG signal simulation
        signal = torch.zeros(batch_size, channels, time_samples)
        
        for b in range(batch_size):
            # Different activity pattern for each batch
            activity_level = torch.rand(1).item()  # Activity level between 0~1
            
            for c in range(channels):
                # Base noise
                noise = 0.05 * torch.randn(time_samples)
                
                if activity_level > 0.7:  # High activity (grasp/pinch)
                    # Signal with many high frequency components
                    for freq in [50, 80, 120, 150]:
                        t = torch.linspace(0, 0.1, time_samples)  # 100ms
                        amplitude = 0.3 * torch.rand(1).item()
                        signal[b, c] += amplitude * torch.sin(2 * np.pi * freq * t)
                    
                elif activity_level > 0.3:  # Medium activity (release)
                    # Medium frequency components
                    for freq in [20, 40, 60]:
                        t = torch.linspace(0, 0.1, time_samples)
                        amplitude = 0.2 * torch.rand(1).item()
                        signal[b, c] += amplitude * torch.sin(2 * np.pi * freq * t)
                
                # Add noise
                signal[b, c] += noise
                
                # Normalize to 0-1 range (since original data normalized 0-255 to 0-1)
                signal[b, c] = (signal[b, c] - signal[b, c].min()) / (signal[b, c].max() - signal[b, c].min())
    
    return signal


def main():
    """Main inference function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EMG Conformer Model Inference')
    parser.add_argument('--model_path', type=str, default='emg_conformer.pth',
                       help='Path to model weight file (default: emg_conformer.pth)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto)')
    args = parser.parse_args()
    
    print("EMG Conformer Final Model Inference")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("\nPlease check one of the following:")
        print("1. Place your model file as 'emg_conformer.pth' in the current directory")
        print("2. Specify the correct model path:")
        print("   python inference.py --model_path your_model.pth")
        print("3. Train a new model using train.py")
        return
    
    # Create inference object with specified model path
    try:
        print(f"\nLoading model: {args.model_path}")
        inference = EMGConformerInference(model_path=args.model_path, device=args.device)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the model file path and try again.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\nStarting inference test with various dummy EMG signals...\n")
    
    # 1. Single random signal inference + visualization
    print("1. Single random EMG signal inference and visualization:")
    single_signal = generate_dummy_emg_signal(1, 4, 30, 'random')
    print(f"   Input signal shape: {single_signal.shape}")
    
    result = inference.predict(single_signal)
    inference.print_prediction(result)
    inference.visualize_prediction(single_signal, result)
    
    # 2. Batch sine wave signal inference
    print("2. Batch sine wave EMG signal inference (3 samples):")
    batch_signal = generate_dummy_emg_signal(3, 4, 30, 'sine')
    print(f"   Input signal shape: {batch_signal.shape}")
    
    result = inference.predict_batch(batch_signal)
    inference.print_prediction(result)
    
    # 3. Realistic EMG signal inference
    print("3. Realistic EMG signal inference (5 samples):")
    realistic_signal = generate_dummy_emg_signal(5, 4, 30, 'realistic')
    print(f"   Input signal shape: {realistic_signal.shape}")
    
    result = inference.predict_batch(realistic_signal)
    inference.print_prediction(result)
    
    # 4. Inference with feature vectors included
    print("4. Inference with feature vectors included:")
    single_signal = generate_dummy_emg_signal(1, 4, 30, 'realistic')
    result = inference.predict(single_signal, return_features=True)
    
    print(f"   Feature vector shape: {result['features'].shape}")
    print(f"   Feature vector statistics:")
    print(f"     Mean: {np.mean(result['features']):.4f}")
    print(f"     Standard deviation: {np.std(result['features']):.4f}")
    print(f"     Minimum: {np.min(result['features']):.4f}")
    print(f"     Maximum: {np.max(result['features']):.4f}")
    
    inference.print_prediction(result)
    
    print("="*60)
    print("All inference tests completed!")
    print("\nFor actual usage in your code:")
    print("```python")
    print("# Load model with specific path")
    print("inference = EMGConformerInference(model_path='your_model.pth')")
    print("# Load EMG signal (4 channels, 30 samples)")
    print("emg_data = your_emg_signal  # shape: (4, 30)")
    print("result = inference.predict(emg_data)")
    print("inference.print_prediction(result)")
    print("inference.visualize_prediction(emg_data, result)")
    print("```")


if __name__ == "__main__":
    main() 