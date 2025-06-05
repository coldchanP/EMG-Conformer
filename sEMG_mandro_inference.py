"""
Real-time sEMG Classification System

Real-time hand gesture classification system that receives data 
from sEMG devices via serial communication.

Requirements:
- pyserial: pip install pyserial
- 300Hz sampling rate
- 4 channel EMG signals
- 30 sample window (100ms)

Usage:
    python sEMG_mandro_inference.py --model_path your_model.pth
    python sEMG_mandro_inference.py  # uses default 'emg_conformer.pth'
"""

import argparse
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
import serial
import serial.tools.list_ports
import threading
import queue
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from sklearn.preprocessing import StandardScaler


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


class SerialEMGReader:
    """Reading sEMG data via serial communication"""
    
    def __init__(self, port="COM7", baudrate=115200, timeout=1):  # Default set to COM7
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_port = None
        self.is_connected = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.read_thread = None
        self.stop_reading = False
        
    def connect_to_port(self, port=None):
        """Direct connection to specified port"""
        if port is None:
            port = self.port
            
        try:
            print(f"Connecting to {port} port...")
            self.serial_port = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(0.2)  # Wait for connection stabilization
            
            # Connection test - receive some data
            print(f"{port} data reception test in progress...")
            test_data = []
            for i in range(5):
                try:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        values = [float(val) for val in line.split(',')]
                        if len(values) == 4:  # 4 channel data
                            test_data.append(values)
                            print(f"Received data: {values}")
                except Exception as e:
                    print(f"Data reception error: {e}")
                    break
            
            if len(test_data) >= 3:
                self.is_connected = True
                print(f"✓ {port} port connection successful!")
                return True
            else:
                print(f"✗ {port} port data reception failed")
                self.serial_port.close()
                return False
                
        except Exception as e:
            print(f"✗ {port} port connection failed: {e}")
            return False
    
    def find_available_ports(self):
        """Find available COM ports"""
        available_ports = []
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            available_ports.append(port.device)
            print(f"Found port: {port.device} - {port.description}")
        
        return available_ports
    
    def test_port_connection(self, port_name):
        """Test specific port connection"""
        try:
            print(f"\nTesting {port_name} port...")
            test_serial = serial.Serial(port_name, self.baudrate, timeout=1)
            time.sleep(0.5)
            
            # Test data reception
            data_count = 0
            valid_data_count = 0
            
            for i in range(10):  # Try 10 times
                try:
                    line = test_serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        data_count += 1
                        print(f"  Raw data: {line}")
                        
                        # Check if it's 4-channel EMG data
                        try:
                            values = [float(val) for val in line.split(',')]
                            if len(values) == 4:
                                valid_data_count += 1
                                print(f"  Valid EMG data: {values}")
                        except:
                            print(f"  Invalid format: {line}")
                    
                    time.sleep(0.1)
                except Exception as e:
                    print(f"  Read error: {e}")
                    continue
            
            test_serial.close()
            
            print(f"  Total data count: {data_count}")
            print(f"  Valid EMG data count: {valid_data_count}")
            
            # If at least 5 valid data received, consider as success
            if valid_data_count >= 5:
                print(f"✓ {port_name} EMG data reception successful!")
                return True
            else:
                print(f"✗ {port_name} insufficient EMG data")
                return False
                
        except Exception as e:
            print(f"✗ {port_name} connection test failed: {e}")
            return False
    
    def auto_connect(self):
        """Auto-discover and connect to sEMG device"""
        print("Auto-discovering sEMG device...")
        
        # First try default port (COM7)
        if self.connect_to_port("COM7"):
            self.port = "COM7"
            return True
        
        # If default port fails, scan all available ports
        available_ports = self.find_available_ports()
        
        if not available_ports:
            print("No available COM ports found")
            return False
        
        print(f"\nSearching among {len(available_ports)} available ports...")
        
        for port_name in available_ports:
            if self.test_port_connection(port_name):
                if self.connect_to_port(port_name):
                    self.port = port_name
                    return True
        
        print("Failed to find connectable sEMG device")
        return False
    
    def start_reading(self):
        """Start data reading thread"""
        if not self.is_connected:
            print("Serial port not connected. Cannot start reading.")
            return False
        
        if self.read_thread and self.read_thread.is_alive():
            print("Data reading already in progress.")
            return True
        
        self.stop_reading = False
        self.read_thread = threading.Thread(target=self._read_data_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        print("Data reading started.")
        return True
    
    def _read_data_loop(self):
        """Data reading loop (runs in thread)"""
        while not self.stop_reading and self.is_connected:
            try:
                if self.serial_port and self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    if line:
                        try:
                            # CSV format parsing (4 channels)
                            values = [float(val) for val in line.split(',')]
                            if len(values) == 4:
                                # Normalize (0-255 -> 0-1)
                                normalized_values = [val / 255.0 for val in values]
                                
                                # Add to queue
                                if not self.data_queue.full():
                                    self.data_queue.put(normalized_values)
                                else:
                                    # If queue is full, remove old data
                                    try:
                                        self.data_queue.get_nowait()
                                        self.data_queue.put(normalized_values)
                                    except queue.Empty:
                                        pass
                        except ValueError:
                            # Skip data with incorrect format
                            pass
                            
                time.sleep(0.001)  # 1ms sleep to prevent excessive CPU usage
                
            except Exception as e:
                print(f"Data reading error: {e}")
                break
    
    def get_data(self):
        """Get one data sample from queue"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop data reading and close serial port"""
        self.stop_reading = True
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.is_connected = False
        print("Serial connection closed.")


class RealTimeEMGClassifier:
    """Real-time EMG signal classification"""
    
    def __init__(self, model_path="emg_conformer.pth", device='auto'):
        # Device configuration
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Class names
        self.class_names = ["Rest", "Pinch Grasp", "Cylinder Grasp", "Extension"]
        
        # Window configuration
        self.window_size = 30  # 100ms at 300Hz
        self.overlap = 0.5
        self.step_size = int(self.window_size * (1 - self.overlap))
        
        # Data buffer
        self.data_buffer = deque(maxlen=self.window_size * 2)  # 2x window size buffer
        
        # Prediction history for voting
        self.prediction_buffer = deque(maxlen=10)  # Store last 10 predictions
        self.voting_interval = 0.5  # 0.5 seconds
        self.last_voting_time = time.time()
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        print(f"Model path: {self.model_path}")
        
        # Model and scaler initialization
        self.model = None
        self.scaler = None
        
        # Initialize
        self.load_model()
        self.setup_scaler()
        
        print("Real-time EMG Classifier initialized")
        print(f"Window size: {self.window_size} samples")
        print(f"Overlap: {self.overlap * 100}%")
        print(f"Voting interval: {self.voting_interval}s")
    
    def load_model(self):
        """Load model"""
        print(f"Loading model: {self.model_path}")
        
        # Model initialization
        self.model = ViT(emb_size=128, depth=10, n_classes=4)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully: {os.path.basename(self.model_path)}")
    
    def setup_scaler(self):
        """Configure data scaler (dummy scaler since data is already normalized)"""
        # Since data from serial is already normalized (0-255 -> 0-1), 
        # use identity scaler
        self.scaler = StandardScaler()
        # Fit with dummy data (0-1 range)
        dummy_data = np.random.rand(100, 4)  # 100 samples, 4 channels
        self.scaler.fit(dummy_data)
        
        # Set scaler parameters for normalized data
        self.scaler.mean_ = np.array([0.5, 0.5, 0.5, 0.5])  # Mean of 0-1 range
        self.scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0])  # No scaling
        
        print("Data scaler configured")
    
    def add_sample(self, emg_sample, debug=True):
        """Add EMG sample to buffer and perform prediction"""
        if len(emg_sample) != 4:
            if debug:
                print(f"Warning: Expected 4 channels, got {len(emg_sample)}")
            return None
        
        # Add to buffer
        self.data_buffer.append(emg_sample)
        
        # If buffer has enough data, perform prediction
        if len(self.data_buffer) >= self.window_size:
            # Prediction with 50% overlap
            if len(self.data_buffer) % self.step_size == 0:
                result = self.predict_current_window(debug=debug)
                
                if result:
                    # Add to prediction buffer
                    self.prediction_buffer.append({
                        'class_id': result['class_id'],
                        'confidence': result['confidence'],
                        'timestamp': time.time()
                    })
                    
                    if debug:
                        self.print_detailed_prediction(result)
                    
                    # Check voting result
                    self.check_voting_result()
                    
                    return result
        
        return None
    
    def predict_current_window(self, debug=False):
        """Predict current window"""
        if len(self.data_buffer) < self.window_size:
            return None
        
        try:
            # Get latest window
            window_data = list(self.data_buffer)[-self.window_size:]
            window_array = np.array(window_data)  # (30, 4)
            
            # Normalization using scaler (though already normalized)
            window_normalized = self.scaler.transform(window_array)
            
            # Transpose to (channels, time) format
            window_tensor = torch.FloatTensor(window_normalized.T).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 30)
            window_tensor = window_tensor.to(self.device)
            
            # Prediction
            with torch.no_grad():
                features, outputs = self.model(window_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_id = predicted.item()
                confidence_val = confidence.item()
                
                if debug:
                    probs = probabilities.cpu().numpy()[0]
                    debug_info = {
                        'window_shape': window_tensor.shape,
                        'features_shape': features.shape,
                        'all_probabilities': {self.class_names[i]: probs[i] for i in range(len(self.class_names))},
                        'raw_data_sample': window_data[-1]  # Latest sample
                    }
                
                return {
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': confidence_val,
                    'probabilities': probabilities.cpu().numpy()[0],
                    'debug_info': debug_info if debug else None
                }
                
        except Exception as e:
            if debug:
                print(f"Prediction error: {e}")
            return None
    
    def check_voting_result(self):
        """Check voting result every 0.5 seconds"""
        current_time = time.time()
        
        if current_time - self.last_voting_time >= self.voting_interval:
            if len(self.prediction_buffer) > 0:
                # Collect predictions from last 0.5 seconds
                recent_predictions = [
                    pred for pred in self.prediction_buffer 
                    if current_time - pred['timestamp'] <= self.voting_interval
                ]
                
                if recent_predictions:
                    # Count votes by class
                    class_votes = {}
                    total_confidence = {}
                    
                    for pred in recent_predictions:
                        class_id = pred['class_id']
                        confidence = pred['confidence']
                        
                        if class_id not in class_votes:
                            class_votes[class_id] = 0
                            total_confidence[class_id] = 0
                        
                        class_votes[class_id] += 1
                        total_confidence[class_id] += confidence
                    
                    # Find winner class
                    winner_class_id = max(class_votes, key=class_votes.get)
                    vote_count = class_votes[winner_class_id]
                    avg_confidence = total_confidence[winner_class_id] / vote_count
                    
                    # Print result
                    self.print_voting_result(winner_class_id, vote_count, avg_confidence, 
                                           len(recent_predictions), current_time)
            
            self.last_voting_time = current_time
    
    def print_voting_result(self, winner_class_id, vote_count, avg_confidence, 
                          total_predictions, timestamp):
        """Print voting result"""
        class_name = self.class_names[winner_class_id]
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
        
        print(f"\n[{timestamp_str}] VOTING RESULT:")
        print(f"   Winner: {class_name}")
        print(f"   Votes: {vote_count}/{total_predictions}")
        print(f"   Confidence: {avg_confidence:.3f}")
        print(f"   {'='*50}")
    
    def get_current_status(self):
        """Get current classification status"""
        return {
            'buffer_size': len(self.data_buffer),
            'prediction_count': len(self.prediction_buffer),
            'last_prediction': self.prediction_buffer[-1] if self.prediction_buffer else None,
            'model_path': self.model_path,
            'device': self.device
        }
    
    def print_detailed_prediction(self, result):
        """Print detailed prediction result"""
        print(f"Prediction: {result['class_name']} (confidence: {result['confidence']:.3f})")
        
        if result.get('debug_info'):
            debug = result['debug_info']
            print(f"  All probabilities:")
            for class_name, prob in debug['all_probabilities'].items():
                print(f"    {class_name}: {prob:.3f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time sEMG Hand Gesture Classification')
    parser.add_argument('--model_path', type=str, default='emg_conformer.pth',
                       help='Path to model weight file (default: emg_conformer.pth)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto)')
    args = parser.parse_args()
    
    print("="*60)
    print("Real-time sEMG Hand Gesture Classification System")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print("="*60)
    
    try:
        # Check if model file exists
        if not os.path.exists(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found!")
            print("\nPlease check one of the following:")
            print("1. Place your model file as 'emg_conformer.pth' in the current directory")
            print("2. Specify the correct model path:")
            print("   python sEMG_mandro_inference.py --model_path your_model.pth")
            print("3. Train a new model using train.py")
            return
        
        # Initialize classifier
        print("\n1. Initializing classifier...")
        classifier = RealTimeEMGClassifier(model_path=args.model_path, device=args.device)
        
        # Initialize serial reader
        print("\n2. Connecting to sEMG device...")
        reader = SerialEMGReader()
        
        # Auto-connect
        if not reader.auto_connect():
            print("Failed to connect to sEMG device. Please check connection and try again.")
            return
        
        # Start data reading
        print("\n3. Starting data reading...")
        if not reader.start_reading():
            print("Failed to start data reading.")
            return
        
        print("\n4. Starting real-time classification...")
        print("   - Press Ctrl+C to stop")
        print("   - Predictions occur every ~100ms (50% overlap)")
        print("   - Voting results every 0.5 seconds")
        print("="*60)
        
        # Real-time classification loop
        sample_count = 0
        try:
            while True:
                # Get data from serial
                emg_data = reader.get_data()
                
                if emg_data is not None:
                    sample_count += 1
                    
                    # Add to classifier and get prediction
                    result = classifier.add_sample(emg_data, debug=False)
                    
                    # Print periodic status
                    if sample_count % 100 == 0:
                        status = classifier.get_current_status()
                        print(f"\nStatus update - Samples: {sample_count}, "
                              f"Buffer: {status['buffer_size']}/30, "
                              f"Predictions: {status['prediction_count']}")
                
                time.sleep(0.001)  # 1ms sleep to prevent excessive CPU usage
                
        except KeyboardInterrupt:
            print("\n\nUser stop requested...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Cleanup
        print("\n5. Cleanup...")
        if 'reader' in locals():
            reader.stop()
        print("Program terminated.")


if __name__ == "__main__":
    main() 