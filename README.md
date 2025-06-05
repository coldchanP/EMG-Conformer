# EMG-Conformer: Real-time Hand Gesture Classification System

This project is a research project that adapts [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) to change EEG (electroencephalogram) signal processing to EMG (electromyogram) signal processing.

A deep learning system that receives data from actual sEMG (surface electromyography) devices via serial communication and classifies hand gestures in real-time. It analyzes 4-channel EMG signals.
We used Mand.ro's sEMG equipment.

## Original Source

This project is based on the following paper and code:

```
@article{song2023eeg,
  title = {{{EEG Conformer}}: {{Convolutional Transformer}} for {{EEG Decoding}} and {{Visualization}}},
  shorttitle = {{{EEG Conformer}}},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2022.3230250}
}
```

- **Original Repository**: [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
- **License**: GPL-3.0 license

## Key Features

- **Real-time Classification**: 300Hz sampling, 30-sample (100ms) window for real-time processing
- **High Accuracy**: 90%+ classification performance with 5-fold cross validation
- **4 Gesture Classification**: Rest, Pinch Grasp, Cylinder Grasp, Extension
- **Advanced Signal Processing**: 50% window overlap and 0.5-second voting system
- **Flexible Connection**: Automatic COM port detection and connection
- **Comprehensive Visualization**: Training curves, confusion matrix, t-SNE, etc.

## System Requirements

### Hardware
- 4-channel sEMG device (with serial communication support)
- Windows 10/11 (COM port support)
- NVIDIA GPU (recommended, CUDA support)

### Software
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU usage)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/EMG-Conformer.git
cd EMG-Conformer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional Dependencies
```bash
# Enhanced visualization
pip install plotly

# FLOPs calculation (optional)
pip install fvcore
```

## Usage

### 1. Model Training

First, train the model with EMG data:

```bash
python train.py
```

**Training Data Format:**
- Place CSV files in `./ProjectFoloder/custom_dataset/` directory
- Each CSV: 5 columns (4 EMG channels + 1 label)
- Labels: 0(Rest), 1(Pinch Grasp), 2(Cylinder Grasp), 3(Extension)

**Output:**
- Trained model: `./runs/emg_conformer_large_YYYYMMDD_HHMMSS/checkpoints/`
- Visualization results: `./runs/emg_conformer_large_YYYYMMDD_HHMMSS/visualizations/`

### 2. Model Weight Preparation

After training, copy your best model weight to the main directory:

```bash
# Copy the best model from training results
cp ./runs/emg_conformer_large_YYYYMMDD_HHMMSS/checkpoints/emg_conformer_large_GLOBAL_BEST_*.pth ./emg_conformer.pth
```

### 3. Real-time Classification

Classify real-time EMG signals with the trained model:

```bash
# Use default model (emg_conformer.pth)
python sEMG_mandro_inference.py

# Use specific model file
python sEMG_mandro_inference.py --model_path your_model.pth

# Use specific device
python sEMG_mandro_inference.py --model_path your_model.pth --device cuda
```

**Features:**
- Automatic connection to COM7 port (auto port scan on failure)
- Continuous prediction with 50% overlap windows
- Final voting results output every 0.5 seconds
- Real-time statistics and performance information

### 4. Model Testing and Validation

Test inference performance with dummy data:

```bash
# Use default model (emg_conformer.pth)
python inference.py

# Use specific model file
python inference.py --model_path your_model.pth

# Use specific device
python inference.py --model_path your_model.pth --device cpu
```

**Features:**
- Various dummy signal generation (random, sine wave, realistic)
- Batch processing capability
- Feature vector extraction
- Signal visualization
- Performance metrics (inference time, accuracy)

### 5. Serial Communication Testing

Check sEMG device connection status and data:

```bash
python Mandro_serial_test.py
```

## Command Line Options

### sEMG_mandro_inference.py
```bash
python sEMG_mandro_inference.py [OPTIONS]

Options:
  --model_path PATH    Path to model weight file (default: emg_conformer.pth)
  --device DEVICE      Device to use: auto, cuda, cpu (default: auto)
  --help              Show help message
```

### inference.py
```bash
python inference.py [OPTIONS]

Options:
  --model_path PATH    Path to model weight file (default: emg_conformer.pth)
  --device DEVICE      Device to use: auto, cuda, cpu (default: auto)
  --help              Show help message
```

## Model Architecture

### EMG Conformer Structure
```
Input (1, 4, 30) 
    ↓
PatchEmbedding (Conv2D + BatchNorm + ELU + Pooling)
    ↓
TransformerEncoder (10 layers)
    ├── MultiHeadAttention (16 heads)
    ├── LayerNorm + Residual
    ├── FeedForward (expansion=6)
    └── LayerNorm + Residual
    ↓
ClassificationHead (Global Pooling + FC layers)
    ↓
Output (4 classes)
```

### Hyperparameters
- **Embedding Size**: 128
- **Transformer Depth**: 10 layers
- **Attention Heads**: 16
- **Window Size**: 30 samples (100ms at 300Hz)
- **Learning Rate**: 0.0005 (AdamW + Cosine Annealing)
- **Batch Size**: 32

## File Structure

```
EMG_Conformer/
├── train.py                    # Model training (5-fold CV)
├── sEMG_mandro_inference.py    # Real-time classification system
├── inference.py                # Model inference test
├── Mandro_serial_test.py       # Serial monitoring tool
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies list
├── runs/                       # Training results storage
│   └── emg_conformer_large_*/
│       ├── checkpoints/        # Model weights
│       └── visualizations/     # Visualization results
├── custom_dataset/             # Training data
│   └── data                    # EMG CSV files
└── weights/                    # Weight files folder
    └── ..pth                   # Weight files
```

## Performance

### Classification Performance
- **Overall Accuracy**: 90%+ (5-fold cross validation)
- **Class-wise Accuracy**: 
  - Rest: 95%+
  - Pinch Grasp: 88%+
  - Cylinder Grasp: 87%+
  - Extension: 89%+

### Real-time Performance
- **Inference Speed**: 2-5ms per window
- **Prediction Frequency**: ~10 times/second (50% overlap)
- **Voting Frequency**: 2 times/second (0.5-second intervals)
- **Memory Usage**: ~500MB (GPU)

## Usage Examples

### Command Line Usage
```bash
# Basic real-time classification (default model)
python sEMG_mandro_inference.py

# Real-time classification with specific model
python sEMG_mandro_inference.py --model_path ./weights/best_model.pth

# Model testing with dummy data
python inference.py --model_path ./weights/best_model.pth

# Serial port testing
python Mandro_serial_test.py
```

### Using in Python Code
```python
from sEMG_mandro_inference import RealTimeEMGClassifier

# Initialize classifier with specific model
classifier = RealTimeEMGClassifier(model_path="./weights/your_model.pth")

# Classify single EMG sample (4 channels x 30 samples)
emg_data = np.random.rand(4, 30)  # Example data
result = classifier.predict(emg_data)

print(f"Predicted class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.3f}")
```

```python
from inference import EMGConformerInference

# Initialize inference object
inference = EMGConformerInference(model_path="./weights/your_model.pth")

# Test with dummy data
emg_signal = generate_dummy_emg_signal(1, 4, 30, 'realistic')
result = inference.predict(emg_signal)

# Print results
inference.print_prediction(result)
inference.visualize_prediction(emg_signal, result)
```

### Serial Data Format
```
# Data format received from COM port (CSV)
100.5,102.3,98.7,101.2
99.8,103.1,97.9,100.5
101.2,101.8,99.1,102.0
...
```

## Troubleshooting

### 1. Model Not Found
```bash
# First run model training
python train.py

# Copy the trained model to main directory
cp ./runs/emg_conformer_large_YYYYMMDD_HHMMSS/checkpoints/emg_conformer_large_GLOBAL_BEST_*.pth ./emg_conformer.pth

# Or specify model path directly
python sEMG_mandro_inference.py --model_path path/to/your/model.pth
python inference.py --model_path path/to/your/model.pth
```

### 2. COM Port Connection Failed
```bash
# Check available ports
python Mandro_serial_test.py

# Check port number in Windows Device Manager
# Verify sEMG device is properly connected
```

### 3. GPU Memory Insufficient
```bash
# Run in CPU mode
python sEMG_mandro_inference.py --device cpu
python inference.py --device cpu
```

### 4. Low Accuracy
- Check training data quality
- Review normalization methods
- Collect more training data
- Tune hyperparameters

## License

This project is distributed under the same GPL-3.0 license as the original EEG-Conformer. See the `LICENSE` file for details.

## Author

**Author**: [tjdcks7570@kw.ac.kr](mailto:tjdcks7570@kw.ac.kr)

This project is a research that applies the EEG-Conformer idea to EMG signal processing to develop a system for real-time hand gesture classification.

## Acknowledgments

- We express deep gratitude to Song, Yonghao et al. for their EEG-Conformer paper and code.
- The innovative ideas of the original architecture have shown excellent performance in the EMG field as well.
- Original repository: [https://github.com/eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)

## Update History

### v1.0.0 (2024-01-XX)
- Initial release based on EEG-Conformer adapted for EMG
- 4-channel EMG real-time classification system
- EMG Conformer model implementation
- 5-fold cross validation
- Real-time serial communication support

