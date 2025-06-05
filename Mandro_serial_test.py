"""
sEMG Serial Monitor - Simple Serial Value Output

Code for real-time monitoring of raw data from sEMG devices
Can check data format, value range, connection status, etc.

Requirements:
- pyserial: pip install pyserial

Usage:
python Mandro_serial_test.py
"""

import serial
import serial.tools.list_ports
import time
import sys
from datetime import datetime

class SimpleSerialMonitor:
    """Simple Serial Monitor"""
    
    def __init__(self, port="COM7", baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_port = None
        
        # Statistics
        self.total_count = 0
        self.valid_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Data analysis
        self.min_values = [float('inf')] * 4
        self.max_values = [float('-inf')] * 4
        self.sum_values = [0] * 4
        
    def find_available_ports(self):
        """Find available COM ports"""
        ports = serial.tools.list_ports.comports()
        print("Available COM ports:")
        for port in ports:
            print(f"  - {port.device}: {port.description}")
        return [port.device for port in ports]
    
    def connect_serial(self, port=None):
        """Connect to serial port"""
        if port is None:
            port = self.port
            
        try:
            print(f"Connecting to {port} port...")
            self.serial_port = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(0.5)  # Wait for connection stabilization
            print(f"✓ {port} connection successful!")
            return True
        except Exception as e:
            print(f"✗ {port} connection failed: {e}")
            return False
    
    def read_and_display_data(self, max_count=None, display_interval=10):
        """Read and display data from serial port"""
        if not self.serial_port:
            print("Serial port not connected!")
            return
        
        print(f"\nStarting data monitoring... (Press Ctrl+C to stop)")
        print(f"Displaying every {display_interval} samples")
        print("=" * 80)
        
        self.start_time = time.time()
        
        try:
            while True:
                if max_count and self.total_count >= max_count:
                    break
                
                try:
                    # Read line
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        self.total_count += 1
                        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # Parse CSV data
                        try:
                            values = [float(val) for val in line.split(',')]
                            
                            if len(values) == 4:  # 4-channel EMG data
                                self.valid_count += 1
                                
                                # Update statistics
                                for i in range(4):
                                    self.min_values[i] = min(self.min_values[i], values[i])
                                    self.max_values[i] = max(self.max_values[i], values[i])
                                    self.sum_values[i] += values[i]
                                
                                # Display data periodically
                                if self.total_count % display_interval == 0:
                                    print(f"[{current_time}] Sample #{self.total_count:4d}: "
                                          f"CH1={values[0]:6.1f}, CH2={values[1]:6.1f}, "
                                          f"CH3={values[2]:6.1f}, CH4={values[3]:6.1f}")
                            else:
                                self.error_count += 1
                                print(f"[{current_time}] Invalid format (expected 4 values): {line}")
                                
                        except ValueError:
                            self.error_count += 1
                            print(f"[{current_time}] Parse error: {line}")
                    
                    # Brief pause to prevent excessive CPU usage
                    time.sleep(0.001)
                    
                except KeyboardInterrupt:
                    print("\n\nUser stop requested...")
                    break
                except Exception as e:
                    self.error_count += 1
                    print(f"Read error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Fatal error: {e}")
        
        finally:
            self.print_statistics()
    
    def print_statistics(self):
        """Print data statistics"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("DATA STATISTICS")
        print("=" * 80)
        print(f"Total samples: {self.total_count}")
        print(f"Valid samples: {self.valid_count}")
        print(f"Error samples: {self.error_count}")
        print(f"Success rate: {(self.valid_count/max(self.total_count,1)*100):.1f}%")
        print(f"Execution time: {elapsed_time:.1f}s")
        
        if self.valid_count > 0:
            print(f"Data rate: {self.valid_count/elapsed_time:.1f} samples/sec")
            
            print("\nCHANNEL STATISTICS:")
            for i in range(4):
                avg_val = self.sum_values[i] / self.valid_count
                print(f"  CH{i+1}: Min={self.min_values[i]:6.1f}, "
                      f"Max={self.max_values[i]:6.1f}, Avg={avg_val:6.1f}")
    
    def test_connection(self, test_duration=5):
        """Test connection for specified duration"""
        print(f"\nTesting connection for {test_duration} seconds...")
        
        start_time = time.time()
        test_count = 0
        
        while (time.time() - start_time) < test_duration:
            try:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    test_count += 1
                    try:
                        values = [float(val) for val in line.split(',')]
                        if len(values) == 4:
                            print(f"Test #{test_count}: VALID - {values}")
                        else:
                            print(f"Test #{test_count}: INVALID - {line}")
                    except ValueError:
                        print(f"Test #{test_count}: PARSE ERROR - {line}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Test error: {e}")
                break
        
        print(f"Connection test completed. Total samples: {test_count}")
    
    def auto_find_and_connect(self):
        """Automatically find and connect to sEMG device"""
        print("Auto-discovering sEMG device...")
        
        available_ports = self.find_available_ports()
        
        if not available_ports:
            print("No available COM ports found!")
            return False
        
        # First try default port
        if "COM7" in available_ports:
            if self.connect_serial("COM7"):
                self.port = "COM7"
                return True
        
        # Try each available port
        for port in available_ports:
            print(f"\nTrying {port}...")
            if self.connect_serial(port):
                # Test briefly to verify it's EMG data
                print(f"Testing {port} for EMG data...")
                self.test_connection(test_duration=2)
                
                # If successful, use this port
                if self.valid_count > 0:
                    self.port = port
                    print(f"✓ sEMG device found on {port}!")
                    return True
                else:
                    print(f"✗ {port} doesn't seem to have EMG data")
                    self.close()
        
        print("Failed to find sEMG device on any port")
        return False
    
    def close(self):
        """Close serial connection"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Serial connection closed.")


def main():
    print("=" * 60)
    print("sEMG Serial Monitor - Simple Data Display")
    print("=" * 60)
    
    monitor = SimpleSerialMonitor()
    
    try:
        # Auto-connect
        if not monitor.auto_find_and_connect():
            return
        
        # Display menu
        print("\nSelect mode:")
        print("1. Continuous monitoring")
        print("2. Connection test (5 seconds)")
        print("3. Limited sample monitoring (100 samples)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("\nStarting continuous monitoring...")
            monitor.read_and_display_data()
        
        elif choice == "2":
            print("\nStarting connection test...")
            monitor.test_connection(test_duration=5)
        
        elif choice == "3":
            print("\nStarting limited monitoring (100 samples)...")
            monitor.read_and_display_data(max_count=100, display_interval=5)
        
        else:
            print("Invalid choice. Defaulting to continuous monitoring...")
            monitor.read_and_display_data()
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    finally:
        monitor.close()
        print("Program terminated.")


if __name__ == "__main__":
    main() 