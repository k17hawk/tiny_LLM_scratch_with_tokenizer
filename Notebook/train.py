import sentencepiece as spm
import threading
import time
import psutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class SPMTrainingMonitor:
    def __init__(self):
        self.training_complete = False
        self.cpu_history = []
        self.time_history = []
        self.start_time = None
        
    def monitor_cpu(self):
        """Monitor CPU usage in a separate thread"""
        while not self.training_complete:
            cpu_percent = psutil.cpu_percent(interval=1)
            elapsed = time.time() - self.start_time
            self.cpu_history.append(cpu_percent)
            self.time_history.append(elapsed)
            time.sleep(1)
    
    def estimate_progress(self, log_file='ne_spm.log'):
        """Estimate progress by monitoring log file size or output"""
        # SentencePiece doesn't provide direct progress, so we monitor activity
        if os.path.exists(log_file):
            return os.path.getsize(log_file)
        return 0
    
    def train_with_monitoring(self, **train_params):
        """Train SentencePiece with live monitoring"""
        self.start_time = time.time()
        
        # Start CPU monitoring thread
        cpu_thread = threading.Thread(target=self.monitor_cpu, daemon=True)
        cpu_thread.start()
        
        # Set up live plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('SentencePiece Training Monitor', fontsize=14, fontweight='bold')
        
        def update_plot(frame):
            if not self.training_complete and len(self.cpu_history) > 0:
                # CPU Usage plot
                ax1.clear()
                ax1.plot(self.time_history, self.cpu_history, 'b-', linewidth=2)
                ax1.fill_between(self.time_history, self.cpu_history, alpha=0.3)
                ax1.set_ylabel('CPU Usage (%)', fontsize=11)
                ax1.set_xlabel('Time (seconds)', fontsize=11)
                ax1.set_title('Live CPU Usage', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([0, 100])
                
                # Stats display
                ax2.clear()
                ax2.axis('off')
                
                elapsed = time.time() - self.start_time
                avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
                current_cpu = self.cpu_history[-1]
                
                stats_text = f"""
                Training Status: {'Complete ✓' if self.training_complete else 'In Progress...'}
                
                Elapsed Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)
                Current CPU: {current_cpu:.1f}%
                Average CPU: {avg_cpu:.1f}%
                Peak CPU: {max(self.cpu_history):.1f}%
                
                Memory Usage: {psutil.virtual_memory().percent:.1f}%
                Active Threads: {threading.active_count()}
                """
                
                ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round',
                        facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
        
        # Start animation
        anim = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
        plt.ion()
        plt.show()
        
        # Start training in separate thread
        def train():
            print("Starting SentencePiece training...")
            print(f"Parameters: {train_params}")
            print("-" * 50)
            
            try:
                spm.SentencePieceTrainer.train(**train_params)
                print("\n" + "="*50)
                print("Training complete!")
                print("="*50)
            except Exception as e:
                print(f"Error during training: {e}")
            finally:
                self.training_complete = True
        
        train_thread = threading.Thread(target=train, daemon=True)
        train_thread.start()
        
        # Keep plot alive until training completes
        try:
            while not self.training_complete:
                plt.pause(0.1)
            
            # Show final plot for 5 seconds
            plt.ioff()
            update_plot(None)
            plt.show(block=False)
            plt.pause(5)
            
        except KeyboardInterrupt:
            print("\nMonitoring interrupted by user")
            self.training_complete = True
        
        train_thread.join(timeout=5)

# Usage
if __name__ == "__main__":
    monitor = SPMTrainingMonitor()
    
    training_params = {
        'input': 'ne.txt',
        'model_prefix': 'ne_spm',
        'vocab_size': 16000,
        'character_coverage': 1.0,
        'model_type': 'bpe',
        'input_sentence_size': 10000000,
        'shuffle_input_sentence': True,
        'train_extremely_large_corpus': True,
        'num_threads': 16
    }
    
    monitor.train_with_monitoring(**training_params)