import os
from datetime import datetime
import psutil
import GPUtil
class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"training_log_{self.timestamp}.txt")
        self.start_time = datetime.now()
        
        # Create header in log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training Log - Started at {self.timestamp}\n")
            f.write("="*50 + "\n\n")
    
    def log_message(self, message, show_timestamp=True, print_to_console=True):
        """Log a message with optional timestamp"""
        timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " if show_timestamp else ""
        formatted_msg = f"{timestamp}{message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + '\n')
            
        if print_to_console:
            print(message)

    def log_system_info(self):
        """Log detailed system information"""
        def get_size_str(bytes):
            gb = bytes / (1024**3)
            return f"{gb:.1f}GB"

        # Memory Information
        memory = psutil.virtual_memory()
        total_ram = get_size_str(memory.total)
        used_ram = get_size_str(memory.used)
        available_ram = get_size_str(memory.available)

        self.log_message("\n╔══════════════ System Information ══════════════╗")
        
        # CPU Information
        cpu_info = {
            'CPU Usage': f"{psutil.cpu_percent()}%",
            'CPU Cores': psutil.cpu_count(),
            'RAM Total': total_ram,
            'RAM Used': used_ram,
            'RAM Available': available_ram,
            'RAM Usage': f"{memory.percent}%"
        }
        
        for key, value in cpu_info.items():
            self.log_message(f"║ {key:<15}: {value:>30} ║")
        
        # GPU Information
        if GPUtil.getGPUs():
            gpu = GPUtil.getGPUs()[0]
            gpu_info = {
                'GPU Model': gpu.name,
                'GPU Memory': f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB",
                'GPU Usage': f"{gpu.load*100:.1f}%",
                'GPU Memory %': f"{gpu.memoryUtil*100:.1f}%",
                'Temperature': f"{gpu.temperature}°C"
            }
            
            self.log_message("║" + "─"*48 + "║")
            for key, value in gpu_info.items():
                self.log_message(f"║ {key:<15}: {value:>30} ║")
        
        self.log_message("╚" + "═"*48 + "╝")
    
    def log_training_start(self, batch_size, epochs, learning_rate):
        """Log training start with parameters"""
        self.log_message("\n╔══════════════ Training Session Started ══════════════╗")
        
        # Log training parameters
        params = {
            'Batch Size': batch_size,
            'Epochs': epochs,
            'Learning Rate': f"{learning_rate:.6f}",
            'Start Time': self.start_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for key, value in params.items():
            self.log_message(f"║ {key:<15}: {value:>33} ║")
        
        self.log_message("╚" + "═"*52 + "╝\n")
        
        # Log system state at start
        self.log_system_info()
    
    def log_epoch_progress(self, epoch, epochs, metrics):
        """Log epoch metrics with proper formatting"""
        self.log_message(f"\n{'─'*20} Epoch {epoch}/{epochs} {'─'*20}")
        
        for name, value in metrics.items():
            if 'loss' in name.lower():
                self.log_message(f"{name:>15}: {value:.4f}")
            else:
                self.log_message(f"{name:>15}: {value:.2%}")
    
    def log_batch_progress(self, batch_idx, total_batches, metrics):
        """Log batch progress (to file only)"""
        msg = (f"Batch {batch_idx:>4}/{total_batches} - " + 
               " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
        self.log_message(msg, print_to_console=False)
    
    def log_training_complete(self, best_accuracy):
        """Log training completion with summary"""
        duration = datetime.now() - self.start_time
        hours = duration.total_seconds() // 3600
        minutes = (duration.total_seconds() % 3600) // 60
        seconds = duration.total_seconds() % 60
        
        self.log_message("\n╔══════════════ Training Complete ══════════════╗")
        summary = {
            'Duration': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            'Best Accuracy': f"{best_accuracy:.2%}",
            'End Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for key, value in summary.items():
            self.log_message(f"║ {key:<15}: {value:>30} ║")
        
        self.log_message("╚" + "═"*48 + "╝")
        
        # Log final system state
        self.log_system_info()
