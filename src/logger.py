import datetime
import os

class TrainingLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(
            log_dir, 
            f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def log_training_start(self, batch_size, epochs, learning_rate):
        """Log training parameters at start"""
        self.log(f"\nTraining Started")
        self.log(f"Batch Size: {batch_size}")
        self.log(f"Epochs: {epochs}")
        self.log(f"Learning Rate: {learning_rate}")
    
    def log_epoch_progress(self, current_epoch, total_epochs, metrics):
        """Log progress after each epoch"""
        self.log(f"\nEpoch {current_epoch}/{total_epochs}")
        for metric_name, value in metrics.items():
            self.log(f"{metric_name}: {value:.4f}")
