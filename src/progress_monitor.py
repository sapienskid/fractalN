import sys
from tqdm.auto import tqdm
import psutil
import GPUtil
from colorama import Fore, Style, Back, init
import numpy as np
from collections import deque
import shutil
import time
from gpu_utils import GPU_AVAILABLE
from datetime import datetime

init(autoreset=True)

class AsciiArt:
    LOGO = f"""{Fore.CYAN}
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃     {Fore.GREEN}███████╗{Fore.YELLOW} █████╗ {Fore.RED}███╗   ██╗{Fore.MAGENTA}███╗   ██╗{Fore.CYAN}       ┃
┃     {Fore.GREEN}██╔════╝{Fore.YELLOW}██╔══██╗{Fore.RED}████╗  ██║{Fore.MAGENTA}████╗  ██║{Fore.CYAN}       ┃
┃     {Fore.GREEN}█████╗  {Fore.YELLOW}███████║{Fore.RED}██╔██╗ ██║{Fore.MAGENTA}██╔██╗ ██║{Fore.CYAN}       ┃
┃     {Fore.GREEN}██╔══╝  {Fore.YELLOW}██╔══██║{Fore.RED}██║╚██╗██║{Fore.MAGENTA}██║╚██╗██║{Fore.CYAN}       ┃
┃     {Fore.GREEN}██║     {Fore.YELLOW}██║  ██║{Fore.RED}██║ ╚████║{Fore.MAGENTA}██║ ╚████║{Fore.CYAN}       ┃
┃     {Fore.GREEN}╚═╝     {Fore.YELLOW}╚═╝  ╚═╝{Fore.RED}╚═╝  ╚═══╝{Fore.MAGENTA}╚═╝  ╚═══╝{Fore.CYAN}       ┃
┃       {Fore.WHITE}Fractal Artificial Neural Network{Fore.CYAN}         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
{Style.RESET_ALL}"""

    SECTION_HEADER = f"""{Fore.CYAN}
╔══════════════════════════════════════════════════════════╗
║{{}}║
╚══════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

    PARAMS_BOX = f"""{Fore.BLUE}
┌──────────────────── Training Setup ────────────────────┐
│                                                        │
│  • Batch Size: {{}}                                      │
│  • Epochs:     {{}}                                      │
│  • Learning Rate: {{:<8}}                             │
│                                                        │
└────────────────────────────────────────────────────────┘
{Style.RESET_ALL}"""

class LiveDisplay:
    def __init__(self, total_epochs):
        self.term_width = shutil.get_terminal_size().columns
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1
        self.show_logo()
    
    def show_logo(self):
        """Display the initial logo"""
        self.clear_screen()
        print(AsciiArt.LOGO)
        sys.stdout.flush()
        
    def clear_screen(self):
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    
    def create_progress_bar(self, current, total, width=40, color=Fore.GREEN):
        percentage = current / total
        filled = int(width * percentage)
        
        # Create gradient effect
        gradient_chars = "░▒▓█"
        bar = ""
        for i in range(width):
            if i < filled:
                char_index = min(3, int((i / filled) * 4))
                bar += f"{color}{gradient_chars[char_index]}{Style.RESET_ALL}"
            else:
                bar += "░"
        
        return f"{Fore.BLUE}▕{bar}{Fore.BLUE}▏ {Fore.YELLOW}{percentage:>7.1%}{Style.RESET_ALL}"
    
    def format_metrics(self, metrics):
        """Format metrics with colors based on values"""
        formatted = []
        for name, value in metrics.items():
            if 'loss' in name.lower():
                color = Fore.GREEN if value < 0.5 else Fore.YELLOW if value < 1.0 else Fore.RED
                formatted.append(f"{name}: {color}{value:.4f}{Style.RESET_ALL}")
            else:
                color = Fore.GREEN if value > 0.8 else Fore.YELLOW if value > 0.6 else Fore.RED
                formatted.append(f"{name}: {color}{value:.2%}{Style.RESET_ALL}")
        return "  ".join(formatted)
    
    def create_layer_progress(self, layer_info):
        """Create a fancy layer progress indicator"""
        if not layer_info:
            return ""
        
        name = layer_info['name']
        current = layer_info['index']
        total = layer_info['total']
        progress = layer_info['progress']
        
        # Create gradient progress bar for layer
        width = 30
        filled = int(width * (current / total))
        gradient = ""
        colors = [Fore.GREEN, Fore.YELLOW, Fore.BLUE]
        
        for i in range(width):
            if i < filled:
                color_idx = min(len(colors)-1, int((i/filled) * len(colors)))
                gradient += f"{colors[color_idx]}█{Style.RESET_ALL}"
            else:
                gradient += "░"
        
        return f"{Fore.CYAN}Layer {current}/{total} [{gradient}] {Fore.YELLOW}{name}{Style.RESET_ALL}"
    
    def print_system_info(self):
        """Print system information in a fancy box"""
        cpu_percent = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        print(f"\n{Fore.CYAN}╔══════════════ System Information ══════════════╗")
        print(f"║ CPU Usage      : {cpu_percent:>24.1f}% ║")
        print(f"║ CPU Cores      : {cpu_count:>24d} ║")
        print(f"║ RAM Total      : {memory.total/1e9:>23.1f}GB ║")
        print(f"║ RAM Used       : {memory.used/1e9:>23.1f}GB ║")
        print(f"║ RAM Available  : {memory.available/1e9:>23.1f}GB ║")
        print(f"║ RAM Usage      : {memory.percent:>23.1f}% ║")
        
        if GPU_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                print(f"║────────────────────────────────────────────────║")
                print(f"║ GPU Model      : {gpu.name:>24s} ║")
                print(f"║ GPU Memory     : {gpu.memoryUsed:>8.1f}MB / {gpu.memoryTotal:>8.1f}MB ║")
                print(f"║ GPU Usage      : {gpu.load*100:>23.1f}% ║")
                print(f"║ GPU Memory %   : {gpu.memoryUtil*100:>23.1f}% ║")
                print(f"║ Temperature    : {gpu.temperature:>23.1f}°C ║")
            except Exception:
                pass
        print(f"╚════════════════════════════════════════════════╝\n{Style.RESET_ALL}")

    def print_training_header(self, batch_size, epochs, learning_rate):
        """Print training session information"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{Fore.CYAN}╔══════════════ Training Session Started ══════════════╗")
        print(f"║ Batch Size     : {batch_size:>33d} ║")
        print(f"║ Epochs         : {epochs:>33d} ║")
        print(f"║ Learning Rate  : {learning_rate:>33.6f} ║")
        print(f"║ Start Time     : {current_time:>33s} ║")
        print(f"╚════════════════════════════════════════════════════╝\n{Style.RESET_ALL}")

    def update_display(self, epoch, batch, total_batches, metrics, layer_info=None):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        self.clear_screen()
        
        # Always show logo first
        print(AsciiArt.LOGO)
        
        # Print epoch progress
        epoch_progress = self.create_progress_bar(epoch, self.total_epochs)
        print(f"\n{Fore.WHITE}Epoch Progress: {epoch_progress}{Style.RESET_ALL}")
        
        # Print batch progress
        batch_progress = self.create_progress_bar(batch, total_batches)
        print(f"{Fore.WHITE}Batch Progress: {batch_progress}{Style.RESET_ALL}")
        
        # Print current layer if available
        if layer_info:
            print(f"\n{Fore.YELLOW}Current Layer: {Fore.WHITE}{layer_info}{Style.RESET_ALL}")
        
        # Print current layer progress
        if layer_info:
            print("\n" + self.create_layer_progress(layer_info))
            
        # Print detailed metrics table
        if metrics:
            print(f"\n{Fore.CYAN}Performance Metrics:{Style.RESET_ALL}")
            headers = ["Metric", "Current", "Average", "Best", "Trend"]
            print(f"{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
            print(f"{headers[0]:<15} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12} {headers[4]:>12}")
            print(f"{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
            
            for name, values in metrics.items():
                current = values.get('current', 0)
                avg = values.get('average', 0)
                best = values.get('best', 0)
                
                if 'loss' in name.lower():
                    color = Fore.GREEN if current < best else Fore.RED
                    trend = "↓" if current < best else "↑"
                    print(f"{Fore.WHITE}{name:<15} "
                          f"{color}{current:>12.4f} "
                          f"{Fore.YELLOW}{avg:>12.4f} "
                          f"{Fore.BLUE}{best:>12.4f} "
                          f"{color}{trend:>12}{Style.RESET_ALL}")
                else:
                    color = Fore.GREEN if current > best else Fore.RED
                    trend = "↑" if current > best else "↓"
                    print(f"{Fore.WHITE}{name:<15} "
                          f"{color}{current:>12.2%} "
                          f"{Fore.YELLOW}{avg:>12.2%} "
                          f"{Fore.BLUE}{best:>12.2%} "
                          f"{color}{trend:>12}{Style.RESET_ALL}")
            
            print(f"{Fore.WHITE}{'=' * 80}{Style.RESET_ALL}")
        
        # Print GPU status if available
        if GPU_AVAILABLE:
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_usage = gpu.load * 100
                gpu_memory = self.create_progress_bar(gpu.memoryUsed, gpu.memoryTotal, color=Fore.MAGENTA)
                print(f"\n{Fore.CYAN}GPU Usage: {gpu_usage:.1f}%")
                print(f"GPU Memory: {gpu_memory}{Style.RESET_ALL}")
                print(f"Temperature: {Fore.YELLOW}{gpu.temperature}°C{Style.RESET_ALL}")
            except Exception:
                pass
        
        # Print elapsed time
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(f"\n{Fore.WHITE}Elapsed Time: {Fore.YELLOW}{elapsed_str}{Style.RESET_ALL}")
        
        sys.stdout.flush()

# Update EnhancedProgressMonitor to use LiveDisplay
class EnhancedProgressMonitor:
    """Enhanced progress monitor for training process"""
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_batch = 0
        self.total_batches = 0
        self.epoch_start_time = None
        self.train_bar = None
        self.val_bar = None
        
    def print_epoch_header(self, epoch):
        """Print formatted epoch header"""
        print("\n" + "="*70)
        print(f"{Fore.CYAN}Epoch {epoch}/{self.total_epochs}".center(70))
        print("="*70)
        self.epoch_start_time = time.time()
        
    def print_metrics(self, metrics):
        """Print formatted metrics"""
        duration = time.time() - self.epoch_start_time
        
        print("\n" + "-"*70)
        print(f"{Fore.YELLOW}Training Metrics:")
        print(f"  Loss: {metrics['train_loss']:.4f}")
        print(f"  Accuracy: {metrics['train_acc']:.4f}")
        
        print(f"\n{Fore.GREEN}Validation Metrics:")
        print(f"  Loss: {metrics['val_loss']:.4f}")
        print(f"  Accuracy: {metrics['val_acc']:.4f}")
        
        print(f"\n{Fore.BLUE}Time: {duration:.2f}s")
        print("-"*70 + "\n")
        sys.stdout.flush()

class EnhancedBatchProgress:
    """Enhanced progress bar for batch processing"""
    def __init__(self, total_batches, total_samples, desc=""):
        self.total_batches = total_batches
        self.total_samples = total_samples
        self.desc = desc
        self.pbar = tqdm(
            total=total_samples,
            desc=desc,
            bar_format="{desc}: {percentage:3.0f}%|{bar:30}| "
                      "{n_fmt}/{total_fmt} samples "
                      "[{elapsed}<{remaining}, {rate_fmt}]"
        )
        self.current_metrics = {}
        
    def update(self, samples_in_batch, loss=None, acc=None, batch=None):
        """Update progress bar with new metrics"""
        self.pbar.update(samples_in_batch)
        
        # Update metrics display
        metrics_str = []
        if loss is not None:
            metrics_str.append(f"loss: {loss:.4f}")
        if acc is not None:
            metrics_str.append(f"acc: {acc:.4f}")
        if batch is not None:
            metrics_str.append(f"batch: {batch}/{self.total_batches}")
            
        if metrics_str:
            self.pbar.set_postfix_str(" - ".join(metrics_str))
        
    def close(self):
        """Close progress bar"""
        self.pbar.close()

def print_gpu_status(gpu_info):
    """Print formatted GPU status"""
    if gpu_info:
        print(f"\n{Fore.CYAN}GPU Status:")
        print(f"  Device: {gpu_info['name']}")
        print(f"  Memory: {gpu_info['total_memory']/1e9:.1f}GB")
        print(f"  CUDA: {gpu_info['cuda_version']}")
    else:
        print(f"\n{Fore.YELLOW}GPU not available")
