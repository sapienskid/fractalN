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
        self.update_interval = 0.1  # Update every 100ms
        
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
    
    def update_display(self, epoch, batch, total_batches, metrics, layer_info=None):
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        self.clear_screen()
        
        # Print logo
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
    def __init__(self, total_epochs):
        self.display = LiveDisplay(total_epochs)
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.term_width = shutil.get_terminal_size().columns
        self.best_metrics = {
            'train_acc': 0,
            'val_acc': 0,
            'train_loss': float('inf'),
            'val_loss': float('inf')
        }
        self.current_layer_info = None
        self.metrics_history = {
            'train_loss': deque(maxlen=100),
            'train_acc': deque(maxlen=100),
            'val_loss': deque(maxlen=100),
            'val_acc': deque(maxlen=100)
        }
        self.current_batch = 0
        self.total_batches = 0
        self.current_metrics = {}  # Add this to store current metrics
    
    def print_header(self):
        """Print the initial header with logo"""
        print(AsciiArt.LOGO)
        if GPU_AVAILABLE:
            print(f"{Fore.GREEN}GPU Mode: Enabled{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}GPU Mode: Disabled (Using CPU){Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * self.term_width}{Style.RESET_ALL}")
    
    def print_section_header(self, text):
        """Print a beautifully formatted section header"""
        header = AsciiArt.SECTION_HEADER.format(text.center(self.term_width - 4))
        print(header)
    
    def print_training_params(self, batch_size, epochs, learning_rate):
        """Print training parameters in a styled box"""
        params_box = AsciiArt.PARAMS_BOX.format(batch_size, epochs, learning_rate)
        print(params_box)
    
    def print_epoch_header(self, epoch):
        self.current_epoch = epoch
        print(f"\n{Fore.CYAN}{'=' * self.term_width}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Epoch {Fore.GREEN}{epoch}/{self.total_epochs}{Style.RESET_ALL}".center(self.term_width))
        print(f"{Fore.CYAN}{'=' * self.term_width}{Style.RESET_ALL}\n")
    
    def update_layer_info(self, layer_info):
        """Update current layer information"""
        self.current_layer_info = layer_info
        if not hasattr(self, 'current_metrics'):
            self.current_metrics = {}
        if not hasattr(self, 'current_batch'):
            self.current_batch = 0
        if not hasattr(self, 'total_batches'):
            self.total_batches = 0
            
        self.display.update_display(
            self.current_epoch,
            self.current_batch,
            self.total_batches,
            self.current_metrics,
            layer_info
        )
    
    def update_batch_progress(self, batch_idx, total_batches, metrics):
        """Update batch and sample progress with metrics"""
        self.current_batch = batch_idx
        self.total_batches = total_batches
        
        # Initialize current_metrics if needed
        if not hasattr(self, 'current_metrics'):
            self.current_metrics = {}
        
        # Update metrics history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = deque(maxlen=100)
            self.metrics_history[name].append(value)
        
        # Calculate averages
        avg_metrics = {name: np.mean(values) for name, values in self.metrics_history.items()}
        self.current_metrics = {
            name: {
                'current': value,
                'average': avg_metrics[name],
                'best': self.best_metrics.get(name, value)
            } 
            for name, value in metrics.items()
        }
        
        self.display.update_display(
            self.current_epoch,
            batch_idx,
            total_batches,
            self.current_metrics,
            self.current_layer_info
        )
    
    def print_metrics(self, current_metrics):
        """Display current and best metrics"""
        print(f"\n\n{Fore.CYAN}Training Statistics:{Style.RESET_ALL}")
        
        headers = [f"{Fore.WHITE}Metric", f"{Fore.WHITE}Current", f"{Fore.WHITE}Best", f"{Fore.WHITE}Trend"]
        print(f"{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
        print(f"{headers[0]:<15} {headers[1]:>12} {headers[2]:>12} {headers[3]:>12}")
        print(f"{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
        
        metrics_info = [
            ('Train Loss', current_metrics['train_loss'], self.best_metrics['train_loss'], 'loss'),
            ('Train Acc', current_metrics['train_acc'], self.best_metrics['train_acc'], 'acc'),
            ('Val Loss', current_metrics['val_loss'], self.best_metrics['val_loss'], 'loss'),
            ('Val Acc', current_metrics['val_acc'], self.best_metrics['val_acc'], 'acc')
        ]
        
        for name, current, best, metric_type in metrics_info:
            if metric_type == 'loss':
                is_better = current < best
                print(f"{Fore.WHITE}{name:<15} "
                      f"{Fore.GREEN if is_better else Fore.RED}{current:>12.4f} "
                      f"{Fore.YELLOW}{best:>12.4f} "
                      f"{Fore.GREEN if is_better else Fore.RED}{'↓' if is_better else '↑':>12}{Style.RESET_ALL}")
            else:
                is_better = current > best
                print(f"{Fore.WHITE}{name:<15} "
                      f"{Fore.GREEN if is_better else Fore.RED}{current:>12.2%} "
                      f"{Fore.YELLOW}{best:>12.2%} "
                      f"{Fore.GREEN if is_better else Fore.RED}{'↑' if is_better else '↓':>12}{Style.RESET_ALL}")
        
        print(f"{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")

class EnhancedBatchProgress:
    def __init__(self, total_batches, total_samples, desc="Processing", leave=True):
        self.total_samples = total_samples
        self.samples_processed = 0
        
        # Simpler bar format that works with tqdm
        self.bar_format = (
            '{desc}: {percentage:3.0f}%|{bar}| '
            '{n_fmt}/{total_fmt} '
            '[{elapsed}<{remaining}, {rate_fmt}] '
            '({postfix})'
        )
        
        # Add immediate feedback
        print(f"\n{desc} starting... Total batches: {total_batches}")
        
        self.progress_bar = tqdm(
            total=total_batches,
            desc=desc,
            leave=leave,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                      '[{elapsed}<{remaining}] [{postfix}]',
            ascii=" ▏▎▍▌▋▊▉█",
            ncols=100,
            dynamic_ncols=True,
            smoothing=0.3,
            mininterval=0.1,  # Update more frequently
            unit='batch'
        )
        
        self.metrics = deque(maxlen=10)
        self.last_value = {}  # Store last values for trend indicators
        
        # Add accuracy to postfix format
        self.postfix_fmt = (
            "loss: {loss} • acc: {acc} • "
            "samples: {samples}"
        )
    
    def format_metric(self, name, value):
        """Format metrics with colors based on type"""
        if 'loss' in name.lower():
            if value < 0.5:
                prefix = "🟢"  # Green circle for good loss
            elif value < 1.0:
                prefix = "🟡"  # Yellow circle for okay loss
            else:
                prefix = "🔴"  # Red circle for bad loss
            return f"{prefix} {value:.4f}"
        else:
            if value > 0.9:
                prefix = "🌟"  # Star for exceptional accuracy
            elif value > 0.8:
                prefix = "🟢"  # Green for good accuracy
            elif value > 0.6:
                prefix = "🟡"  # Yellow for okay accuracy
            else:
                prefix = "🔴"  # Red for poor accuracy
            # Add percentage and trend indicators
            trend = "↗️" if value > self.last_value[name] else "↘️" if value < self.last_value[name] else "➡️"
            self.last_value[name] = value
            return f"{prefix} {value:.2%} {trend}"
    
    def update(self, samples_in_batch, **metrics):
        self.samples_processed += samples_in_batch
        self.metrics.append(metrics)
        
        # Format current metrics
        current_metrics = {
            k: f"{v:.4f}" if 'loss' in k else f"{v:.2%}"
            for k, v in metrics.items()
        }
        
        # Add batch counter
        current_metrics['batch'] = f"{self.progress_bar.n + 1}/{self.progress_bar.total}"
        
        # Add sample progress
        current_metrics['samples'] = f"{self.samples_processed}/{self.total_samples}"
        
        # Update progress bar
        self.progress_bar.set_postfix(current_metrics, refresh=True)
        self.progress_bar.update(1)
    
    def close(self):
        self.progress_bar.close()
