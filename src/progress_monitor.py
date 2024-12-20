import sys
from tqdm.auto import tqdm
import psutil
import GPUtil
from colorama import Fore, Style, Back, init
import os
import shutil
import time
from gpu_utils import GPU_AVAILABLE
from datetime import datetime

init(autoreset=True)

class AsciiArt:
    LOGO = f"""{Fore.CYAN}

     {Fore.GREEN}███████╗{Fore.YELLOW} █████╗ {Fore.RED}███╗   ██╗{Fore.MAGENTA}███╗   ██╗{Fore.CYAN}       
     {Fore.GREEN}██╔════╝{Fore.YELLOW}██╔══██╗{Fore.RED}████╗  ██║{Fore.MAGENTA}████╗  ██║{Fore.CYAN}       
     {Fore.GREEN}█████╗  {Fore.YELLOW}███████║{Fore.RED}██╔██╗ ██║{Fore.MAGENTA}██╔██╗ ██║{Fore.CYAN}       
     {Fore.GREEN}██╔══╝  {Fore.YELLOW}██╔══██║{Fore.RED}██║╚██╗██║{Fore.MAGENTA}██║╚██╗██║{Fore.CYAN}       
     {Fore.GREEN}██║     {Fore.YELLOW}██║  ██║{Fore.RED}██║ ╚████║{Fore.MAGENTA}██║ ╚████║{Fore.CYAN}       
     {Fore.GREEN}╚═╝     {Fore.YELLOW}╚═╝  ╚═╝{Fore.RED}╚═╝  ╚═══╝{Fore.MAGENTA}╚═╝  ╚═══╝{Fore.CYAN}       
       {Fore.WHITE}Fractal Artificial Neural Network{Fore.CYAN}

{Style.RESET_ALL}""".center(shutil.get_terminal_size().columns-2)

    SECTION_HEADER = f"""{Fore.CYAN}
╔══════════════════════════════════════════════════════════╗
║{{}}║
╚══════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

    PARAMS_BOX = f"""{Fore.BLUE}
\n{Fore.GREEN}┌{'─' * (shutil.get_terminal_size().columns-2)}┐
│{' Training Setup '.center(shutil.get_terminal_size().columns-2)}│
├{'─' * (shutil.get_terminal_size().columns-2)}┤
│                                                        
│  • Batch Size: {{}}                                    
│  • Epochs:     {{}}                                    
│  • Learning Rate: {{:<8}}                                                                                      
└{'─' * (shutil.get_terminal_size().columns-2)}┘{Style.RESET_ALL}
{Style.RESET_ALL}"""

class LiveDisplay:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.term_width = shutil.get_terminal_size().columns
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.5  # Increase minimum time between updates
        self.last_update_time = time.time()
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.current_layer = None
        self.layer_progress = 0.0
        self.last_epoch = 0
        self.last_batch = 0
        self.last_total_batches = 1  # Default to 1 to avoid division by zero
        self.batch_start_time = time.time()
        self.batch_times = []  # Store batch times for average calculation
        self.mini_logo = f"""{Fore.CYAN}╔{'═' * (self.term_width-2)}╗
║{' Neural Network Training Configuration '.center(self.term_width-2)}║
╚{'═' * (self.term_width-2)}╝{Style.RESET_ALL}
"""
        self.last_metrics = None  # Add this to store previous batch metrics
        self.last_layer_info = None  # Add this to store previous layer info
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        sys.stdout.flush()

    def show_logo(self):
        """Display the initial logo"""
        self.clear_screen()
        print(AsciiArt.LOGO)
        time.sleep(0.5)  # Give time to render
        sys.stdout.flush()

    def print_training_header(self, batch_size, epochs, learning_rate):
        """Print training session information in a fancy box"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # First show the mini logo
        print(self.mini_logo)
        
        # Print main header
        print(f"{Fore.CYAN}╔{'═' * (self.term_width-2)}╗")
        print(f"║{' Neural Network Training Configuration '.center(self.term_width-2)}║")
        print(f"╚{'═' * (self.term_width-2)}╝{Style.RESET_ALL}\n")
        
        # Show training configuration in similar style as training display
        print(f"{Fore.BLUE}┌{'─' * (self.term_width-2)}┐")
        print(f"│{' Training Parameters '.center(self.term_width-2)}│")
        print(f"├{'─' * (self.term_width-2)}┤")
        print(f"│ Start Time     : {current_time:<{self.term_width-20}}│")
        print(f"│ Batch Size     : {batch_size:<{self.term_width-20}}│")
        print(f"│ Total Epochs   : {epochs:<{self.term_width-20}}│")
        print(f"│ Learning Rate  : {learning_rate:<{self.term_width-20}.6f}│")
        print(f"└{'─' * (self.term_width-2)}┘{Style.RESET_ALL}")
        
        sys.stdout.flush()
        time.sleep(0.5)

    def print_system_info(self):
        """Print system information with fancy formatting"""
        try:
            # Get detailed system information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            print(f"\n{Fore.GREEN}┌{'─' * (self.term_width-2)}┐")
            print(f"│{' System Information '.center(self.term_width-2)}│")
            print(f"├{'─' * (self.term_width-2)}┤")
            
            # CPU Information
            print(f"│ CPU Usage    : {self._create_mini_bar(cpu_percent/100, Fore.RED)} ({cpu_count} cores)")
            if cpu_freq:
                print(f"│ CPU Freq     : {cpu_freq.current:.1f} MHz")
            
            # Memory Information
            print(f"│ RAM Usage    : {self._create_mini_bar(memory.percent/100, Fore.BLUE)} ({memory.used//1024//1024}MB/{memory.total//1024//1024}MB)")
            
            # GPU Information
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        print(f"├{'─' * (self.term_width-2)}┤")
                        print(f"│ GPU Model    : {gpu.name:<{self.term_width-20}}")
                        print(f"│ GPU Memory   : {self._create_mini_bar(gpu.memoryUtil, Fore.GREEN)} ({gpu.memoryUsed}MB/{gpu.memoryTotal}MB)")
                        print(f"│ GPU Load     : {self._create_mini_bar(gpu.load, Fore.YELLOW)} ({gpu.load*100:.1f}%)")
                        print(f"│ Temperature  : {self._create_temp_bar(gpu.temperature)} ({gpu.temperature}°C)")
                except Exception as e:
                    print(f"│ GPU Error: {str(e):<{self.term_width-15}}│")
            else:
                print(f"│ GPU Status   : Not Available{' ' * (self.term_width-25)}│")
                
            print(f"└{'─' * (self.term_width-2)}┘{Style.RESET_ALL}")
            
            sys.stdout.flush()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error getting system info: {str(e)}")

    def _create_temp_bar(self, temp, width=20):
        """Create a temperature bar with color gradient"""
        try:
            progress = min(1.0, temp / 100.0)  # Normalize to 0-1 range
            filled = int(width * progress)
            
            # Temperature color gradient
            if temp < 50:
                color = Fore.GREEN
            elif temp < 70:
                color = Fore.YELLOW
            else:
                color = Fore.RED
                
            bar = f"{color}{'█' * filled}{'░' * (width - filled)}{Style.RESET_ALL}"
            return f"[{bar}]"
            
        except Exception:
            return "[" + "?" * width + "]"

    def _create_mini_bar(self, progress, color, width=20):
        """Create a mini progress bar with percentage"""
        filled = int(width * progress)
        bar = f"{color}{'█' * filled}{'░' * (width - filled)}{Style.RESET_ALL}"
        return f"[{bar}] {progress*100:>5.1f}%"

    def update_display(self, epoch, batch, total_batches, metrics=None, layer_info=None):
        """Update the display with rate limiting"""
        current_time = time.time()
        
        # Only update if enough time has passed since last update
        if current_time - self.last_update_time < self.update_interval:
            return
            
        try:
            # Store valid values with safety checks
            self.last_epoch = max(1, epoch if epoch is not None else self.last_epoch)
            self.last_batch = max(0, batch if batch is not None else self.last_batch)
            self.last_total_batches = max(1, total_batches if total_batches is not None else self.last_total_batches)
            
            # Store metrics and layer info
            if metrics is not None:
                self.last_metrics = metrics
            if layer_info is not None:
                self.last_layer_info = layer_info
            
            # Update display efficiently
            self.clear_screen()
            self._print_header()
            self._print_progress_section(self.last_epoch, self.last_batch, self.last_total_batches)
            
            if self.last_layer_info:
                self._print_layer_section(self.last_layer_info)
            if self.last_metrics:
                self._print_metrics_section(self.last_metrics)
            
            self._print_system_status()
            sys.stdout.flush()
            
            self.last_update_time = current_time
            
        except Exception as e:
            print(f"Display update error: {str(e)}")

    def _print_header(self):
        print(f"{Fore.CYAN}╔{'═' * (self.term_width-2)}╗")
        print(f"║{' Neural Network Training Progress '.center(self.term_width-2)}║")
        print(f"╚{'═' * (self.term_width-2)}╝{Style.RESET_ALL}\n")

    def _print_progress_section(self, epoch, batch, total_batches):
        """Print progress with safety checks and better formatting"""
        try:
            # Calculate progress percentages
            epoch_progress = (epoch / self.total_epochs) * 100
            batch_progress = (batch / max(1, total_batches)) * 100
            
            # Calculate batch timing
            current_time = time.time()
            batch_time = current_time - self.batch_start_time if batch > 0 else 0
            
            if batch > 0:  # Only update timing for non-zero batch
                self.batch_times.append(batch_time)
                self.batch_start_time = current_time  # Update start time for next batch
            
            # Calculate averages
            recent_times = self.batch_times[-10:] if self.batch_times else [0]
            avg_batch_time = sum(recent_times) / len(recent_times)
            
            # Create progress bars
            epoch_bar = self._create_progress_bar(epoch, self.total_epochs, Fore.GREEN)
            batch_bar = self._create_progress_bar(batch, max(1, total_batches), Fore.BLUE)
            
            print(f"{Fore.YELLOW}┌{'─' * (self.term_width-2)}┐")
            print(f"│ Epoch: {epoch}/{self.total_epochs} ({epoch_progress:.1f}%)")
            print(f"│ {epoch_bar}")
            print(f"│ Batch: {batch}/{total_batches} ({batch_progress:.1f}%)")
            print(f"│ {batch_bar}")
            if batch > 0:
                print(f"│ Last Batch Time: {batch_time:.2f}s")
                print(f"│ Avg Batch Time: {avg_batch_time:.2f}s")
                remaining_batches = total_batches - batch
                eta = remaining_batches * avg_batch_time
                print(f"│ ETA: {eta:.1f}s")
            print(f"└{'─' * (self.term_width-2)}┘{Style.RESET_ALL}\n")
            
        except Exception as e:
            print(f"Error in progress section: {str(e)}")

    def _print_layer_section(self, layer_info):
        progress = layer_info['progress']
        layer_bar = self._create_gradient_bar(progress, 40)
        
        print(f"{Fore.MAGENTA}┌{'─' * (self.term_width-2)}┐")
        print(f"│ Current Layer: {layer_info['name']:<{self.term_width-20}}")
        print(f"│ Operation   : {layer_info['operation']:<{self.term_width-20}}")
        if 'shape' in layer_info:
            shape_str = f"Input: {layer_info['shape']}"
            print(f"│ Shape       : {shape_str:<{self.term_width-20}}")
        print(f"│ Progress    : {layer_bar}")
        print(f"└{'─' * (self.term_width-2)}┘{Style.RESET_ALL}\n")

    def _print_metrics_section(self, metrics):
        print(f"{Fore.CYAN}┌{'─' * (self.term_width-2)}┐")
        print("│ Current Metrics:")
        for name, values in metrics.items():
            current = values['current']
            best = values.get('best', current)
            avg = values.get('average', current)
            
            color = Fore.GREEN if current == best else Fore.YELLOW
            print(f"│ {name}: {color}{current:.4f}{Style.RESET_ALL} "
                  f"(Best: {Fore.BLUE}{best:.4f}{Style.RESET_ALL}, "
                  f"Avg: {Fore.YELLOW}{avg:.4f}{Style.RESET_ALL})")
        print(f"└{'─' * (self.term_width-2)}┘{Style.RESET_ALL}\n")

    def _create_gradient_bar(self, progress, width=40):
        filled = int(width * progress)
        gradient = ""
        colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
        
        for i in range(width):
            if i < filled:
                color_idx = min(len(colors)-1, int((i/width) * len(colors)))
                gradient += f"{colors[color_idx]}█{Style.RESET_ALL}"
            else:
                gradient += "░"
                
        return f"[{gradient}] {progress:.1%}"

    def _create_progress_bar(self, current, total, color):
        """Create progress bar with safety checks"""
        try:
            # Ensure valid numbers
            current = max(0, float(current))
            total = max(1, float(total))  # Prevent division by zero
            
            progress = min(1.0, current / total)  # Clamp to maximum of 100%
            width = 40
            filled = int(width * progress)
            
            bar = f"{color}{'█' * filled}{'░' * (width - filled)}{Style.RESET_ALL}"
            return f"[{bar}] {progress:.1%}"
            
        except Exception as e:
            return f"[{'?' * 40}] ???%"  # Fallback display

    def _print_system_status(self):
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            print(f"{Fore.WHITE}System Status:")
            print(f"CPU: {self._create_mini_bar(cpu_percent/100, Fore.RED)}")
            print(f"RAM: {self._create_mini_bar(memory.percent/100, Fore.BLUE)}")
            
            if GPU_AVAILABLE:
                gpu = GPUtil.getGPUs()[0]
                print(f"GPU: {self._create_mini_bar(gpu.memoryUtil, Fore.GREEN)}")
                print(f"GPU Temp: {gpu.temperature}°C")
        except Exception:
            pass

    def _create_mini_bar(self, progress, color, width=20):
        filled = int(width * progress)
        bar = f"{color}{'█' * filled}{'░' * (width - filled)}{Style.RESET_ALL}"
        return f"[{bar}] {progress:.1%}"

    def handle_error(self, error_msg):
        """Handle and display errors gracefully"""
        print(f"\n{Fore.RED}╔{'═' * (self.term_width-2)}╗")
        print(f"║{' ERROR '.center(self.term_width-2)}║")
        print(f"╠{'═' * (self.term_width-2)}╣")
        print(f"║{error_msg.center(self.term_width-2)}║")
        print(f"╚{'═' * (self.term_width-2)}╝{Style.RESET_ALL}")
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