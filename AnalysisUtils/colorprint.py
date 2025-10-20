"""
ColorPrint Utility - Jupyter-friendly colored printing for data analysis
Just drop this file in your utils/ folder and import!

Usage:
    from utils.colorprint import cprint, success, error, warning, info
    
    cprint("Analysis complete!", "green")
    cprint(f"Accuracy: {accuracy:.2%}", "green", bold=True)
    success("Model training completed!")
"""

from IPython.display import HTML, display
import sys

def cprint(text, color='black', bg=None, bold=False, style=None):
    """
    Print colored text optimized for Jupyter notebooks and analysis workflows
    
    Args:
        text (str): Text to print - supports f-strings and any string formatting
        color (str): Text color - 'red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'black'
        bg (str): Background color - same options as color
        bold (bool): Make text bold for emphasis
        style (str): Predefined styles - 'success', 'error', 'warning', 'info', 'metric', 'result'
    
    Examples:
        cprint("Model accuracy: 95.2%", "green")
        cprint(f"Training completed: {epochs} epochs", "blue", bold=True)
        cprint("Data validation passed!", style="success")
        cprint("R² = 0.94", "white", bg="green", bold=True)
    """
    
    # Color palette optimized for data analysis readability
    color_map = {
        'red': '#e74c3c',      'green': '#27ae60',    'blue': '#3498db',
        'yellow': '#f1c40f',   'purple': '#9b59b6',   'magenta': '#9b59b6',
        'cyan': '#1abc9c',     'orange': '#e67e22',   'black': '#2c3e50',
        'white': '#ffffff',    'gray': '#7f8c8d',     'grey': '#7f8c8d',
        'pink': '#e91e63',     'brown': '#8d6e63',    'lime': '#8bc34a',
        'navy': '#34495e'
    }
    
    # Predefined styles for common analysis scenarios
    if style:
        style_configs = {
            'success': {'color': 'green', 'bold': True, 'prefix': '✅ '},
            'error': {'color': 'red', 'bold': True, 'prefix': '❌ '},
            'warning': {'color': 'orange', 'bold': True, 'prefix': '⚠️ '},
            'info': {'color': 'blue', 'bold': True, 'prefix': 'ℹ️ '},
            'metric': {'color': 'purple', 'bold': True, 'prefix': '📊 '},
            'result': {'color': 'green', 'bold': True, 'prefix': '🎯 '},
            'debug': {'color': 'gray', 'bold': False, 'prefix': '🐛 '},
            'note': {'color': 'cyan', 'bold': False, 'prefix': '📝 '}
        }
        
        if style in style_configs:
            config = style_configs[style]
            color = config.get('color', color)
            bold = config.get('bold', bold)
            text = config.get('prefix', '') + text
    
    # Get colors with fallbacks
    text_color = color_map.get(color.lower(), color_map['black'])
    bg_color = color_map.get(bg.lower(), 'transparent') if bg else 'transparent'
    
    # Build CSS styles
    styles = [
        f"color: {text_color}",
        f"background-color: {bg_color}",
        "font-family: 'Consolas', 'Monaco', 'Courier New', monospace"
    ]
    
    if bold:
        styles.append("font-weight: bold")
    
    if bg:
        styles.append("padding: 2px 6px")
        styles.append("border-radius: 3px")
    
    # Create and display HTML
    css_style = "; ".join(styles)
    html = f'<span style="{css_style}">{text}</span>'
    
    try:
        if 'ipykernel' in sys.modules:
            display(HTML(html))
        else:
            print(text)  # Fallback for non-Jupyter
    except:
        print(text)  # Final fallback


# Convenience functions for common analysis tasks
def success(text):
    """Print success message - perfect for completed analysis steps"""
    cprint(text, style='success')

def error(text):
    """Print error message - for failed operations or data issues"""
    cprint(text, style='error')

def warning(text):
    """Print warning message - for data quality issues or important notes"""
    cprint(text, style='warning')

def info(text):
    """Print info message - for general status updates"""
    cprint(text, style='info')

def metric(text):
    """Print metric/result - for displaying key analysis results"""
    cprint(text, style='metric')

def result(text):
    """Print final result - for highlighting main findings"""
    cprint(text, style='result')

def debug(text):
    """Print debug message - for troubleshooting"""
    cprint(text, style='debug')

def note(text):
    """Print note - for additional context"""
    cprint(text, style='note')

def highlight(text, color='yellow'):
    """Print highlighted text with background - for emphasis"""
    cprint(text, color='black', bg=color, bold=True)


# Analysis-specific helper functions
def print_dataframe_info(df, name="DataFrame"):
    """Print formatted DataFrame information"""
    info(f"{name} Info:")
    cprint(f"Shape: {df.shape}", "blue")
    cprint(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", "blue")
    if df.isnull().sum().sum() > 0:
        warning(f"Missing values: {df.isnull().sum().sum()}")
    else:
        success("No missing values")

def print_model_results(model_name, accuracy, **metrics):
    """Print formatted model results"""
    result(f"Model: {model_name}")
    cprint(f"Accuracy: {accuracy:.4f}", "green", bold=True)
    for name, value in metrics.items():
        if isinstance(value, float):
            cprint(f"{name}: {value:.4f}", "blue")
        else:
            cprint(f"{name}: {value}", "blue")

def print_progress(current, total, message="Processing"):
    """Print progress update"""
    percentage = (current / total) * 100
    color = "green" if percentage == 100 else "blue"
    cprint(f"{message}: {current}/{total} ({percentage:.1f}%)", color)


# Quick demo function
def demo():
    """Demo all the colorprint functions"""
    cprint("=== ColorPrint Demo ===", "purple", bold=True)
    
    cprint("Basic colors:", "black", bold=True)
    cprint("Red text", "red")
    cprint("Green text", "green") 
    cprint("Blue bold text", "blue", bold=True)
    
    cprint("\nPredefined styles:", "black", bold=True)
    print_success("Operation successful!")
    print_error("An error occurred!")
    print_warning("Warning message")
    print_info("Information message")
    print_metric("Accuracy: 95.2%")
    print_result("Best model found!")
    
    cprint("\nF-string example:", "black", bold=True)
    name, score = "Alice", 0.952
    cprint(f"User {name} achieved {score:.1%} accuracy", "green", bold=True)
    
    highlight("This is highlighted!", "yellow")


if __name__ == "__main__":
    demo()