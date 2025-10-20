import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def fade_colormap(left_color="blue", right_color="red", mid_color="white", vmin=-1, vmax=1, 
                  n_colors=256, power=1.7, name="custom_fade"):
    """
    Create a colormap that fades from left_color through mid_color to right_color.

    Parameters:
    -----------
    left_color : str or tuple
        Color for the minimum value (left side of colormap)
    right_color : str or tuple  
        Color for the maximum value (right side of colormap)
    vmin : float
        Minimum value for the colormap normalization
    vmax : float
        Maximum value for the colormap normalization
    n_colors : int
        Total number of color steps in the colormap (default: 256)
    power : float
        Power for nonlinear fade (power > 1 = stronger fade to white, default: 1.7)
    name : str
        Name for the colormap (default: "custom_fade")
        
    Returns:
    --------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The generated colormap
    norm : matplotlib.colors.Normalize
        Normalization object for the given vmin/vmax
    """

    def fade_to_color(color, n=128, power=2.0, mid_color="white"):
        """Return n shades fading from mid_color to the given color."""
        rgb = np.array(mcolors.to_rgb(color))
        mid = np.array(mcolors.to_rgb(mid_color))
        steps = np.linspace(0, 1, n)[:, None]
        steps = steps**power

        return (1 - steps) * mid + steps * rgb

    # Determine how to split colors based on the range
    range_total = vmax - vmin
    
    if vmin >= 0:
        # All positive range - fade from mid_color to right_color
        colors = fade_to_color(right_color, n_colors, power, mid_color)
        positions = np.linspace(0, 1, n_colors)
        
    elif vmax <= 0:
        # All negative range - fade from left_color to mid_color
        left_fade = fade_to_color(left_color, n_colors, power, mid_color)
        colors = left_fade[::-1]  # reverse to go from color to mid_color
        positions = np.linspace(0, 1, n_colors)
        
    else:
        # Mixed range - need both sides
        # Calculate proportions
        neg_proportion = abs(vmin) / range_total
        pos_proportion = vmax / range_total
        
        # Calculate number of steps for each side
        n_left = max(1, int(n_colors * neg_proportion))
        n_right = max(1, int(n_colors * pos_proportion))
        
        # Create fades
        left_fade = fade_to_color(left_color, n_left, power, mid_color)
        right_fade = fade_to_color(right_color, n_right, power, mid_color)

        # Stack them around mid color
        mid_fade = np.array(mcolors.to_rgb(mid_color))
        colors = np.vstack([left_fade[::-1], [mid_fade], right_fade])
        positions = np.linspace(0, 1, len(colors))
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    return cmap, norm

def demo_colormap(cmap, norm, title="Custom Fade Colormap", figsize=(8, 1.5)):
    """
    Create a demo plot showing the colormap as a horizontal colorbar.
    
    Parameters:
    -----------
    cmap : matplotlib colormap
        The colormap to display
    norm : matplotlib.colors.Normalize
        Normalization object
    title : str
        Title for the colorbar
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal"
    )
    cb.set_label(title, fontsize=12)
    plt.tight_layout()
    plt.show()

# Example usage and demos
if __name__ == "__main__":
    # Example 1: Original blue-white-red
    print("Example 1: Blue-white-red (-0.15 to 0.3)")
    cmap1, norm1 = fade_colormap("blue", "red", mid_color='black',vmin=-0.15, vmax=0.3)
    demo_colormap(cmap1, norm1, "Blue-White-Red Fade")
    
    # Example 2: Green-white-purple with symmetric range
    print("\nExample 2: Green-white-purple (-1 to 1)")
    cmap2, norm2 = fade_colormap("green", "purple", vmin=-1, vmax=1)
    demo_colormap(cmap2, norm2, "Green-White-Purple Fade")
    
    # Example 3: Only positive range
    print("\nExample 3: Black-to-orange (0 to 5)")
    cmap3, norm3 = fade_colormap("black", "orange", mid_color='black',vmin=0, vmax=5)
    demo_colormap(cmap3, norm3, "Black-to-Orange Fade")

    # Example 4: Only negative range
    print("\nExample 4: Dark blue-to-white (-10 to 0)")
    cmap4, norm4 = fade_colormap("darkblue", "white", vmin=-10, vmax=0)
    demo_colormap(cmap4, norm4, "Dark Blue-to-White Fade")
    
    # Example 5: Custom colors with hex codes
    print("\nExample 5: Custom hex colors")
    cmap5, norm5 = fade_colormap("#FF6B6B", "#4ECDC4", vmin=-2, vmax=3, power=1.2)
    demo_colormap(cmap5, norm5, "Custom Hex Colors Fade")
    
    # Example with actual data plot
    print("\nExample 8: Applied to sample data")
    # Generate sample data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(x, y)
    Z = X * np.exp(-(X**2 + Y**2))
    
    # Create colormap based on data range
    data_min, data_max = Z.min(), Z.max()
    cmap_data, norm_data = fade_colormap("navy", "gold", 
                                               vmin=data_min, vmax=data_max, 
                                               power=1.5)
    
    # Plot the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    im = ax1.contourf(X, Y, Z, levels=50, cmap=cmap_data, norm=norm_data)
    ax1.set_title("Sample Data with Custom Colormap")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # Show the colormap
    cb = fig.colorbar(im, ax=ax2, orientation="horizontal")
    ax2.axis('off')
    cb.set_label("Navy-White-Gold Fade")
    
    plt.tight_layout()
    plt.show()
    
    # Example 9: Compare smooth vs stepped on same data
    print("\nExample 9: Comparison of smooth vs stepped colormaps")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Smooth colormap
    cmap_smooth, norm_smooth = fade_colormap("purple", "orange", 
                                                   vmin=data_min, vmax=data_max, 
                                                   n_colors=256)
    im1 = ax1.contourf(X, Y, Z, levels=50, cmap=cmap_smooth, norm=norm_smooth)
    ax1.set_title("Smooth Gradient (n_colors=256)")
    
    # Stepped colormap
    cmap_stepped, norm_stepped = fade_colormap("purple", "orange", 
                                                     vmin=data_min, vmax=data_max, 
                                                     n_colors=8)
    im2 = ax2.contourf(X, Y, Z, levels=8, cmap=cmap_stepped, norm=norm_stepped)
    ax2.set_title("Stepped/Banded (n_steps=8)")
    
    # Show colormaps
    fig.colorbar(im1, ax=ax3, orientation="horizontal")
    ax3.axis('off')
    ax3.set_title("Smooth Colormap")
    
    fig.colorbar(im2, ax=ax4, orientation="horizontal") 
    ax4.axis('off')
    ax4.set_title("Stepped Colormap")
    
    plt.tight_layout()
    plt.show()