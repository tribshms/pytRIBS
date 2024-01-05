def add_scale_bar(ax, length, location=(0.5, 0.05)):
    """
    Add a scale bar to the map.

    Parameters:
    - ax: Matplotlib axis
    - length: Length of the scale bar in map units
    - location: Tuple (x, y) representing the location of the scale bar
    """
    # Calculate the width of the scale bar in inches
    width = length / ax.get_xlim()[1] * fig.get_size_inches()[0]

    # Create a Rectangle patch for the scale bar
    scale_bar = plt.Rectangle(location, width, 0.01, ec='black', fc='white', transform=ax.transAxes)

    # Add the scale bar to the map
    ax.add_patch(scale_bar)

    # Add a label to the scale bar
    ax.text(location[0] + width / 2, location[1] + 0.02, f'{length:.0f} units', ha='center', va='bottom')
