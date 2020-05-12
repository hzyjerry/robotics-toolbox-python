from graphics.graphics_grid import *


def init_canvas(height=500, width=1000, title='', caption='', grid=True):
    """
    Set up the scene with initial conditions.
        - White background
        - Width, height
        - Title, caption
        - Axes drawn (if applicable)

    :param height: Height of the canvas on screen (Pixels), defaults to 500.
    :type height: int, optional
    :param width: Width of the canvas on screen (Pixels), defaults to 1000.
    :type width: int, optional
    :param title: Title of the plot. Gets displayed above canvas, defaults to ''.
    :type title: str, optional
    :param caption: Caption (subtitle) of the plot. Gets displayed below the canvas, defaults to ''.
    :type caption: str, optional
    :param grid: Whether a grid should be displayed in the plot, defaults to True.
    :type grid: bool, optional
    :return: Returns an array of the grid and the numbers displayed along the grid.
    :rtype: Array of vpython objects: [compound object, array of labels]
    """

    # Apply the settings
    scene.background = color.white
    scene.width = width
    scene.height = height
    scene.autoscale = False

    if title != '':
        scene.title = title

    if caption != '':
        scene.caption = caption

    plot_grid = draw_grid()
    if not grid:
        # If no grid, turn the objects invisible
        plot_grid[0].visible = False
        for number in plot_grid[1]:
            number.visible = False

    return plot_grid


def draw_reference_frame_axes(origin, x_axis_vector, x_axis_rotation):
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.

    :param origin: 3D vector representing the point to draw the reference from at.
    :type origin: vpython.vector
    :param x_axis_vector: 3D vector representing the direction of the positive x axis.
    :type x_axis_vector: vpython.vector
    :param x_axis_rotation: Angle in radians to rotate the frame around the x-axis.
    :type x_axis_rotation: float
    :return: Compound object of the 3 axis arrows.
    :rtype: vpython.compound
    """

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(pos=origin, axis=vector(1, 0, 0), length=1, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(pos=origin, axis=vector(0, 1, 0), length=1, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(pos=origin, axis=vector(0, 0, 1), length=1, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of the resulting object bounding box)
    frame_ref = compound([x_arrow, y_arrow, z_arrow], origin=origin)

    # Rotate frame around x, y, z axes as required
    # Set x-axis along required vector, and rotate around the x-axis to corresponding angle to align last two axes
    frame_ref.axis = x_axis_vector
    frame_ref.rotate(angle=x_axis_rotation)

    return frame_ref
