# config.py
# Configuration settings for the pen plotter text-to-G-code pipeline

CONFIG = {
    # Page settings
    'page_width': 215.9,  # mm
    'page_height': 279.4,  # mm
    'margin_left': 50.8,   # mm (2 inches)
    'margin_right': 20,    # mm
    'margin_top': 220,     # mm (very top)
    'margin_bottom': 20,   # mm

    # Text layout
    'font_name': 'roman_simplex',   # Hershey font name, roman_simplex, script_simplex, roman_duplex
    'font_size': 5,         # mm, approximate height
    'line_spacing': 1.4,    # multiplier of font_size (e.g. 1.8 = 1.8× font height between lines)
    'letter_spacing_factor': 0.2,  # reduce for closer letters (0.3 = closest without touching)

    # Feed rates (mm/min) — set high; firmware caps at M203 limit
    'feed_rate_draw': 99999,
    'feed_rate_travel': 99999,
    'feed_rate_initial': 3000,  # mm/min for first travel from home (~50 mm/s, avoids step loss)

    # Speed multiplier
    'speed_multiplier': 10,

    # Pen Z heights (mm)
    #'pen_up_height': 0.5,      # Z height for pen up
    #'pen_down_height': 0,     # Z height for pen down

    # Pen commands
    'pen_up': 'G0 Z4',
    'pen_down': 'G0 Z3.1',

    # Fan speed (0-255, lower for quieter)
    'fan_speed': 50,

    # Page pause
    'page_pause': 'M0 Change paper',  # pause command

    # Output filename
    'output_filename': 'output_8.gcode',

    # G-code settings
    'start_commands': [
        'M140 S0',             # set bed temp to 0
        'M104 S0',             # set hotend temp to 0
        'G28',                 # home all axes
        'G90',                 # absolute positioning
        'G21',                 # units mm
        'M106 S50',            # set fan to low speed for quiet operation
        'M203 X500 Y500 Z50',  # max feedrate (mm/s)
        'M201 X5000 Y5000 Z500', # max acceleration (mm/s²)
        'M205 X20 Y20',        # jerk (mm/s)
        'G0 Z15',               # pen up to initialize
    ],
}