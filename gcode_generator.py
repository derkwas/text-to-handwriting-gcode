# gcode_generator.py
# Module for generating G-code from vector paths

import config


def generate_gcode(paths):
    """
    Generate G-code from the list of paths.
    """
    cfg = config.CONFIG
    multiplier = cfg.get('speed_multiplier', 1)
    feed_draw = int(cfg['feed_rate_draw'] * multiplier)
    feed_travel = int(cfg['feed_rate_travel'] * multiplier)
    feed_initial = cfg.get('feed_rate_initial', 3000)
    pen_up = cfg['pen_up']
    pen_down = cfg['pen_down']
    gcode = []

    # Start commands (last command leaves Z at 15)
    gcode.extend(cfg['start_commands'])

    first_path = True
    for path in paths:
        if not path:
            continue
        start_x, start_y = path[0]
        if first_path:
            # Z is already at 15 from start commands — travel slowly to avoid step loss on long move
            gcode.append(f'G0 X{start_x:.3f} Y{start_y:.3f} F{feed_initial}')
            first_path = False
        else:
            gcode.append(pen_up)
            gcode.append('G4 P0')  # Z retracts fully before XY moves
            gcode.append(f'G0 X{start_x:.3f} Y{start_y:.3f} F{feed_travel}')
        gcode.append('G4 P0')  # flush motion buffer so carriage fully arrives before pen down
        # Pen down
        gcode.append(pen_down)
        # Draw the path
        for x, y in path[1:]:
            gcode.append(f'G1 X{x:.3f} Y{y:.3f} F{feed_draw}')
        # Pen up (Z only — next iteration combines travel into next pen-up)
        gcode.append(pen_up)

    # End
    gcode.append('G0 Z50')               # raise pen clear
    gcode.append(f'G0 X220 Y220 F{feed_travel}')  # move to home corner
    gcode.append('M84')  # disable motors

    return '\n'.join(gcode)