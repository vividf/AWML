debug = False  # 或 False

custom_hooks = []
if debug:
    custom_hooks.append(dict(type="ResultVisualizationHook", save_dir="./projection_vis_origin/"))
