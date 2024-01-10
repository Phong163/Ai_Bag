from pathlib import Path
def get_gif(name_gif):
    gif_path= r'C:\Users\OS\Desktop\AiBag\gif'
    return str(Path('.') / gif_path / name_gif)


