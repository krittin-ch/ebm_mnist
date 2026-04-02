from PIL import Image

for file_loc in [
    "ssl/tsne_animation.gif",
    "ssl/umap_animation.gif",
    "ssl_vit/tsne_animation.gif",
    "ssl_vit/umap_animation.gif",
    "sup/tsne_animation.gif",
    "sup/umap_animation.gif",
]:

    gif = Image.open(file_loc)
    sec = 5

    frames = []
    durations = []

    # -- extract frames and durations
    try:
        while True:
            frames.append(gif.copy())
            durations.append(gif.info.get("duration", 100))  # default 100ms
            gif.seek(len(frames))
    except EOFError:
        pass

    total_duration = sum(durations)

    scale = sec * 1000 / total_duration
    new_durations = [int(d * scale) for d in durations]

    # -- save new GIF
    frames[0].save(
        file_loc,
        save_all=True,
        append_images=frames[1:],
        duration=new_durations,
        loop=0,
    )
