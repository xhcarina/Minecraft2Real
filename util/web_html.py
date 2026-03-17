"""
Save training epoch visuals and build HTML gallery (CycleGAN/pix2pix style).
Use this to view intermediate results in checkpoints/<name>/web/index.html.
"""

import os


def save_epoch_figure(web_dir: str, epoch: int, fig) -> None:
    """Save the current matplotlib figure for this epoch."""
    os.makedirs(web_dir, exist_ok=True)
    images_dir = os.path.join(web_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, f"epoch_{epoch:03d}.png")
    fig.savefig(path, bbox_inches="tight", dpi=100)
    update_index_html(web_dir, epoch + 1)


def update_index_html(web_dir: str, num_epochs: int) -> None:
    """Write index.html that displays all epoch images (latest first)."""
    images_dir = os.path.join(web_dir, "images")
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head><meta charset='utf-8'><title>Minecraft2Real — Training Progress</title>",
        "<style>",
        "body { font-family: system-ui, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }",
        "h1 { color: #fff; }",
        ".epoch { margin: 24px 0; padding: 16px; background: #2a2a2a; border-radius: 8px; }",
        ".epoch h2 { margin-top: 0; color: #7bed9f; }",
        "img { max-width: 100%; height: auto; border-radius: 4px; }",
        "a { color: #70a1ff; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Minecraft → Real (CycleGAN) — Training Progress</h1>",
        "<p>X→Y: Minecraft to real landscape &nbsp;|&nbsp; Y→X: Real to Minecraft</p>",
    ]
    for e in range(num_epochs - 1, -1, -1):
        rel_path = os.path.join("images", f"epoch_{e:03d}.png")
        full_path = os.path.join(web_dir, rel_path)
        if os.path.isfile(full_path):
            lines.append(f"<div class='epoch'><h2>Epoch {e}</h2>")
            lines.append(f"<img src='{rel_path}' alt='Epoch {e}' />")
            lines.append("</div>")
    lines.extend(["</body>", "</html>"])
    index_path = os.path.join(web_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_index_from_existing(web_dir: str) -> None:
    """Scan web_dir/images for epoch_*.png and rebuild index.html."""
    images_dir = os.path.join(web_dir, "images")
    if not os.path.isdir(images_dir):
        return
    epochs = []
    for name in os.listdir(images_dir):
        if name.startswith("epoch_") and name.endswith(".png"):
            try:
                e = int(name[6:9])
                epochs.append(e)
            except ValueError:
                pass
    if epochs:
        update_index_html(web_dir, max(epochs) + 1)
