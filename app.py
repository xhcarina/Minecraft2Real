import io
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from streamlit_cropper import st_cropper

# ── Locate checkpoints bundled with the app ───────────────────────────────────
REPO_DIR = Path(__file__).parent
BUNDLED_CHECKPOINTS = {
    p.name: p for p in sorted(REPO_DIR.glob("*.pth"))
}

# ── Model definitions (must match training code) ──────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_features, num_features, 3),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_features, num_features, 3),
            nn.InstanceNorm2d(num_features),
        )

    def forward(self, x):
        return x + self.conv2(self.dropout(self.conv1(x)))


class Generator(nn.Module):
    def __init__(self, num_res=9):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, 2, 0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res)])
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1, 0),
            nn.InstanceNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.res(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x


# ── Helpers ───────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@st.cache_resource(show_spinner="Loading model weights…")
def load_generators(checkpoint_path: str):
    """Load G_XY and G_YX from a .pth checkpoint path."""
    states = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    g_xy = Generator(num_res=9).to(DEVICE)
    g_yx = Generator(num_res=9).to(DEVICE)
    g_xy.load_state_dict(states["G_XY"])
    g_yx.load_state_dict(states["G_YX"])
    g_xy.eval()
    g_yx.eval()
    return g_xy, g_yx


def run_inference(generator: nn.Module, pil_image: Image.Image) -> Image.Image:
    """Preprocess → forward pass → denormalize → PIL Image."""
    tensor = PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = generator(tensor).squeeze(0).cpu()
    # [-1, 1] → [0, 255]
    output = (output.permute(1, 2, 0).numpy() + 1.0) / 2.0
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output)


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Minecraft ↔ Real", page_icon="🎮", layout="wide")

st.title("Minecraft ↔ Real")
st.caption("CycleGAN — translate between Minecraft and real landscapes.")

# Sidebar: checkpoint selection
with st.sidebar:
    st.header("⚙️ Setup")

    if BUNDLED_CHECKPOINTS:
        ckpt_name = st.selectbox(
            "Checkpoint",
            options=list(BUNDLED_CHECKPOINTS.keys()),
            index=0,
            help="Pre-trained checkpoints found in the project folder.",
        )
        ckpt_path = str(BUNDLED_CHECKPOINTS[ckpt_name])
    else:
        st.warning("No `.pth` files found in the project folder.")
        st.stop()

    st.divider()
    direction = st.radio(
        "Translation direction",
        ["Minecraft → Real", "Real → Minecraft"],
        index=0,
    )
    st.divider()
    st.markdown(
        "**Model:** CycleGAN  \n"
        "**Input size:** 256 × 256  \n"
        f"**Device:** `{DEVICE}`"
    )

# Load model (cached per checkpoint path)
g_xy, g_yx = load_generators(ckpt_path)
generator = g_xy if direction == "Minecraft → Real" else g_yx

# Image upload
label = "Minecraft screenshot" if direction == "Minecraft → Real" else "Real landscape photo"
uploaded_img = st.file_uploader(
    f"Upload {label}",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
)

if uploaded_img is None:
    st.info(f"Upload a {label.lower()} above to translate it.")
    st.stop()

raw_pil = Image.open(uploaded_img).convert("RGB")
w, h = raw_pil.size
is_square = w == h

domain = "Minecraft" if direction == "Minecraft → Real" else "Real"
target = "Real" if direction == "Minecraft → Real" else "Minecraft"

# Clear cached output whenever the source image or checkpoint changes
img_key = (uploaded_img.name, uploaded_img.size, ckpt_name, direction)
if st.session_state.get("_img_key") != img_key:
    st.session_state["_img_key"] = img_key
    st.session_state.pop("output_bytes", None)
    st.session_state.pop("output_pil", None)

# ── Crop UI ───────────────────────────────────────────────────────────────────
if not is_square:
    st.subheader("Crop your image")
    crop_mode = st.radio(
        "Crop mode",
        ["Auto-crop (center square)", "Manual crop"],
        horizontal=True,
    )
    if crop_mode == "Auto-crop (center square)":
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        input_pil = raw_pil.crop((left, top, left + side, top + side))
        st.image(input_pil, caption=f"Auto-cropped {side}×{side}", width=400)
    else:
        st.caption("Drag the box to select a square region, then click Translate.")
        input_pil = st_cropper(
            raw_pil,
            realtime_update=True,
            box_color="#00FF88",
            aspect_ratio=(1, 1),
        )
        st.image(input_pil, caption=f"Cropped preview ({input_pil.size[0]}×{input_pil.size[1]})", width=400)
else:
    input_pil = raw_pil
    col_in, _ = st.columns(2)
    with col_in:
        st.subheader(f"Input ({domain})")
        st.image(input_pil, use_container_width=True)

st.divider()

# ── Inference ─────────────────────────────────────────────────────────────────
if st.button("Translate", type="primary", use_container_width=True):
    with st.spinner("Translating…"):
        output_pil = run_inference(generator, input_pil)
    st.session_state["output_pil"] = output_pil
    st.session_state["output_bytes"] = pil_to_bytes(output_pil)
    st.session_state["output_input"] = input_pil  # snapshot of input used

# Show result if available (persists through download button reruns)
if "output_pil" in st.session_state:
    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader(f"Input ({domain})")
        st.image(st.session_state["output_input"], use_container_width=True)
    with col_out:
        st.subheader(f"Output ({target})")
        st.image(st.session_state["output_pil"], use_container_width=True)

    st.download_button(
        label="⬇️ Download result",
        data=st.session_state["output_bytes"],
        file_name=f"translated_{target.lower()}.png",
        mime="image/png",
    )
