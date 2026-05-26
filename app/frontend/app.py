"""
app/frontend/app.py
-------------------
Stage 4 – Streamlit UI for the Monocular 3D Reconstruction demo.

Features:
  - Upload any photo
  - Real-time DaV2 depth estimation via FastAPI
  - Colourised depth map display
  - Interactive 3D point cloud (Plotly)
  - Download depth map as PNG

Run locally:
    streamlit run app/frontend/app.py
    API_URL=http://localhost:8000 streamlit run app/frontend/app.py
"""

import base64
import io
import os

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monocular 3D Reconstruction",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #1a1035 50%, #0d1a2a 100%); }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

.hero { text-align: center; padding: 10px 0 24px 0; }
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p { color: rgba(255,255,255,0.45); font-size: 1rem; margin-top: 6px; }

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
}
.card b { color: #a78bfa; }
.card span { color: rgba(255,255,255,0.85); }

.label {
    text-align: center;
    color: rgba(255,255,255,0.45);
    font-size: 0.8rem;
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.badge-ok   { color: #34d399; font-weight: 700; }
.badge-warn { color: #fbbf24; font-weight: 700; }
.badge-fail { color: #f87171; font-weight: 700; }

.stButton>button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 8px;
    padding: 8px 20px; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def check_api_health(url: str) -> dict:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "offline"}


def call_predict_json(pil_img: Image.Image, subsample: int = 4) -> dict | None:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    try:
        resp = requests.post(
            f"{API_URL}/predict/json",
            files={"file": ("image.jpg", buf, "image/jpeg")},
            params={"subsample": subsample},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot reach API at **{API_URL}**\n\n"
            "Start the server: `uvicorn app.api.main:app --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(
            f"API error {e.response.status_code}: {e.response.text[:300]}")
        return None


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def build_plotly_figure(pc: dict) -> go.Figure:
    fig = go.Figure(data=[go.Scatter3d(
        x=pc["x"], y=pc["z"], z=pc["y"],
        mode="markers",
        marker=dict(
            size=1.2,
            color=pc["colors"],
            opacity=0.9,
        ),
        hovertemplate="x: %{x:.3f}<br>y: %{z:.3f}<br>depth: %{y:.3f}<extra></extra>",
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", showbackground=False,
                       color="rgba(255,255,255,0.4)"),
            yaxis=dict(title="Depth", showbackground=False,
                       color="rgba(255,255,255,0.4)"),
            zaxis=dict(title="Y", showbackground=False,
                       color="rgba(255,255,255,0.4)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    health = check_api_health(API_URL)
    if health.get("status") == "ok" and health.get("model_loaded"):
        badge, label = "badge-ok", "✓ Online"
    elif health.get("status") == "ok":
        badge, label = "badge-warn", "⚠ Model loading…"
    else:
        badge, label = "badge-fail", "✗ Offline"

    st.markdown(
        f"**API:** <span class='{badge}'>{label}</span>",
        unsafe_allow_html=True,
    )
    if health.get("model"):
        st.markdown(f"**Model:** `{health['model']}`")
    st.caption(API_URL)

    st.divider()
    st.markdown("### 3D Settings")
    show_3d = st.toggle("Show 3D Point Cloud", value=True)
    subsample = st.slider(
        "Point density", 2, 10, 4, step=2,
        help="Lower value = more points rendered (slower)"
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "**Depth-Anything-V2** (ViT-B) monocular depth estimation "
        "wrapped in a full MLOps pipeline.\n\n"
        "Upload any photo → get depth map → explore in 3D."
    )
    st.markdown("**Group 15** — Almira · Kanhaiya · Livia")


# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔭 Monocular 3D Reconstruction</h1>
  <p>Depth-Anything-V2 · FastAPI · Streamlit · Docker · W&B</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    # Show original immediately
    col_orig, col_depth = st.columns(2)
    with col_orig:
        st.image(pil_img, use_column_width=True)
        st.markdown('<p class="label">📷 Original Image</p>',
                    unsafe_allow_html=True)

    with col_depth:
        placeholder = st.empty()
        placeholder.markdown(
            '<div style="height:300px;display:flex;align-items:center;'
            'justify-content:center;color:rgba(255,255,255,0.3);">'
            '<span style="font-size:3rem;">⏳</span></div>',
            unsafe_allow_html=True,
        )

    # Run inference
    with st.spinner("Running DaV2 depth inference…"):
        result = call_predict_json(pil_img, subsample=subsample)

    if result:
        depth_pil = b64_to_pil(result["depth_colormap_b64"])

        with col_depth:
            placeholder.image(depth_pil, use_column_width=True)
            st.markdown('<p class="label">🌈 Predicted Depth Map (INFERNO)</p>',
                        unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Inference", f"{result['inference_ms']} ms")
        c2.metric("Depth Min", f"{result['depth_min']:.3f}")
        c3.metric("Depth Max", f"{result['depth_max']:.3f}")
        c4.metric("3D Points", f"{result['point_cloud']['n_points']:,}")

        # Download button
        buf = io.BytesIO()
        depth_pil.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Depth Map",
            data=buf.getvalue(),
            file_name="depth_map.png",
            mime="image/png",
        )

        # 3D Point Cloud
        if show_3d:
            st.divider()
            st.markdown("### 🌐 Interactive 3D Point Cloud")
            st.caption(
                f"Rendered {result['point_cloud']['n_points']:,} points "
                f"(subsample={subsample}). Drag to rotate, scroll to zoom."
            )
            fig = build_plotly_figure(result["point_cloud"])
            st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
<div style="text-align:center;padding:80px 0;color:rgba(255,255,255,0.2);">
  <div style="font-size:6rem;margin-bottom:16px;">🖼️</div>
  <p style="font-size:1.2rem;">Upload any photo to start</p>
  <p style="font-size:0.9rem;margin-top:8px;">
    JPG or PNG · Any resolution · Instant depth estimation
  </p>
</div>
""", unsafe_allow_html=True)
