import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Gauss Map & Curvature Analysis",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= SIDEBAR CONTROLS =================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    curve_type = st.selectbox(
        "Select Curve Type",
        ["Straight Line", "Exponential Curve (e^x)", "Circle (closed)", 
         "Cubic Curve (x³)", "Sinusoidal Curve (sin(x))", 
         "Figure-8 (Lemniscate)", "Limaçon (Heart-like)", "Rhodonea (Rose Curve)",
         "Epitrochoid (Spirograph)", "Cassini Oval", "Deltoid (3-cusped)"],
        help="Choose a curve to analyze its curvature properties"
    )
    
    st.divider()
    
    # Parameter controls based on curve type
    if curve_type not in ["Circle (closed)", "Figure-8 (Lemniscate)", "Limaçon (Heart-like)", 
                          "Rhodonea (Rose Curve)", "Epitrochoid (Spirograph)", 
                          "Cassini Oval", "Deltoid (3-cusped)"]:
        st.subheader("Domain")
        x_min = st.number_input("x_min", value=-2.0, step=0.5)
        x_max = st.number_input("x_max", value=2.0, step=0.5)
    
    # Curve-specific parameters
    if curve_type == "Circle (closed)":
        st.subheader("Circle Parameters")
        r = st.slider("Radius r", 0.5, 5.0, 1.0, step=0.1)
        orientation = st.radio("Orientation", ["Anti-clockwise", "Clockwise"])
    
    elif curve_type == "Figure-8 (Lemniscate)":
        st.subheader("Lemniscate Parameters")
        scale = st.slider("Scale", 0.5, 3.0, 1.0, step=0.1)
    
    elif curve_type == "Limaçon (Heart-like)":
        st.subheader("Limaçon Parameters")
        a = st.slider("Parameter a", 0.5, 3.0, 1.5, step=0.1)
        b = st.slider("Parameter b", 0.5, 3.0, 1.0, step=0.1)
        st.caption("Tip: Try a=1.5, b=1.0 for dimpled limaçon")
    
    elif curve_type == "Rhodonea (Rose Curve)":
        st.subheader("Rose Parameters")
        a = st.slider("Amplitude a", 0.5, 3.0, 1.5, step=0.1)
        k = st.slider("Petals k", 2.0, 7.0, 3.0, step=0.5)
        st.caption("k determines number of petals")
    
    elif curve_type == "Epitrochoid (Spirograph)":
        st.subheader("Epitrochoid Parameters")
        R = st.slider("Large radius R", 2.0, 5.0, 3.0, step=0.5)
        r = st.slider("Small radius r", 0.5, 3.0, 1.0, step=0.1)
        d = st.slider("Distance d", 0.5, 3.0, 1.5, step=0.1)
    
    elif curve_type == "Cassini Oval":
        st.subheader("Cassini Oval Parameters")
        a = st.slider("Focus distance a", 0.5, 2.0, 1.0, step=0.1)
        b = st.slider("Parameter b", 0.5, 3.0, 1.5, step=0.1)
        st.caption("b > a√2: single loop")
    
    elif curve_type == "Deltoid (3-cusped)":
        st.subheader("Deltoid Parameters")
        a = st.slider("Radius a", 0.5, 3.0, 1.5, step=0.1)

# ================= MAIN TITLE =================
st.title("📐 Gauss Map & Signed Curvature Analysis")
st.markdown("**Differential Geometry Visualization Tool**")
st.markdown("---")

N = 800  # High resolution for smooth curves

# ================= CURVE GENERATION =================
if curve_type == "Straight Line":
    x = np.linspace(x_min, x_max, N)
    y = x

elif curve_type == "Exponential Curve (e^x)":
    x = np.linspace(x_min, x_max, N)
    y = np.exp(x)

elif curve_type == "Cubic Curve (x³)":
    x = np.linspace(x_min, x_max, N)
    y = x**3

elif curve_type == "Sinusoidal Curve (sin(x))":
    x = np.linspace(x_min, x_max, N)
    y = np.sin(x)

elif curve_type == "Figure-8 (Lemniscate)":
    t = np.linspace(0, 2*np.pi, N)
    x = scale * np.cos(t)
    y = scale * np.sin(2*t) / 2

elif curve_type == "Limaçon (Heart-like)":
    t = np.linspace(0, 2*np.pi, N)
    r = a + b * np.cos(t)
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Rhodonea (Rose Curve)":
    t = np.linspace(0, 2*np.pi, N)
    r = a * np.cos(k * t)
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Epitrochoid (Spirograph)":
    revolutions = int(np.ceil(r / np.gcd(int(R*10), int(r*10)) * 10))
    t = np.linspace(0, 2*np.pi*revolutions, N*revolutions)
    x = (R + r) * np.cos(t) - d * np.cos((R + r) / r * t)
    y = (R + r) * np.sin(t) - d * np.sin((R + r) / r * t)

elif curve_type == "Cassini Oval":
    t = np.linspace(0, 2*np.pi, N)
    r_squared = b**2 * np.cos(2*t) + np.sqrt(b**4 - a**4 * np.sin(2*t)**2)
    r = np.sqrt(np.maximum(r_squared, 0))
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Deltoid (3-cusped)":
    t = np.linspace(0, 2*np.pi, N)
    x = 2 * a * np.cos(t) + a * np.cos(2*t)
    y = 2 * a * np.sin(t) - a * np.sin(2*t)

elif curve_type == "Circle (closed)":
    t = np.linspace(0, 2*np.pi, N)
    if orientation == "Anti-clockwise":
        x = r * np.cos(t)
        y = r * np.sin(t)
    else:
        x = r * np.cos(t)
        y = -r * np.sin(t)

# ================= CURVATURE COMPUTATION =================
if curve_type in ["Straight Line", "Exponential Curve (e^x)", "Cubic Curve (x³)", "Sinusoidal Curve (sin(x))"]:
    dx = x[1] - x[0]
    dy = np.gradient(y, dx, edge_order=2)
    d2y = np.gradient(dy, dx, edge_order=2)
    
    kappa = d2y / (1 + dy**2)**(3/2)
    ds = np.sqrt(1 + dy**2) * dx
    
    Tx = 1 / np.sqrt(1 + dy**2)
    Ty = dy / np.sqrt(1 + dy**2)

else:  # Parametric curves
    dt = t[1] - t[0]
    dx_dt = np.gradient(x, dt, edge_order=2)
    dy_dt = np.gradient(y, dt, edge_order=2)
    d2x_dt2 = np.gradient(dx_dt, dt, edge_order=2)
    d2y_dt2 = np.gradient(dy_dt, dt, edge_order=2)
    
    speed_squared = dx_dt**2 + dy_dt**2
    speed_cubed = speed_squared**(3/2)
    speed_cubed = np.where(speed_cubed < 1e-10, 1e-10, speed_cubed)
    
    kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / speed_cubed
    ds = np.sqrt(speed_squared) * dt
    
    speed = np.sqrt(speed_squared)
    speed = np.where(speed < 1e-10, 1e-10, speed)
    
    Tx = dx_dt / speed
    Ty = dy_dt / speed

# ================= GAUSS INDEX CALCULATION =================
turning_angle = np.sum(kappa * ds)
index = turning_angle / (2*np.pi)

num_positive = np.sum(kappa >= 0)
num_negative = np.sum(kappa < 0)
total_points = num_positive + num_negative
sign_changes = len(np.where(np.diff(np.sign(kappa)))[0])

# ================= VISUALIZATION =================
st.subheader("📊 Curve and Gauss Map Visualization")

# Create figure with larger size for mobile
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25)

# Original curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, y, color='#2E86AB', linewidth=2.5)
ax1.set_title("Original Curve γ(t)", fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel("x", fontsize=13, fontweight='bold')
ax1.set_ylabel("y", fontsize=13, fontweight='bold')
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.tick_params(labelsize=11)
ax1.margins(0.15)

# Gauss map
ax2 = fig.add_subplot(gs[0, 1])
theta = np.linspace(0, 2*np.pi, 400)
ax2.plot(np.cos(theta), np.sin(theta), color='#2D3142', linewidth=2, linestyle='-', alpha=0.8)

colors = np.where(kappa >= 0, '#0077B6', '#DC2F02')
ax2.scatter(Tx, Ty, c=colors, s=25, alpha=0.7, edgecolors='none')

ax2.set_aspect('equal')
ax2.set_title("Gauss Map: T(t) → S¹", fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel("Tₓ", fontsize=13, fontweight='bold')
ax2.set_ylabel("Tᵧ", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-1.25, 1.25)
ax2.tick_params(labelsize=11)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#0077B6', label='κ > 0 (Convex)'),
    Patch(facecolor='#DC2F02', label='κ < 0 (Concave)')
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ================= MATHEMATICAL RESULTS =================
st.markdown("---")
st.subheader("🔢 Mathematical Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Turning Angle",
        value=f"{turning_angle:.4f} rad",
        help="∫ κ(s) ds over the curve"
    )
    st.latex(r"\int_\gamma \kappa \, ds")

with col2:
    st.metric(
        label="Gauss-Bonnet Index",
        value=f"{index:.4f}",
        help="Turning angle divided by 2π"
    )
    st.latex(r"\text{index} = \frac{1}{2\pi}\int_\gamma \kappa \, ds")

with col3:
    st.metric(
        label="Curvature Sign Changes",
        value=f"{sign_changes}",
        help="Number of inflection points"
    )

# ================= CURVATURE STATISTICS =================
st.markdown("---")
st.subheader("📈 Curvature Distribution")

col4, col5, col6, col7 = st.columns(4)

with col4:
    st.metric("Points with κ > 0", f"{num_positive}", delta=f"{(num_positive/total_points*100):.1f}%")

with col5:
    st.metric("Points with κ < 0", f"{num_negative}", delta=f"{(num_negative/total_points*100):.1f}%")

with col6:
    st.metric("max |κ|", f"{np.max(np.abs(kappa)):.4f}")

with col7:
    st.metric("mean |κ|", f"{np.mean(np.abs(kappa)):.4f}")

# ================= CURVATURE ANALYSIS =================
st.markdown("---")
st.subheader("🎯 Curvature Analysis")

if num_positive > 0 and num_negative > 0:
    st.success("✅ **Mixed Curvature Detected**: This curve exhibits both positive and negative curvature regions")
    
    percentage_positive = (num_positive / total_points) * 100
    
    # Visual progress bar for curvature distribution
    st.markdown("**Curvature Distribution:**")
    progress_col1, progress_col2 = st.columns([percentage_positive, 100-percentage_positive])
    with progress_col1:
        st.markdown(f"🔵 Positive: **{percentage_positive:.1f}%**")
    with progress_col2:
        st.markdown(f"🔴 Negative: **{100-percentage_positive:.1f}%**")
    
    st.progress(percentage_positive/100)
    
elif num_positive > 0:
    st.info("ℹ️ **Uniformly Positive Curvature**: The curve is entirely convex (κ > 0)")
elif num_negative > 0:
    st.warning("⚠️ **Uniformly Negative Curvature**: The curve is entirely concave (κ < 0)")
else:
    st.error("⚪ **Zero Curvature**: The curve is a straight line (κ = 0)")

# ================= MATHEMATICAL INTERPRETATION =================
st.markdown("---")
st.subheader("📚 Mathematical Interpretation")

with st.expander("ℹ️ Understanding the Gauss Map", expanded=False):
    st.markdown("""
    **Definition**: The Gauss map T: γ → S¹ maps each point on the curve to its unit tangent vector on the unit circle.
    
    **Signed Curvature**: 
    - κ(t) = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    - **Blue points (κ > 0)**: Curve bends counter-clockwise (convex)
    - **Red points (κ < 0)**: Curve bends clockwise (concave)
    
    **Gauss-Bonnet Theorem**: For closed curves:
    """)
    st.latex(r"\frac{1}{2\pi}\oint_\gamma \kappa \, ds = \text{winding number}")
    st.markdown("""
    - Circle (counter-clockwise): index = +1
    - Circle (clockwise): index = -1
    - Figure-8: index = 0
    """)

with st.expander("🔍 About This Curve", expanded=False):
    if curve_type == "Limaçon (Heart-like)":
        st.markdown("""
        **Limaçon Curve**: r = a + b cos(θ)
        - When **a = b**: Cardioid (heart shape)
        - When **a > b**: Dimpled limaçon with curvature changes
        - When **a < b**: Limaçon with inner loop
        """)
    elif curve_type == "Rhodonea (Rose Curve)":
        st.markdown("""
        **Rose Curve**: r = a cos(kθ)
        - **k odd**: k petals
        - **k even**: 2k petals
        - **Non-integer k**: Creates interesting patterns
        """)
    elif curve_type == "Deltoid (3-cusped)":
        st.markdown("""
        **Deltoid (Hypocycloid)**: 
        - x = 2a cos(t) + a cos(2t)
        - y = 2a sin(t) - a sin(2t)
        - Has 3 cusps with dramatic curvature changes
        """)
    elif curve_type == "Cubic Curve (x³)":
        st.markdown("""
        **Cubic Function**: y = x³
        - Inflection point at x = 0
        - Perfect example of curvature sign change
        - κ = 6x / (1 + 9x⁴)^(3/2)
        """)
    elif curve_type == "Cassini Oval":
        st.markdown("""
        **Cassini Oval**: Product of distances to two foci is constant
        - When **b > a√2**: Single smooth loop
        - When **b = a**: Lemniscate (figure-8)
        - When **b < a**: Two separate loops
        """)

# ================= FOOTER =================
st.markdown("---")
st.caption("💡 **Tip**: Adjust parameters in the sidebar to explore different curves and their curvature properties")
st.caption("📱 **Mobile-friendly**: This interface is optimized for mobile viewing with large, clear visualizations")