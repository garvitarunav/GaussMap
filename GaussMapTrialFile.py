import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Gauss Map and Signed Curvature (Orientation-Correct)")

# ================= USER CONTROLS =================
curve_type = st.selectbox(
    "Choose a curve",
    ["Straight Line", "Exponential Curve (e^x)", "Circle (closed)", 
     "Cubic Curve (x³)", "Sinusoidal Curve (sin(x))", 
     "Figure-8 (Lemniscate)", "Limaçon (Heart-like)", "Rhodonea (Rose Curve)",
     "Epitrochoid (Spirograph)", "Cassini Oval", "Deltoid (3-cusped)"]
)

x_min, x_max = st.slider("Select x-range", -10.0, 10.0, (-2.0, 2.0), step=0.5)
N = 800  # Increased for smoother curves

# ================= CURVES =================
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
    # Gerono lemniscate
    t = np.linspace(0, 2*np.pi, N)
    scale = st.slider("Select scale", 0.5, 3.0, 1.0, step=0.1)
    x = scale * np.cos(t)
    y = scale * np.sin(2*t) / 2

elif curve_type == "Limaçon (Heart-like)":
    # Limaçon: shows clear curvature changes
    # r = a + b*cos(θ), with a/b ratio determining shape
    t = np.linspace(0, 2*np.pi, N)
    a = st.slider("Parameter a", 0.5, 3.0, 1.5, step=0.1)
    b = st.slider("Parameter b", 0.5, 3.0, 1.0, step=0.1)
    
    r = a + b * np.cos(t)
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Rhodonea (Rose Curve)":
    # Rose curve: r = a*cos(k*θ)
    # k determines number of petals (if k is odd, k petals; if even, 2k petals)
    t = np.linspace(0, 2*np.pi, N)
    a = st.slider("Amplitude a", 0.5, 3.0, 1.5, step=0.1)
    k = st.slider("Petals parameter k", 2.0, 7.0, 3.0, step=0.5)
    
    r = a * np.cos(k * t)
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Epitrochoid (Spirograph)":
    # Epitrochoid: like spirograph patterns
    # Shows complex curvature changes
    t = np.linspace(0, 2*np.pi, N)
    R = st.slider("Large circle radius R", 2.0, 5.0, 3.0, step=0.5)
    r = st.slider("Small circle radius r", 0.5, 3.0, 1.0, step=0.1)
    d = st.slider("Distance from center d", 0.5, 3.0, 1.5, step=0.1)
    
    # Extend t range for complete pattern
    revolutions = int(np.ceil(r / np.gcd(int(R*10), int(r*10)) * 10))
    t = np.linspace(0, 2*np.pi*revolutions, N*revolutions)
    
    x = (R + r) * np.cos(t) - d * np.cos((R + r) / r * t)
    y = (R + r) * np.sin(t) - d * np.sin((R + r) / r * t)

elif curve_type == "Cassini Oval":
    # Cassini oval: generalization of lemniscate
    # ((x-a)² + y²) * ((x+a)² + y²) = b⁴
    t = np.linspace(0, 2*np.pi, N)
    a = st.slider("Focus distance a", 0.5, 2.0, 1.0, step=0.1)
    b = st.slider("Parameter b", 0.5, 3.0, 1.5, step=0.1)
    
    # Parametric form
    r_squared = b**2 * np.cos(2*t) + np.sqrt(b**4 - a**4 * np.sin(2*t)**2)
    r = np.sqrt(np.maximum(r_squared, 0))  # Ensure non-negative
    
    x = r * np.cos(t)
    y = r * np.sin(t)

elif curve_type == "Deltoid (3-cusped)":
    # Deltoid (hypocycloid with k=3)
    # Shows three cusps with clear curvature changes
    t = np.linspace(0, 2*np.pi, N)
    a = st.slider("Radius a", 0.5, 3.0, 1.5, step=0.1)
    
    x = 2 * a * np.cos(t) + a * np.cos(2*t)
    y = 2 * a * np.sin(t) - a * np.sin(2*t)

elif curve_type == "Circle (closed)":
    r = st.slider("Select radius r", 0.5, 5.0, 1.0, step=0.1)
    orientation = st.selectbox("Orientation", ["Anti-clockwise", "Clockwise"])

    t = np.linspace(0, 2*np.pi, N)
    if orientation == "Anti-clockwise":
        x = r * np.cos(t)
        y = r * np.sin(t)
    else:  # Clockwise
        x = r * np.cos(t)
        y = -r * np.sin(t)

# ================= DERIVATIVES =================
if curve_type in ["Straight Line", "Exponential Curve (e^x)", "Cubic Curve (x³)", "Sinusoidal Curve (sin(x))"]:
    dx = x[1] - x[0]
    dy = np.gradient(y, dx, edge_order=2)
    d2y = np.gradient(dy, dx, edge_order=2)

    # curvature for y=f(x)
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

    # SIGNED curvature (orientation matters)
    # κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    speed_squared = dx_dt**2 + dy_dt**2
    speed_cubed = speed_squared**(3/2)
    
    # Avoid division by zero
    speed_cubed = np.where(speed_cubed < 1e-10, 1e-10, speed_cubed)
    
    kappa = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / speed_cubed
    ds = np.sqrt(speed_squared) * dt

    # Unit tangent vector
    speed = np.sqrt(speed_squared)
    speed = np.where(speed < 1e-10, 1e-10, speed)
    
    Tx = dx_dt / speed
    Ty = dy_dt / speed

# ================= GAUSS INDEX =================
turning_angle = np.sum(kappa * ds)
index = turning_angle / (2*np.pi)

# ================= VISUALIZATION =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Original curve
ax1.plot(x, y, color='black', linewidth=2)
ax1.set_title("Original Curve", fontsize=14, fontweight='bold')
ax1.set_xlabel("x", fontsize=12)
ax1.set_ylabel("y", fontsize=12)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, alpha=0.3)
ax1.margins(0.1)

# Gauss map
theta = np.linspace(0, 2*np.pi, 400)
ax2.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.5)

colors = np.where(kappa >= 0, 'blue', 'red')
ax2.scatter(Tx, Ty, c=colors, s=15, alpha=0.6)
ax2.set_aspect('equal')
ax2.set_title("Gauss Map (Signed)", fontsize=14, fontweight='bold')
ax2.set_xlabel("Tx", fontsize=12)
ax2.set_ylabel("Ty", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)

plt.tight_layout()
st.pyplot(fig)

# ================= OUTPUT =================
st.markdown("### Geometric Results")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Turning Angle (radians)", f"{turning_angle:.4f}")
    st.metric("Gauss Index", f"{index:.4f}")

with col2:
    num_positive = np.sum(kappa >= 0)
    num_negative = np.sum(kappa < 0)
    st.metric("Blue dots (positive κ)", num_positive)
    st.metric("Red dots (negative κ)", num_negative)

st.caption("🔵 Blue: positive curvature (convex, anticlockwise) | 🔴 Red: negative curvature (concave, clockwise)")

# Show curvature sign changes
if num_positive > 0 and num_negative > 0:
    st.success("✅ This curve shows BOTH red and blue dots - curvature changes sign!")
    percentage_positive = (num_positive / (num_positive + num_negative)) * 100
    st.write(f"Curvature distribution: {percentage_positive:.1f}% positive, {100-percentage_positive:.1f}% negative")
elif num_positive > 0:
    st.info("ℹ️ This curve only has positive curvature (blue dots only)")
elif num_negative > 0:
    st.warning("⚠️ This curve only has negative curvature (red dots only)")

# Curvature statistics
st.markdown("### Curvature Statistics")
col3, col4, col5 = st.columns(3)
with col3:
    st.write(f"**Max curvature:** {np.max(np.abs(kappa)):.4f}")
with col4:
    st.write(f"**Mean |κ|:** {np.mean(np.abs(kappa)):.4f}")
with col5:
    sign_changes = len(np.where(np.diff(np.sign(kappa)))[0])
    st.write(f"**Sign changes:** {sign_changes}")

# Information about the curve
st.markdown("### Curve Information")
if curve_type == "Limaçon (Heart-like)":
    st.info("💡 **Limaçon**: When a=b, you get a cardioid (heart shape). Try a=1.5, b=1.0 for dimpled limaçon with curvature changes.")
elif curve_type == "Rhodonea (Rose Curve)":
    st.info("💡 **Rose Curve**: The parameter k controls the number of petals. Non-integer k creates interesting patterns with curvature changes.")
elif curve_type == "Epitrochoid (Spirograph)":
    st.info("💡 **Epitrochoid**: Like a spirograph! Adjust R, r, and d to see complex patterns. Shows rich curvature variation.")
elif curve_type == "Cassini Oval":
    st.info("💡 **Cassini Oval**: When b > a√2, it's a single loop. When b < a, it splits into two loops. Try b=1.2, a=1.0.")
elif curve_type == "Deltoid (3-cusped)":
    st.info("💡 **Deltoid**: A hypocycloid with 3 cusps. Shows dramatic curvature changes near the cusps.")