import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from io import BytesIO

# Configurar la página
st.set_page_config(page_title="Simulación de Tiro Parabólico", layout="centered")

# Título de la app
st.title("🎯 Simulación de Tiro Parabólico")

# Entradas del usuario
v0 = st.number_input("Velocidad inicial (m/s):", min_value=1.0, max_value=100.0, value=30.0)
angle_deg = st.slider("Ángulo de lanzamiento (°):", min_value=1, max_value=89, value=45)
g = 9.81  # gravedad (m/s²)

# Convertir a radianes
angle_rad = np.radians(angle_deg)

# Componentes de la velocidad
v0x = v0 * np.cos(angle_rad)
v0y = v0 * np.sin(angle_rad)

# Tiempo de vuelo total
t_flight = 2 * v0y / g
t = np.linspace(0, t_flight, num=100)  # Reducido para mejor rendimiento

# Coordenadas del proyectil
x = v0x * t
y = v0y * t - 0.5 * g * t**2

# Botón para generar animación
if st.button("🎬 Generar Animación"):
    with st.spinner("Generando animación..."):
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'ro', markersize=8, label='Proyectil')
        trail, = ax.plot([], [], 'b-', alpha=0.5, label='Trayectoria')
        ax.plot(x, y, '--', alpha=0.3, color='gray', label='Trayectoria completa')
        
        ax.set_xlim(0, max(x)*1.1)
        ax.set_ylim(0, max(y)*1.1)
        ax.set_xlabel("Distancia (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_title(f"Tiro Parabólico - v₀={v0} m/s, θ={angle_deg}°")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Función de actualización para animación
        def update(frame):
            # Punto actual
            line.set_data([x[frame]], [y[frame]])
            # Trayectoria hasta el punto actual
            trail.set_data(x[:frame+1], y[:frame+1])
            return line, trail

        # Crear la animación
        ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True, repeat=True)

        # Guardar como GIF
        buf = BytesIO()
        ani.save(buf, writer='pillow', fps=20)
        buf.seek(0)
        
        # Mostrar el GIF en Streamlit
        st.image(buf, caption="Animación del Tiro Parabólico")
        
        plt.close(fig)  # Liberar memoria

# Mostrar información adicional
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Datos del Movimiento")
    st.write(f"**Velocidad inicial:** {v0} m/s")
    st.write(f"**Ángulo:** {angle_deg}°")
    st.write(f"**Componente Vx:** {v0x:.2f} m/s")
    st.write(f"**Componente Vy:** {v0y:.2f} m/s")

with col2:
    st.subheader("🎯 Resultados")
    alcance_max = (v0**2 * np.sin(2 * angle_rad)) / g
    altura_max = (v0y**2) / (2 * g)
    st.write(f"**Tiempo de vuelo:** {t_flight:.2f} s")
    st.write(f"**Alcance máximo:** {alcance_max:.2f} m")
    st.write(f"**Altura máxima:** {altura_max:.2f} m")

# Gráfico estático
st.subheader("📈 Trayectoria Completa")
fig_static, ax_static = plt.subplots(figsize=(10, 6))
ax_static.plot(x, y, 'b-', linewidth=2, label='Trayectoria')
ax_static.plot(x[0], y[0], 'go', markersize=8, label='Inicio')
ax_static.plot(x[-1], y[-1], 'ro', markersize=8, label='Final')
ax_static.set_xlim(0, max(x)*1.1)
ax_static.set_ylim(0, max(y)*1.1)
ax_static.set_xlabel("Distancia (m)")
ax_static.set_ylabel("Altura (m)")
ax_static.set_title("Trayectoria del Proyectil")
ax_static.legend()
ax_static.grid(True, alpha=0.3)

st.pyplot(fig_static)