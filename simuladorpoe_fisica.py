import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Simulador de Física", page_icon="⚡", layout="wide")

# Título principal
st.title("⚡ Simulador de Física")

# Sidebar para navegación
st.sidebar.title("🎯 Menú de Simulaciones")
simulacion = st.sidebar.selectbox(
    "Selecciona una simulación:",
    [
        "🎯 Colisión 1D",
        "🌐 Colisión 2D con Trayectorias",
        "🏔️Péndulo Balístico"
    ]
)

# Función para calcular colisiones 1D
def colision_1d(m1, m2, v1i, v2i, e=1):
    """Calcula las velocidades finales en colisión 1D."""
    v1f = ((m1 - e*m2) * v1i + (1 + e) * m2 * v2i) / (m1 + m2)
    v2f = ((m2 - e*m1) * v2i + (1 + e) * m1 * v1i) / (m1 + m2)
    return v1f, v2f

# Visualización para colisión 1D
def crear_visualizacion_1d(m1, m2, v1i, v2i, e):
    """Crea la visualización de colisión 1D."""
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    t = np.linspace(0, 4, 100)
    x1 = np.where(t < 2, v1i * t, v1i * 2 + v1f * (t - 2))
    x2 = np.where(t < 2, v2i * t, v2i * 2 + v2f * (t - 2))

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Posición vs Tiempo', 'Velocidad vs Tiempo'))

    # Graficar posiciones
    fig.add_trace(go.Scatter(x=t, y=x1, mode='lines', name=f'Objeto 1 (m={m1} kg)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t, y=x2, mode='lines', name=f'Objeto 2 (m={m2} kg)', line=dict(color='red')))
    
    # Graficar velocidades
    v1_array = np.where(t < 2, v1i, v1f)
    v2_array = np.where(t < 2, v2i, v2f)
    fig.add_trace(go.Scatter(x=t, y=v1_array, mode='lines', name='Velocidad Objeto 1', line=dict(color='blue'), showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=v2_array, mode='lines', name='Velocidad Objeto 2', line=dict(color='red'), showlegend=False))
    
    fig.update_layout(title_text="Análisis de Colisión 1D", height=600)
    return fig

# Animación de colisión 1D
def crear_animacion_1d(m1, m2, v1i, v2i, e):
    """Crea la animación de colisión 1D."""
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    t_total = 4.0
    frames_per_second = 20
    frames = []

    for frame in range(int(t_total * frames_per_second)):
        tiempo = frame / frames_per_second
        if tiempo < 2:
            x1 = v1i * tiempo
            x2 = v2i * tiempo
        else:
            t_post = tiempo - 2
            x1 = v1i * 2 + v1f * t_post
            x2 = v2i * 2 + v2f * t_post

        frames.append(go.Frame(data=[
            go.Scatter(x=[x1], y=[0], mode='markers', marker=dict(size=20, color='blue')),
            go.Scatter(x=[x2], y=[0], mode='markers', marker=dict(size=20, color='red'))
        ]))

    fig = go.Figure(data=[
        go.Scatter(x=[-3], y=[0], mode='markers', marker=dict(size=20, color='blue')),
        go.Scatter(x=[3], y=[0], mode='markers', marker=dict(size=20, color='red'))
    ], frames=frames)

    fig.update_layout(title="Animación de Colisión 1D", xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-1, 1]))
    fig.update_layout(updatemenus=[{
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
            }
        ]
    }])
    return fig

# Simulación de Colisión 1D
if simulacion == "Colisión 1D - Elástica e Inelástica":
    st.header("🔵 Colisión 1D - Elástica e Inelástica")
    
    m1 = st.number_input("Masa del objeto 1 (kg)", value=2.0)
    m2 = st.number_input("Masa del objeto 2 (kg)", value=1.0)
    v1i = st.number_input("Velocidad inicial objeto 1 (m/s)", value=5.0)
    v2i = st.number_input("Velocidad inicial objeto 2 (m/s)", value=-2.0)
    
    tipo_colision = st.selectbox("Tipo de colisión:", ["Elástica", "Inelástica", "Perfectamente Inelástica"])
    e = 1.0 if tipo_colision == "Elástica" else 0.0 if tipo_colision == "Perfectamente Inelástica" else st.slider("Coeficiente de restitución (e)", 0.0, 1.0, 0.5)
    
    # Visualización gráfica
    st.subheader("📊 Análisis Gráfico")
    fig_analysis = crear_visualizacion_1d(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_analysis)
    
    # Animación
    st.subheader("🎬 Animación de la Colisión")
    fig_animation = crear_animacion_1d(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_animation)

# Funciones para colisiones 2D
def colision_2d(m1, m2, v1, v2):
    """Calcula las velocidades finales en colisión 2D."""
    v1x_i, v1y_i = v1
    v2x_i, v2y_i = v2

    # Conservación del momento en 2D
    v1x_f = (v1x_i * (m1 - m2) + 2 * m2 * v2x_i) / (m1 + m2)
    v1y_f = (v1y_i * (m1 - m2) + 2 * m2 * v2y_i) / (m1 + m2)
    v2x_f = (v2x_i * (m2 - m1) + 2 * m1 * v1x_i) / (m1 + m2)
    v2y_f = (v2y_i * (m2 - m1) + 2 * m1 * v1y_i) / (m1 + m2)

    return (v1x_f, v1y_f), (v2x_f, v2y_f)

# Visualización para colisión 2D
def crear_visualizacion_2d(m1, m2, v1, v2):
    """Crea la visualización de colisión 2D."""
    (v1x_f, v1y_f), (v2x_f, v2y_f) = colision_2d(m1, m2, v1, v2)
    
    t = np.linspace(0, 4, 100)
    x1 = v1[0] * t
    y1 = v1[1] * t
    x2 = v2[0] * t
    y2 = v2[1] * t

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines', name='Objeto 1', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines', name='Objeto 2', line=dict(color='red')))
    fig.update_layout(title='Trayectorias de Colisión 2D', xaxis_title='X (m)', yaxis_title='Y (m)')
    
    return fig

# Animación de colisión 2D
def crear_animacion_2d(m1, m2, v1, v2):
    """Crea la animación de colisión 2D."""
    frames = []
    t_total = 4.0
    frames_per_second = 20

    for frame in range(int(t_total * frames_per_second)):
        tiempo = frame / frames_per_second
        (v1x_f, v1y_f), (v2x_f, v2y_f) = colision_2d(m1, m2, v1, v2)

        if tiempo < 2:
            x1 = v1[0] * tiempo
            y1 = v1[1] * tiempo
            x2 = v2[0] * tiempo
            y2 = v2[1] * tiempo
        else:
            t_post = tiempo - 2
            x1 = v1[0] * 2 + v1x_f * t_post
            y1 = v1[1] * 2 + v1y_f * t_post
            x2 = v2[0] * 2 + v2x_f * t_post
            y2 = v2[1] * 2 + v2y_f * t_post

        frames.append(go.Frame(data=[
            go.Scatter(x=[x1], y=[y1], mode='markers', marker=dict(size=20, color='blue')),
            go.Scatter(x=[x2], y=[y2], mode='markers', marker=dict(size=20, color='red'))
        ]))

    fig = go.Figure(data=[
        go.Scatter(x=[-5], y=[0], mode='markers', marker=dict(size=20, color='blue')),
        go.Scatter(x=[5], y=[0], mode='markers', marker=dict(size=20, color='red'))
    ], frames=frames)

    fig.update_layout(title="Animación de Colisión 2D", xaxis=dict(range=[-5, 5]), yaxis=dict(range=[-5, 5]))
    fig.update_layout(updatemenus=[{
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
            }
        ]
    }])
    return fig

# Simulación de Colisión 2D
if simulacion == "Colisión 2D con Trayectorias":
    st.header("🎯 Colisión 2D con Trayectorias")
    
    # Parámetros de entrada
    m1 = st.number_input("Masa del objeto 1 (kg)", value=2.0)
    m2 = st.number_input("Masa del objeto 2 (kg)", value=1.0)
    v1x = st.number_input("Velocidad objeto 1 en x (m/s)", value=4.0)
    v1y = st.number_input("Velocidad objeto 1 en y (m/s)", value=0.0)
    v2x = st.number_input("Velocidad objeto 2 en x (m/s)", value=0.0)
    v2y = st.number_input("Velocidad objeto 2 en y (m/s)", value=0.0)
    
    # Calcular y mostrar visualización y animación
    st.subheader("📊 Análisis Gráfico")
    fig_analysis_2d = crear_visualizacion_2d(m1, m2, (v1x, v1y), (v2x, v2y))
    st.plotly_chart(fig_analysis_2d)
    
    st.subheader("🎬 Animación de la Colisión 2D")
    fig_animation_2d = crear_animacion_2d(m1, m2, (v1x, v1y), (v2x, v2y))
    st.plotly_chart(fig_animation_2d)

# Simulación del Péndulo Balístico
elif simulacion == "Péndulo Balístico":
    st.header("🎯 Péndulo Balístico")
    
    m_proyectil = st.number_input("Masa del proyectil (kg)", value=0.01)
    m_pendulo = st.number_input("Masa del péndulo (kg)", value=2.0)
    L = st.number_input("Longitud del péndulo (m)", value=1.0)
    v_proyectil = st.number_input("Velocidad del proyectil (m/s)", value=300.0)

    # Cálculos y visualización
    g = 9.81  # gravedad
    v_conjunto = (m_proyectil * v_proyectil) / (m_proyectil + m_pendulo)
    h_max = v_conjunto**2 / (2 * g)

    # Mostrar resultados
    st.metric("Velocidad del conjunto después del impacto", f"{v_conjunto:.2f} m/s")
    st.metric("Altura máxima alcanzada", f"{h_max:.2f} m")

    # Animación del péndulo
    st.subheader("🎬 Animación del Péndulo Balístico")
    n_frames = 100
    frames = []
    for i in range(n_frames):
        t = i / n_frames * 4  # 4 segundos de animación
        if t <= 1:
            x = -2 + 2 * t  # Proyectil moviéndose
            y = 0
        elif t <= 1.1:
            x = 0  # Momento de impacto
            y = 0
        else:
            t_osc = t - 1.1
            theta = np.pi / 4 * np.cos(3 * np.pi * t_osc)  # Oscilación simplificada
            x = L * np.sin(theta)
            y = -L * np.cos(theta)

        frames.append(go.Frame(data=[
            go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20, color='blue'))
        ]))

    fig_pendulo = go.Figure(data=[
        go.Scatter(x=[0], y=[-L], mode='markers', marker=dict(size=20, color='blue'), name='Péndulo')
    ], frames=frames)

    fig_pendulo.update_layout(title="Animación del Péndulo Balístico", xaxis=dict(range=[-L-1, L+1]), yaxis=dict(range=[-L-1, 1]))
    fig_pendulo.update_layout(updatemenus=[{
        "buttons": [
            {
                "label": "Play",
                "method": "animate",
                "args": [None, {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
            }
        ]
    }])
    
    st.plotly_chart(fig_pendulo)

st.success("Simulación completada. ¡Explora y experimenta!")