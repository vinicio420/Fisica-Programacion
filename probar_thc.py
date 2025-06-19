import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Simulador de Colisiones 2D", page_icon="⚛️", layout="wide")

# Título principal
st.title("⚛️ Simulador Avanzado de Colisiones 2D")
st.subheader("Física de Partículas con Conservación de Momentum y Energía")

# Sidebar para navegación
st.sidebar.title("🧭 Menú de Simulaciones")
simulacion = st.sidebar.selectbox(
    "Selecciona una simulación:",
    [
        "Colisión 1D - Elástica e Inelástica",
        "Colisión 2D con Trayectorias",
        "Sistema de Partículas Múltiples",
        "Péndulo Balístico",
        "Disco de Newton"
    ]
)

# ======================================================================
# FUNCIONES FÍSICAS COMUNES (ACTUALIZADAS)
# ======================================================================
def colision_1d(m1, m2, v1i, v2i, e=1):
    """Calcula las velocidades finales en colisión 1D."""
    v1f = ((m1 - e * m2) * v1i + (1 + e) * m2 * v2i) / (m1 + m2)
    v2f = ((m2 - e * m1) * v2i + (1 + e) * m1 * v1i) / (m1 + m2)
    return v1f, v2f

def colision_2d(m1, m2, v1x, v1y, v2x, v2y, e=1.0):
    """Calcula velocidades post-colisión para cualquier tipo de choque."""
    v1x_f = ((m1 - e*m2)*v1x + (1 + e)*m2*v2x) / (m1 + m2)
    v2x_f = ((m2 - e*m1)*v2x + (1 + e)*m1*v1x) / (m1 + m2)
    v1y_f = ((m1 - e*m2)*v1y + (1 + e)*m2*v2y) / (m1 + m2)
    v2y_f = ((m2 - e*m1)*v2y + (1 + e)*m1*v1y) / (m1 + m2)
    return v1x_f, v1y_f, v2x_f, v2y_f

def calcular_colision(x1, y1, x2, y2, v1x, v1y, v2x, v2y, radio1, radio2):
    """Calcula el tiempo y posición de colisión en 2D."""
    dx = x2 - x1
    dy = y2 - y1
    dvx = v2x - v1x
    dvy = v2y - v1y
    
    a = dvx**2 + dvy**2
    b = 2*(dx*dvx + dy*dvy)
    c = dx**2 + dy**2 - (radio1 + radio2)**2
    
    discriminante = b**2 - 4*a*c
    
    if a == 0 or discriminante < 0:
        return None, None, None
    
    t = (-b - np.sqrt(discriminante)) / (2*a)
    
    if t < 0:
        return None, None, None
    
    x_col = x1 + v1x * t
    y_col = y1 + v1y * t
    
    return t, x_col, y_col

# ======================================================================
# FUNCIONES DE VISUALIZACIÓN 2D (ACTUALIZADAS)
# ======================================================================
def crear_animacion_2d(m1, m2, v1x, v1y, v2x, v2y, duracion=5, e=1.0):
    """Crea animación completa 2D con colisiones elásticas/inelásticas."""
    fps = 30
    total_frames = int(fps * duracion)
    radio1 = 0.3 * np.sqrt(m1)
    radio2 = 0.3 * np.sqrt(m2)
    
    x1, y1 = -3.0, 1.0
    x2, y2 = 3.0, -1.0
    
    t_col, x_col, y_col = calcular_colision(x1, y1, x2, y2, v1x, v1y, v2x, v2y, radio1, radio2)
    
    if t_col is None:
        t_col = max(0.1, min(duracion * 0.4, duracion - 0.1))
        st.warning("⚠️ Las partículas no colisionarán con estas condiciones iniciales")
        colision_ocurre = False
    else:
        colision_ocurre = True
    
    t_values = np.linspace(0, duracion, total_frames)
    trayectoria1_x, trayectoria1_y = [], []
    trayectoria2_x, trayectoria2_y = [], []
    
    if colision_ocurre:
        v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
        
        # Calcular energía disipada si es inelástica
        if e < 1.0:
            energia_inicial = 0.5*m1*(v1x**2 + v1y**2) + 0.5*m2*(v2x**2 + v2y**2)
            energia_final = 0.5*m1*(v1x_f**2 + v1y_f**2) + 0.5*m2*(v2x_f**2 + v2y_f**2)
            st.warning(f"⚠️ Energía disipada: {energia_inicial - energia_final:.2f} J ({100*(1 - e**2):.0f}% del total)")
    
    frames = []
    for frame in range(total_frames):
        t = t_values[frame]
        
        if colision_ocurre and t >= t_col:
            t_post = t - t_col
            x1_t = x_col + v1x_f * t_post
            y1_t = y_col + v1y_f * t_post
            x2_t = x_col + v2x_f * t_post
            y2_t = y_col + v2y_f * t_post
        else:
            x1_t = x1 + v1x * t
            y1_t = y1 + v1y * t
            x2_t = x2 + v2x * t
            y2_t = y2 + v2y * t
        
        trayectoria1_x.append(x1_t)
        trayectoria1_y.append(y1_t)
        trayectoria2_x.append(x2_t)
        trayectoria2_y.append(y2_t)
        
        frame_data = [
            go.Scatter(
                x=[x1_t], y=[y1_t],
                mode='markers',
                marker=dict(size=20*np.sqrt(m1), color='#1f77b4', line=dict(width=2, color='darkblue')),
                name=f'Partícula 1 ({m1} kg)',
                hovertemplate=f'Masa: {m1} kg<br>Velocidad: {np.sqrt(v1x**2 + v1y**2):.2f} m/s<br>Posición: ({x1_t:.2f}, {y1_t:.2f})'
            ),
            go.Scatter(
                x=[x2_t], y=[y2_t],
                mode='markers',
                marker=dict(size=20*np.sqrt(m2), color='#ff7f0e', line=dict(width=2, color='darkred')),
                name=f'Partícula 2 ({m2} kg)',
                hovertemplate=f'Masa: {m2} kg<br>Velocidad: {np.sqrt(v2x**2 + v2y**2):.2f} m/s<br>Posición: ({x2_t:.2f}, {y2_t:.2f})'
            ),
            go.Scatter(
                x=trayectoria1_x[:frame+1],
                y=trayectoria1_y[:frame+1],
                mode='lines',
                line=dict(color='#1f77b4', width=2, dash='dot'),
                showlegend=False
            ),
            go.Scatter(
                x=trayectoria2_x[:frame+1],
                y=trayectoria2_y[:frame+1],
                mode='lines',
                line=dict(color='#ff7f0e', width=2, dash='dot'),
                showlegend=False
            )
        ]
        
        if colision_ocurre and t >= t_col and frame == int(t_col * fps):
            frame_data.append(
                go.Scatter(
                    x=[x_col], y=[y_col],
                    mode='markers',
                    marker=dict(size=12, color='lime', symbol='x'),
                    name='Punto de colisión',
                    showlegend=False
                )
            )
        
        frames.append(go.Frame(data=frame_data))
    
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            title=f'Simulación de Colisión {"Elástica" if e == 1 else "Inelástica"} 2D (e={e})',
            xaxis=dict(title='Posición X (m)', range=[-5, 5]),
            yaxis=dict(title='Posición Y (m)', scaleanchor='x', scaleratio=1, range=[-5, 5]),
            hovermode='closest',
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="▶️", method="animate", args=[None, {"frame": {"duration": 1000/fps}, "fromcurrent": True}]),
                    dict(label="⏸", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )]
        )
    )
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=50, r=50, b=50, t=80),
        annotations=[
            dict(
                text=f"Simulación de colisión {'elástica' if e == 1 else 'inelástica'} | m₁={m1} kg, m₂={m2} kg, e={e}",
                x=0.5, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(size=12)
            )
        ]
    )
    
    return fig

# ======================================================================
# INTERFAZ DE COLISIÓN 2D (ACTUALIZADA)
# ======================================================================
if simulacion == "Colisión 2D con Trayectorias":
    st.header("🌀 Colisión Bidimensional con Trayectorias")
    
    with st.expander("📚 Teoría de Colisiones 2D", expanded=False):
        st.markdown(r"""
        ### Fundamentos Físicos
        
        **1. Conservación del Momentum Lineal:**
        $$ \sum \vec{p}_{\text{inicial}} = \sum \vec{p}_{\text{final}} $$
        
        **2. Coeficiente de Restitución (e):**
        $$ e = \frac{v_{\text{separación}}}{v_{\text{aproximación}}} $$
        - **e = 1**: Colisión elástica (conserva energía cinética)
        - **0 < e < 1**: Colisión inelástica (pérdida de energía)
        - **e = 0**: Colisión perfectamente inelástica (máxima disipación)
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Parámetros de Simulación")
        
        with st.container(border=True):
            st.markdown("**⚡ Tipo de Colisión**")
            tipo_colision = st.selectbox(
                "Tipo de colisión",
                ["Elástica (e=1)", "Inelástica (0 ≤ e < 1)"],
                key="tipo_colision"
            )
            if tipo_colision == "Inelástica (0 ≤ e < 1)":
                e = st.slider("Coeficiente de restitución (e)", 0.0, 0.99, 0.7, 0.05)
            else:
                e = 1.0
            
            st.markdown("**Propiedades de las Partículas**")
            m1 = st.number_input("Masa 1 (kg)", value=2.0, min_value=0.1, step=0.1, key='2d_m1')
            m2 = st.number_input("Masa 2 (kg)", value=1.0, min_value=0.1, step=0.1, key='2d_m2')
            
            st.markdown("**Velocidad Inicial - Partícula 1**")
            v1x = st.number_input("Componente X (m/s)", value=1.0, key='2d_v1x')
            v1y = st.number_input("Componente Y (m/s)", value=-0.5, key='2d_v1y')
            
            st.markdown("**Velocidad Inicial - Partícula 2**")
            v2x = st.number_input("Componente X (m/s)", value=-1.0, key='2d_v2x')
            v2y = st.number_input("Componente Y (m/s)", value=0.5, key='2d_v2y')
        
        with st.container(border=True):
            st.markdown("**⚙️ Configuración de Visualización**")
            duracion = st.slider("Duración de simulación (s)", 2.0, 10.0, 5.0, 0.5)
    
    with col2:
        st.subheader("🎬 Animación Interactiva")
        
        try:
            fig = crear_animacion_2d(m1, m2, v1x, v1y, v2x, v2y, duracion, e)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cálculo de velocidades finales
            v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
            
            # Cálculo de momentos lineales
            momento_inicial_p1 = m1 * np.array([v1x, v1y])
            momento_inicial_p2 = m2 * np.array([v2x, v2y])
            momento_final_p1 = m1 * np.array([v1x_f, v1y_f])
            momento_final_p2 = m2 * np.array([v2x_f, v2y_f])

            # Cálculo de energías cinéticas
            energia_inicial_p1 = 0.5 * m1 * (v1x**2 + v1y**2)
            energia_inicial_p2 = 0.5 * m2 * (v2x**2 + v2y**2)
            energia_final_p1 = 0.5 * m1 * (v1x_f**2 + v1y_f**2)
            energia_final_p2 = 0.5 * m2 * (v2x_f**2 + v2y_f**2)

            # Mostrar resultados en la columna derecha (debajo de la animación)
            with st.container(border=True):
                st.subheader("📊 Resultados de la Colisión")
                
                # Mostrar solo magnitudes de velocidades
                st.markdown("**Velocidades Finales (Magnitud)**")
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.metric("Partícula 1", f"{np.sqrt(v1x_f**2 + v1y_f**2):.2f} m/s")
                with col_v2:
                    st.metric("Partícula 2", f"{np.sqrt(v2x_f**2 + v2y_f**2):.2f} m/s")
                
                st.markdown("**Momento Lineal Final**")
                st.latex(fr"""
                \begin{{aligned}}
                \sum \vec{{p}}_{{f}} &= {momento_final_p1[0] + momento_final_p2[0]:.2f}\hat{{i}} + {momento_final_p1[1] + momento_final_p2[1]:.2f}\hat{{j}}\ \text{{kg·m/s}}
                \end{{aligned}}
                """)

                st.markdown("**Energía Cinética**")
                col_ener1, col_ener2 = st.columns(2)
                with col_ener1:
                    st.metric("Partícula 1", f"{energia_final_p1:.2f} J", 
                            delta=f"{energia_final_p1 - energia_inicial_p1:.2f} J")
                with col_ener2:
                    st.metric("Partícula 2", f"{energia_final_p2:.2f} J", 
                            delta=f"{energia_final_p2 - energia_inicial_p2:.2f} J")

                st.metric("Energía Total del Sistema", 
                        f"{energia_final_p1 + energia_final_p2:.2f} J", 
                        delta=f"{(energia_final_p1 + energia_final_p2) - (energia_inicial_p1 + energia_inicial_p2):.2f} J")
                
        except Exception as e:
            st.error(f"Error al generar animación: {str(e)}")
        
    with st.expander("📊 Análisis de Momentum (Magnitudes)", expanded=True):
     # ---- Cálculos ----
     v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
    
    # Magnitudes de momentum (antes/después)
    p1_ini = m1 * np.sqrt(v1x**2 + v1y**2)
    p2_ini = m2 * np.sqrt(v2x**2 + v2y**2)
    p1_fin = m1 * np.sqrt(v1x_f**2 + v1y_f**2)
    p2_fin = m2 * np.sqrt(v2x_f**2 + v2y_f**2)
    total_ini = p1_ini + p2_ini
    total_fin = p1_fin + p2_fin

    # ---- Visualización ----
    st.subheader("📊 Momentum Lineal (Solo Magnitudes)")
    
    # Tabla comparativa
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**ANTES (kg·m/s)**")
        st.metric("Partícula 1", f"{p1_ini:.2f}")
        st.metric("Partícula 2", f"{p2_ini:.2f}")
        st.metric("TOTAL", f"{total_ini:.2f}", delta="Referencia")

    with col2:
        st.markdown("**DESPUÉS (kg·m/s)**")
        st.metric("Partícula 1", f"{p1_fin:.2f}", delta=f"{p1_fin - p1_ini:.2f}")
        st.metric("Partícula 2", f"{p2_fin:.2f}", delta=f"{p2_fin - p2_ini:.2f}")
        st.metric("TOTAL", f"{total_fin:.2f}", delta=f"{total_fin - total_ini:.2f}")

    # Barra de progreso para visualizar conservación
    st.progress(min(1.0, total_fin/total_ini))
    st.caption(f"Conservación del momentum: {100*total_fin/total_ini:.1f}%")

    # Diagnóstico
    if np.isclose(total_ini, total_fin, atol=0.01):
        st.success("✅ El momentum total SE CONSERVA (ley física cumplida)")
    else:
        st.error(f"❌ Hay una discrepancia de {abs(total_ini - total_fin):.3f} kg·m/s")