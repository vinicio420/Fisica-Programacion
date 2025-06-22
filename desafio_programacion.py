import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Simulador de Impulso y Momentum", page_icon="⚡", layout="wide")

# Título principal
st.title("⚛️ Desafío de Programación Física II")
st.subheader("Integrantes: Erik Alqui, Marco Muñoz, Ariel Santana")

# Sidebar para navegación y parámetros
st.sidebar.title("📚 Menú de Simulaciones")
simulacion = st.sidebar.selectbox(
    "Selecciona una simulación:",
    [
        "Colisión 1D - Elástica e Inelástica",
        "Colisión 2D con Trayectorias",
        "Cálculo de Impulso y Fuerza",
        "Péndulo Balístico"
    ]
)

# Función para calcular colisiones 1D
def colision_1d(m1, m2, v1i, v2i, e=1):
    """Calcula las velocidades finales en colisión 1D."""
    v1f = ((m1 - e * m2) * v1i + (1 + e) * m2 * v2i) / (m1 + m2)
    v2f = ((m2 - e * m1) * v2i + (1 + e) * m1 * v1i) / (m1 + m2)
    return v1f, v2f

# Función para calcular el tiempo y posición de colisión
def calcular_colision(x1_inicial, x2_inicial, v1i, v2i):
    """Calcula cuándo y dónde ocurre la colisión."""
    # Si los objetos se mueven en la misma dirección, verificar si hay colisión
    if (v1i - v2i) == 0:
        # Se mueven a la misma velocidad, no hay colisión
        return None, None
    
    # Calcular tiempo de colisión: x1 + v1*t = x2 + v2*t
    t_colision = (x2_inicial - x1_inicial) / (v1i - v2i)
    
    # Solo hay colisión si t > 0 (en el futuro)
    if t_colision <= 0:
        return None, None
    
    # Calcular posición de colisión
    x_colision = x1_inicial + v1i * t_colision
    
    return t_colision, x_colision

# Visualización para colisión 1D (CORREGIDA)
def crear_visualizacion_1d_plotly(m1, m2, v1i, v2i, e):
    # Posiciones iniciales
    x1_inicial = -2.0
    x2_inicial = 2.0
    
    # Calcular tiempo y posición de colisión
    t_colision, x_colision = calcular_colision(x1_inicial, x2_inicial, v1i, v2i)
    
    # Si no hay colisión, usar tiempo fijo
    if t_colision is None:
        t_colision = 2.0
        st.warning("⚠️ Con estas velocidades, los objetos no colisionan")
    
    # Calcular velocidades finales
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    
    # Parámetros de animación
    t_total = max(4.0, t_colision + 2.0)  # Asegurar tiempo suficiente después de la colisión
    dt = 0.05
    
    # Crear arrays de tiempo y posición
    t = np.arange(0, t_total, dt)
    x1 = np.zeros_like(t)
    x2 = np.zeros_like(t)
    
    for i, tiempo in enumerate(t):
        if tiempo < t_colision:
            # Antes de la colisión
            x1[i] = x1_inicial + v1i * tiempo
            x2[i] = x2_inicial + v2i * tiempo
        else:
            # Después de la colisión
            t_post = tiempo - t_colision
            x1[i] = x_colision + v1f * t_post
            x2[i] = x_colision + v2f * t_post
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Posición vs Tiempo', 'Velocidad vs Tiempo'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Graficar posiciones vs tiempo
    fig.add_trace(
        go.Scatter(
            x=t, y=x1, 
            mode='lines',
            name=f'Objeto 1 (m={m1} kg)',
            line=dict(color='blue', width=3),
            hovertemplate='Tiempo: %{x:.2f}s<br>Posición: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t, y=x2, 
            mode='lines',
            name=f'Objeto 2 (m={m2} kg)',
            line=dict(color='red', width=3),
            hovertemplate='Tiempo: %{x:.2f}s<br>Posición: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Línea vertical del momento de colisión
    if t_colision is not None:
        fig.add_vline(
            x=t_colision, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Colisión (t={t_colision:.2f}s)",
            row=1, col=1
        )
    
    # Graficar velocidades
    v1_array = np.where(t < t_colision, v1i, v1f)
    v2_array = np.where(t < t_colision, v2i, v2f)
    
    fig.add_trace(
        go.Scatter(
            x=t, y=v1_array, 
            mode='lines',
            name='Velocidad Objeto 1',
            line=dict(color='blue', width=3),
            hovertemplate='Tiempo: %{x:.2f}s<br>Velocidad: %{y:.2f}m/s<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t, y=v2_array, 
            mode='lines',
            name='Velocidad Objeto 2',
            line=dict(color='red', width=3),
            hovertemplate='Tiempo: %{x:.2f}s<br>Velocidad: %{y:.2f}m/s<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Línea vertical del momento de colisión
    if t_colision is not None:
        fig.add_vline(
            x=t_colision, 
            line_dash="dash", 
            line_color="green",
            row=2, col=1
        )
    
    # Actualizar layout
    fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
    fig.update_yaxes(title_text="Posición (m)", row=1, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig.update_yaxes(title_text="Velocidad (m/s)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Análisis de Colisión 1D",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# Función para crear animación de los objetos (CORREGIDA)
def crear_animacion_objetos_plotly(m1, m2, v1i, v2i, e):
    # Posiciones iniciales
    x1_inicial = -2.0
    x2_inicial = 2.0
    
    # Calcular tiempo y posición de colisión
    t_colision, x_colision = calcular_colision(x1_inicial, x2_inicial, v1i, v2i)
    
    # Si no hay colisión, usar tiempo fijo y posiciones separadas
    if t_colision is None:
        t_colision = 2.0
        x_colision = 0.0
    
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    
    # Parámetros
    t_total = max(4.0, t_colision + 2.0)
    frames_per_second = 20
    total_frames = int(t_total * frames_per_second)
    
    # Crear frames para la animación
    frames = []
    
    for frame in range(total_frames):
        tiempo = frame / frames_per_second
        
        if tiempo < t_colision:
            # Antes de la colisión
            x1 = x1_inicial + v1i * tiempo
            x2 = x2_inicial + v2i * tiempo
        else:
            # Después de la colisión - ambos objetos parten desde el punto de colisión
            t_post = tiempo - t_colision
            x1 = x_colision + v1f * t_post
            x2 = x_colision + v2f * t_post
        
        frame_data = go.Frame(
            data=[
                go.Scatter(
                    x=[x1], y=[0],
                    mode='markers',
                    marker=dict(size=30, color='blue'),
                    name=f'Objeto 1 (m={m1}kg)',
                    hovertemplate=f'Objeto 1<br>Posición: {x1:.2f}m<br>Tiempo: {tiempo:.2f}s<extra></extra>'
                ),
                go.Scatter(
                    x=[x2], y=[0],
                    mode='markers',
                    marker=dict(size=30, color='red'),
                    name=f'Objeto 2 (m={m2}kg)',
                    hovertemplate=f'Objeto 2<br>Posición: {x2:.2f}m<br>Tiempo: {tiempo:.2f}s<extra></extra>'
                )
            ]
        )
        frames.append(frame_data)
    
    # Crear figura inicial
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[x1_inicial], y=[0],
                mode='markers',
                marker=dict(size=30, color='blue'),
                name=f'Objeto 1 (m={m1}kg)'
            ),
            go.Scatter(
                x=[x2_inicial], y=[0],
                mode='markers',
                marker=dict(size=30, color='red'),
                name=f'Objeto 2 (m={m2}kg)'
            )
        ],
        frames=frames
    )
    
    # Configurar animación
    fig.update_layout(
        xaxis=dict(range=[-4, 4], title="Posición (m)"),
        yaxis=dict(range=[-1, 1], title="", showticklabels=False),
        title=f"Animación de Colisión 1D (Colisión en t={t_colision:.2f}s, x={x_colision:.2f}m)",
        updatemenus=[{
            "buttons": [
    {
        "args": [None, {"frame": {"duration": 50, "redraw": True},
                "fromcurrent": True, "transition": {"duration": 300}}],
        "label": "▶️ Play",
        "method": "animate"
    },
    {
        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                "mode": "immediate", "transition": {"duration": 0}}],
        "label": "⏸️ Pause",
        "method": "animate"
    },
    {
        "args": [{"frame": {"duration": 0, "redraw": True}}, 
                {"frame": {"duration": 0}, "mode": "immediate", 
                 "fromcurrent": False, "transition": {"duration": 0}}],
        "label": "⏹️ Stop",
        "method": "animate"
    }
],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

# Simulación de Colisión 1D
if simulacion == "Colisión 1D - Elástica e Inelástica":
    st.header("🔵 Colisión 1D - Elástica e Inelástica")
    
    # PARÁMETROS EN EL SIDEBAR
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Parámetros de Entrada")
    m1 = st.sidebar.number_input("Masa del objeto 1 (kg)", value=2.0, min_value=0.1)
    m2 = st.sidebar.number_input("Masa del objeto 2 (kg)", value=1.0, min_value=0.1)
    v1i = st.sidebar.number_input("Velocidad inicial objeto 1 (m/s)", value=5.0)
    v2i = st.sidebar.number_input("Velocidad inicial objeto 2 (m/s)", value=-2.0)
    
    tipo_colision = st.sidebar.selectbox("Tipo de colisión:", ["Elástica", "Inelástica", "Perfectamente Inelástica"])
    
    if tipo_colision == "Elástica":
        e = 1.0
        st.sidebar.info("🔵 **Colisión Elástica:** Se conserva la energía cinética total")
    elif tipo_colision == "Perfectamente Inelástica":
        e = 0.0
        st.sidebar.warning("🟡 **Colisión Perfectamente Inelástica:** Los objetos se quedan unidos")
    else:
        e = st.sidebar.slider("Coeficiente de restitución (e)", 0.0, 1.0, 0.5)
        st.sidebar.info(f"🟠 **Colisión Inelástica:** e = {e:.2f} - Se pierde energía cinética")
    
    # 1. EXPLICACIÓN FÍSICA
    with st.expander("📚 Explicación Física - Colisiones 1D", expanded=True):
        st.markdown("""
        ### 🔬 Fundamentos Teóricos
        
        **1. Conservación del Momentum (Cantidad de Movimiento):**
        - El momentum total del sistema se conserva en todas las colisiones
        - Ecuación: `p₁ᵢ + p₂ᵢ = p₁f + p₂f`
        - Donde: `p = m × v`
        
        **2. Coeficiente de Restitución (e):**
        - Mide la "elasticidad" de la colisión
        - `e = |velocidad de separación| / |velocidad de aproximación|`
        - `e = |v₂f - v₁f| / |v₁ᵢ - v₂ᵢ|`
        
        **3. Tipos de Colisiones:**
        - **Elástica (e = 1):** Se conserva la energía cinética
        - **Inelástica (0 < e < 1):** Se pierde energía cinética
        - **Perfectamente Inelástica (e = 0):** Los objetos se quedan unidos
        
        **4. Fórmulas para Velocidades Finales:**
        ```
        v₁f = [(m₁ - e×m₂)×v₁ᵢ + (1 + e)×m₂×v₂ᵢ] / (m₁ + m₂)
        v₂f = [(m₂ - e×m₁)×v₂ᵢ + (1 + e)×m₁×v₁ᵢ] / (m₁ + m₂)
        ```
        
        **5. Casos Especiales:**
        - **Objeto en reposo:** Si v₂ᵢ = 0, las fórmulas se simplifican
        - **Masas iguales:** En colisión elástica, los objetos intercambian velocidades
        - **Masa muy grande:** El objeto pequeño rebota, el grande apenas se mueve
        
        **6. Principios Físicos Aplicados:**
        - **Conservación del momentum:** Principio fundamental en todas las colisiones
        - **Conservación de energía:** Solo en colisiones elásticas
        - **Impulso:** J = Δp = F × Δt durante la colisión
        - **Deformación:** En colisiones inelásticas, parte de la energía se disipa como calor, sonido o deformación
        """)
    
    # Calcular resultados una vez para usar en todas las secciones
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    
    # Calcular información de la colisión
    x1_inicial = -2.0
    x2_inicial = 2.0
    t_colision, x_colision = calcular_colision(x1_inicial, x2_inicial, v1i, v2i)
    
    # 2. ANIMACIÓN DE LA COLISIÓN
    st.subheader("🎬 Animación de la Colisión")
    fig_animation = crear_animacion_objetos_plotly(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_animation, use_container_width=True)
    
    # 3. ANÁLISIS GRÁFICO
    st.subheader("📊 Análisis Gráfico")
    fig_analysis = crear_visualizacion_1d_plotly(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_analysis, use_container_width=True)
    
    # 4. RESULTADOS
    st.subheader("📈 Resultados de la Simulación")
    
    # Información de la colisión
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        if t_colision is not None:
            st.success(f"⏱️ **Colisión en:** t = {t_colision:.2f} s")
        else:
            st.warning("⚠️ **No hay colisión** con estas velocidades")
    
    with col_info2:
        if x_colision is not None:
            st.success(f"📍 **Posición de colisión:** x = {x_colision:.2f} m")
    
    # Velocidades finales
    col_vel1, col_vel2 = st.columns(2)
    with col_vel1:
        st.metric("Velocidad final objeto 1", f"{v1f:.2f} m/s", f"{v1f - v1i:.2f} m/s")
    with col_vel2:
        st.metric("Velocidad final objeto 2", f"{v2f:.2f} m/s", f"{v2f - v2i:.2f} m/s")
    
    # Cálculos para conservación
    p_inicial = m1 * v1i + m2 * v2i
    p_final = m1 * v1f + m2 * v2f
    ke_inicial = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
    ke_final = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
    
    # Momentum y energía del sistema
    st.subheader("⚖️ Conservación del Sistema")
    col_mom, col_energy = st.columns(2)
    with col_mom:
        st.metric("Momento inicial (Sistema)", f"{p_inicial:.2f} kg⋅m/s")
        st.metric("Momento final (Sistema)", f"{p_final:.2f} kg⋅m/s")
    with col_energy:
        st.metric("Energía Cinética inicial (Sistema)", f"{ke_inicial:.2f} J")   
        st.metric("Energía Cinética final (Sistema)", f"{ke_final:.2f} J")
    
    # Cambio en cantidad de movimiento por partícula
    delta_p1 = m1 * v1f - m1 * v1i
    delta_p2 = m2 * v2f - m2 * v2i
    
    st.subheader("💥 Cambio en Cantidad de Movimiento (Por Partícula)")
    col_delta_p1, col_delta_p2 = st.columns(2)
    with col_delta_p1:
        st.metric("Objeto 1 (Δp₁)", f"{delta_p1:.2f} kg⋅m/s")
    with col_delta_p2:
        st.metric("Objeto 2 (Δp₂)", f"{delta_p2:.2f} kg⋅m/s")
    
    # Cambio en energía cinética por partícula
    ke1_inicial = 0.5 * m1 * v1i**2
    ke2_inicial = 0.5 * m2 * v2i**2
    ke1_final = 0.5 * m1 * v1f**2
    ke2_final = 0.5 * m2 * v2f**2
    delta_ke1 = ke1_final - ke1_inicial
    delta_ke2 = ke2_final - ke2_inicial
    
    st.subheader("⚡ Cambio en Energía Cinética (Por Partícula)")
    col_delta_ke1, col_delta_ke2 = st.columns(2)
    with col_delta_ke1:
        st.metric("Objeto 1 (ΔKE₁)", f"{delta_ke1:.2f} J")
    with col_delta_ke2:
        st.metric("Objeto 2 (ΔKE₂)", f"{delta_ke2:.2f} J")

    # Verificaciones finales
    st.subheader("✅ Verificacion de Leyes Fisicas")
    conservacion_momentum = abs(p_inicial - p_final) < 0.01
    energia_perdida = ke_inicial - ke_final
    
    col_check1, col_check2 = st.columns(2)
    with col_check1:
        if conservacion_momentum:
            st.success("✅ **Momentum conservado**")
        else:
            st.error("❌ **Error en conservación del momentum**")
    
    with col_check2:
        if energia_perdida > 0.01:
            st.warning(f"⚠️ **Energía perdida:** {energia_perdida:.2f} J ({(energia_perdida/ke_inicial)*100:.1f}%)")
        elif abs(energia_perdida) < 0.01:
            st.success("✅ **Energía cinética conservada**")
    
    # Análisis detallado
    with st.expander("🔍 Análisis Detallado de Resultados"):
        t_colision_display = "No hay colisión" if t_colision is None else f"{t_colision:.3f} s"
        x_colision_display = "N/A" if x_colision is None else f"{x_colision:.3f} m"
        
        st.markdown(f"""
        ### Análisis de la Colisión
        
        **Parámetros de entrada:**
        - Masa objeto 1: {m1} kg, velocidad inicial: {v1i} m/s
        - Masa objeto 2: {m2} kg, velocidad inicial: {v2i} m/s
        - Coeficiente de restitución: {e}
        
        **Información de la colisión:**
        - Tiempo de colisión: {t_colision_display}
        - Posición de colisión: {x_colision_display}
        
        **Resultados calculados:**
        - Velocidad final objeto 1: {v1f:.3f} m/s
        - Velocidad final objeto 2: {v2f:.3f} m/s
        
        **Verificación de conservación:**
        - Momentum inicial: {p_inicial:.3f} kg⋅m/s
        - Momentum final: {p_final:.3f} kg⋅m/s
        - Diferencia: {abs(p_inicial - p_final):.6f} kg⋅m/s
        
        **Análisis energético:**
        - Energía cinética inicial: {ke_inicial:.3f} J
        - Energía cinética final: {ke_final:.3f} J
        - Energía perdida: {energia_perdida:.3f} J
        - Porcentaje de energía perdida: {(energia_perdida/ke_inicial)*100:.2f}%
        """)

# ======================================================================
# COLISION 2D INICIO (ACTUALIZADAS)
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

def calcular_angulos_post_colision(v1x_f, v1y_f, v2x_f, v2y_f):
    """Calcula los ángulos de las velocidades finales respecto al eje X."""
    angulo1 = np.degrees(np.arctan2(v1y_f, v1x_f))
    angulo2 = np.degrees(np.arctan2(v2y_f, v2x_f))
    return angulo1, angulo2

# ======================================================================
# FUNCIÓN PARA CREAR TABLA DE RESULTADOS
# ======================================================================
def crear_tabla_resultados(m1, m2, v1x, v1y, v2x, v2y, v1x_f, v1y_f, v2x_f, v2y_f, e):
    """Crea una tabla estilizada con los resultados del análisis de choque."""
    
    # Cálculos para ANTES del choque
    v1_mag_inicial = np.sqrt(v1x**2 + v1y**2)
    v2_mag_inicial = np.sqrt(v2x**2 + v2y**2)
    ke1_inicial = 0.5 * m1 * v1_mag_inicial**2
    ke2_inicial = 0.5 * m2 * v2_mag_inicial**2
    ke_total_inicial = ke1_inicial + ke2_inicial
    
    p1_inicial = m1 * v1_mag_inicial
    p2_inicial = m2 * v2_mag_inicial
    p_total_inicial = m1 * np.array([v1x, v1y]) + m2 * np.array([v2x, v2y])
    p_total_mag_inicial = np.linalg.norm(p_total_inicial)
    
    # Cálculos para DESPUÉS del choque
    v1_mag_final = np.sqrt(v1x_f**2 + v1y_f**2)
    v2_mag_final = np.sqrt(v2x_f**2 + v2y_f**2)
    ke1_final = 0.5 * m1 * v1_mag_final**2
    ke2_final = 0.5 * m2 * v2_mag_final**2
    ke_total_final = ke1_final + ke2_final
    
    p1_final = m1 * v1_mag_final
    p2_final = m2 * v2_mag_final
    p_total_final = m1 * np.array([v1x_f, v1y_f]) + m2 * np.array([v2x_f, v2y_f])
    p_total_mag_final = np.linalg.norm(p_total_final)
    
    # Crear DataFrame para la tabla
    datos_tabla = {
        'Magnitud': [
            'Energía Cinética (J)',
            'Cantidad de Movimiento (kg·m/s)'
        ],
        'Partícula 1 - ANTES': [
            f"{ke1_inicial:.3f}",
            f"{p1_inicial:.3f}"
        ],
        'Partícula 2 - ANTES': [
            f"{ke2_inicial:.3f}",
            f"{p2_inicial:.3f}"
        ],
        'Sistema - ANTES': [
            f"{ke_total_inicial:.3f}",
            f"{p_total_mag_inicial:.3f}"
        ],
        'Partícula 1 - DESPUÉS': [
            f"{ke1_final:.3f}",
            f"{p1_final:.3f}"
        ],
        'Partícula 2 - DESPUÉS': [
            f"{ke2_final:.3f}",
            f"{p2_final:.3f}"
        ],
        'Sistema - DESPUÉS': [
            f"{ke_total_final:.3f}",
            f"{p_total_mag_final:.3f}"
        ]
    }
    
    df = pd.DataFrame(datos_tabla)
    
    return df, ke_total_inicial, ke_total_final, p_total_mag_inicial, p_total_mag_final

# ======================================================================
# FUNCIONES DE VISUALIZACIÓN 2D (ACTUALIZADAS)
# ======================================================================
def crear_animacion_2d(m1, m2, v1x, v1y, v2x, v2y, duracion=5, e=1.0, x1=0.0, y1=0.5, x2=4.0, y2=-0.5):
    """Crea animación completa 2D con colisiones elásticas/inelásticas."""
    fps = 30
    total_frames = int(fps * duracion)
    radio1 = 0.3 
    radio2 = 0.3 
    
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
                marker=dict(size=20, color="#0a2538", line=dict(width=2, color='darkblue')),
                name=f'Partícula 1 ({m1} kg)',
                hovertemplate=f'Masa: {m1} kg<br>Velocidad: {np.sqrt(v1x**2 + v1y**2):.2f} m/s<br>Posición: ({x1_t:.2f}, {y1_t:.2f})'
            ),
            go.Scatter(
                x=[x2_t], y=[y2_t],
                mode='markers',
                marker=dict(size=20, color='#ff7f0e', line=dict(width=2, color='darkred')),
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
    # NUEVA TABLA DE RESULTADOS ESTILIZADA
    # ======================================================================
    
    # Calcular velocidades finales si no se han calculado
    v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
    
    # Crear y mostrar la tabla de resultados
    st.markdown("---")
    st.subheader("📊 TABLA DE RESULTADOS DEL ANÁLISIS DE CHOQUE")
    
    df_tabla, ke_inicial, ke_final, p_inicial, p_final = crear_tabla_resultados(
        m1, m2, v1x, v1y, v2x, v2y, v1x_f, v1y_f, v2x_f, v2y_f, e
    )
    
    # Aplicar estilo personalizado a la tabla
    def estilizar_tabla(df):
        return df.style.apply(lambda x: [
            'background-color: #f8f9fa; font-weight: bold; color: #2c3e50' if i == 0 
            else 'background-color: #e8f4f8; font-weight: bold; color: #2c3e50' if i == 1
            else 'background-color: #ffffff'
            for i in range(len(x))
        ], axis=0).set_properties(**{
            'text-align': 'center',
            'border': '1px solid #dee2e6',
            'padding': '10px'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#343a40'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '12px'),
                ('border', '1px solid #6c757d')
            ]},
            {'selector': 'td', 'props': [
                ('border', '1px solid #dee2e6'),
                ('padding', '10px')
            ]}
        ])
    
    # Mostrar la tabla estilizada
    st.dataframe(
        estilizar_tabla(df_tabla),
        use_container_width=True,
        hide_index=True
    )


## CAMBIO 2: Mover parámetros al sidebar

# BUSCAR esta línea (aproximadamente línea 158):
if simulacion == "Colisión 2D con Trayectorias":

# Y REEMPLAZAR todo el bloque desde ahí hasta antes de "# Análisis detallado anterior" con:

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
    
    # MOVER PARÁMETROS AL SIDEBAR
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Parámetros de Simulación")
    
    with st.sidebar.container():
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
        
        st.markdown("**Posiciones Iniciales**")
        st.markdown("*Partícula 1*")
        x1_input = st.number_input("Posición X₁ (m)", value=0.0, step=0.1, key='pos_x1')
        y1_input = st.number_input("Posición Y₁ (m)", value=0.5, step=0.1, key='pos_y1')
        st.markdown("*Partícula 2*")
        x2_input = st.number_input("Posición X₂ (m)", value=4.0, step=0.1, key='pos_x2')
        y2_input = st.number_input("Posición Y₂ (m)", value=-0.5, step=0.1, key='pos_y2')           
        
        st.markdown("**Velocidad Inicial - Partícula 1**")
        v1x = st.number_input("Componente X (m/s)", value=1.0, key='2d_v1x')
        v1y = st.number_input("Componente Y (m/s)", value=-0.5, key='2d_v1y')
        
        st.markdown("**Velocidad Inicial - Partícula 2**")
        v2x = st.number_input("Componente X (m/s)", value=-1.0, key='2d_v2x')
        v2y = st.number_input("Componente Y (m/s)", value=0.5, key='2d_v2y')
        
        st.markdown("**⚙️ Configuración de Visualización**")
        duracion = st.slider("Duración de simulación (s)", 2.0, 10.0, 5.0, 0.5)
    
    # AHORA LA ANIMACIÓN OCUPA TODO EL ANCHO
    st.subheader("🎬 Animación Interactiva")
    
    try:
        fig = crear_animacion_2d(m1, m2, v1x, v1y, v2x, v2y, duracion, e, x1_input, y1_input, x2_input, y2_input)
        st.plotly_chart(fig, use_container_width=True)
        # Calcular velocidades finales
        v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
    except Exception as error:
        st.error(f"Error al crear la animación: {error}")
        st.info("Verifica que todos los parámetros sean válidos")    
        # Cálculo de velocidades finales
        v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
        # AGREGAR ESTE CÓDIGO después del try-except de la animación 
# (aproximadamente después de la línea donde calculas v1x_f, v1y_f, v2x_f, v2y_f)

    # Calcular velocidades finales si no se han calculado
    v1x_f, v1y_f, v2x_f, v2y_f = colision_2d(m1, m2, v1x, v1y, v2x, v2y, e)
    
    # DEFINIR LAS VARIABLES QUE FALTABAN (agregar después del cálculo de velocidades finales)
    # Cálculos para ANTES del choque
    v1_mag_inicial = np.sqrt(v1x**2 + v1y**2)
    v2_mag_inicial = np.sqrt(v2x**2 + v2y**2)
    ke1_inicial = 0.5 * m1 * v1_mag_inicial**2
    ke2_inicial = 0.5 * m2 * v2_mag_inicial**2
    ke_inicial = ke1_inicial + ke2_inicial  # ke_total_inicial
    
    p1_inicial = m1 * np.array([v1x, v1y])
    p2_inicial = m2 * np.array([v2x, v2y])
    p_total_inicial = p1_inicial + p2_inicial
    p_inicial = np.linalg.norm(p_total_inicial)  # p_total_mag_inicial
    
    # Cálculos para DESPUÉS del choque
    v1_mag_final = np.sqrt(v1x_f**2 + v1y_f**2)
    v2_mag_final = np.sqrt(v2x_f**2 + v2y_f**2)
    ke1_final = 0.5 * m1 * v1_mag_final**2
    ke2_final = 0.5 * m2 * v2_mag_final**2
    ke_final = ke1_final + ke2_final  # ke_total_final
    
    p1_final = m1 * np.array([v1x_f, v1y_f])
    p2_final = m2 * np.array([v2x_f, v2y_f])
    p_total_final = p1_final + p2_final
    p_final = np.linalg.norm(p_total_final)  # p_total_mag_final
    
    # ======================================================================
    # ANÁLISIS Y VERIFICACIONES
    # ======================================================================
    
    st.markdown("---")
    with st.container(border=True):
        st.subheader("🔬 Verificación de Leyes Físicas")
        
        col_ley1, col_ley2, col_ley3 = st.columns(3)
        
        # Verificación de conservación de momentum
        with col_ley1:
            error_momentum = abs(p_final - p_inicial)
            st.metric(
                "⚖️ Conservación de Momentum",
                "✅ SE CONSERVA" if error_momentum < 0.01 else "❌ NO SE CONSERVA",
                delta=f"Error: {error_momentum:.4f} kg·m/s"
            )
        
        # Verificación de conservación de energía
        with col_ley2:
            if e == 1.0:  # Colisión elástica
                error_energia = abs(ke_final - ke_inicial)
                st.metric(
                    "⚡ Conservación de Energía",
                    "✅ SE CONSERVA" if error_energia < 0.01 else "❌ NO SE CONSERVA",
                    delta=f"Error: {error_energia:.4f} J"
                )
            else:  # Colisión inelástica
                energia_perdida = ke_inicial - ke_final
                porcentaje_perdido = (energia_perdida / ke_inicial) * 100
                st.metric(
                    "⚡ Energía Disipada",
                    f"{energia_perdida:.3f} J",
                    delta=f"{porcentaje_perdido:.1f}% perdida"
                )
        
        # Información del tipo de choque
        with col_ley3:
            st.metric(
                "🎯 Tipo de Choque",
                "Elástica" if e == 1.0 else "Inelástica",
                delta=f"e = {e}"
            )
    
    # ======================================================================
    # ANÁLISIS DETALLADO ANTERIOR (MANTENIDO)
    # ======================================================================
    
    # Calcular ángulos post-colisión
    angulo1, angulo2 = calcular_angulos_post_colision(v1x_f, v1y_f, v2x_f, v2y_f)
    
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
        st.subheader("📊 Análisis Completo de la Colisión")
        
        # Crear dos columnas principales
        col_antes, col_despues = st.columns(2)
        
        # ============ COLUMNA IZQUIERDA - ANTES DEL CHOQUE ============
        with col_antes:
            st.markdown("### 🔵 ANTES del Choque")
            
            # Cálculos iniciales
            v1_mag_inicial = np.sqrt(v1x**2 + v1y**2)
            v2_mag_inicial = np.sqrt(v2x**2 + v2y**2)
            
            # Energía cinética inicial
            st.markdown("**⚡ Energía Cinética**")
            ke1_inicial = 0.5 * m1 * v1_mag_inicial**2
            ke2_inicial = 0.5 * m2 * v2_mag_inicial**2
            ke_total_inicial = ke1_inicial + ke2_inicial
            
            st.metric("Partícula 1", f"{ke1_inicial:.3f} J")
            st.metric("Partícula 2", f"{ke2_inicial:.3f} J")
            st.metric("Sistema Total", f"{ke_total_inicial:.3f} J", help="Energía total del sistema")
            
            # Cantidad de movimiento inicial
            st.markdown("**🎯 Cantidad de Movimiento**")
            p1_inicial = m1 * np.array([v1x, v1y])
            p2_inicial = m2 * np.array([v2x, v2y])
            p_total_inicial = p1_inicial + p2_inicial
            
            p1_mag_inicial = np.linalg.norm(p1_inicial)
            p2_mag_inicial = np.linalg.norm(p2_inicial)
            p_total_mag_inicial = np.linalg.norm(p_total_inicial)
            
            st.metric("Partícula 1", f"{p1_mag_inicial:.3f} kg·m/s")
            st.metric("Partícula 2", f"{p2_mag_inicial:.3f} kg·m/s")
            st.metric("Sistema Total", f"{p_total_mag_inicial:.3f} kg·m/s")
            
            # Velocidades iniciales
            st.markdown("**🏃 Velocidades**")
            st.metric("Partícula 1", f"{v1_mag_inicial:.3f} m/s")
            st.metric("Partícula 2", f"{v2_mag_inicial:.3f} m/s")
            
            # Componentes vectoriales (expandible)
            with st.expander("🔍 Componentes Vectoriales"):
                st.write("**Velocidades (componentes):**")
                st.write(f"v₁ = ({v1x:.2f}, {v1y:.2f}) m/s")
                st.write(f"v₂ = ({v2x:.2f}, {v2y:.2f}) m/s")
                st.write("**Momentum (componentes):**")
                st.write(f"p₁ = ({p1_inicial[0]:.2f}, {p1_inicial[1]:.2f}) kg·m/s")
                st.write(f"p₂ = ({p2_inicial[0]:.2f}, {p2_inicial[1]:.2f}) kg·m/s")
                st.write(f"p_total = ({p_total_inicial[0]:.2f}, {p_total_inicial[1]:.2f}) kg·m/s")
        
        # ============ COLUMNA DERECHA -
        # ============ COLUMNA DERECHA - DESPUÉS DEL CHOQUE ============
        with col_despues:
            st.markdown("### 🟢 DESPUÉS del Choque")
            
            # Cálculos finales
            v1_mag_final = np.sqrt(v1x_f**2 + v1y_f**2)
            v2_mag_final = np.sqrt(v2x_f**2 + v2y_f**2)
            
            # Energía cinética final
            st.markdown("**⚡ Energía Cinética**")
            ke1_final = 0.5 * m1 * v1_mag_final**2
            ke2_final = 0.5 * m2 * v2_mag_final**2
            ke_total_final = ke1_final + ke2_final
            
            st.metric("Partícula 1", f"{ke1_final:.3f} J")
            st.metric("Partícula 2", f"{ke2_final:.3f} J")
            st.metric("Sistema Total", f"{ke_total_final:.3f} J", help="Energía total del sistema")
            
            # Cantidad de movimiento final
            st.markdown("**🎯 Cantidad de Movimiento**")
            p1_final = m1 * np.array([v1x_f, v1y_f])
            p2_final = m2 * np.array([v2x_f, v2y_f])
            p_total_final = p1_final + p2_final
            
            p1_mag_final = np.linalg.norm(p1_final)
            p2_mag_final = np.linalg.norm(p2_final)
            p_total_mag_final = np.linalg.norm(p_total_final)
            
            st.metric("Partícula 1", f"{p1_mag_final:.3f} kg·m/s")
            st.metric("Partícula 2", f"{p2_mag_final:.3f} kg·m/s")
            st.metric("Sistema Total", f"{p_total_mag_final:.3f} kg·m/s")
            
            # Velocidades finales
            st.markdown("**🏃 Velocidades**")
            st.metric("Partícula 1", f"{v1_mag_final:.3f} m/s")
            st.metric("Partícula 2", f"{v2_mag_final:.3f} m/s")
            
            # Ángulos de deflexión
            st.markdown("**📐 Ángulos de Deflexión**")
            st.metric("Partícula 1", f"{angulo1:.1f}°")
            st.metric("Partícula 2", f"{angulo2:.1f}°")
            
            # Componentes vectoriales (expandible)
            with st.expander("🔍 Componentes Vectoriales"):
                st.write("**Velocidades (componentes):**")
                st.write(f"v₁' = ({v1x_f:.2f}, {v1y_f:.2f}) m/s")
                st.write(f"v₂' = ({v2x_f:.2f}, {v2y_f:.2f}) m/s")
                st.write("**Momentum (componentes):**")
                st.write(f"p₁' = ({p1_final[0]:.2f}, {p1_final[1]:.2f}) kg·m/s")
                st.write(f"p₂' = ({p2_final[0]:.2f}, {p2_final[1]:.2f}) kg·m/s")
                st.write(f"p_total' = ({p_total_final[0]:.2f}, {p_total_final[1]:.2f}) kg·m/s")



# Simulación de Cálculo de Impulso y Fuerza
elif simulacion == "Cálculo de Impulso y Fuerza":
    st.header("⚡ Cálculo de Impulso y Fuerza")
    
    # Explicación física completa
    with st.expander("📚 Explicación Física - Impulso y Fuerza", expanded=True):
        st.markdown("""
        ### 🔬 Fundamentos Teóricos - Impulso y Fuerza
        
        **1. Definición de Impulso:**
        - El impulso es el cambio en la cantidad de movimiento (momentum)
        - `J = Δp = pf - pi = m(vf - vi)`
        - También se define como: `J = F × Δt` (para fuerza constante)
        
        **2. Teorema Impulso-Momentum:**
        - El impulso aplicado a un objeto es igual al cambio en su momentum
        - `F × Δt = m × Δv`
        - Esta relación conecta fuerza, tiempo y cambio de velocidad
        
        **3. Tipos de Fuerzas:**
        - **Fuerza constante:** F no cambia con el tiempo
        - **Fuerza variable:** F cambia durante la interacción
        - **Fuerza impulsiva:** Fuerza muy grande aplicada durante tiempo muy corto
        
        **4. Fuerza Promedio:**
        - Para fuerzas variables: `F_promedio = J / Δt`
        - Útil para calcular efectos de colisiones complejas
        
        **5. Relación con Colisiones:**
        - Durante una colisión, se aplica una fuerza grande durante poco tiempo
        - El impulso total determina el cambio de velocidad
        - `J = ∫ F dt` (integral de la fuerza en el tiempo)
        
        **6. Aplicaciones Prácticas:**
        - Diseño de airbags (aumentar Δt para reducir F)
        - Deportes (técnica de golpeo en tenis, béisbol)
        - Ingeniería automotriz (zonas de deformación)
        - Propulsión de cohetes
        
        **7. Unidades:**
        - Impulso: N⋅s o kg⋅m/s
        - Fuerza: N (Newton) = kg⋅m/s²
        - Momentum: kg⋅m/s
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        
        modo_calculo = st.selectbox(
            "Modo de cálculo:",
            ["Conociendo Fuerza y Tiempo", "Conociendo Cambio de Velocidad", "Análisis de Colisión"]
        )
        
        masa = st.number_input("Masa del objeto (kg)", value=5.0, min_value=0.1)
        
        if modo_calculo == "Conociendo Fuerza y Tiempo":
            fuerza = st.number_input("Fuerza aplicada (N)", value=100.0)
            tiempo = st.number_input("Tiempo de aplicación (s)", value=0.5, min_value=0.01)
            
            # Calcular impulso y cambio de velocidad
            impulso = fuerza * tiempo
            delta_v = impulso / masa
            
        elif modo_calculo == "Conociendo Cambio de Velocidad":
            vi = st.number_input("Velocidad inicial (m/s)", value=0.0)
            vf = st.number_input("Velocidad final (m/s)", value=20.0)
            tiempo = st.number_input("Tiempo de cambio (s)", value=2.0, min_value=0.01)
            
            # Calcular impulso y fuerza
            delta_v = vf - vi
            impulso = masa * delta_v
            fuerza = impulso / tiempo
            
        else:  # Análisis de Colisión
            vi = st.number_input("Velocidad antes colisión (m/s)", value=30.0)
            vf = st.number_input("Velocidad después colisión (m/s)", value=-10.0)
            tiempo_colision = st.number_input("Duración de colisión (s)", value=0.01, min_value=0.001, step=0.001, format="%.3f")
            
            # Calcular para colisión
            delta_v = vf - vi
            impulso = masa * delta_v
            fuerza = impulso / tiempo_colision
    
    with col2:
        st.subheader("Resultados del Análisis")
        
        st.metric("Impulso (J)", f"{impulso:.2f} N⋅s")
        st.metric("Fuerza", f"{fuerza:.2f} N", f"{fuerza/masa:.2f} m/s² (aceleración)")
        st.metric("Cambio de velocidad", f"{delta_v:.2f} m/s")
        
        # Análisis adicional
        if modo_calculo == "Análisis de Colisión":
            g = 9.81  # Aceleración de gravedad
            fuerza_peso = masa * g
            relacion_fuerza = fuerza / fuerza_peso
            
            st.metric("Fuerza vs Peso", f"{relacion_fuerza:.1f}x", "veces el peso del objeto")
            
            if abs(fuerza) > 1000:
                st.warning(f"⚠️ **Fuerza muy alta:** {fuerza:.0f} N puede causar daños estructurales")
            elif abs(fuerza) > 500:
                st.info(f"🔶 **Fuerza considerable:** {fuerza:.0f} N requiere estructura resistente")
            else:
                st.success(f"✅ **Fuerza moderada:** {fuerza:.0f} N es manejable")
        
        # Información sobre la magnitud de la fuerza
        if abs(fuerza) > 10000:
            st.error("🚨 **Fuerza extrema** - Comparable a impactos destructivos")
        elif abs(fuerza) > 1000:
            st.warning("⚠️ **Fuerza alta** - Requiere consideraciones de seguridad")
        elif abs(fuerza) > 100:
            st.info("🔵 **Fuerza moderada** - Típica en aplicaciones cotidianas")
        else:
            st.success("✅ **Fuerza baja** - Segura para la mayoría de aplicaciones")
    
    # Crear gráfico de fuerza vs tiempo
    st.subheader("📊 Visualización del Impulso")
    
    if modo_calculo == "Conociendo Fuerza y Tiempo":
        # Gráfico de fuerza constante
        t_grafico = np.linspace(0, tiempo * 1.5, 1000)
        f_grafico = np.where((t_grafico >= 0) & (t_grafico <= tiempo), fuerza, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_grafico, y=f_grafico,
            mode='lines',
            name='Fuerza aplicada',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            hovertemplate='Tiempo: %{x:.3f}s<br>Fuerza: %{y:.2f}N<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Fuerza vs Tiempo (Impulso = {impulso:.2f} N⋅s)",
            xaxis_title="Tiempo (s)",
            yaxis_title="Fuerza (N)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Gráfico de velocidad vs tiempo
        if modo_calculo == "Conociendo Cambio de Velocidad":
            t_grafico = np.linspace(0, tiempo, 100)
            v_grafico = vi + (delta_v / tiempo) * t_grafico
        else:  # Análisis de Colisión
            t_antes = np.linspace(-tiempo_colision*2, 0, 50)
            t_colision = np.linspace(0, tiempo_colision, 20)
            t_despues = np.linspace(tiempo_colision, tiempo_colision*3, 50)
            
            v_antes = np.full_like(t_antes, vi)
            v_colision = vi + (delta_v / tiempo_colision) * t_colision
            v_despues = np.full_like(t_despues, vf)
            
            t_grafico = np.concatenate([t_antes, t_colision, t_despues])
            v_grafico = np.concatenate([v_antes, v_colision, v_despues])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t_grafico, y=v_grafico,
            mode='lines',
            name='Velocidad',
            line=dict(color='blue', width=3),
            hovertemplate='Tiempo: %{x:.4f}s<br>Velocidad: %{y:.2f}m/s<extra></extra>'
        ))
        
        if modo_calculo == "Análisis de Colisión":
            fig.add_vline(x=0, line_dash="dash", line_color="red", 
                         annotation_text="Inicio colisión")
            fig.add_vline(x=tiempo_colision, line_dash="dash", line_color="red", 
                         annotation_text="Fin colisión")
        
        fig.update_layout(
            title=f"Velocidad vs Tiempo (Δv = {delta_v:.2f} m/s)",
            xaxis_title="Tiempo (s)",
            yaxis_title="Velocidad (m/s)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis detallado
    with st.expander("🔍 Análisis Detallado"):
        st.markdown(f"""
        ### Análisis Completo del Impulso
        
        **Parámetros:**
        - Masa del objeto: {masa} kg
        - Modo de cálculo: {modo_calculo}
        
        **Resultados:**
        - Impulso total: {impulso:.3f} N⋅s
        - Fuerza {'promedio' if modo_calculo != 'Conociendo Fuerza y Tiempo' else 'aplicada'}: {fuerza:.3f} N
        - Cambio de velocidad: {delta_v:.3f} m/s
        
        **Verificaciones físicas:**
        - Relación impulso-momentum: J = m × Δv = {masa} × {delta_v:.3f} = {impulso:.3f} N⋅s ✓
        - Relación fuerza-tiempo: F × Δt = {fuerza:.3f} × {tiempo if modo_calculo != 'Análisis de Colisión' else tiempo_colision:.3f} = {impulso:.3f} N⋅s ✓
        
        **Interpretación física:**
        - {'Una fuerza constante aplicada durante un período de tiempo' if modo_calculo == 'Conociendo Fuerza y Tiempo' else 'Un cambio de velocidad que requiere cierta fuerza promedio'}
        - El impulso representa la transferencia total de momentum al objeto
        - La magnitud de la fuerza depende inversamente del tiempo de aplicación
        """)

# Simulación de Péndulo Balístico
elif simulacion == "Péndulo Balístico":
    st.header("🎯 Péndulo Balístico")
    
    # Parámetros en la barra lateral
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚙️ Parámetros del Sistema")
        
        # Parámetros del proyectil
        m_proyectil = st.number_input("Masa del proyectil (kg)", value=0.01, min_value=0.001, step=0.001, format="%.3f")
        
        # Parámetros del péndulo
        m_pendulo = st.number_input("Masa del péndulo (kg)", value=2.0, min_value=0.1)
        L_pendulo = st.number_input("Longitud del péndulo (m)", value=1.0, min_value=0.1)
        
        # Método de entrada
        metodo_entrada = st.selectbox(
            "Método de cálculo:",
            ["Conociendo velocidad del proyectil", "Conociendo ángulo máximo", "Conociendo altura máxima"]
        )
        
        if metodo_entrada == "Conociendo velocidad del proyectil":
            v_proyectil = st.number_input("Velocidad del proyectil (m/s)", value=300.0, min_value=1.0)
            
            # Calcular velocidad después de la colisión
            v_conjunto = (m_proyectil * v_proyectil) / (m_proyectil + m_pendulo)
            
            # Calcular altura máxima
            g = 9.81
            h_max = v_conjunto**2 / (2 * g)
            
            # Calcular ángulo máximo
            if h_max <= L_pendulo:
                theta_max = np.arccos(1 - h_max/L_pendulo)
                theta_max_grados = np.degrees(theta_max)
            else:
                theta_max = np.pi
                theta_max_grados = 180.0
                h_max = L_pendulo  # Corrección física
                
        elif metodo_entrada == "Conociendo ángulo máximo":
            theta_max_grados = st.number_input("Ángulo máximo (grados)", value=30.0, min_value=0.1, max_value=90.0)
            theta_max = np.radians(theta_max_grados)
            
            # Calcular altura máxima
            h_max = L_pendulo * (1 - np.cos(theta_max))
            
            # Calcular velocidad del conjunto después de la colisión
            g = 9.81
            v_conjunto = np.sqrt(2 * g * h_max)
            
            # Calcular velocidad original del proyectil
            v_proyectil = v_conjunto * (m_proyectil + m_pendulo) / m_proyectil
            
        else:  # Conociendo altura máxima
            h_max = st.number_input("Altura máxima (m)", value=0.5, min_value=0.01, max_value=float(L_pendulo))
            
            # Calcular ángulo máximo
            if h_max <= L_pendulo:
                theta_max = np.arccos(1 - h_max/L_pendulo)
                theta_max_grados = np.degrees(theta_max)
            else:
                st.error("⚠️ La altura no puede ser mayor que la longitud del péndulo")
                h_max = L_pendulo
                theta_max = np.pi/2
                theta_max_grados = 90.0
            
            # Calcular velocidad del conjunto
            g = 9.81
            v_conjunto = np.sqrt(2 * g * h_max)
            
            # Calcular velocidad original del proyectil
            v_proyectil = v_conjunto * (m_proyectil + m_pendulo) / m_proyectil
    
    # Explicación física completa
    with st.expander("📚 Explicación Física - Péndulo Balístico", expanded=True):
        st.markdown("""
        ### 🔬 Fundamentos Teóricos - Péndulo Balístico
        
        **1. Concepto del Péndulo Balístico:**
        - Dispositivo usado para medir la velocidad de proyectiles
        - Consiste en un proyectil que impacta un péndulo masivo
        - La colisión es perfectamente inelástica (proyectil se incrusta)
        
        **2. Física del Proceso:**
        - **Fase 1:** Colisión inelástica (conservación de momentum)
        - **Fase 2:** Movimiento pendular (conservación de energía)
        
        **3. Ecuaciones Fundamentales:**
        
        **Fase de Colisión (Conservación de Momentum):**
        - Antes: `p_inicial = m_proyectil × v_proyectil`
        - Después: `p_final = (m_proyectil + m_péndulo) × v_conjunto`
        - `m₁v₁ = (m₁ + m₂)v'`
        
        **Fase Pendular (Conservación de Energía):**
        - Energía cinética → Energía potencial
        - `½(m₁ + m₂)v'² = (m₁ + m₂)gh`
        - `v' = √(2gh)`
        
        **4. Altura Máxima:**
        - `h = L(1 - cos θ)` donde L = longitud del péndulo
        - θ = ángulo máximo de oscilación
        
        **5. Velocidad del Proyectil:**
        - Combinando las ecuaciones:
        - `v₁ = ((m₁ + m₂)/m₁) × √(2gh)`
        - `v₁ = ((m₁ + m₂)/m₁) × √(2gL(1 - cos θ))`
        
        **6. Ventajas del Método:**
        - No requiere cronómetros de alta precisión
        - Solo necesita medir masa, longitud y ángulo
        - Muy preciso para proyectiles de alta velocidad
        
        **7. Aplicaciones Históricas:**
        - Medición de velocidad de balas de cañón
        - Determinación de propiedades balísticas
        - Estudios de física experimental del siglo XVIII-XIX
        
        **8. Limitaciones:**
        - Solo funciona si el proyectil se incrusta (colisión inelástica)
        - Pérdidas de energía por fricción y deformación
        - Efectos de rotación no considerados en el modelo simple
        """)
    
    # Crear visualización animada del péndulo balístico
    st.subheader("🎬 Animación del Péndulo Balístico")
    
    # Crear animación
    n_frames = 100
    t_total = 4.0  # Duración total de la animación
    
    frames = []
    for i in range(n_frames):
        t = i / n_frames * t_total
        
        if t <= 1.0:  # Proyectil aproximándose
            # Proyectil moviéndose hacia el péndulo
            x_proyectil = -2 + 2*t
            y_proyectil = -1
            
            # Péndulo en reposo
            x_pendulo = L_pendulo * np.sin(0)
            y_pendulo = -L_pendulo * np.cos(0)
            
        elif t <= 1.1:  # Momento de impacto
            x_proyectil = 0
            y_proyectil = -1
            x_pendulo = 0
            y_pendulo = -L_pendulo
            
        else:  # Péndulo oscilando
            t_oscilacion = t - 1.1
            periodo = 2 * np.pi * np.sqrt(L_pendulo / g)
            frecuencia = 2 * np.pi / periodo
            
            # Ángulo del péndulo (oscilación amortiguada)
            amortiguamiento = np.exp(-0.1 * t_oscilacion)
            angulo = theta_max * np.cos(frecuencia * t_oscilacion) * amortiguamiento
            
            x_pendulo = L_pendulo * np.sin(angulo)
            y_pendulo = -L_pendulo * np.cos(angulo)
            
            # Proyectil incrustado en el péndulo
            x_proyectil = x_pendulo
            y_proyectil = y_pendulo
        
        # Crear frame
        frame_data = [
            # Hilo del péndulo
            go.Scatter(
                x=[0, x_pendulo], y=[0, y_pendulo],
                mode='lines',
                line=dict(color='black', width=2),
                name='Hilo',
                showlegend=False
            ),
            # Masa del péndulo
            go.Scatter(
                x=[x_pendulo], y=[y_pendulo],
                mode='markers',
                marker=dict(size=20, color='blue'),
                name='Péndulo',
                showlegend=False
            ),
            # Proyectil (si está separado)
            go.Scatter(
                x=[x_proyectil] if t <= 1.0 else [],
                y=[y_proyectil] if t <= 1.0 else [],
                mode='markers',
                marker=dict(size=20, color='red'),
                name='Proyectil',
                showlegend=False
            )
        ]
        
        frames.append(go.Frame(data=frame_data))
    
    # Crear figura inicial
    fig = go.Figure(
        data=[
            go.Scatter(x=[0, 0], y=[0, -L_pendulo], mode='lines', 
                      line=dict(color='black', width=2), name='Hilo'),
            go.Scatter(x=[0], y=[-L_pendulo], mode='markers', 
                      marker=dict(size=20, color='blue'), name='Péndulo'),
            go.Scatter(x=[-2], y=[-1], mode='markers', 
                      marker=dict(size=20, color='red'), name='Proyectil')
        ],
        frames=frames
    )                         
    fig.update_layout(
        xaxis=dict(range=[-2.2, 2.2], title="Posición X (m)"),
        yaxis=dict(range=[-L_pendulo-0.2, 0.5], title="Posición Y (m)"),
        title="Simulación del Péndulo Balístico",
        showlegend=True,
        updatemenus=[{
            "buttons": [
    {
        "args": [None, {"frame": {"duration": 50, "redraw": True},
                "fromcurrent": True, "transition": {"duration": 300}}],
        "label": "▶️ Play",
        "method": "animate"
    },
    {
        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                "mode": "immediate", "transition": {"duration": 0}}],
        "label": "⏸️ Pause",
        "method": "animate"
    },
    {
        "args": [{"frame": {"duration": 0, "redraw": True}}, 
                {"frame": {"duration": 0}, "mode": "immediate", 
                 "fromcurrent": False, "transition": {"duration": 0}}],
        "label": "⏹️ Stop",
        "method": "animate"
    }
],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        height=500
    )
    
    # Mantener proporción
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========== SECCIÓN DE RESULTADOS EN DOS COLUMNAS ==========
    st.subheader("📊 Resultados del Análisis")
    
    # Calcular todas las energías necesarias
    # Energía cinética inicial del proyectil
    E_cin_inicial = 0.5 * m_proyectil * v_proyectil**2
    
    # Energía cinética después de la colisión
    E_cin_despues = 0.5 * (m_proyectil + m_pendulo) * v_conjunto**2
    
    # Energía potencial máxima
    E_pot_max = (m_proyectil + m_pendulo) * g * h_max
    
    # Energía perdida en la colisión
    E_perdida = E_cin_inicial - E_cin_despues
    
    # Relación de masas
    relacion_masas = (m_proyectil + m_pendulo) / m_proyectil
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Resultados Principales")
        
        # Mostrar resultados principales
        st.metric("Velocidad del proyectil", f"{v_proyectil:.1f} m/s")
        st.metric("Velocidad después colisión", f"{v_conjunto:.3f} m/s")
        st.metric("Altura máxima", f"{h_max:.3f} m")
        st.metric("Ángulo máximo", f"{theta_max_grados:.1f}°")
        
        # Verificaciones
        error_energia = abs(E_cin_despues - E_pot_max)
        if error_energia < 0.01:
            st.success("✅ **Energía conservada en fase pendular**")
        else:
            st.warning(f"⚠️ **Error energético:** {error_energia:.3f} J")
        
        # Relación de masas
        st.metric("Factor de amplificación", f"{relacion_masas:.2f}x")
    
    with col2:
        st.markdown("### ⚡ Análisis Energético")
        
        st.metric("Energía inicial", f"{E_cin_inicial:.2f} J")
        st.metric("Energía después colisión", f"{E_cin_despues:.2f} J")
        st.metric("Energía perdida", f"{E_perdida:.2f} J", 
                 delta=f"{(E_perdida/E_cin_inicial)*100:.1f}%")
        
        # Información adicional sobre eficiencia
        eficiencia = (E_cin_despues / E_cin_inicial) * 100
        st.metric("Eficiencia de transferencia", f"{eficiencia:.1f}%")
        
        # Período de oscilación
        periodo = 2 * np.pi * np.sqrt(L_pendulo / g)
        st.metric("Período de oscilación", f"{periodo:.2f} s")
    
    # ========== GRÁFICAS DEBAJO EN DOS COLUMNAS ==========
    st.markdown("---")  # Separador visual
    
    col_graf1, col_graf2 = st.columns(2)
    
    with col_graf1:
        st.subheader("📊 Energía vs Tiempo")
        
        # Crear gráfico de energía
        t_energia = np.linspace(0, 4, 200)
        E_cinetica = []
        E_potencial = []
        E_total = []
        
        for t in t_energia:
            if t <= 1.0:  # Proyectil aproximándose
                E_k = 0.5 * m_proyectil * v_proyectil**2
                E_p = 0
                E_t = E_k
            elif t <= 1.1:  # Impacto
                E_k = 0.5 * (m_proyectil + m_pendulo) * v_conjunto**2
                E_p = 0
                E_t = E_k
            else:  # Oscilación
                t_osc = t - 1.1
                periodo = 2 * np.pi * np.sqrt(L_pendulo / g)
                frecuencia = 2 * np.pi / periodo
                amort = np.exp(-0.1 * t_osc)
                angulo = theta_max * np.cos(frecuencia * t_osc) * amort
                
                altura_actual = L_pendulo * (1 - np.cos(abs(angulo)))
                velocidad_angular = -theta_max * frecuencia * np.sin(frecuencia * t_osc) * amort
                velocidad_lineal = abs(velocidad_angular) * L_pendulo
                
                E_k = 0.5 * (m_proyectil + m_pendulo) * velocidad_lineal**2
                E_p = (m_proyectil + m_pendulo) * g * altura_actual
                E_t = E_k + E_p
            
            E_cinetica.append(E_k)
            E_potencial.append(E_p)
            E_total.append(E_t)
        
        fig_energia = go.Figure()
        fig_energia.add_trace(go.Scatter(
            x=t_energia, y=E_cinetica,
            mode='lines', name='Energía Cinética',
            line=dict(color='red', width=2)
        ))
        fig_energia.add_trace(go.Scatter(
            x=t_energia, y=E_potencial,
            mode='lines', name='Energía Potencial',
            line=dict(color='blue', width=2)
        ))
        fig_energia.add_trace(go.Scatter(
            x=t_energia, y=E_total,
            mode='lines', name='Energía Total',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig_energia.add_vline(x=1.0, line_dash="dot", line_color="orange", 
                             annotation_text="Impacto")
        
        fig_energia.update_layout(
            title="Análisis Energético del Sistema",
            xaxis_title="Tiempo (s)",
            yaxis_title="Energía (J)",
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_energia, use_container_width=True)
    
    with col_graf2:
        st.subheader("📈 Ángulo vs Tiempo")
        
        # Gráfico del ángulo de oscilación
        t_angulo = np.linspace(1.1, 4, 200)
        angulos = []
        
        for t in t_angulo:
            t_osc = t - 1.1
            periodo = 2 * np.pi * np.sqrt(L_pendulo / g)
            frecuencia = 2 * np.pi / periodo
            amort = np.exp(-0.1 * t_osc)
            angulo = theta_max * np.cos(frecuencia * t_osc) * amort
            angulos.append(np.degrees(angulo))
        
        fig_angulo = go.Figure()
        fig_angulo.add_trace(go.Scatter(
            x=t_angulo, y=angulos,
            mode='lines',
            name='Ángulo de oscilación',
            line=dict(color='purple', width=2)
        ))
        
        fig_angulo.add_hline(y=theta_max_grados, line_dash="dash", line_color="red",
                            annotation_text=f"Ángulo máximo: {theta_max_grados:.1f}°")
        fig_angulo.add_hline(y=-theta_max_grados, line_dash="dash", line_color="red")
        
        fig_angulo.update_layout(
            title="Oscilación del Péndulo",
            xaxis_title="Tiempo (s)",
            yaxis_title="Ángulo (grados)",
            height=400,
            annotations=[
                dict(
                    x=3.5, y=theta_max_grados*0.7,
                    text=f"Amplitud inicial: {theta_max_grados:.1f}°<br>Período: {periodo:.2f}s",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="purple",
                    borderwidth=1
                )
            ]
        )
        
        st.plotly_chart(fig_angulo, use_container_width=True)
    
    # Análisis detallado
    with st.expander("🔍 Análisis Detallado del Péndulo Balístico"):
        st.markdown(f"""
        ### Análisis Completo del Péndulo Balístico
        
        **Parámetros del Sistema:**
        - Masa del proyectil: {m_proyectil:.3f} kg
        - Masa del péndulo: {m_pendulo:.1f} kg
        - Longitud del péndulo: {L_pendulo:.1f} m
        - Relación de masas: {(m_proyectil + m_pendulo)/m_proyectil:.2f}
        
        **Resultados Experimentales:**
        - Velocidad inicial del proyectil: {v_proyectil:.1f} m/s
        - Velocidad después del impacto: {v_conjunto:.3f} m/s
        - Altura máxima alcanzada: {h_max:.3f} m
        - Ángulo máximo: {theta_max_grados:.1f}°
        
        **Análisis Energético:**
        - Energía cinética inicial: {E_cin_inicial:.2f} J
        - Energía cinética después del impacto: {E_cin_despues:.2f} J
        - Energía potencial máxima: {E_pot_max:.2f} J
        - Energía perdida en la colisión: {E_perdida:.2f} J ({(E_perdida/E_cin_inicial)*100:.1f}%)
        
        **Verificaciones Físicas:**
        - Conservación de momentum: ✓ {m_proyectil:.3f} × {v_proyectil:.1f} = {(m_proyectil + m_pendulo):.3f} × {v_conjunto:.3f}
        - Conservación de energía (fase pendular): ✓ {E_cin_despues:.2f} J ≈ {E_pot_max:.2f} J
        - Error energético: {abs(E_cin_despues - E_pot_max):.4f} J
        
        **Parámetros de Oscilación:**
        - Período natural: {2 * np.pi * np.sqrt(L_pendulo / g):.2f} s
        - Frecuencia natural: {1/(2 * np.pi * np.sqrt(L_pendulo / g)):.2f} Hz
        - Amplitud inicial: {theta_max_grados:.1f}°
        
        **Interpretación Balística:**
        - El péndulo balístico convierte el momentum del proyectil en altura observable
        - La amplificación de velocidad permite medir proyectiles muy rápidos
        - Factor de amplificación: {relacion_masas:.2f}x (proyectil debe ser mucho más ligero)
        - Pérdida energética del {(E_perdida/E_cin_inicial)*100:.1f}% es típica en colisiones inelásticas
        
        **Aplicaciones Históricas:**
        - Medición de velocidad de balas de mosquete y cañón
        - Determinación de propiedades balísticas en los siglos XVIII-XIX
        - Base para el desarrollo de cronómetros balísticos modernos
        """)
    
    # Comparación con otros métodos
    st.subheader("⚖️ Comparación con Otros Métodos")
    
    # Crear tabla comparativa
    metodos_data = {
        "Método": ["Péndulo Balístico", "Cronómetro Balístico", "Radar Doppler", "Photogate"],
        "Precisión": ["Alta (±2%)", "Muy Alta (±0.5%)", "Muy Alta (±0.1%)", "Alta (±1%)"],
        "Rango de Velocidad": ["50-1000 m/s", "100-2000 m/s", "10-3000 m/s", "1-500 m/s"],
        "Costo": ["Bajo", "Medio", "Alto", "Medio"],
        "Complejidad": ["Simple", "Media", "Alta", "Media"],
        "Ventajas": [
            "Simple, robusto, económico",
            "Muy preciso, versátil",
            "Sin contacto, muy preciso",
            "Fácil de usar, digital"
        ]
    }
    
    df_metodos = pd.DataFrame(metodos_data)
    st.dataframe(df_metodos, use_container_width=True)