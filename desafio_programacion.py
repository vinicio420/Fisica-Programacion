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
st.subheader("Grupo: M ")
st.subheader("Semestre: Segundo B")
st.subheader("Docente: Ing Diego Nuñez")

# Sidebar para navegación
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
                    marker=dict(size=20*np.sqrt(m1), color='blue'),
                    name=f'Objeto 1 (m={m1}kg)',
                    hovertemplate=f'Objeto 1<br>Posición: {x1:.2f}m<br>Tiempo: {tiempo:.2f}s<extra></extra>'
                ),
                go.Scatter(
                    x=[x2], y=[0],
                    mode='markers',
                    marker=dict(size=20*np.sqrt(m2), color='red'),
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
                marker=dict(size=20*np.sqrt(m1), color='blue'),
                name=f'Objeto 1 (m={m1}kg)'
            ),
            go.Scatter(
                x=[x2_inicial], y=[0],
                mode='markers',
                marker=dict(size=20*np.sqrt(m2), color='red'),
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
    
    # Explicación física completa
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        m1 = st.number_input("Masa del objeto 1 (kg)", value=2.0, min_value=0.1)
        m2 = st.number_input("Masa del objeto 2 (kg)", value=1.0, min_value=0.1)
        v1i = st.number_input("Velocidad inicial objeto 1 (m/s)", value=5.0)
        v2i = st.number_input("Velocidad inicial objeto 2 (m/s)", value=-2.0)
        
        tipo_colision = st.selectbox("Tipo de colisión:", ["Elástica", "Inelástica", "Perfectamente Inelástica"])
        
        if tipo_colision == "Elástica":
            e = 1.0
            st.info("🔵 **Colisión Elástica:** Se conserva la energía cinética total")
        elif tipo_colision == "Perfectamente Inelástica":
            e = 0.0
            st.warning("🟡 **Colisión Perfectamente Inelástica:** Los objetos se quedan unidos")
        else:
            e = st.slider("Coeficiente de restitución (e)", 0.0, 1.0, 0.5)
            st.info(f"🟠 **Colisión Inelástica:** e = {e:.2f} - Se pierde energía cinética")
    
    with col2:
        st.subheader("Resultados")
        v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
        
        # Calcular información de la colisión
        x1_inicial = -2.0
        x2_inicial = 2.0
        t_colision, x_colision = calcular_colision(x1_inicial, x2_inicial, v1i, v2i)
        
        # Momentum antes y después
        p_inicial = m1 * v1i + m2 * v2i
        p_final = m1 * v1f + m2 * v2f
        
        # Energía cinética antes y después
        ke_inicial = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
        ke_final = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        
        # Energía cinética antes y después (del sistema)
        ke_inicial_sistema = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
        ke_final_sistema = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        
        # Momentum antes y después
        p_inicial = m1 * v1i + m2 * v2i
        p_final = m1 * v1f + m2 * v2f

        # Energía cinética antes y después
        ke_inicial = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
        ke_final = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2

        # --- CÁLCULO DEL CAMBIO EN CANTIDAD DE MOVIMIENTO (IMPULSO) ---
        delta_p1 = m1 * v1f - m1 * v1i
        delta_p2 = m2 * v2f - m2 * v2i
        # -------------------------------------------------------------
        
        # --- CÁLCULO DE ENERGÍA CINÉTICA INDIVIDUAL Y SU CAMBIO ---
        # Energía cinética individual inicial
        ke1_inicial = 0.5 * m1 * v1i**2
        ke2_inicial = 0.5 * m2 * v2i**2

        # Energía cinética individual final
        ke1_final = 0.5 * m1 * v1f**2
        ke2_final = 0.5 * m2 * v2f**2

        # Cambio en la energía cinética para cada partícula
        delta_ke1 = ke1_final - ke1_inicial
        delta_ke2 = ke2_final - ke2_inicial
        # ---------------------------------------------------------
        
        # Mostrar información de la colisión
        if t_colision is not None:
            st.success(f"⏱️ **Colisión en:** t = {t_colision:.2f} s, x = {x_colision:.2f} m")
        else:
            st.warning("⚠️ **No hay colisión** con estas velocidades")
        
        # Mostrar resultados con formato mejorado
        st.metric("Velocidad final objeto 1", f"{v1f:.2f} m/s", f"{v1f - v1i:.2f} m/s")
        st.metric("Velocidad final objeto 2", f"{v2f:.2f} m/s", f"{v2f - v2i:.2f} m/s")
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Momento inicial (Sistema)", f"{p_inicial:.2f} kg⋅m/s")
            st.metric("Momento final (Sistema)", f"{p_final:.2f} kg⋅m/s")
        with col2b:
            st.metric("Energía Cinetica inicial (Sistema)", f"{ke_inicial_sistema:.2f} J")   # ¡Asegúrate de tener esta línea!
            st.metric("Energía Cinetica final (Sistema)", f"{ke_final_sistema:.2f} J")
            
              # --- NUEVA SECCIÓN PARA IMPULSO ---
        st.subheader("Cambio en Cantidad de Movimiento (Para cada particula) 💥")
        col_delta_p1, col_delta_p2 = st.columns(2)
        with col_delta_p1:
            st.metric("Objeto 1 ($\Delta p_1$)", f"{delta_p1:.2f} kg⋅m/s")
        with col_delta_p2:
            st.metric("Objeto 2 ($\Delta p_2$)", f"{delta_p2:.2f} kg⋅m/s")
        
        # --- NUEVA SECCIÓN PARA CAMBIO EN ENERGÍA CINÉTICA POR PARTÍCULA ---
        st.subheader("Cambio en Energía Cinética (Para cada particula) ⚡")
        col_delta_ke1, col_delta_ke2 = st.columns(2)
        with col_delta_ke1:
            st.metric("Objeto 1 ($\Delta KE_1$)", f"{delta_ke1:.2f} J")
        with col_delta_ke2:
            st.metric("Objeto 2 ($\Delta KE_2$)", f"{delta_ke2:.2f} J")
        

        # Verificaciones
        conservacion_momentum = abs(p_inicial - p_final) < 0.01
        energia_perdida = ke_inicial - ke_final
        
        # Verificaciones
        conservacion_momentum = abs(p_inicial - p_final) < 0.01
        energia_perdida = ke_inicial - ke_final
        
        if conservacion_momentum:
            st.success("✅ **Momentum conservado**")
        else:
            st.error("❌ **Error en conservación del momentum**")
        
        if energia_perdida > 0.01:
            st.warning(f"⚠️ **Energía perdida:** {energia_perdida:.2f} J ({(energia_perdida/ke_inicial)*100:.1f}%)")
        elif abs(energia_perdida) < 0.01:
            st.success("✅ **Energía cinética conservada**")
    
    # Mostrar gráficos interactivos con Plotly
    st.subheader("📊 Análisis Gráfico")
    
    # Gráficos de posición y velocidad vs tiempo
    fig_analysis = crear_visualizacion_1d_plotly(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_analysis, use_container_width=True)
    
    # Animación de los objetos
    st.subheader("🎬 Animación de la Colisión")
    fig_animation = crear_animacion_objetos_plotly(m1, m2, v1i, v2i, e)
    st.plotly_chart(fig_animation, use_container_width=True)
    
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
                
                st.markdown("**Momento Lineal Final (Magnitud)**")
                # Calcular el momento lineal total final (suma vectorial de p1_final y p2_final)
                momento_total_final_x = momento_final_p1[0] + momento_final_p2[0]
                momento_total_final_y = momento_final_p1[1] + momento_final_p2[1]
                
                # Calcular la magnitud del momento lineal total final
                magnitud_momento_total_final = np.sqrt(momento_total_final_x**2 + momento_total_final_y**2)
                
                st.metric("Momento Lineal Total", f"{magnitud_momento_total_final:.2f} kg·m/s")

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros del Sistema")
        
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
    
    with col2:
        st.subheader("Resultados del Análisis")
        
        # Mostrar resultados principales
        st.metric("Velocidad del proyectil", f"{v_proyectil:.1f} m/s")
        st.metric("Velocidad después colisión", f"{v_conjunto:.3f} m/s")
        st.metric("Altura máxima", f"{h_max:.3f} m")
        st.metric("Ángulo máximo", f"{theta_max_grados:.1f}°")
        
        # Análisis energético
        # Energía cinética inicial del proyectil
        E_cin_inicial = 0.5 * m_proyectil * v_proyectil**2
        
        # Energía cinética después de la colisión
        E_cin_despues = 0.5 * (m_proyectil + m_pendulo) * v_conjunto**2
        
        # Energía potencial máxima
        E_pot_max = (m_proyectil + m_pendulo) * g * h_max
        
        # Energía perdida en la colisión
        E_perdida = E_cin_inicial - E_cin_despues
        
        st.markdown("### Análisis Energético")
        st.metric("Energía inicial", f"{E_cin_inicial:.2f} J")
        st.metric("Energía después colisión", f"{E_cin_despues:.2f} J")
        st.metric("Energía perdida", f"{E_perdida:.2f} J", f"{(E_perdida/E_cin_inicial)*100:.1f}%")
        
        # Verificaciones
        error_energia = abs(E_cin_despues - E_pot_max)
        if error_energia < 0.01:
            st.success("✅ **Energía conservada en fase pendular**")
        else:
            st.warning(f"⚠️ **Error energético:** {error_energia:.3f} J")
        
        # Relación de masas
        relacion_masas = (m_proyectil + m_pendulo) / m_proyectil
        st.metric("Factor de amplificación", f"{relacion_masas:.2f}x")
    
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
                marker=dict(size=8, color='red'),
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
                      marker=dict(size=8, color='red'), name='Proyectil')
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
    
    # Gráficos de análisis
    col3, col4 = st.columns(2)
    
    with col3:
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
            title="Análisis Energético",
            xaxis_title="Tiempo (s)",
            yaxis_title="Energía (J)",
            height=350
        )
        
        st.plotly_chart(fig_energia, use_container_width=True)
    
    with col4:
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
            height=350
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
   