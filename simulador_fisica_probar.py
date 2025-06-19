import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(page_title="Simulador de Impulso y Momentum", page_icon="⚡", layout="wide")

# Título principal
st.title("⚡ Desafío de Programación Física II")
st.subheader("Simulador de Impulso y Cantidad de Movimiento Lineal")

# Sidebar para navegación
st.sidebar.title("🎯 Menú de Simulaciones")
simulacion = st.sidebar.selectbox(
    "Selecciona una simulación:",
    [
        "Colisión 1D - Elástica e Inelástica",
         "Colisión 2D con Trayectorias",
        "Cálculo de Impulso y Fuerza",
        "Péndulo Balístico",
        "Caída Libre con Impacto",
        "Flecha en Saco (Inelástica)"
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
                                  "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
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
            st.metric("Momentum inicial", f"{p_inicial:.2f} kg⋅m/s")
            st.metric("Momentum final", f"{p_final:.2f} kg⋅m/s")
        with col2b:
            st.metric("Energía inicial", f"{ke_inicial:.2f} J")
            st.metric("Energía final", f"{ke_final:.2f} J")
        
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

# Simulación de Colisión 2D
elif simulacion == "Colisión 2D con Trayectorias":
    st.header("🎯 Colisión 2D con Trayectorias")
    
    # Explicación física para colisiones 2D
    with st.expander("📚 Explicación Física - Colisiones 2D", expanded=True):
        st.markdown("""
        ### 🔬 Fundamentos Teóricos - Colisiones 2D
        
        **1. Conservación del Momentum en 2D:**
        - El momentum se conserva en ambas direcciones (x e y)
        - Componente x: `m₁v₁ₓᵢ + m₂v₂ₓᵢ = m₁v₁ₓf + m₂v₂ₓf`
        - Componente y: `m₁v₁ᵧᵢ + m₂v₂ᵧᵢ = m₁v₁ᵧf + m₂v₂ᵧf`
        
        **2. Análisis Vectorial:**
        - Cada velocidad tiene componentes x e y
        - `v = √(vₓ² + vᵧ²)` (magnitud)
        - `θ = arctan(vᵧ/vₓ)` (dirección)
        
        **3. Tipos de Colisiones 2D:**
        - **Frontales:** Objetos se mueven en la misma línea
        - **Oblicuas:** Ángulo entre trayectorias ≠ 0°
        - **Tangenciales:** Contacto mínimo, pequeño cambio de dirección
        
        **4. Conservación de Energía:**
        - Elástica: `½m₁v₁ᵢ² + ½m₂v₂ᵢ² = ½m₁v₁f² + ½m₂v₂f²`
        - Inelástica: Se pierde energía cinética
        
        **5. Factores que Afectan la Colisión:**
        - **Ángulo de impacto:** Determina la dirección de las velocidades finales
        - **Punto de contacto:** Influye en la transferencia de momentum
        - **Geometría de los objetos:** Esferas, cilindros, etc.
        
        **6. Aplicaciones Prácticas:**
        - Colisiones de vehículos
        - Deportes (billar, hockey, fútbol)
        - Colisiones de partículas subatómicas
        - Simulaciones de videojuegos
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        m1 = st.number_input("Masa del objeto 1 (kg)", value=2.0, min_value=0.1, key="2d_m1")
        m2 = st.number_input("Masa del objeto 2 (kg)", value=1.5, min_value=0.1, key="2d_m2")
        
        st.markdown("**Velocidades iniciales objeto 1:**")
        v1x_i = st.number_input("Componente x (m/s)", value=4.0, key="2d_v1x")
        v1y_i = st.number_input("Componente y (m/s)", value=0.0, key="2d_v1y")
        
        st.markdown("**Velocidades iniciales objeto 2:**")
        v2x_i = st.number_input("Componente x (m/s)", value=0.0, key="2d_v2x")
        v2y_i = st.number_input("Componente y (m/s)", value=0.0, key="2d_v2y")
        
        angulo_colision = st.slider("Ángulo de dispersión (grados)", 0, 90, 45)
        tipo_colision_2d = st.selectbox("Tipo de colisión:", ["Elástica", "Inelástica"], key="tipo_2d")
        
        if tipo_colision_2d == "Inelástica":
            e_2d = st.slider("Coeficiente de restitución", 0.0, 1.0, 0.7, key="e_2d")
        else:
            e_2d = 1.0
    
    with col2:
        st.subheader("Cálculos y Resultados")
        
        # Convertir ángulo a radianes
        theta = np.radians(angulo_colision)
        
        # Cálculo simplificado para colisión 2D
        # Velocidad inicial del objeto 1
        v1i_mag = np.sqrt(v1x_i**2 + v1y_i**2)
        
        # Para simplificar, asumimos que el objeto 2 está inicialmente en reposo
        # y calculamos las velocidades finales usando conservación de momentum y energía
        
        # Velocidades finales (aproximación)
        if v1i_mag > 0:
            v1f_mag = v1i_mag * (m1 - e_2d*m2) / (m1 + m2)
            v2f_mag = v1i_mag * (1 + e_2d) * m1 / (m1 + m2)
            
            # Componentes finales
            v1x_f = v1f_mag * np.cos(theta)
            v1y_f = v1f_mag * np.sin(theta)
            v2x_f = v2f_mag * np.cos(theta + np.pi/4)
            v2y_f = v2f_mag * np.sin(theta + np.pi/4)
        else:
            v1x_f = v1y_f = v2x_f = v2y_f = 0
        
        # Mostrar resultados
        st.markdown("**Velocidades finales objeto 1:**")
        st.write(f"vₓ = {v1x_f:.2f} m/s")
        st.write(f"vᵧ = {v1y_f:.2f} m/s")
        st.write(f"Magnitud: {np.sqrt(v1x_f**2 + v1y_f**2):.2f} m/s")
        
        st.markdown("**Velocidades finales objeto 2:**")
        st.write(f"vₓ = {v2x_f:.2f} m/s")
        st.write(f"vᵧ = {v2y_f:.2f} m/s")
        st.write(f"Magnitud: {np.sqrt(v2x_f**2 + v2y_f**2):.2f} m/s")
        
        # Verificar conservación del momentum
        px_inicial = m1 * v1x_i + m2 * v2x_i
        py_inicial = m1 * v1y_i + m2 * v2y_i
        px_final = m1 * v1x_f + m2 * v2x_f
        py_final = m1 * v1y_f + m2 * v2y_f
        
        # Energías
        ke_inicial_2d = 0.5 * m1 * (v1x_i**2 + v1y_i**2) + 0.5 * m2 * (v2x_i**2 + v2y_i**2)
        ke_final_2d = 0.5 * m1 * (v1x_f**2 + v1y_f**2) + 0.5 * m2 * (v2x_f**2 + v2y_f**2)
        
        conservacion_px = abs(px_inicial - px_final) < 0.1
        conservacion_py = abs(py_inicial - py_final) < 0.1
        
        if conservacion_px:
            st.success("✅ **Momentum x conservado**")
        else:
            st.error("❌ **Error en momentum x**")
            
        if conservacion_py:
            st.success("✅ **Momentum y conservado**")
        else:
            st.error("❌ **Error en momentum y**")
        
        energia_perdida_2d = ke_inicial_2d - ke_final_2d
        if energia_perdida_2d > 0.01:
            st.warning(f"⚠️ **Energía perdida:** {energia_perdida_2d:.2f} J")
        else:
            st.success("✅ **Energía conservada**")

    # Crear visualización 2D con Plotly (gráfico estático de referencia)
    fig = go.Figure()
    
    # Trayectorias antes de la colisión
    t_antes = np.linspace(-1, 0, 50)
    x1_antes = v1x_i * t_antes
    y1_antes = v1y_i * t_antes
    x2_antes = v2x_i * t_antes
    y2_antes = v2y_i * t_antes
    
    # Trayectorias después de la colisión
    t_despues = np.linspace(0, 2, 100)
    x1_despues = v1x_f * t_despues
    y1_despues = v1y_f * t_despues
    x2_despues = v2x_f * t_despues
    y2_despues = v2y_f * t_despues
    
    # Añadir trayectorias
    fig.add_trace(go.Scatter(
        x=x1_antes, y=y1_antes,
        mode='lines',
        name='Objeto 1 (antes)',
        line=dict(color='blue', dash='dash', width=2),
        opacity=0.7,
        hovertemplate='Objeto 1 (antes)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=x2_antes, y=y2_antes,
        mode='lines',
        name='Objeto 2 (antes)',
        line=dict(color='red', dash='dash', width=2),
        opacity=0.7,
        hovertemplate='Objeto 2 (antes)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=x1_despues, y=y1_despues,
        mode='lines',
        name='Objeto 1 (después)',
        line=dict(color='blue', width=3),
        hovertemplate='Objeto 1 (después)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=x2_despues, y=y2_despues,
        mode='lines',
        name='Objeto 2 (después)',
        line=dict(color='red', width=3),
        hovertemplate='Objeto 2 (después)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    # Punto de colisión
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        name='Punto de colisión',
        marker=dict(color='gold', size=15, symbol='star'),
        hovertemplate='Punto de colisión<br>x: 0m<br>y: 0m<extra></extra>'
    ))
    
    # Objetos en posiciones iniciales y finales
    fig.add_trace(go.Scatter(
        x=[x1_antes[0], x1_despues[-1]], y=[y1_antes[0], y1_despues[-1]],
        mode='markers',
        name='Posiciones Objeto 1',
        marker=dict(color='blue', size=15*np.sqrt(m1), symbol='circle'),
        hovertemplate='Objeto 1<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x2_antes[0], x2_despues[-1]], y=[y2_antes[0], y2_despues[-1]],
        mode='markers',
        name='Posiciones Objeto 2',
        marker=dict(color='red', size=15*np.sqrt(m2), symbol='circle'),
        hovertemplate='Objeto 2<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
    ))
    
    # Configurar layout
    fig.update_layout(
        title="Colisión 2D - Trayectorias y Análisis",
        xaxis_title="Posición X (m)",
        yaxis_title="Posición Y (m)",
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    # Mantener proporción
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # ================================
    # SECCIÓN DE ANIMACIÓN INTERACTIVA
    # ================================
    
    # Botón para controlar la animación
    col_anim1, col_anim2, col_anim3 = st.columns([1, 1, 1])
    with col_anim1:
        mostrar_animacion = st.checkbox("🎬 Mostrar Animación", value=True)
    with col_anim2:
        velocidad_animacion = st.slider("Velocidad animación", 50, 500, 200, step=50)
    with col_anim3:
        mostrar_vectores = st.checkbox("➡️ Mostrar vectores velocidad", value=True)

    if mostrar_animacion:
        # Crear animación de la colisión
        st.subheader("🎬 Animación de la Colisión 2D")
        
        # Parámetros de la animación
        fps = 30
        duracion_total = 4.0  # segundos
        frames_total = int(fps * duracion_total)
        tiempo_colision = duracion_total / 2  # La colisión ocurre a la mitad
        
        # Posiciones iniciales de los objetos
        pos_inicial_1 = np.array([-4.0, 0.0])
        pos_inicial_2 = np.array([2.0, -1.0])
        
        # Crear datos para todos los frames
        frames_data = []
        
        for frame in range(frames_total):
            t = frame / fps
            
            if t < tiempo_colision:
                # Antes de la colisión - movimiento con velocidades iniciales
                pos_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * t
                pos_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * t
                vel_1 = np.array([v1x_i, v1y_i])
                vel_2 = np.array([v2x_i, v2y_i])
                fase = "antes"
            else:
                # Después de la colisión - movimiento con velocidades finales
                t_post = t - tiempo_colision
                pos_colision_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * tiempo_colision
                pos_colision_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * tiempo_colision
                
                pos_1 = pos_colision_1 + np.array([v1x_f, v1y_f]) * t_post
                pos_2 = pos_colision_2 + np.array([v2x_f, v2y_f]) * t_post
                vel_1 = np.array([v1x_f, v1y_f])
                vel_2 = np.array([v2x_f, v2y_f])
                fase = "despues"
            
            frames_data.append({
                'frame': frame,
                'tiempo': t,
                'pos_1': pos_1,
                'pos_2': pos_2,
                'vel_1': vel_1,
                'vel_2': vel_2,
                'fase': fase
            })
        
        # Crear figura de animación
        fig_anim = go.Figure()
        
        # Configurar el rango de los ejes basado en las trayectorias
        todas_pos_x = []
        todas_pos_y = []
        for data in frames_data:
            todas_pos_x.extend([data['pos_1'][0], data['pos_2'][0]])
            todas_pos_y.extend([data['pos_1'][1], data['pos_2'][1]])
        
        x_min, x_max = min(todas_pos_x) - 1, max(todas_pos_x) + 1
        y_min, y_max = min(todas_pos_y) - 1, max(todas_pos_y) + 1
        
        # Crear trazas para las trayectorias completas (líneas de referencia)
        t_completo = np.linspace(0, tiempo_colision, 100)
        tray_1_antes_x = pos_inicial_1[0] + v1x_i * t_completo
        tray_1_antes_y = pos_inicial_1[1] + v1y_i * t_completo
        tray_2_antes_x = pos_inicial_2[0] + v2x_i * t_completo
        tray_2_antes_y = pos_inicial_2[1] + v2y_i * t_completo
        
        t_despues = np.linspace(0, tiempo_colision, 100)
        pos_colision_1_ref = pos_inicial_1 + np.array([v1x_i, v1y_i]) * tiempo_colision
        pos_colision_2_ref = pos_inicial_2 + np.array([v2x_i, v2y_i]) * tiempo_colision
        tray_1_despues_x = pos_colision_1_ref[0] + v1x_f * t_despues
        tray_1_despues_y = pos_colision_1_ref[1] + v1y_f * t_despues
        tray_2_despues_x = pos_colision_2_ref[0] + v2x_f * t_despues
        tray_2_despues_y = pos_colision_2_ref[1] + v2y_f * t_despues
        
        # Trayectorias de referencia (líneas punteadas)
        fig_anim.add_trace(go.Scatter(
            x=tray_1_antes_x, y=tray_1_antes_y,
            mode='lines', name='Trayectoria Obj 1 (antes)',
            line=dict(color='lightblue', dash='dot', width=1),
            opacity=0.5, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_1_despues_x, y=tray_1_despues_y,
            mode='lines', name='Trayectoria Obj 1 (después)',
            line=dict(color='blue', dash='dot', width=1),
            opacity=0.5, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_2_antes_x, y=tray_2_antes_y,
            mode='lines', name='Trayectoria Obj 2 (antes)',
            line=dict(color='lightcoral', dash='dot', width=1),
            opacity=0.5, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_2_despues_x, y=tray_2_despues_y,
            mode='lines', name='Trayectoria Obj 2 (después)',
            line=dict(color='red', dash='dot', width=1),
            opacity=0.5, showlegend=False
        ))
        
        # Objetos animados
        fig_anim.add_trace(go.Scatter(
            x=[frames_data[0]['pos_1'][0]], y=[frames_data[0]['pos_1'][1]],
            mode='markers', name=f'Objeto 1 ({m1}kg)',
            marker=dict(size=20*np.sqrt(m1), color='blue', symbol='circle',
                       line=dict(width=2, color='darkblue')),
            hovertemplate='Objeto 1<br>Masa: %.1f kg<br>x: %%{x:.2f}m<br>y: %%{y:.2f}m<extra></extra>' % m1
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=[frames_data[0]['pos_2'][0]], y=[frames_data[0]['pos_2'][1]],
            mode='markers', name=f'Objeto 2 ({m2}kg)',
            marker=dict(size=20*np.sqrt(m2), color='red', symbol='circle',
                       line=dict(width=2, color='darkred')),
            hovertemplate='Objeto 2<br>Masa: %.1f kg<br>x: %%{x:.2f}m<br>y: %%{y:.2f}m<extra></extra>' % m2
        ))
        
        # Vectores de velocidad (flechas)
        if mostrar_vectores:
            # Escalar los vectores para visualización
            escala_vector = 0.5
            
            vel_1_inicial = frames_data[0]['vel_1'] * escala_vector
            vel_2_inicial = frames_data[0]['vel_2'] * escala_vector
            
            # Flecha objeto 1
            fig_anim.add_trace(go.Scatter(
                x=[frames_data[0]['pos_1'][0], frames_data[0]['pos_1'][0] + vel_1_inicial[0]],
                y=[frames_data[0]['pos_1'][1], frames_data[0]['pos_1'][1] + vel_1_inicial[1]],
                mode='lines', name='Vector vel. Obj 1',
                line=dict(color='blue', width=4),
                showlegend=False
            ))
            
            # Punta de flecha objeto 1
            fig_anim.add_trace(go.Scatter(
                x=[frames_data[0]['pos_1'][0] + vel_1_inicial[0]],
                y=[frames_data[0]['pos_1'][1] + vel_1_inicial[1]],
                mode='markers', name='',
                marker=dict(size=12, color='blue', symbol='triangle-up'),
                showlegend=False
            ))
            
            # Flecha objeto 2
            fig_anim.add_trace(go.Scatter(
                x=[frames_data[0]['pos_2'][0], frames_data[0]['pos_2'][0] + vel_2_inicial[0]],
                y=[frames_data[0]['pos_2'][1], frames_data[0]['pos_2'][1] + vel_2_inicial[1]],
                mode='lines', name='Vector vel. Obj 2',
                line=dict(color='red', width=4),
                showlegend=False
            ))
            
            # Punta de flecha objeto 2
            fig_anim.add_trace(go.Scatter(
                x=[frames_data[0]['pos_2'][0] + vel_2_inicial[0]],
                y=[frames_data[0]['pos_2'][1] + vel_2_inicial[1]],
                mode='markers', name='',
                marker=dict(size=12, color='red', symbol='triangle-up'),
                showlegend=False
            ))
        
        # Crear frames para la animación
        frames = []
        for i, data in enumerate(frames_data):
            frame_data = []
            trace_idx = 4  # Empezar después de las trayectorias de referencia
            
            # Actualizar posición objeto 1
            frame_data.append(go.Scatter(
                x=[data['pos_1'][0]], y=[data['pos_1'][1]]
            ))
            
            # Actualizar posición objeto 2
            frame_data.append(go.Scatter(
                x=[data['pos_2'][0]], y=[data['pos_2'][1]]
            ))
            
            if mostrar_vectores:
                escala_vector = 0.5
                vel_1_escalada = data['vel_1'] * escala_vector
                vel_2_escalada = data['vel_2'] * escala_vector
                
                # Calcular ángulo para rotar la flecha
                def calcular_angulo_flecha(vel):
                    if np.linalg.norm(vel) < 0.01:  # Velocidad muy pequeña
                        return 0
                    return np.degrees(np.arctan2(vel[1], vel[0]))
                
                angulo_1 = calcular_angulo_flecha(data['vel_1'])
                angulo_2 = calcular_angulo_flecha(data['vel_2'])
                
                # Vector velocidad objeto 1
                frame_data.append(go.Scatter(
                    x=[data['pos_1'][0], data['pos_1'][0] + vel_1_escalada[0]],
                    y=[data['pos_1'][1], data['pos_1'][1] + vel_1_escalada[1]]
                ))
                
                # Punta flecha objeto 1
                frame_data.append(go.Scatter(
                    x=[data['pos_1'][0] + vel_1_escalada[0]],
                    y=[data['pos_1'][1] + vel_1_escalada[1]],
                    marker=dict(
                        size=12, color='blue', 
                        symbol='triangle-up',
                        angle=angulo_1
                    )
                ))
                
                # Vector velocidad objeto 2
                frame_data.append(go.Scatter(
                    x=[data['pos_2'][0], data['pos_2'][0] + vel_2_escalada[0]],
                    y=[data['pos_2'][1], data['pos_2'][1] + vel_2_escalada[1]]
                ))
                
                # Punta flecha objeto 2
                frame_data.append(go.Scatter(
                    x=[data['pos_2'][0] + vel_2_escalada[0]],
                    y=[data['pos_2'][1] + vel_2_escalada[1]],
                    marker=dict(
                        size=12, color='red', 
                        symbol='triangle-up',
                        angle=angulo_2
                    )
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(i),
                layout=go.Layout(
                    title=f"Colisión 2D - t = {data['tiempo']:.2f}s - Fase: {data['fase'].title()}<br>"
                          f"Tipo: {tipo_colision_2d} (e = {e_2d:.2f})"
                )
            ))
        
        fig_anim.frames = frames
        
        # Configurar layout de la animación
        fig_anim.update_layout(
            title=f"Animación Colisión 2D - {tipo_colision_2d}",
            xaxis=dict(range=[x_min, x_max], title="Posición X (m)"),
            yaxis=dict(range=[y_min, y_max], title="Posición Y (m)", scaleanchor="x", scaleratio=1),
            height=600,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': velocidad_animacion, 'redraw': True},
                            'transition': {'duration': 50},
                            'fromcurrent': True
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏹️ Stop',
                        'method': 'animate',
                        'args': [[frames_data[0]['frame']], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top'
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[frame['name']], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'{data["tiempo"]:.1f}s',
                        'method': 'animate'
                    }
                    for frame, data in zip(frames, frames_data)
                ][::5],  # Mostrar cada 5to frame en el slider
                'active': 0,
                'currentvalue': {'prefix': 'Tiempo: '},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'xanchor': 'left',
                'yanchor': 'top'
            }]
        )
        
        st.plotly_chart(fig_anim, use_container_width=True)
    
    # Mostrar vectores de velocidad
    st.subheader("📐 Análisis Vectorial")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Vectores de Velocidad Inicial:**")
        st.write(f"Objeto 1: ({v1x_i:.2f}, {v1y_i:.2f}) m/s")
        st.write(f"Magnitud: {np.sqrt(v1x_i**2 + v1y_i**2):.2f} m/s")
        st.write(f"Objeto 2: ({v2x_i:.2f}, {v2y_i:.2f}) m/s")
        st.write(f"Magnitud: {np.sqrt(v2x_i**2 + v2y_i**2):.2f} m/s")
    
    with col4:
        st.markdown("**Vectores de Velocidad Final:**")
        st.write(f"Objeto 1: ({v1x_f:.2f}, {v1y_f:.2f}) m/s")
        st.write(f"Magnitud: {np.sqrt(v1x_f**2 + v1y_f**2):.2f} m/s")
        st.write(f"Objeto 2: ({v2x_f:.2f}, {v2y_f:.2f}) m/s")
        st.write(f"Magnitud: {np.sqrt(v2x_f**2 + v2y_f**2):.2f} m/s")

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
            y_proyectil = 0
            
            # Péndulo en reposo
            x_pendulo = L_pendulo * np.sin(0)
            y_pendulo = -L_pendulo * np.cos(0)
            
        elif t <= 1.1:  # Momento de impacto
            x_proyectil = 0
            y_proyectil = 0
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
            go.Scatter(x=[-2], y=[0], mode='markers', 
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
    
    # Calculadora de incertidumbre
    st.subheader("📏 Análisis de Incertidumbre")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**Fuentes de Error:**")
        
        error_masa_proj = st.number_input("Error en masa proyectil (%)", value=1.0, min_value=0.1, max_value=10.0)
        error_masa_pend = st.number_input("Error en masa péndulo (%)", value=0.5, min_value=0.1, max_value=5.0)
        error_longitud = st.number_input("Error en longitud (%)", value=0.5, min_value=0.1, max_value=5.0)
        error_angulo = st.number_input("Error en ángulo (grados)", value=0.5, min_value=0.1, max_value=5.0)
    
    with col6:
        st.markdown("**Propagación de Errores:**")
        
        # Calcular propagación de errores usando derivadas parciales
        # Para v₁ = ((m₁ + m₂)/m₁) × √(2gL(1 - cos θ))
        
        # Error relativo en velocidad debido a cada fuente
        error_v_masa_proj = error_masa_proj  # Error proporcional
        error_v_masa_pend = (m_pendulo/(m_proyectil + m_pendulo)) * error_masa_pend
        error_v_longitud = 0.5 * error_longitud  # Raíz cuadrada
        error_v_angulo = 0.5 * (np.sin(theta_max)/(1 - np.cos(theta_max))) * np.radians(error_angulo)
        
        # Error total (suma cuadrática)
        error_total = np.sqrt(error_v_masa_proj**2 + error_v_masa_pend**2 + 
                             error_v_longitud**2 + (np.degrees(error_v_angulo)*100)**2)
        
        st.metric("Error por masa proyectil", f"±{error_v_masa_proj:.2f}%")
        st.metric("Error por masa péndulo", f"±{error_v_masa_pend:.2f}%")
        st.metric("Error por longitud", f"±{error_v_longitud:.2f}%")
        st.metric("Error por ángulo", f"±{np.degrees(error_v_angulo)*100:.2f}%")
        st.metric("**Error total estimado**", f"**±{error_total:.2f}%**")
        
        velocidad_error = v_proyectil * error_total / 100
        st.metric("Velocidad final", f"{v_proyectil:.1f} ± {velocidad_error:.1f} m/s")

# Simulación de Cohete de Múltiples Etapas
elif simulacion == "Cohete de Múltiples Etapas":
    st.header("🚀 Cohete de Múltiples Etapas")
    
    # Explicación física completa
    with st.expander("📚 Explicación Física - Cohete de Múltiples Etapas", expanded=True):
        st.markdown("""
        ### 🔬 Fundamentos Teóricos - Cohetes de Múltiples Etapas
        
        **1. Ecuación Fundamental del Cohete (Tsiolkovsky):**
        - `Δv = v_e × ln(m_inicial / m_final)`
        - Donde v_e es la velocidad de escape de los gases
        - Esta ecuación relaciona el cambio de velocidad con la relación de masas
        
        **2. Principio de Múltiples Etapas:**
        - Cada etapa tiene su propia masa estructural y combustible
        - Al separar etapas vacías, se reduce la masa total a acelerar
        - Permite alcanzar velocidades mucho mayores que cohetes de una sola etapa
        
        **3. Ventajas del Diseño Multi-etapa:**
        - **Eficiencia:** Elimina masa muerta después de agotar combustible
        - **Velocidad:** Permite alcanzar velocidades orbitales (>7.8 km/s)
        - **Flexibilidad:** Diferentes etapas pueden optimizarse para diferentes fases
        
        **4. Ecuaciones para Múltiples Etapas:**
        - Velocidad total: `Δv_total = Σ(v_e_i × ln(m_i_inicial / m_i_final))`
        - Cada etapa contribuye independientemente al Δv total
        - La masa inicial de cada etapa incluye todas las etapas superiores
        
        **5. Parámetros Críticos:**
        - **Impulso específico (Isp):** Eficiencia del propelente
        - **Relación estructural:** Masa estructura / Masa total de la etapa
        - **Relación de masas:** Masa inicial / Masa final por etapa
        
        **6. Tipos de Combustibles:**
        - **Químicos líquidos:** H₂/O₂ (Isp ~450s), RP-1/O₂ (Isp ~350s)
        - **Químicos sólidos:** Isp ~250-280s
        - **Iónicos:** Isp ~3000-10000s (empuje muy bajo)
        
        **7. Secuencia de Vuelo:**
        - **Fase 1:** Despegue con primera etapa (máximo empuje)
        - **Separación:** Etapas se separan al agotar combustible
        - **Fases superiores:** Aceleración en vacío (mayor eficiencia)
        - **Inserción orbital:** Última etapa ajusta órbita final
        
        **8. Aplicaciones:**
        - **Lanzadores orbitales:** Falcon 9, Atlas V, Ariane 5
        - **Misiones interplanetarias:** Voyager, Cassini
        - **Misiones lunares:** Saturn V (3 etapas)
        - **Satélites:** Inserción en órbitas específicas
        """)
    
    st.subheader("Configuración del Cohete")
    
    # Número de etapas
    num_etapas = st.selectbox("Número de etapas:", [2, 3, 4], index=1)
    
    # Crear tabs para cada etapa
    tabs = st.tabs([f"Etapa {i+1}" for i in range(num_etapas)] + ["Carga Útil", "Resultados"])
    
    # Parámetros de cada etapa
    etapas = []
    
    for i in range(num_etapas):
        with tabs[i]:
            st.subheader(f"Etapa {i+1} - {'Primera' if i == 0 else 'Segunda' if i == 1 else f'{i+1}ª'} Etapa")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Masas:**")
                masa_estructura = st.number_input(f"Masa estructura (kg)", value=10000*(3-i), min_value=100, key=f"est_{i}")
                masa_combustible = st.number_input(f"Masa combustible (kg)", value=50000*(3-i), min_value=1000, key=f"comb_{i}")
                
                st.markdown("**Propulsión:**")
                isp = st.number_input(f"Impulso específico (s)", value=350-i*20, min_value=200, max_value=500, key=f"isp_{i}")
                empuje = st.number_input(f"Empuje (kN)", value=1000*(3-i), min_value=50, key=f"emp_{i}")
            
            with col2:
                st.markdown("**Cálculos:**")
                masa_total_etapa = masa_estructura + masa_combustible
                relacion_masas = masa_total_etapa / masa_estructura
                v_escape = isp * 9.81  # Convertir a m/s
                
                st.metric("Masa total etapa", f"{masa_total_etapa/1000:.1f} t")
                st.metric("Relación de masas", f"{relacion_masas:.2f}")
                st.metric("Velocidad escape", f"{v_escape:.0f} m/s")
                st.metric("Tiempo de quemado", f"{masa_combustible*isp*9.81/empuje/1000:.1f} s")
            
            # Guardar parámetros de la etapa
            etapa = {
                'masa_estructura': masa_estructura,
                'masa_combustible': masa_combustible,
                'masa_total': masa_total_etapa,
                'isp': isp,
                'v_escape': v_escape,
                'empuje': empuje * 1000,  # Convertir a N
                'tiempo_quemado': masa_combustible * isp * 9.81 / (empuje * 1000)
            }
            etapas.append(etapa)
    
    # Carga útil
    with tabs[num_etapas]:
        st.subheader("Carga Útil")
        
        masa_carga = st.number_input("Masa de carga útil (kg)", value=1000, min_value=10)
        
        # Destino
        destino = st.selectbox("Destino de la misión:", [
            "Órbita Baja Terrestre (LEO) - 200 km",
            "Órbita Geoestacionaria (GEO) - 35,786 km",
            "Escape Terrestre - Velocidad >11.2 km/s",
            "Órbita Lunar",
            "Órbita Marciana"
        ])
        
        # Velocidades características según destino
        delta_v_requerido = {
            "Órbita Baja Terrestre (LEO) - 200 km": 9400,
            "Órbita Geoestacionaria (GEO) - 35,786 km": 12500,
            "Escape Terrestre - Velocidad >11.2 km/s": 16500,
            "Órbita Lunar": 18000,
            "Órbita Marciana": 20000
        }
        
        delta_v_objetivo = delta_v_requerido[destino]
        
        st.metric("Δv requerido para la misión", f"{delta_v_objetivo/1000:.1f} km/s")
        
        # Pérdidas por gravedad y atmósfera
        perdidas_gravedad = st.number_input("Pérdidas por gravedad (m/s)", value=1500, min_value=0)
        perdidas_atmosfera = st.number_input("Pérdidas atmosféricas (m/s)", value=300, min_value=0)
        
        delta_v_total_requerido = delta_v_objetivo + perdidas_gravedad + perdidas_atmosfera
        
        st.metric("Δv total requerido", f"{delta_v_total_requerido/1000:.2f} km/s")
    
    # Resultados y análisis
    with tabs[num_etapas + 1]:
        st.subheader("Análisis de Rendimiento")
        
        # Calcular el análisis completo
        # Empezar desde la última etapa hacia abajo
        masa_actual = masa_carga
        delta_v_total = 0
        resultados_etapas = []
        
        for i in range(num_etapas-1, -1, -1):  # De la última a la primera etapa
            etapa = etapas[i]
            
            # Masa inicial de esta etapa (incluye etapas superiores + carga)
            masa_inicial = masa_actual + etapa['masa_total']
            
            # Masa final de esta etapa (sin combustible)
            masa_final = masa_actual + etapa['masa_estructura']
            
            # Delta-V de esta etapa
            delta_v_etapa = etapa['v_escape'] * np.log(masa_inicial / masa_final)
            
            # Acumulación
            delta_v_total += delta_v_etapa
            
            # Guardar resultados
            resultado = {
                'etapa': i + 1,
                'masa_inicial': masa_inicial,
                'masa_final': masa_final,
                'relacion_masas': masa_inicial / masa_final,
                'delta_v': delta_v_etapa,
                'delta_v_acumulado': delta_v_total,
                'tiempo_quemado': etapa['tiempo_quemado'],
                'aceleracion_inicial': etapa['empuje'] / masa_inicial,
                'aceleracion_final': etapa['empuje'] / masa_final
            }
            resultados_etapas.append(resultado)
            
            # Para la siguiente iteración
            masa_actual = masa_inicial
        
        # Invertir la lista para mostrar desde la primera etapa
        resultados_etapas.reverse()
        
        # Mostrar resultados principales
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Δv total alcanzado", f"{delta_v_total/1000:.2f} km/s")
            st.metric("Δv requerido", f"{delta_v_total_requerido/1000:.2f} km/s")
            
            if delta_v_total >= delta_v_total_requerido:
                st.success(f"✅ **Misión POSIBLE** - Exceso: {(delta_v_total - delta_v_total_requerido)/1000:.2f} km/s")
            else:
                st.error(f"❌ **Misión IMPOSIBLE** - Déficit: {(delta_v_total_requerido - delta_v_total)/1000:.2f} km/s")
        
        with col2:
            masa_total_cohete = resultados_etapas[0]['masa_inicial']
            relacion_carga = masa_carga / masa_total_cohete
            
            st.metric("Masa total del cohete", f"{masa_total_cohete/1000:.1f} toneladas")
            st.metric("Relación carga útil", f"{relacion_carga*100:.2f}%")
            
            if relacion_carga > 0.05:
                st.success("✅ **Eficiencia alta** (>5%)")
            elif relacion_carga > 0.02:
                st.warning("⚠️ **Eficiencia moderada** (2-5%)")
            else:
                st.error("❌ **Eficiencia baja** (<2%)")
        
        # Tabla detallada de resultados
        st.subheader("📊 Análisis Detallado por Etapas")
        
        datos_tabla = []
        for resultado in resultados_etapas:
            datos_tabla.append({
                "Etapa": f"Etapa {resultado['etapa']}",
                "Masa Inicial (t)": f"{resultado['masa_inicial']/1000:.1f}",
                "Masa Final (t)": f"{resultado['masa_final']/1000:.1f}",
                "Relación Masas": f"{resultado['relacion_masas']:.2f}",
                "Δv (km/s)": f"{resultado['delta_v']/1000:.2f}",
                "Δv Acum. (km/s)": f"{resultado['delta_v_acumulado']/1000:.2f}",
                "Tiempo (s)": f"{resultado['tiempo_quemado']:.0f}",
                "Acel. Inicial (g)": f"{resultado['aceleracion_inicial']/9.81:.1f}",
                "Acel. Final (g)": f"{resultado['aceleracion_final']/9.81:.1f}"
            })
        
        df_resultados = pd.DataFrame(datos_tabla)
        st.dataframe(df_resultados, use_container_width=True)
        
        # Gráficos de análisis
        st.subheader("📈 Análisis Gráfico")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Gráfico de velocidad vs tiempo
            tiempos = [0]
            velocidades = [0]
            tiempo_actual = 0
            
            for resultado in resultados_etapas:
                tiempo_actual += resultado['tiempo_quemado']
                tiempos.append(tiempo_actual)
                velocidades.append(resultado['delta_v_acumulado']/1000)
            
            fig_velocidad = go.Figure()
            fig_velocidad.add_trace(go.Scatter(
                x=tiempos, y=velocidades,
                mode='lines+markers',
                name='Velocidad acumulada',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Línea de velocidad objetivo
            fig_velocidad.add_hline(
                y=delta_v_total_requerido/1000,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Objetivo: {delta_v_total_requerido/1000:.1f} km/s"
            )
            
            fig_velocidad.update_layout(
                title="Velocidad vs Tiempo",
                xaxis_title="Tiempo (s)",
                yaxis_title="Velocidad (km/s)",
                height=400
            )
            
            st.plotly_chart(fig_velocidad, use_container_width=True)
        
        with col4:
            # Gráfico de masa vs tiempo
            tiempos_masa = [0]
            masas = [resultados_etapas[0]['masa_inicial']/1000]
            tiempo_actual = 0
            
            for i, resultado in enumerate(resultados_etapas):
                tiempo_actual += resultado['tiempo_quemado']
                tiempos_masa.append(tiempo_actual)
                masas.append(resultado['masa_final']/1000)
            
            fig_masa = go.Figure()
            fig_masa.add_trace(go.Scatter(
                x=tiempos_masa, y=masas,
                mode='lines+markers',
                name='Masa del cohete',
                line=dict(color='green', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(0,255,0,0.1)'
            ))
            
            fig_masa.update_layout(
                title="Masa vs Tiempo",
                xaxis_title="Tiempo (s)",
                yaxis_title="Masa (toneladas)",
                height=400
            )
            
            st.plotly_chart(fig_masa, use_container_width=True)
        
       