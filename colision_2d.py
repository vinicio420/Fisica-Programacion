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

