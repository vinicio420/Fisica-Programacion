import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# Cualquier otra librería que estés usando

simulacion = st.selectbox(
    "Selecciona el tipo de simulación:",
    ["Colisión 2D con Trayectorias", "Otro tipo de simulación"]
)
if simulacion == "Colisión 2D con Trayectorias":
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
        
        # Cálculo mejorado para colisión 2D usando conservación de momentum
        # Velocidad inicial del objeto 1
        v1i_mag = np.sqrt(v1x_i**2 + v1y_i**2)
        v2i_mag = np.sqrt(v2x_i**2 + v2y_i**2)
        
        # Cálculo de velocidades finales usando conservación de momentum y energía
        if v1i_mag > 0 or v2i_mag > 0:
            # Momentum inicial
            px_i = m1 * v1x_i + m2 * v2x_i
            py_i = m1 * v1y_i + m2 * v2y_i
            
            # Para simplificar, usamos una aproximación basada en colisión elástica/inelástica
            # Velocidades finales aproximadas
            if v2i_mag == 0:  # Objeto 2 en reposo
                v1f_mag = v1i_mag * (m1 - e_2d*m2) / (m1 + m2)
                v2f_mag = v1i_mag * (1 + e_2d) * m1 / (m1 + m2)
                
                # Aplicar ángulo de dispersión
                v1x_f = v1f_mag * np.cos(theta)
                v1y_f = v1f_mag * np.sin(theta)
                v2x_f = v2f_mag * np.cos(-theta/2)
                v2y_f = v2f_mag * np.sin(-theta/2)
            else:
                # Caso general - conservación de momentum
                masa_total = m1 + m2
                v1x_f = ((m1 - e_2d*m2)*v1x_i + (1+e_2d)*m2*v2x_i) / masa_total
                v1y_f = ((m1 - e_2d*m2)*v1y_i + (1+e_2d)*m2*v2y_i) / masa_total
                v2x_f = ((m2 - e_2d*m1)*v2x_i + (1+e_2d)*m1*v1x_i) / masa_total
                v2y_f = ((m2 - e_2d*m1)*v2y_i + (1+e_2d)*m1*v1y_i) / masa_total
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

    # ================================
    # SECCIÓN DE ANIMACIÓN INTERACTIVA CORREGIDA
    # ================================
    
    # Botón para controlar la animación
    col_anim1, col_anim2, col_anim3 = st.columns([1, 1, 1])
    with col_anim1:
        mostrar_animacion = st.checkbox("🎬 Mostrar Animación", value=True)
    with col_anim2:
        velocidad_animacion = st.slider("Velocidad animación (ms)", 50, 500, 150, step=50)
    with col_anim3:
        mostrar_vectores = st.checkbox("➡️ Mostrar vectores velocidad", value=True)

    if mostrar_animacion:
        # Crear animación de la colisión
        st.subheader("🎬 Animación de la Colisión 2D")
        
        # Parámetros de la animación
        fps = 20
        duracion_total = 4.0  # segundos
        frames_total = int(fps * duracion_total)
        tiempo_colision = duracion_total / 2  # La colisión ocurre a la mitad
        
        # Posiciones iniciales de los objetos (más separados)
        pos_inicial_1 = np.array([-6.0, 1.0])
        pos_inicial_2 = np.array([4.0, -1.5])
        
        # Crear figura de animación
        fig_anim = go.Figure()
        
        # Calcular todas las posiciones para determinar los límites
        tiempos = np.linspace(0, duracion_total, frames_total)
        todas_pos_x = []  
        todas_pos_y = []
        
        for t in tiempos:
            if t < tiempo_colision:
                pos_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * t
                pos_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * t
            else:
                t_post = t - tiempo_colision
                pos_colision_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * tiempo_colision
                pos_colision_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * tiempo_colision
                pos_1 = pos_colision_1 + np.array([v1x_f, v1y_f]) * t_post
                pos_2 = pos_colision_2 + np.array([v2x_f, v2y_f]) * t_post
                
            todas_pos_x.extend([pos_1[0], pos_2[0]])
            todas_pos_y.extend([pos_1[1], pos_2[1]])
        
        x_min, x_max = min(todas_pos_x) - 2, max(todas_pos_x) + 2
        y_min, y_max = min(todas_pos_y) - 2, max(todas_pos_y) + 2
        
        # Trayectorias de referencia
        t_ref = np.linspace(0, tiempo_colision, 50)
        tray_1_antes_x = pos_inicial_1[0] + v1x_i * t_ref
        tray_1_antes_y = pos_inicial_1[1] + v1y_i * t_ref
        tray_2_antes_x = pos_inicial_2[0] + v2x_i * t_ref
        tray_2_antes_y = pos_inicial_2[1] + v2y_i * t_ref
        
        pos_colision_1_ref = pos_inicial_1 + np.array([v1x_i, v1y_i]) * tiempo_colision
        pos_colision_2_ref = pos_inicial_2 + np.array([v2x_i, v2y_i]) * tiempo_colision
        tray_1_despues_x = pos_colision_1_ref[0] + v1x_f * t_ref
        tray_1_despues_y = pos_colision_1_ref[1] + v1y_f * t_ref
        tray_2_despues_x = pos_colision_2_ref[0] + v2x_f * t_ref
        tray_2_despues_y = pos_colision_2_ref[1] + v2y_f * t_ref
        
        # Añadir trayectorias de referencia
        fig_anim.add_trace(go.Scatter(
            x=tray_1_antes_x, y=tray_1_antes_y,
            mode='lines', name='Trayectoria antes',
            line=dict(color='lightblue', dash='dot', width=2),
            opacity=0.6, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_1_despues_x, y=tray_1_despues_y,
            mode='lines', name='Trayectoria después',
            line=dict(color='blue', dash='dot', width=2),
            opacity=0.6, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_2_antes_x, y=tray_2_antes_y,
            mode='lines', name='Trayectoria antes',
            line=dict(color='lightcoral', dash='dot', width=2),
            opacity=0.6, showlegend=False
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=tray_2_despues_x, y=tray_2_despues_y,
            mode='lines', name='Trayectoria después',
            line=dict(color='red', dash='dot', width=2),
            opacity=0.6, showlegend=False
        ))
        
        # Posición inicial de los objetos
        pos_1_inicial = pos_inicial_1
        pos_2_inicial = pos_inicial_2
        
        # Objetos principales
        fig_anim.add_trace(go.Scatter(
            x=[pos_1_inicial[0]], y=[pos_1_inicial[1]],
            mode='markers', name=f'Objeto 1 ({m1:.1f}kg)',
            marker=dict(size=25*np.sqrt(m1), color='blue', symbol='circle',
                       line=dict(width=3, color='darkblue')),
            hovertemplate='Objeto 1<br>Masa: %.1f kg<br>x: %%{x:.2f}m<br>y: %%{y:.2f}m<extra></extra>' % m1
        ))
        
        fig_anim.add_trace(go.Scatter(
            x=[pos_2_inicial[0]], y=[pos_2_inicial[1]],
            mode='markers', name=f'Objeto 2 ({m2:.1f}kg)',
            marker=dict(size=25*np.sqrt(m2), color='red', symbol='circle',
                       line=dict(width=3, color='darkred')),
            hovertemplate='Objeto 2<br>Masa: %.1f kg<br>x: %%{x:.2f}m<br>y: %%{y:.2f}m<extra></extra>' % m2
        ))
        
        # Vectores de velocidad iniciales
        if mostrar_vectores:
            escala_vector = 0.8
            vel_1_inicial = np.array([v1x_i, v1y_i]) * escala_vector
            vel_2_inicial = np.array([v2x_i, v2y_i]) * escala_vector
            
            # Vector objeto 1
            fig_anim.add_trace(go.Scatter(
                x=[pos_1_inicial[0], pos_1_inicial[0] + vel_1_inicial[0]],
                y=[pos_1_inicial[1], pos_1_inicial[1] + vel_1_inicial[1]],
                mode='lines+markers', name='Vector vel. Obj 1',
                line=dict(color='blue', width=4),
                marker=dict(size=[0, 15], color='blue', symbol=['circle', 'triangle-up']),
                showlegend=False
            ))
            
            # Vector objeto 2
            fig_anim.add_trace(go.Scatter(
                x=[pos_2_inicial[0], pos_2_inicial[0] + vel_2_inicial[0]],
                y=[pos_2_inicial[1], pos_2_inicial[1] + vel_2_inicial[1]],
                mode='lines+markers', name='Vector vel. Obj 2',
                line=dict(color='red', width=4),
                marker=dict(size=[0, 15], color='red', symbol=['circle', 'triangle-up']),
                showlegend=False
            ))
        
        # Crear frames para la animación
frames = []
for i in range(frames_total):
    t = tiempos[i]
    
    # Calcular posiciones
    if t < tiempo_colision:
        pos_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * t
        pos_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * t
        vel_1 = np.array([v1x_i, v1y_i])
        vel_2 = np.array([v2x_i, v2y_i])
        fase = "Antes de la colisión"
    else:
        t_post = t - tiempo_colision
        pos_colision_1 = pos_inicial_1 + np.array([v1x_i, v1y_i]) * tiempo_colision
        pos_colision_2 = pos_inicial_2 + np.array([v2x_i, v2y_i]) * tiempo_colision
        pos_1 = pos_colision_1 + np.array([v1x_f, v1y_f]) * t_post
        pos_2 = pos_colision_2 + np.array([v2x_f, v2y_f]) * t_post
        vel_1 = np.array([v1x_f, v1y_f])
        vel_2 = np.array([v2x_f, v2y_f])
        fase = "Después de la colisión"
    
    # Preparar datos del frame - ORDEN CORRECTO:
    # 0-3: trayectorias de referencia (no se actualizan)
    # 4-5: objetos principales (SE ACTUALIZAN)
    # 6-7: vectores velocidad (SE ACTUALIZAN si están habilitados)
    
    frame_data = [
        # Mantener las trayectorias de referencia (índices 0-3)
        go.Scatter(x=tray_1_antes_x, y=tray_1_antes_y, opacity=0.3),  # Más transparentes
        go.Scatter(x=tray_1_despues_x, y=tray_1_despues_y, opacity=0.3),
        go.Scatter(x=tray_2_antes_x, y=tray_2_antes_y, opacity=0.3),
        go.Scatter(x=tray_2_despues_x, y=tray_2_despues_y, opacity=0.3),
        
        # Actualizar objetos principales (índices 4-5)
        go.Scatter(
            x=[pos_1[0]], y=[pos_1[1]],
            mode='markers', 
            marker=dict(size=25*np.sqrt(m1), color='blue', symbol='circle',
                       line=dict(width=3, color='darkblue'))
        ),
        go.Scatter(
            x=[pos_2[0]], y=[pos_2[1]],
            mode='markers',
            marker=dict(size=25*np.sqrt(m2), color='red', symbol='circle',
                       line=dict(width=3, color='darkred'))
        )
    ]
    
    # Si hay vectores de velocidad, actualizar (índices 6-7)
    if mostrar_vectores:
        escala_vector = 0.8
        vel_1_escalada = vel_1 * escala_vector
        vel_2_escalada = vel_2 * escala_vector
        
        frame_data.extend([
            go.Scatter(
                x=[pos_1[0], pos_1[0] + vel_1_escalada[0]],
                y=[pos_1[1], pos_1[1] + vel_1_escalada[1]],
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=[0, 15], color='blue', symbol=['circle', 'triangle-up'])
            ),
            go.Scatter(
                x=[pos_2[0], pos_2[0] + vel_2_escalada[0]],
                y=[pos_2[1], pos_2[1] + vel_2_escalada[1]],
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=[0, 15], color='red', symbol=['circle', 'triangle-up'])
            )
        ])
    
    frames.append(go.Frame(
        data=frame_data,
        name=str(i),
        layout=go.Layout(
            title=f"Colisión 2D - t = {t:.2f}s - {fase}<br>"
                  f"Tipo: {tipo_colision_2d} (e = {e_2d:.2f}) | "
                  f"Obj1: v=({vel_1[0]:.1f}, {vel_1[1]:.1f}) m/s | "
                  f"Obj2: v=({vel_2[0]:.1f}, {vel_2[1]:.1f}) m/s"
        )
    ))
        # Crear frames para la animación
    fig_anim.frames = frames
        
        # Configurar layout de la animación
    fig_anim.update_layout(
            title=f"Animación Colisión 2D - {tipo_colision_2d} (e = {e_2d})",
            xaxis=dict(range=[x_min, x_max], title="Posición X (m)"),
            yaxis=dict(range=[y_min, y_max], title="Posición Y (m)", scaleanchor="x", scaleratio=1),
            height=700,
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': velocidad_animacion, 'redraw': True},
                            'transition': {'duration': 20},
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
                        'label': '⏹️ Reset',
                        'method': 'animate',
                        'args': [['0'], {
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
                        'args': [[str(i)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'{tiempos[i]:.1f}s',
                        'method': 'animate'
                    }
                    for i in range(0, frames_total, 5)  # Cada 5 frames en el slider
                ],
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
        
        # Información adicional sobre la animación
    st.info("""
        **Controles de la animación:**
        - ▶️ **Play**: Inicia la animación
        - ⏸️ **Pause**: Pausa la animación
        - ⏹️ **Reset**: Reinicia la animación desde el principio
        - **Slider**: Permite navegar manualmente por la animación
        
        **Elementos visualizados:**
        - **Círculos**: Representan los objetos (tamaño proporcional a la masa)
        - **Líneas punteadas**: Trayectorias de referencia
        - **Flechas**: Vectores de velocidad (si están habilitados)
        - **Colores**: Azul para objeto 1, Rojo para objeto 2
       """)

    else:
        # Mostrar gráfico estático si no se quiere animación
        st.subheader('📊 Análisis Estático de Trayectorías')
        
        # Crear visualización 2D estática con Plotly
        fig = go.Figure()
        
        # Parámetros para el gráfico estático
        tiempo_total = 3.0
        tiempo_colision_static = 1.5
        
        # Posiciones para gráfico estático
        pos_inicial_1_static = np.array([-5.0, 0.5])
        pos_inicial_2_static = np.array([3.0, -1.0])
        
        # Trayectorias antes de la colisión
        t_antes = np.linspace(0, tiempo_colision_static, 50)
        x1_antes = pos_inicial_1_static[0] + v1x_i * t_antes
        y1_antes = pos_inicial_1_static[1] + v1y_i * t_antes
        x2_antes = pos_inicial_2_static[0] + v2x_i * t_antes
        y2_antes = pos_inicial_2_static[1] + v2y_i * t_antes
        
        # Trayectorias después de la colisión
        t_despues = np.linspace(0, tiempo_total - tiempo_colision_static, 50)
        pos_colision_1_static = pos_inicial_1_static + np.array([v1x_i, v1y_i]) * tiempo_colision_static
        pos_colision_2_static = pos_inicial_2_static + np.array([v2x_i, v2y_i]) * tiempo_colision_static
        x1_despues = pos_colision_1_static[0] + v1x_f * t_despues
        y1_despues = pos_colision_1_static[1] + v1y_f * t_despues
        x2_despues = pos_colision_2_static[0] + v2x_f * t_despues
        y2_despues = pos_colision_2_static[1] + v2y_f * t_despues
        
        # Añadir trayectorias
        fig.add_trace(go.Scatter(
            x=x1_antes, y=y1_antes,
            mode='lines',
            name='Objeto 1 (antes)',
            line=dict(color='blue', dash='dash', width=3),
            hovertemplate='Objeto 1 (antes colisión)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x2_antes, y=y2_antes,
            mode='lines',
            name='Objeto 2 (antes)',
            line=dict(color='red', dash='dash', width=3),
            hovertemplate='Objeto 2 (antes colisión)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x1_despues, y=y1_despues,
            mode='lines',
            name='Objeto 1 (después)',
            line=dict(color='blue', width=4),
            hovertemplate='Objeto 1 (después colisión)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=x2_despues, y=y2_despues,
            mode='lines',
            name='Objeto 2 (después)',
            line=dict(color='red', width=4),
            hovertemplate='Objeto 2 (después colisión)<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        # Posiciones iniciales y finales
        fig.add_trace(go.Scatter(
            x=[pos_inicial_1_static[0]], y=[pos_inicial_1_static[1]],
            mode='markers',
            name='Objeto 1 (inicio)',
            marker=dict(size=20*np.sqrt(m1), color='lightblue', symbol='circle',
                       line=dict(width=2, color='blue')),
            hovertemplate='Objeto 1 - Posición inicial<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[pos_inicial_2_static[0]], y=[pos_inicial_2_static[1]],
            mode='markers',
            name='Objeto 2 (inicio)',
            marker=dict(size=20*np.sqrt(m2), color='lightcoral', symbol='circle',
                       line=dict(width=2, color='red')),
            hovertemplate='Objeto 2 - Posición inicial<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        # Punto de colisión
        fig.add_trace(go.Scatter(
            x=[pos_colision_1_static[0]], y=[pos_colision_1_static[1]],
            mode='markers',
            name='Punto de colisión',
            marker=dict(size=30, color='yellow', symbol='star',
                       line=dict(width=3, color='orange')),
            hovertemplate='Punto de colisión<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        # Posiciones finales
        x1_final = x1_despues[-1]
        y1_final = y1_despues[-1]
        x2_final = x2_despues[-1]
        y2_final = y2_despues[-1]
        
        fig.add_trace(go.Scatter(
            x=[x1_final], y=[y1_final],
            mode='markers',
            name='Objeto 1 (final)',
            marker=dict(size=25*np.sqrt(m1), color='blue', symbol='circle',
                       line=dict(width=3, color='darkblue')),
            hovertemplate='Objeto 1 - Posición final<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x2_final], y=[y2_final],
            mode='markers',
            name='Objeto 2 (final)',
            marker=dict(size=25*np.sqrt(m2), color='red', symbol='circle',
                       line=dict(width=3, color='darkred')),
            hovertemplate='Objeto 2 - Posición final<br>x: %{x:.2f}m<br>y: %{y:.2f}m<extra></extra>'
        ))
        
        # Vectores de velocidad en el punto de colisión
        if mostrar_vectores:
            escala = 0.5
            # Velocidades antes de la colisión
            fig.add_trace(go.Scatter(
                x=[pos_colision_1_static[0], pos_colision_1_static[0] + v1x_i*escala],
                y=[pos_colision_1_static[1], pos_colision_1_static[1] + v1y_i*escala],
                mode='lines+markers',
                name='v₁ inicial',
                line=dict(color='lightblue', width=3),
                marker=dict(size=[0, 12], color='lightblue', symbol=['circle', 'triangle-up']),
                hovertemplate='Velocidad inicial Obj 1<br>vₓ: %.2f m/s<br>vᵧ: %.2f m/s<extra></extra>' % (v1x_i, v1y_i)
            ))
            
            fig.add_trace(go.Scatter(
                x=[pos_colision_2_static[0], pos_colision_2_static[0] + v2x_i*escala],
                y=[pos_colision_2_static[1], pos_colision_2_static[1] + v2y_i*escala],
                mode='lines+markers',
                name='v₂ inicial',
                line=dict(color='lightcoral', width=3),
                marker=dict(size=[0, 12], color='lightcoral', symbol=['circle', 'triangle-up']),
                hovertemplate='Velocidad inicial Obj 2<br>vₓ: %.2f m/s<br>vᵧ: %.2f m/s<extra></extra>' % (v2x_i, v2y_i)
            ))
        
        # Configurar layout
        fig.update_layout(
            title=f"Análisis de Colisión 2D - {tipo_colision_2d} (e = {e_2d})",
            xaxis_title="Posición X (m)",
            yaxis_title="Posición Y (m)",
            height=600,
            hovermode='closest',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Añadir líneas de cuadrícula
        fig.update_layout(
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de resumen
        st.subheader("📋 Resumen de Resultados")
        
        datos_resumen = {
            "Parámetro": [
                "Momentum inicial X", "Momentum final X", "Momentum inicial Y", "Momentum final Y",
                "Energía cinética inicial", "Energía cinética final", "Energía perdida",
                "Velocidad inicial Obj 1", "Velocidad final Obj 1", "Velocidad inicial Obj 2", "Velocidad final Obj 2"
            ],
            "Valor": [
                f"{px_inicial:.3f} kg⋅m/s", f"{px_final:.3f} kg⋅m/s",
                f"{py_inicial:.3f} kg⋅m/s", f"{py_final:.3f} kg⋅m/s",
                f"{ke_inicial_2d:.3f} J", f"{ke_final_2d:.3f} J", f"{energia_perdida_2d:.3f} J",
                f"{np.sqrt(v1x_i**2 + v1y_i**2):.3f} m/s", f"{np.sqrt(v1x_f**2 + v1y_f**2):.3f} m/s",
                f"{np.sqrt(v2x_i**2 + v2y_i**2):.3f} m/s", f"{np.sqrt(v2x_f**2 + v2y_f**2):.3f} m/s"
            ],
            "Estado": [
                "✅" if conservacion_px else "❌", "✅" if conservacion_px else "❌",
                "✅" if conservacion_py else "❌", "✅" if conservacion_py else "❌",
                "📊", "📊", "⚠️" if energia_perdida_2d > 0.01 else "✅",
                "📍", "📍", "📍", "📍"
            ]
        }
        
        df_resumen = pd.DataFrame(datos_resumen)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        # Sección de análisis avanzado
        st.subheader("🔬 Análisis Avanzado")
        
        col_analisis1, col_analisis2 = st.columns(2)
        
        with col_analisis1:
            st.markdown("**Análisis Vectorial:**")
            
            # Ángulos de las velocidades
            angulo_v1i = np.degrees(np.arctan2(v1y_i, v1x_i)) if v1x_i != 0 else (90 if v1y_i > 0 else -90)
            angulo_v1f = np.degrees(np.arctan2(v1y_f, v1x_f)) if v1x_f != 0 else (90 if v1y_f > 0 else -90)
            angulo_v2i = np.degrees(np.arctan2(v2y_i, v2x_i)) if v2x_i != 0 else (90 if v2y_i > 0 else -90)
            angulo_v2f = np.degrees(np.arctan2(v2y_f, v2x_f)) if v2x_f != 0 else (90 if v2y_f > 0 else -90)
            
            st.write(f"🔵 **Objeto 1:**")
            st.write(f"  • Ángulo inicial: {angulo_v1i:.1f}°")
            st.write(f"  • Ángulo final: {angulo_v1f:.1f}°")
            st.write(f"  • Cambio de dirección: {abs(angulo_v1f - angulo_v1i):.1f}°")
            
            st.write(f"🔴 **Objeto 2:**")
            st.write(f"  • Ángulo inicial: {angulo_v2i:.1f}°")
            st.write(f"  • Ángulo final: {angulo_v2f:.1f}°")
            st.write(f"  • Cambio de dirección: {abs(angulo_v2f - angulo_v2i):.1f}°")
    
        with col_analisis2:
            st.markdown("**Características de la Colisión:**")
            
            # Impulso
            impulso_1_x = m1 * (v1x_f - v1x_i)
            impulso_1_y = m1 * (v1y_f - v1y_i)
            impulso_1_mag = np.sqrt(impulso_1_x**2 + impulso_1_y**2)
            
            impulso_2_x = m2 * (v2x_f - v2x_i)
            impulso_2_y = m2 * (v2y_f - v2y_i)
            impulso_2_mag = np.sqrt(impulso_2_x**2 + impulso_2_y**2)
            
            st.write(f"**Impulso sobre Objeto 1:** {impulso_1_mag:.3f} N⋅s")
            st.write(f"**Impulso sobre Objeto 2:** {impulso_2_mag:.3f} N⋅s")
            
            # Porcentaje de energía perdida
            if ke_inicial_2d > 0:
                porcentaje_perdida = (energia_perdida_2d / ke_inicial_2d) * 100
                st.write(f"**Energía perdida:** {porcentaje_perdida:.1f}%")
            
            # Clasificación de la colisión
            if e_2d >= 0.95:
                clasificacion = "Casi perfectamente elástica"
            elif e_2d >= 0.7:
                clasificacion = "Moderadamente elástica"
            elif e_2d >= 0.3:
                clasificacion = "Moderadamente inelástica"
            else:
                clasificacion = "Altamente inelástica"
            
            st.write(f"**Clasificación:** {clasificacion}")
        
        # Ejercicios propuestos
        with st.expander("📝 Ejercicios Propuestos", expanded=False):
            st.markdown("""
            ### 🎯 Ejercicios para Practicar
            
            **Ejercicio 1 - Colisión de Billar:**
            - Configura: m₁ = m₂ = 0.16 kg, v₁ᵢ = (5, 0) m/s, v₂ᵢ = (0, 0) m/s
            - Ángulo de dispersión: 30°, Colisión elástica
            - Analiza si se conserva tanto el momentum como la energía
            
            **Ejercicio 2 - Colisión de Automóviles:**
            - Configura: m₁ = 1200 kg, m₂ = 800 kg
            - v₁ᵢ = (15, 0) m/s, v₂ᵢ = (0, 10) m/s
            - Colisión inelástica con e = 0.3
            - Calcula la energía perdida y explica qué pasó con ella
            
            **Ejercicio 3 - Partículas Subatómicas:**
            - Masas muy pequeñas: m₁ = 2m₂
            - Velocidades altas: v₁ᵢ = (1000, 0) m/s, v₂ᵢ = (0, 500) m/s
            - Colisión elástica, ángulo 45°
            - Compara con colisiones macroscópicas
            
            **Desafío Avanzado:**
            - Encuentra las condiciones para que después de la colisión:
              - El objeto 1 se detenga completamente
              - El objeto 2 cambie su dirección 90°
              - La energía cinética total se reduzca a la mitad
            
            ### 🧪 Investigación Adicional
            1. ¿Cómo afecta la forma de los objetos en colisiones reales?
            2. ¿Qué papel juega la fricción en colisiones sobre superficies?
            3. ¿Cómo se analizan colisiones en 3 dimensiones?
            4. ¿Qué sucede en colisiones a velocidades relativistas?
            """)