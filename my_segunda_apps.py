import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import math
import time

# Configuración de página
st.set_page_config(
    page_title="Simulador de Física - Colisiones y Movimiento",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚀 Simulador Avanzado de Física")
st.markdown("### Colisiones, Impulso y Movimiento en Planos Inclinados")

# Sidebar para navegación
st.sidebar.title("🎯 Selecciona una Simulación")
opcion = st.sidebar.selectbox(
    "Tipo de Simulación:",
    [
        "🎯 Colisión 1D",
        "🌐 Colisión 2D",
        "💥 Impulso y Fuerza",
        "🏔️ Plano Inclinado + Impacto",
        "📊 Comparaciones Reales"
    ]
)

# Funciones auxiliares
def calcular_colision_1d(m1, v1, m2, v2, coef_restitucion):
    """Calcula velocidades después de colisión 1D"""
    v1_final = ((m1 - coef_restitucion * m2) * v1 + m2 * (1 + coef_restitucion) * v2) / (m1 + m2)
    v2_final = ((m2 - coef_restitucion * m1) * v2 + m1 * (1 + coef_restitucion) * v1) / (m1 + m2)
    return v1_final, v2_final

def simular_trayectoria_1d(m1, v1, m2, v2, coef_rest, tiempo_total=5):
    """Simula la trayectoria completa de colisión 1D"""
    dt = 0.01
    tiempo = np.arange(0, tiempo_total, dt)
    pos1 = np.zeros(len(tiempo))
    pos2 = np.zeros(len(tiempo))
    vel1 = np.zeros(len(tiempo))
    vel2 = np.zeros(len(tiempo))
    
    # Posiciones iniciales
    x1_inicial = -2
    x2_inicial = 2
    
    colision_ocurrida = False
    t_colision = 0
    
    for i, t in enumerate(tiempo):
        if not colision_ocurrida:
            pos1[i] = x1_inicial + v1 * t
            pos2[i] = x2_inicial + v2 * t
            vel1[i] = v1
            vel2[i] = v2
            
            # Detectar colisión
            if pos1[i] >= pos2[i] and v1 > v2:
                colision_ocurrida = True
                t_colision = t
                v1_new, v2_new = calcular_colision_1d(m1, v1, m2, v2, coef_rest)
                v1, v2 = v1_new, v2_new
        else:
            t_post = t - t_colision
            pos1[i] = pos1[i-1] + v1 * dt
            pos2[i] = pos2[i-1] + v2 * dt
            vel1[i] = v1
            vel2[i] = v2
    
    return tiempo, pos1, pos2, vel1, vel2, t_colision

def calcular_colision_2d(m1, v1x, v1y, m2, v2x, v2y, coef_rest):
    """Calcula velocidades después de colisión 2D"""
    # Componente normal (línea de centros)
    v1_final_x = ((m1 - coef_rest * m2) * v1x + m2 * (1 + coef_rest) * v2x) / (m1 + m2)
    v1_final_y = v1y  # Componente tangencial se conserva
    v2_final_x = ((m2 - coef_rest * m1) * v2x + m1 * (1 + coef_rest) * v1x) / (m1 + m2)
    v2_final_y = v2y
    
    return v1_final_x, v1_final_y, v2_final_x, v2_final_y

# SIMULACIÓN 1D
if opcion == "🎯 Colisión 1D":
    st.header("🎯 Simulación de Colisión 1D")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Parámetros")
        
        # Controles para objeto 1
        st.markdown("**Objeto 1 (Rojo)**")
        m1 = st.slider("Masa 1 (kg)", 0.1, 10.0, 2.0, 0.1)
        v1 = st.slider("Velocidad inicial 1 (m/s)", -10.0, 10.0, 5.0, 0.1)
        
        # Controles para objeto 2
        st.markdown("**Objeto 2 (Azul)**")
        m2 = st.slider("Masa 2 (kg)", 0.1, 10.0, 1.5, 0.1)
        v2 = st.slider("Velocidad inicial 2 (m/s)", -10.0, 10.0, -2.0, 0.1)
        
        # Tipo de colisión
        tipo_colision = st.selectbox("Tipo de colisión", ["Elástica", "Inelástica"])
        coef_rest = 1.0 if tipo_colision == "Elástica" else st.slider("Coeficiente de restitución", 0.0, 1.0, 0.5, 0.01)
        
        # Botón de simulación
        if st.button("🚀 Simular Colisión 1D"):
            # Calcular resultados
            v1_final, v2_final = calcular_colision_1d(m1, v1, m2, v2, coef_rest)
            
            st.success("✅ Simulación completada!")
            st.write(f"**Velocidad final objeto 1:** {v1_final:.2f} m/s")
            st.write(f"**Velocidad final objeto 2:** {v2_final:.2f} m/s")
            
            # Conservación de momento
            momento_inicial = m1 * v1 + m2 * v2
            momento_final = m1 * v1_final + m2 * v2_final
            st.write(f"**Momento inicial:** {momento_inicial:.2f} kg⋅m/s")
            st.write(f"**Momento final:** {momento_final:.2f} kg⋅m/s")
            
            # Energía cinética
            ec_inicial = 0.5 * (m1 * v1**2 + m2 * v2**2)
            ec_final = 0.5 * (m1 * v1_final**2 + m2 * v2_final**2)
            st.write(f"**Energía inicial:** {ec_inicial:.2f} J")
            st.write(f"**Energía final:** {ec_final:.2f} J")
            st.write(f"**Energía perdida:** {ec_inicial - ec_final:.2f} J")
    
    with col2:
        st.subheader("📊 Visualización")
        
        # Simular trayectoria
        tiempo, pos1, pos2, vel1, vel2, t_colision = simular_trayectoria_1d(m1, v1, m2, v2, coef_rest)
        
        # Gráfico de posiciones
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=tiempo, y=pos1,
            mode='lines',
            name=f'Objeto 1 (m={m1}kg)',
            line=dict(color='red', width=3)
        ))
        fig1.add_trace(go.Scatter(
            x=tiempo, y=pos2,
            mode='lines',
            name=f'Objeto 2 (m={m2}kg)',
            line=dict(color='blue', width=3)
        ))
        
        # Marcar punto de colisión
        if t_colision > 0:
            fig1.add_vline(x=t_colision, line_dash="dash", line_color="green",
                          annotation_text="Colisión")
        
        fig1.update_layout(
            title="Posiciones vs Tiempo",
            xaxis_title="Tiempo (s)",
            yaxis_title="Posición (m)",
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico de velocidades
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=tiempo, y=vel1,
            mode='lines',
            name=f'Velocidad Objeto 1',
            line=dict(color='red', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=tiempo, y=vel2,
            mode='lines',
            name=f'Velocidad Objeto 2',
            line=dict(color='blue', width=3)
        ))
        
        if t_colision > 0:
            fig2.add_vline(x=t_colision, line_dash="dash", line_color="green",
                          annotation_text="Colisión")
        
        fig2.update_layout(
            title="Velocidades vs Tiempo",
            xaxis_title="Tiempo (s)",
            yaxis_title="Velocidad (m/s)",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Explicación física
    with st.expander("📚 Explicación Física - Colisión 1D"):
        st.markdown("""
        **Principios Físicos:**
        
        1. **Conservación del Momento:** El momento total del sistema se conserva antes y después de la colisión.
           - Momento = masa × velocidad
           - p₁ᵢ + p₂ᵢ = p₁f + p₂f
        
        2. **Coeficiente de Restitución (e):**
           - e = 1: Colisión elástica (se conserva la energía cinética)
           - 0 < e < 1: Colisión inelástica parcial
           - e = 0: Colisión perfectamente inelástica
        
        3. **Ecuaciones de colisión:**
           - v₁f = [(m₁-e⋅m₂)v₁ᵢ + m₂(1+e)v₂ᵢ]/(m₁+m₂)
           - v₂f = [(m₂-e⋅m₁)v₂ᵢ + m₁(1+e)v₁ᵢ]/(m₁+m₂)
        """)

# SIMULACIÓN 2D
elif opcion == "🌐 Colisión 2D":
    st.header("🌐 Simulación de Colisión 2D")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Parámetros 2D")
        
        # Objeto 1
        st.markdown("**Objeto 1 (Rojo)**")
        m1_2d = st.slider("Masa 1 (kg)", 0.1, 10.0, 2.0, 0.1, key="m1_2d")
        v1x = st.slider("Velocidad X₁ (m/s)", -10.0, 10.0, 5.0, 0.1)
        v1y = st.slider("Velocidad Y₁ (m/s)", -10.0, 10.0, 0.0, 0.1)
        
        # Objeto 2
        st.markdown("**Objeto 2 (Azul)**")
        m2_2d = st.slider("Masa 2 (kg)", 0.1, 10.0, 1.5, 0.1, key="m2_2d")
        v2x = st.slider("Velocidad X₂ (m/s)", -10.0, 10.0, -3.0, 0.1)
        v2y = st.slider("Velocidad Y₂ (m/s)", -10.0, 10.0, 2.0, 0.1)
        
        coef_rest_2d = st.slider("Coeficiente de restitución", 0.0, 1.0, 0.8, 0.01, key="coef_2d")
        
        if st.button("🚀 Simular Colisión 2D"):
            # Calcular velocidades finales
            v1fx, v1fy, v2fx, v2fy = calcular_colision_2d(m1_2d, v1x, v1y, m2_2d, v2x, v2y, coef_rest_2d)
            
            st.success("✅ Simulación 2D completada!")
            st.write(f"**Objeto 1 final:** ({v1fx:.2f}, {v1fy:.2f}) m/s")
            st.write(f"**Objeto 2 final:** ({v2fx:.2f}, {v2fy:.2f}) m/s")
    
    with col2:
        st.subheader("📊 Trayectorias 2D")
        
        # Simular trayectorias 2D
        t_max = 3.0
        dt = 0.01
        tiempo_2d = np.arange(0, t_max, dt)
        
        # Posiciones antes de la colisión
        x1_pre = -5 + v1x * tiempo_2d[:len(tiempo_2d)//2]
        y1_pre = 0 + v1y * tiempo_2d[:len(tiempo_2d)//2]
        x2_pre = 5 + v2x * tiempo_2d[:len(tiempo_2d)//2]
        y2_pre = 0 + v2y * tiempo_2d[:len(tiempo_2d)//2]
        
        # Punto de colisión (simplificado)
        t_col = t_max / 2
        x_col, y_col = 0, 0
        
        # Calcular velocidades post-colisión
        v1fx, v1fy, v2fx, v2fy = calcular_colision_2d(m1_2d, v1x, v1y, m2_2d, v2x, v2y, coef_rest_2d)
        
        # Posiciones después de la colisión
        tiempo_post = tiempo_2d[len(tiempo_2d)//2:]
        x1_post = x_col + v1fx * (tiempo_post - t_col)
        y1_post = y_col + v1fy * (tiempo_post - t_col)
        x2_post = x_col + v2fx * (tiempo_post - t_col)
        y2_post = y_col + v2fy * (tiempo_post - t_col)
        
        # Crear gráfico 2D
        fig_2d = go.Figure()
        
        # Trayectorias pre-colisión
        fig_2d.add_trace(go.Scatter(
            x=x1_pre, y=y1_pre,
            mode='lines',
            name='Objeto 1 (pre)',
            line=dict(color='red', width=3, dash='solid')
        ))
        fig_2d.add_trace(go.Scatter(
            x=x2_pre, y=y2_pre,
            mode='lines',
            name='Objeto 2 (pre)',
            line=dict(color='blue', width=3, dash='solid')
        ))
        
        # Trayectorias post-colisión
        fig_2d.add_trace(go.Scatter(
            x=x1_post, y=y1_post,
            mode='lines',
            name='Objeto 1 (post)',
            line=dict(color='red', width=3, dash='dash')
        ))
        fig_2d.add_trace(go.Scatter(
            x=x2_post, y=y2_post,
            mode='lines',
            name='Objeto 2 (post)',
            line=dict(color='blue', width=3, dash='dash')
        ))
        
        # Punto de colisión
        fig_2d.add_trace(go.Scatter(
            x=[x_col], y=[y_col],
            mode='markers',
            name='Colisión',
            marker=dict(color='green', size=15, symbol='star')
        ))
        
        fig_2d.update_layout(
            title="Trayectorias de Colisión 2D",
            xaxis_title="Posición X (m)",
            yaxis_title="Posición Y (m)",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_2d, use_container_width=True)

# IMPULSO Y FUERZA
elif opcion == "💥 Impulso y Fuerza":
    st.header("💥 Cálculo de Impulso y Fuerza Promedio")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Parámetros de Impulso")
        
        masa_imp = st.slider("Masa del objeto (kg)", 0.1, 50.0, 5.0, 0.1)
        v_inicial_imp = st.slider("Velocidad inicial (m/s)", -20.0, 20.0, 0.0, 0.1)
        v_final_imp = st.slider("Velocidad final (m/s)", -20.0, 20.0, 10.0, 0.1)
        tiempo_contacto = st.slider("Tiempo de contacto (s)", 0.001, 2.0, 0.1, 0.001)
        
        # Cálculos
        delta_v = v_final_imp - v_inicial_imp
        impulso = masa_imp * delta_v
        fuerza_promedio = impulso / tiempo_contacto
        
        st.subheader("📊 Resultados")
        st.metric("Cambio de velocidad", f"{delta_v:.2f} m/s")
        st.metric("Impulso", f"{impulso:.2f} N⋅s")
        st.metric("Fuerza promedio", f"{fuerza_promedio:.2f} N")
        
        # Comparación con casos reales
        st.subheader("🌍 Comparaciones Reales")
        caso_real = st.selectbox("Selecciona un caso real:", ["Pelota de tenis", "Balón de fútbol", "Martillo", "Airbag"])
        if st.button("Ver comparación"):
            if caso_real == "Pelota de tenis":
                st.info("Pelota de tenis: ~150-300 N durante ~0.005s")
            elif caso_real == "Balón de fútbol":
                st.info("Balón de fútbol: ~500-800 N durante ~0.01s")
            elif caso_real == "Martillo":
                st.info("Martillo: ~2000-5000 N durante ~0.002s")
            elif caso_real == "Airbag":
                st.info("Airbag: ~1000-3000 N durante ~0.1s")
            st.info("Tu simulación: {:.0f} N durante {:.3f}s".format(fuerza_promedio, tiempo_contacto))
    
    with col2:
        st.subheader("📈 Visualización de Fuerza vs Tiempo")
        
        # Simular perfil de fuerza
        t_impulso = np.linspace(0, tiempo_contacto * 3, 300)
        fuerza_t = np.zeros_like(t_impulso)
        
        # Pulso de fuerza (modelo triangular)
        for i, t in enumerate(t_impulso):
            if t <= tiempo_contacto:
                if t <= tiempo_contacto/2:
                    fuerza_t[i] = (2 * fuerza_promedio * t) / tiempo_contacto
                else:
                    fuerza_t[i] = 2 * fuerza_promedio * (1 - t/tiempo_contacto)
        
        fig_impulso = go.Figure()
        fig_impulso.add_trace(go.Scatter(
            x=t_impulso, y=fuerza_t,
            mode='lines',
            fill='tozeroy',
            name='Fuerza aplicada',
            line=dict(color='orange', width=3)
        ))
        
        fig_impulso.add_hline(y=fuerza_promedio, line_dash="dash", 
                             annotation_text=f"Fuerza promedio: {fuerza_promedio:.1f} N")
        
        fig_impulso.update_layout(
            title="Perfil de Fuerza durante el Contacto",
            xaxis_title="Tiempo (s)",
            yaxis_title="Fuerza (N)",
            height=400
        )
        st.plotly_chart(fig_impulso, use_container_width=True)
        
        # Gráfico de velocidad vs tiempo
        t_vel = np.linspace(-tiempo_contacto, tiempo_contacto*2, 300)
        velocidad_t = np.full_like(t_vel, v_inicial_imp)
        velocidad_t[t_vel >= tiempo_contacto] = v_final_imp
        
        # Transición durante el contacto
        mask_contacto = (t_vel >= 0) & (t_vel <= tiempo_contacto)
        velocidad_t[mask_contacto] = v_inicial_imp + (v_final_imp - v_inicial_imp) * t_vel[mask_contacto] / tiempo_contacto
        
        fig_vel = go.Figure()
        fig_vel.add_trace(go.Scatter(
            x=t_vel, y=velocidad_t,
            mode='lines',
            name='Velocidad',
            line=dict(color='blue', width=3)
        ))
        
        fig_vel.update_layout(
            title="Velocidad vs Tiempo",
            xaxis_title="Tiempo (s)",
            yaxis_title="Velocidad (m/s)",
            height=300
        )
        st.plotly_chart(fig_vel, use_container_width=True)

# PLANO INCLINADO
elif opcion == "🏔️ Plano Inclinado + Impacto":
    st.header("🏔️ Caída por Plano Inclinado + Impacto")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Parámetros del Sistema")
        
        masa_plano = st.slider("Masa del objeto (kg)", 0.1, 20.0, 2.0, 0.1)
        angulo = st.slider("Ángulo del plano (°)", 0, 89, 30)
        altura_inicial = st.slider("Altura inicial (m)", 0.1, 20.0, 5.0, 0.1)
        coef_friccion = st.slider("Coeficiente de fricción", 0.0, 1.0, 0.1, 0.01)
        
        # Cálculos del plano inclinado
        g = 9.81
        angulo_rad = np.radians(angulo)
        
        # Aceleración en el plano
        a_plano = g * (np.sin(angulo_rad) - coef_friccion * np.cos(angulo_rad))
        
        # Distancia recorrida en el plano
        distancia_plano = altura_inicial / np.sin(angulo_rad)
        
        # Tiempo en el plano
        if a_plano > 0:
            tiempo_plano = np.sqrt(2 * distancia_plano / a_plano)
            velocidad_final = a_plano * tiempo_plano
        else:
            tiempo_plano = 0
            velocidad_final = 0
        
        st.subheader("📊 Resultados del Plano")
        st.metric("Aceleración", f"{a_plano:.2f} m/s²")
        st.metric("Tiempo de deslizamiento", f"{tiempo_plano:.2f} s")
        st.metric("Velocidad al final", f"{velocidad_final:.2f} m/s")
        
        # Parámetros del impacto
        st.subheader("💥 Parámetros del Impacto")
        coef_rest_impacto = st.slider("Coeficiente de restitución", 0.0, 1.0, 0.6, 0.01)
        
        velocidad_rebote = coef_rest_impacto * velocidad_final
        st.metric("Velocidad de rebote", f"{velocidad_rebote:.2f} m/s")
    
    with col2:
        st.subheader("📊 Simulación Completa")
        
        if a_plano > 0:
            # Tiempo total de simulación
            t_total = tiempo_plano + 2
            dt = 0.01
            tiempo_sim = np.arange(0, t_total, dt)
            
            # Posiciones y velocidades
            x_pos = np.zeros_like(tiempo_sim)
            y_pos = np.zeros_like(tiempo_sim)
            velocidad = np.zeros_like(tiempo_sim)
            
            for i, t in enumerate(tiempo_sim):
                if t <= tiempo_plano:
                    # Movimiento en el plano inclinado
                    s = 0.5 * a_plano * t**2
                    x_pos[i] = s * np.cos(angulo_rad)
                    y_pos[i] = altura_inicial - s * np.sin(angulo_rad)
                    velocidad[i] = a_plano * t
                else:
                    # Después del impacto (rebote)
                    t_rebote = t - tiempo_plano
                    x_pos[i] = x_pos[int(tiempo_plano/dt)]
                    y_pos[i] = velocidad_rebote * t_rebote - 0.5 * g * t_rebote**2
                    velocidad[i] = velocidad_rebote - g * t_rebote
                    
                    # No permitir que vaya bajo el suelo
                    if y_pos[i] < 0:
                        y_pos[i] = 0
                        velocidad[i] = 0
            
            # Gráfico de trayectoria
            fig_tray = go.Figure()
            fig_tray.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines',
                name='Trayectoria',
                line=dict(color='red', width=3)
            ))
            
            # Dibujar el plano inclinado
            x_plano = [0, distancia_plano * np.cos(angulo_rad)]
            y_plano = [altura_inicial, 0]
            fig_tray.add_trace(go.Scatter(
                x=x_plano, y=y_plano,
                mode='lines',
                name='Plano inclinado',
                line=dict(color='brown', width=5)
            ))
            
            # Punto de impacto
            x_impacto = distancia_plano * np.cos(angulo_rad)
            fig_tray.add_trace(go.Scatter(
                x=[x_impacto], y=[0],
                mode='markers',
                name='Punto de impacto',
                marker=dict(color='orange', size=12, symbol='star')
            ))
            
            fig_tray.update_layout(
                title="Trayectoria: Plano Inclinado + Rebote",
                xaxis_title="Posición X (m)",
                yaxis_title="Altura Y (m)",
                height=400
            )
            st.plotly_chart(fig_tray, use_container_width=True)
            
            # Gráfico de velocidad vs tiempo
            fig_vel_plano = go.Figure()
            fig_vel_plano.add_trace(go.Scatter(
                x=tiempo_sim, y=velocidad,
                mode='lines',
                name='Velocidad',
                line=dict(color='blue', width=3)
            ))
            
            fig_vel_plano.add_vline(x=tiempo_plano, line_dash="dash")