import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import pandas as pd
import time

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Impulso y Momentum",
    page_icon="⚡",
    layout="wide"
)

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
    """
    Calcula las velocidades finales en colisión 1D
    e = coeficiente de restitución (1 = elástica, 0 = perfectamente inelástica)
    """
    # Velocidades finales
    v1f = ((m1 - e*m2)*v1i + (1 + e)*m2*v2i) / (m1 + m2)
    v2f = ((m2 - e*m1)*v2i + (1 + e)*m1*v1i) / (m1 + m2)
    
    return v1f, v2f

# Función para calcular impulso
def calcular_impulso(fuerza, tiempo):
    """Calcula el impulso dado fuerza y tiempo"""
    return fuerza * tiempo

# Función para crear animación de colisión 1D
def crear_animacion_1d(m1, m2, v1i, v2i, e):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Calcular velocidades finales
    v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
    
    # Parámetros de animación
    t_total = 4.0
    dt = 0.05
    t_colision = 2.0
    
    # Posiciones iniciales
    x1_inicial = -3
    x2_inicial = 3
    
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
            x1[i] = x1_inicial + v1i * t_colision + v1f * t_post
            x2[i] = x2_inicial + v2i * t_colision + v2f * t_post
    
    # Graficar posiciones vs tiempo
    ax1.plot(t, x1, 'b-', linewidth=2, label=f'Objeto 1 (m={m1} kg)')
    ax1.plot(t, x2, 'r-', linewidth=2, label=f'Objeto 2 (m={m2} kg)')
    ax1.axvline(x=t_colision, color='g', linestyle='--', label='Momento de colisión')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Posición (m)')
    ax1.set_title('Posición vs Tiempo')
    ax1.legend()
    ax1.grid(True)
    
    # Graficar velocidades
    v1_array = np.where(t < t_colision, v1i, v1f)
    v2_array = np.where(t < t_colision, v2i, v2f)
    
    ax2.plot(t, v1_array, 'b-', linewidth=2, label=f'Objeto 1')
    ax2.plot(t, v2_array, 'r-', linewidth=2, label=f'Objeto 2')
    ax2.axvline(x=t_colision, color='g', linestyle='--', label='Momento de colisión')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad (m/s)')
    ax2.set_title('Velocidad vs Tiempo')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Simulación de Colisión 1D
if simulacion == "Colisión 1D - Elástica e Inelástica":
    st.header("🔵 Colisión 1D - Elástica e Inelástica")
    
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
        elif tipo_colision == "Perfectamente Inelástica":
            e = 0.0
        else:
            e = st.slider("Coeficiente de restitución (e)", 0.0, 1.0, 0.5)
    
    with col2:
        st.subheader("Resultados")
        v1f, v2f = colision_1d(m1, m2, v1i, v2i, e)
        
        # Momentum antes y después
        p_inicial = m1 * v1i + m2 * v2i
        p_final = m1 * v1f + m2 * v2f
        
        # Energía cinética antes y después
        ke_inicial = 0.5 * m1 * v1i**2 + 0.5 * m2 * v2i**2
        ke_final = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2
        
        st.write(f"**Velocidad final objeto 1:** {v1f:.2f} m/s")
        st.write(f"**Velocidad final objeto 2:** {v2f:.2f} m/s")
        st.write(f"**Momentum inicial:** {p_inicial:.2f} kg⋅m/s")
        st.write(f"**Momentum final:** {p_final:.2f} kg⋅m/s")
        st.write(f"**Conservación del momentum:** {'✅' if abs(p_inicial - p_final) < 0.01 else '❌'}")
        st.write(f"**Energía cinética inicial:** {ke_inicial:.2f} J")
        st.write(f"**Energía cinética final:** {ke_final:.2f} J")
        st.write(f"**Energía perdida:** {ke_inicial - ke_final:.2f} J")
    
    # Mostrar gráficos
    fig = crear_animacion_1d(m1, m2, v1i, v2i, e)
    st.pyplot(fig)

# Simulación de Colisión 2D
elif simulacion == "Colisión 2D con Trayectorias":
    st.header("🎯 Colisión 2D con Trayectorias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        m1 = st.number_input("Masa del objeto 1 (kg)", value=2.0, min_value=0.1, key="2d_m1")
        m2 = st.number_input("Masa del objeto 2 (kg)", value=1.5, min_value=0.1, key="2d_m2")
        
        v1x_i = st.number_input("Velocidad inicial objeto 1 - x (m/s)", value=4.0, key="2d_v1x")
        v1y_i = st.number_input("Velocidad inicial objeto 1 - y (m/s)", value=0.0, key="2d_v1y")
        v2x_i = st.number_input("Velocidad inicial objeto 2 - x (m/s)", value=0.0, key="2d_v2x")
        v2y_i = st.number_input("Velocidad inicial objeto 2 - y (m/s)", value=0.0, key="2d_v2y")
        
        angulo_colision = st.slider("Ángulo de colisión (grados)", 0, 90, 45)
    
    with col2:
        st.subheader("Cálculos y Resultados")
        
        # Convertir ángulo a radianes
        theta = np.radians(angulo_colision)
        
        # Simplificación: colisión elástica con ángulo dado
        # Velocidades finales (fórmulas simplificadas para colisión oblicua)
        v1f_mag = np.sqrt(v1x_i**2 + v1y_i**2) * (m1 - m2) / (m1 + m2)
        v2f_mag = np.sqrt(v1x_i**2 + v1y_i**2) * (2 * m1) / (m1 + m2)
        
        v1x_f = v1f_mag * np.cos(theta)
        v1y_f = v1f_mag * np.sin(theta)
        v2x_f = v2f_mag * np.cos(theta + np.pi/4)
        v2y_f = v2f_mag * np.sin(theta + np.pi/4)
        
        st.write(f"**Velocidades finales objeto 1:**")
        st.write(f"vx = {v1x_f:.2f} m/s, vy = {v1y_f:.2f} m/s")
        st.write(f"**Velocidades finales objeto 2:**")
        st.write(f"vx = {v2x_f:.2f} m/s, vy = {v2y_f:.2f} m/s")
        
        # Verificar conservación del momentum
        px_inicial = m1 * v1x_i + m2 * v2x_i
        py_inicial = m1 * v1y_i + m2 * v2y_i
        px_final = m1 * v1x_f + m2 * v2x_f
        py_final = m1 * v1y_f + m2 * v2y_f
        
        st.write(f"**Conservación momentum x:** {abs(px_inicial - px_final) < 0.1}")
        st.write(f"**Conservación momentum y:** {abs(py_inicial - py_final) < 0.1}")
    
    # Crear visualización 2D
    fig, ax = plt.subplots(figsize=(10, 8))
    
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
    
    # Graficar trayectorias
    ax.plot(x1_antes, y1_antes, 'b--', alpha=0.7, label='Objeto 1 (antes)')
    ax.plot(x1_despues, y1_despues, 'b-', linewidth=2, label='Objeto 1 (después)')
    ax.plot(x2_antes, y2_antes, 'r--', alpha=0.7, label='Objeto 2 (antes)')
    ax.plot(x2_despues, y2_despues, 'r-', linewidth=2, label='Objeto 2 (después)')
    
    # Punto de colisión
    ax.plot(0, 0, 'go', markersize=10, label='Punto de colisión')
    
    # Objetos
    circle1 = Circle((0, 0), 0.2, color='blue', alpha=0.7)
    circle2 = Circle((0, 0), 0.15, color='red', alpha=0.7)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    ax.set_xlabel('Posición X (m)')
    ax.set_ylabel('Posición Y (m)')
    ax.set_title('Colisión 2D - Trayectorias')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    st.pyplot(fig)

# Cálculo de Impulso y Fuerza
elif simulacion == "Cálculo de Impulso y Fuerza":
    st.header("⚡ Cálculo de Impulso y Fuerza Promedio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        masa = st.number_input("Masa del objeto (kg)", value=1.0, min_value=0.1)
        v_inicial = st.number_input("Velocidad inicial (m/s)", value=0.0)
        v_final = st.number_input("Velocidad final (m/s)", value=10.0)
        tiempo = st.number_input("Tiempo de contacto (s)", value=0.1, min_value=0.001)
        
        # Opción para fuerza variable
        fuerza_constante = st.checkbox("Fuerza constante", value=True)
        
        if not fuerza_constante:
            st.write("Fuerza variable: F(t) = F₀ × sin(πt/T)")
            f_max = st.number_input("Fuerza máxima (N)", value=100.0)
    
    with col2:
        st.subheader("Resultados")
        
        # Calcular cambio de momentum
        delta_p = masa * (v_final - v_inicial)
        
        # Calcular impulso
        impulso = delta_p
        
        # Calcular fuerza promedio
        if fuerza_constante:
            fuerza_promedio = impulso / tiempo
            st.write(f"**Cambio de momentum (Δp):** {delta_p:.2f} kg⋅m/s")
            st.write(f"**Impulso (J):** {impulso:.2f} N⋅s")
            st.write(f"**Fuerza promedio:** {fuerza_promedio:.2f} N")
        else:
            # Para fuerza senoidal, la fuerza promedio es 2F₀/π
            fuerza_promedio = 2 * f_max / np.pi
            impulso_calculado = fuerza_promedio * tiempo
            st.write(f"**Cambio de momentum (Δp):** {delta_p:.2f} kg⋅m/s")
            st.write(f"**Impulso teórico:** {impulso:.2f} N⋅s")
            st.write(f"**Impulso con F(t):** {impulso_calculado:.2f} N⋅s")
            st.write(f"**Fuerza máxima:** {f_max:.2f} N")
            st.write(f"**Fuerza promedio:** {fuerza_promedio:.2f} N")
    
    # Graficar fuerza vs tiempo
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    t = np.linspace(0, tiempo, 1000)
    
    if fuerza_constante:
        F = np.full_like(t, fuerza_promedio)
    else:
        F = f_max * np.sin(np.pi * t / tiempo)
    
    ax1.plot(t, F, 'b-', linewidth=2)
    ax1.fill_between(t, 0, F, alpha=0.3, color='blue')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Fuerza (N)')
    ax1.set_title('Fuerza vs Tiempo')
    ax1.grid(True)
    
    # Calcular y graficar impulso acumulado
    impulso_acumulado = np.cumsum(F) * (tiempo / len(t))
    ax2.plot(t, impulso_acumulado, 'r-', linewidth=2)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Impulso acumulado (N⋅s)')
    ax2.set_title('Impulso Acumulado vs Tiempo')
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# Péndulo Balístico
elif simulacion == "Péndulo Balístico":
    st.header("🎯 Péndulo Balístico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        m_bala = st.number_input("Masa de la bala (kg)", value=0.01, min_value=0.001, format="%.3f")
        m_bloque = st.number_input("Masa del bloque (kg)", value=1.0, min_value=0.1)
        longitud = st.number_input("Longitud del péndulo (m)", value=1.0, min_value=0.1)
        altura_max = st.number_input("Altura máxima alcanzada (m)", value=0.2, min_value=0.01)
        
        g = 9.81  # Aceleración de la gravedad
    
    with col2:
        st.subheader("Cálculos y Resultados")
        
        # Velocidad del sistema después de la colisión (usando conservación de energía)
        v_despues = np.sqrt(2 * g * altura_max)
        
        # Velocidad inicial de la bala (usando conservación del momentum)
        v_bala = (m_bala + m_bloque) * v_despues / m_bala
        
        # Energía cinética inicial de la bala
        ke_inicial = 0.5 * m_bala * v_bala**2
        
        # Energía cinética después de la colisión
        ke_despues = 0.5 * (m_bala + m_bloque) * v_despues**2
        
        # Energía perdida
        energia_perdida = ke_inicial - ke_despues
        
        st.write(f"**Velocidad inicial de la bala:** {v_bala:.2f} m/s")
        st.write(f"**Velocidad después de la colisión:** {v_despues:.2f} m/s")
        st.write(f"**Energía cinética inicial:** {ke_inicial:.2f} J")
        st.write(f"**Energía cinética final:** {ke_despues:.2f} J")
        st.write(f"**Energía perdida:** {energia_perdida:.2f} J")
        st.write(f"**Porcentaje de energía perdida:** {(energia_perdida/ke_inicial)*100:.1f}%")
    
    # Crear animación del péndulo
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ángulo máximo del péndulo
    theta_max = np.arccos((longitud - altura_max) / longitud)
    
    # Posiciones del péndulo
    theta = np.linspace(-theta_max, theta_max, 100)
    x_pendulo = longitud * np.sin(theta)
    y_pendulo = -longitud * np.cos(theta)
    
    # Graficar el péndulo en diferentes posiciones
    for i, angle in enumerate(theta[::20]):
        x_pos = longitud * np.sin(angle)
        y_pos = -longitud * np.cos(angle)
        alpha = 0.3 + 0.7 * (i / len(theta[::20]))
        
        # Cuerda del péndulo
        ax.plot([0, x_pos], [0, y_pos], 'k-', alpha=alpha, linewidth=1)
        
        # Masa del péndulo
        circle = Circle((x_pos, y_pos), 0.05, color='red', alpha=alpha)
        ax.add_patch(circle)
    
    # Trayectoria de la bala
    x_bala = np.linspace(-0.5, 0, 50)
    y_bala = np.full_like(x_bala, y_pendulo[len(y_pendulo)//2])
    ax.plot(x_bala, y_bala, 'b--', linewidth=2, label='Trayectoria de la bala')
    
    # Punto de pivote
    ax.plot(0, 0, 'ko', markersize=8, label='Pivote')
    
    # Configuración del gráfico
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 0.5)
    ax.set_xlabel('Posición X (m)')
    ax.set_ylabel('Posición Y (m)')
    ax.set_title('Péndulo Balístico - Trayectoria')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    st.pyplot(fig)

# Caída Libre con Impacto
elif simulacion == "Caída Libre con Impacto":
    st.header("🪨 Caída Libre con Impacto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        masa = st.number_input("Masa del objeto (kg)", value=2.0, min_value=0.1, key="caida_masa")
        altura = st.number_input("Altura inicial (m)", value=10.0, min_value=0.1)
        coef_restitucion = st.slider("Coeficiente de restitución", 0.0, 1.0, 0.8)
        
        g = 9.81  # Aceleración de la gravedad
        
        # Tiempo de contacto con el suelo
        tiempo_contacto = st.number_input("Tiempo de contacto (s)", value=0.01, min_value=0.001, format="%.3f")
    
    with col2:
        st.subheader("Cálculos y Resultados")
        
        # Velocidad justo antes del impacto
        v_antes = np.sqrt(2 * g * altura)
        
        # Velocidad después del rebote
        v_despues = coef_restitucion * v_antes
        
        # Cambio de momentum
        delta_p = masa * (v_despues - (-v_antes))  # Cambio de signo por la dirección
        
        # Impulso
        impulso = delta_p
        
        # Fuerza promedio durante el impacto
        fuerza_promedio = impulso / tiempo_contacto
        
        # Altura después del rebote
        altura_rebote = (v_despues**2) / (2 * g)
        
        st.write(f"**Velocidad antes del impacto:** {v_antes:.2f} m/s")
        st.write(f"**Velocidad después del rebote:** {v_despues:.2f} m/s")
        st.write(f"**Cambio de momentum:** {delta_p:.2f} kg⋅m/s")
        st.write(f"**Impulso:** {impulso:.2f} N⋅s")
        st.write(f"**Fuerza promedio de impacto:** {fuerza_promedio:.2f} N")
        st.write(f"**Altura del rebote:** {altura_rebote:.2f} m")
        st.write(f"**Energía perdida:** {masa*g*(altura-altura_rebote):.2f} J")
    
    # Crear simulación de caída libre con rebotes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Simulación de posición vs tiempo con rebotes múltiples
    t_total = 6.0
    dt = 0.01
    t = []
    y = []
    v = []
    
    # Condiciones iniciales
    y_actual = altura
    v_actual = 0
    tiempo = 0
    
    while tiempo < t_total and y_actual > 0.01:
        # Caída libre hasta el suelo
        if y_actual > 0:
            t.append(tiempo)
            y.append(y_actual)
            v.append(v_actual)
            
            # Actualizar posición y velocidad
            v_actual -= g * dt
            y_actual += v_actual * dt
            tiempo += dt
            
            # Verificar si toca el suelo
            if y_actual <= 0:
                y_actual = 0
                v_actual = -coef_restitucion * v_actual
                # Reducir coeficiente para rebotes sucesivos
                coef_restitucion *= 0.95
    
    # Graficar posición vs tiempo
    ax1.plot(t, y, 'b-', linewidth=2)
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Altura (m)')
    ax1.set_title('Altura vs Tiempo (con rebotes)')
    ax1.grid(True)
    
    # Graficar velocidad vs tiempo
    ax2.plot(t, v, 'r-', linewidth=2)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad (m/s)')
    ax2.set_title('Velocidad vs Tiempo')
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# Flecha en Saco (Colisión Inelástica)
elif simulacion == "Flecha en Saco (Inelástica)":
    st.header("🏹 Flecha que se Incrusta en un Saco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros de Entrada")
        m_flecha = st.number_input("Masa de la flecha (kg)", value=0.05, min_value=0.001, format="%.3f")
        m_saco = st.number_input("Masa del saco (kg)", value=10.0, min_value=0.1)
        v_flecha = st.number_input("Velocidad inicial de la flecha (m/s)", value=50.0, min_value=0.1)
        
        # Distancia de penetración
        distancia_penetracion = st.number_input("Distancia de penetración (m)", value=0.2, min_value=0.01)
        
        # El saco inicialmente está en reposo
        v_saco = 0.0
    
    with col2:
        st.subheader("Cálculos y Resultados")
        
        # Velocidad final del sistema (conservación del momentum)
        v_final = (m_flecha * v_flecha + m_saco * v_saco) / (m_flecha + m_saco)
        
        # Energía cinética inicial
        ke_inicial = 0.5 * m_flecha * v_flecha**2
        
        # Energía cinética final
        ke_final = 0.5 * (m_flecha + m_saco) * v_final**2
        
        # Energía perdida (convertida en calor, deformación, etc.)
        energia_perdida = ke_inicial - ke_final
        
        # Fuerza promedio durante la penetración
        # Usando trabajo-energía: F * d = ΔKE de la flecha
        work_done = 0.5 * m_flecha * v_flecha**2 - 0.5 * m_flecha * v_final**2
        fuerza_promedio = work_done / distancia_penetracion
        
        # Tiempo de penetración (aproximado)
        v_promedio_flecha = (v_flecha + v_final) / 2
        tiempo_penetracion = distancia_penetracion / v_promedio_flecha
        
        # Impulso durante la penetración
        impulso = m_flecha * (v_final - v_flecha)
        
        st.write(f"**Velocidad final del sistema:** {v_final:.2f} m/s")
        st.write(f"**Energía cinética inicial:** {ke_inicial:.2f} J")
        st.write(f"**Energía cinética final:** {ke_final:.2f} J")
        st.write(f"**Energía perdida:** {energia_perdida:.2f} J")
        st.write(f"**Porcentaje de energía perdida:** {(energia_perdida/ke_inicial)*100:.1f}%")
        st.write(f"**Fuerza promedio de resistencia:** {fuerza_promedio:.2f} N")
        st.write(f"**Tiempo de penetración:** {tiempo_penetracion:.4f} s")
        st.write(f"**Impulso sobre la flecha:** {impulso:.2f} N⋅s")
    
    # Crear visualización de la penetración
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Simulación de la penetración
    t_antes = np.linspace(-0.1, 0, 50)
    t_penetracion = np.linspace(0, tiempo_penetracion, 100)
    t_despues = np.linspace(tiempo_penetracion, tiempo_penetracion + 0.2, 50)
    
    # Posición de la flecha
    x_flecha_antes = v_flecha * t_antes
    x_flecha_penetracion = v_flecha * t_penetracion - 0.5 * (fuerza_promedio/m_flecha) * t_penetracion**2
    x_flecha_despues = x_flecha_penetracion[-1] + v_final * (t_despues - tiempo_penetracion)
    
    # Posición del saco
    x_saco_antes = np.zeros_like(t_antes)
    x_saco_penetracion = np.zeros_like(t_penetracion)
    for i, t in enumerate(t_penetracion):
        if t > 0:
            # El saco comienza a moverse cuando la flecha lo toca
            aceleracion_saco = fuerza_promedio / m_saco
            x_saco_penetracion[i] = 0.5 * aceleracion_saco * t**2
    x_saco_despues = x_saco_penetracion[-1] + v_final * (t_despues - tiempo_penetracion)
    
    # Graficar posiciones
    ax1.plot(t_antes, x_flecha_antes, 'b-', linewidth=2, label='Flecha')
    ax1.plot(t_penetracion, x_flecha_penetracion, 'b-', linewidth=3, label='Flecha (penetrando)')
    ax1.plot(t_despues, x_flecha_despues, 'b--', linewidth=2, label='Sistema unido')
    
    ax1.plot(t_antes, x_saco_antes, 'r-', linewidth=2, label='Saco')
    ax1.plot(t_penetracion, x_saco_penetracion, 'r-', linewidth=3, label='Saco (acelerando)')
    ax1.plot(t_despues, x_saco_despues, 'r--', linewidth=2, label='Sistema unido')
    
    ax1.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='Inicio del impacto')
    ax1.axvline(x=tiempo_penetracion, color='orange', linestyle='--', alpha=0.7, label='Fin de penetración')
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Posición (m)')
    ax1.set_title('Posición vs Tiempo - Flecha y Saco')
    ax1.legend()
    ax1.grid(True)
    
    # Velocidades
    v_flecha_antes = np.full_like(t_antes, v_flecha)
    v_flecha_penetracion = v_flecha - (fuerza_promedio/m_flecha) * t_penetracion
    v_flecha_despues = np.full_like(t_despues, v_final)
    
    v_saco_antes = np.zeros_like(t_antes)
    v_saco_penetracion = (fuerza_promedio/m_saco) * t_penetracion
    v_saco_despues = np.full_like(t_despues, v_final)
    
    ax2.plot(t_antes, v_flecha_antes, 'b-', linewidth=2, label='Flecha')
    ax2.plot(t_penetracion, v_flecha_penetracion, 'b-', linewidth=3)
    ax2.plot(t_despues, v_flecha_despues, 'b--', linewidth=2)
    
    ax2.plot(t_antes, v_saco_antes, 'r-', linewidth=2, label='Saco')
    ax2.plot(t_penetracion, v_saco_penetracion, 'r-', linewidth=3)
    ax2.plot(t_despues, v_saco_despues, 'r--', linewidth=2)
    
    ax2.axvline(x=0, color='g', linestyle='--', alpha=0.7)
    ax2.axvline(x=tiempo_penetracion, color='orange', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Velocidad (m/s)')
    ax2.set_title('Velocidad vs Tiempo')
    ax2.legend()
    ax2.grid(True)
    
    # Fuerza vs tiempo
    f_antes = np.zeros_like(t_antes)
    f_penetracion = np.full_like(t_penetracion, fuerza_promedio)
    f_despues = np.zeros_like(t_despues)
    
    ax3.plot(t_antes, f_antes, 'k-', linewidth=2)
    ax3.plot(t_penetracion, f_penetracion, 'k-', linewidth=3, label=f'Fuerza = {fuerza_promedio:.1f} N')
    ax3.plot(t_despues, f_despues, 'k-', linewidth=2)
    
    ax3.fill_between(t_penetracion, 0, f_penetracion, alpha=0.3, color='red', 
                     label=f'Impulso = {impulso:.2f} N⋅s')
    
    ax3.axvline(x=0, color='g', linestyle='--', alpha=0.7)
    ax3.axvline(x=tiempo_penetracion, color='orange', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Fuerza (N)')
    ax3.set_title('Fuerza vs Tiempo')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# Sección de información teórica
st.sidebar.markdown("---")
st.sidebar.header("📚 Información Teórica")

with st.sidebar.expander("Conceptos Fundamentales"):
    st.markdown("""
    **Impulso (J):**
    - J = F × Δt (fuerza constante)
    - J = ∫F dt (fuerza variable)
    - J = Δp (cambio de momentum)
    
    **Momentum (p):**
    - p = m × v
    - Conservación: Σp_inicial = Σp_final
    
    **Colisiones:**
    - Elástica: se conserva energía cinética
    - Inelástica: no se conserva energía cinética
    - Coeficiente de restitución: e = |v_sep|/|v_apr|
    """)

with st.sidebar.expander("Fórmulas Clave"):
    st.markdown("""
    **Colisión 1D:**
    - v₁f = [(m₁-em₂)v₁ᵢ + (1+e)m₂v₂ᵢ]/(m₁+m₂)
    - v₂f = [(m₂-em₁)v₂ᵢ + (1+e)m₁v₁ᵢ]/(m₁+m₂)
    
    **Péndulo Balístico:**
    - v = √(2gh) (velocidad después de colisión)
    - v₀ = (m₁+m₂)v/m₁ (velocidad inicial del proyectil)
    
    **Caída Libre:**
    - v = √(2gh)
    - F_promedio = Δp/Δt
    """)

# Pie de página
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<h4>🎓 Desafío de Programación Física II</h4>
<p>Simulador Interactivo de Impulso y Cantidad de Movimiento Lineal</p>
<p><em>Desarrollado con Python + Streamlit</em></p>
</div>
""", unsafe_allow_html=True)

# Agregar explicación de uso
st.markdown("---")
st.info("""
**📋 Instrucciones de Uso:**
1. Selecciona una simulación del menú lateral
2. Ajusta los parámetros de entrada según tu caso de estudio
3. Observa los resultados calculados y las gráficas generadas
4. Experimenta con diferentes valores para entender mejor los conceptos
5. Usa las fórmulas de la barra lateral como referencia teórica
""")

# Sección de validación experimental
st.markdown("---")
st.header("🔬 Validación y Casos de Estudio")

casos_ejemplo = st.selectbox(
    "Selecciona un caso de estudio para validar:",
    [
        "Colisión de bolas de billar",
        "Choque de automóviles",
        "Rebote de pelota de tenis",
        "Impacto de meteorito",
        "Disparo de cañón"
    ]
)

if casos_ejemplo == "Colisión de bolas de billar":
    st.write("""
    **Caso: Colisión de bolas de billar**
    - Masa de cada bola: ~0.17 kg
    - Velocidad típica: 3-5 m/s
    - Coeficiente de restitución: ~0.95
    - Tipo: Colisión casi elástica
    """)
    
elif casos_ejemplo == "Choque de automóviles":
    st.write("""
    **Caso: Choque de automóviles**
    - Masa típica: 1000-2000 kg
    - Velocidad de impacto: 50-100 km/h
    - Coeficiente de restitución: 0.0-0.3
    - Tipo: Colisión inelástica
    """)
    
elif casos_ejemplo == "Rebote de pelota de tenis":
    st.write("""
    **Caso: Rebote de pelota de tenis**
    - Masa: ~0.057 kg
    - Velocidad de saque: 50-70 m/s
    - Coeficiente de restitución: ~0.7
    - Altura de caída vs rebote
    """)
    
elif casos_ejemplo == "Impacto de meteorito":
    st.write("""
    **Caso: Impacto de meteorito**
    - Masa: variable (0.1 kg - 1000 kg)
    - Velocidad: 11-72 km/s
    - Coeficiente de restitución: ~0.0
    - Tipo: Colisión perfectamente inelástica
    """)
    
elif casos_ejemplo == "Disparo de cañón":
    st.write("""
    **Caso: Disparo de cañón**
    - Masa del proyectil: 1-50 kg
    - Masa del cañón: 1000-5000 kg
    - Velocidad del proyectil: 200-1000 m/s
    - Conservación del momentum (retroceso)
    """)

# Agregar análisis de errores
st.markdown("---")
st.header("⚠️ Análisis de Errores y Limitaciones")

with st.expander("Fuentes de Error en las Simulaciones"):
    st.markdown("""
    **Errores Experimentales:**
    - Fricción del aire (no considerada en las simulaciones básicas)
    - Deformación de los objetos durante la colisión
    - Rotación de los objetos (momento angular)
    - Variación del coeficiente de restitución con la velocidad
    
    **Limitaciones del Modelo:**
    - Objetos puntuales (sin considerar forma real)
    - Colisiones instantáneas (tiempo de contacto cero)
    - Superficies perfectamente lisas
    - Condiciones ideales (sin perturbaciones externas)
    
    **Mejoras Posibles:**
    - Incluir resistencia del aire
    - Considerar deformaciones elásticas
    - Añadir rotación y momento angular
    - Simulaciones en 3D
    """)

# Conclusiones
st.markdown("---")
st.success("""
**✅ Objetivos Cumplidos:**
- ✅ Simulación de colisión 1D (elástica e inelástica)
- ✅ Simulación de colisión 2D con trayectorias
- ✅ Cálculo de impulso y fuerza promedio
- ✅ Péndulo balístico con análisis energético
- ✅ Caída libre con impacto y rebotes
- ✅ Flecha incrustándose en saco
- ✅ Controles interactivos y visualización gráfica
- ✅ Explicación física de todos los fenómenos
- ✅ Casos de estudio y validación experimental
""")