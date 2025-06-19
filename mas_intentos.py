import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Simulaciones de Impulso y Movimiento", layout="wide")
st.title("Simulaciones Avanzadas de Impulso y Cantidad de Movimiento")

st.sidebar.title("Opciones de Simulación")
opcion = st.sidebar.selectbox("Selecciona una opción:", [
    "Colisión 1D",
    "Colisión 2D",
    "Cálculo de Impulso",
    "Péndulo Balístico"
])

if opcion == "Colisión 1D":
    st.header("Simulación de Colisión 1D")
    m1 = st.slider("Masa del objeto 1 (kg)", 0.1, 10.0, 2.0)
    v1 = st.slider("Velocidad inicial del objeto 1 (m/s)", -10.0, 10.0, 5.0)
    m2 = st.slider("Masa del objeto 2 (kg)", 0.1, 10.0, 3.0)
    v2 = st.slider("Velocidad inicial del objeto 2 (m/s)", -10.0, 10.0, -2.0)
    tipo = st.selectbox("Tipo de colisión", ["Elástica", "Inelástica"])

    if tipo == "Elástica":
        v1f = (v1*(m1 - m2) + 2*m2*v2)/(m1 + m2)
        v2f = (v2*(m2 - m1) + 2*m1*v1)/(m1 + m2)
    else:
        v1f = v2f = (m1*v1 + m2*v2)/(m1 + m2)

    st.write(f"Velocidad final del objeto 1: {v1f:.2f} m/s")
    st.write(f"Velocidad final del objeto 2: {v2f:.2f} m/s")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Obj 1", "Obj 2"], y=[v1, v2], name="Inicial", marker_color="blue"))
    fig.add_trace(go.Bar(x=["Obj 1", "Obj 2"], y=[v1f, v2f], name="Final", marker_color="green"))
    fig.update_layout(title="Velocidades antes y después de la colisión", barmode="group")
    st.plotly_chart(fig)

elif opcion == "Colisión 2D":
    st.header("Simulación de Colisión 2D con Trayectorias")
    m1 = st.slider("Masa del objeto 1 (kg)", 0.1, 10.0, 1.0)
    m2 = st.slider("Masa del objeto 2 (kg)", 0.1, 10.0, 1.0)
    v1x = st.slider("Velocidad inicial X del objeto 1", -10.0, 10.0, 5.0)
    v1y = st.slider("Velocidad inicial Y del objeto 1", -10.0, 10.0, 0.0)
    v2x = st.slider("Velocidad inicial X del objeto 2", -10.0, 10.0, -3.0)
    v2y = st.slider("Velocidad inicial Y del objeto 2", -10.0, 10.0, 0.0)

    tipo = st.selectbox("Tipo de colisión", ["Elástica", "Inelástica"])

    if tipo == "Elástica":
        # Fórmulas para colisión 2D elástica entre masas iguales simplificadas
        v1fx, v1fy = v2x, v2y
        v2fx, v2fy = v1x, v1y
    else:
        v1fx = v2fx = (m1 * v1x + m2 * v2x) / (m1 + m2)
        v1fy = v2fy = (m1 * v1y + m2 * v2y) / (m1 + m2)

    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=[0, v1x], y=[0, v1y], mode='lines+markers', name='Obj1 Inicial'))
    fig2d.add_trace(go.Scatter(x=[0, v2x], y=[0, v2y], mode='lines+markers', name='Obj2 Inicial'))
    fig2d.add_trace(go.Scatter(x=[0, v1fx], y=[0, v1fy], mode='lines+markers', name='Obj1 Final'))
    fig2d.add_trace(go.Scatter(x=[0, v2fx], y=[0, v2fy], mode='lines+markers', name='Obj2 Final'))
    fig2d.update_layout(title="Vectores de velocidad antes y después", xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig2d)

elif opcion == "Cálculo de Impulso":
    st.header("Cálculo de Impulso y Fuerza Promedio")
    masa = st.number_input("Masa del objeto (kg)", 0.1, 100.0, 5.0)
    vi = st.number_input("Velocidad inicial (m/s)", -50.0, 50.0, 0.0)
    vf = st.number_input("Velocidad final (m/s)", -50.0, 50.0, 10.0)
    tiempo = st.number_input("Tiempo de interacción (s)", 0.01, 10.0, 2.0)

    impulso = masa * (vf - vi)
    fuerza_prom = impulso / tiempo

    st.write(f"Impulso: {impulso:.2f} N·s")
    st.write(f"Fuerza promedio: {fuerza_prom:.2f} N")

elif opcion == "Péndulo Balístico":
    st.header("Simulación de Péndulo Balístico")
    m_bala = st.slider("Masa de la bala (kg)", 0.01, 2.0, 0.05)
    v_bala = st.slider("Velocidad de la bala (m/s)", 1.0, 500.0, 200.0)
    m_pendulo = st.slider("Masa del péndulo (kg)", 0.5, 20.0, 5.0)
    g = 9.81

    v_conjunta = (m_bala * v_bala) / (m_bala + m_pendulo)
    h = v_conjunta**2 / (2 * g)

    st.write(f"Velocidad conjunta después del impacto: {v_conjunta:.2f} m/s")
    st.write(f"Altura máxima alcanzada: {h:.2f} m")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Velocidad conjunta", "Altura máxima"], y=[v_conjunta, h]))
    fig.update_layout(title="Resultado del péndulo balístico")
    st.plotly_chart(fig)