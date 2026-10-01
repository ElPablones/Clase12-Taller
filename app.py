import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import random

st.set_page_config(page_title="AI Blackjack", page_icon="🃏", layout="wide")

# --- LÓGICA DEL BLACKJACK ---

def crear_baraja():
    palos = ['♠️', '♥️', '♣', '♦️']
    valores = [('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), 
               ('8', 8), ('9', 9), ('10', 10), ('J', 10), ('Q', 10), ('K', 10), ('A', 11)]
    baraja = [(val[0], val[1], palo) for palo in palos for val in valores]
    random.shuffle(baraja)
    return baraja

def calcular_puntaje(mano):
    puntaje = sum([carta[1] for carta in mano])
    ases = sum([1 for carta in mano if carta[0] == 'A'])
    while puntaje > 21 and ases > 0:
        puntaje -= 10
        ases -= 1
    return puntaje

def mostrar_mano(mano, ocultar_segunda=False):
    texto = ""
    for i, carta in enumerate(mano):
        if ocultar_segunda and i == 1:
            texto += "[ ❓ ] "
        else:
            texto += f"[ {carta[0]}{carta[2]} ] "
    return texto

# --- INICIALIZACIÓN DE VARIABLES DE SESIÓN ---

if 'estado_juego' not in st.session_state:
    st.session_state.estado_juego = 'apostar'
if 'baraja' not in st.session_state:
    st.session_state.baraja = []
if 'mano_jugador' not in st.session_state:
    st.session_state.mano_jugador = []
if 'mano_dealer' not in st.session_state:
    st.session_state.mano_dealer = []
if 'apuesta' not in st.session_state:
    st.session_state.apuesta = 0
if 'mensaje_final' not in st.session_state:
    st.session_state.mensaje_final = ""
# NUEVO: Billetera del jugador
if 'dinero' not in st.session_state:
    st.session_state.dinero = 2000 

# --- CARGAR MODELO MNIST ---

@st.cache_resource
def load_mnist_model():
    for path in ['mnist_model.h5', 'mnist_model.keras']:
        if os.path.exists(path):
            return keras.models.load_model(path)
    return None

model = load_mnist_model()

# --- BILLETERA EN SIDEBAR ---
with st.sidebar:
    st.header("🏦 Tu Billetera")
    
    # Mostrar el saldo con colores dependiendo de si estás ganando o perdiendo
    if st.session_state.dinero > 0:
        st.metric(label="Saldo Actual", value=f"${st.session_state.dinero}")
    else:
        st.metric(label="Saldo Actual", value=f"${st.session_state.dinero}", delta="- Bancarrota")
        st.error("¡Te quedaste sin fondos!")
        if st.button("💸 Pedir préstamo al casino (Reiniciar)"):
            st.session_state.dinero = 2000
            st.rerun()

# --- INTERFAZ PRINCIPAL ---

st.title("🃏 Blackjack Online vs IA")

if st.session_state.estado_juego == 'apostar':
    st.markdown("### 💰 Fase de Apuestas")
    st.write("Dibuja un número del **1 al 9** en el tablero para definir tu apuesta inicial. (Ej: Si dibujas un 5, apostarás $500).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            width=300,
            height=300,
            drawing_mode="freedraw",
            key="canvas_bet",
            return_image_data=True 
        )
        
    with col2:
        st.write("### Confirmar")
        if not model:
            st.error("⚠️ Falta el modelo `mnist_model.h5` en la carpeta.")
            
        if st.button("🎰 Leer Apuesta y Repartir Cartas", type="primary", use_container_width=True):
            if canvas_result is not None and canvas_result.json_data is not None:
                try:
                    if canvas_result.image_data is not None and np.max(canvas_result.image_data) > 0:
                        if st.session_state.dinero <= 0:
                            st.warning("No tienes dinero para apostar. Pide un préstamo en el menú lateral.")
                        else:
                            with st.spinner("La IA está leyendo tus fichas..."):
                                image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
                                image_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
                                img_array = np.array(image_resized) / 255.0
                                img_array = img_array.reshape(1, 28, 28, 1)
                                
                                prediccion = model.predict(img_array, verbose=0)
                                digito = np.argmax(prediccion)
                                
                                if digito == 0:
                                    digito = 10 
                                    
                                apuesta_calculada = digito * 100
                                # Si la apuesta es mayor al dinero que tiene, apuesta todo lo que le queda (All-in)
                                if apuesta_calculada > st.session_state.dinero:
                                    st.session_state.apuesta = st.session_state.dinero
                                else:
                                    st.session_state.apuesta = apuesta_calculada
                                    
                                st.session_state.baraja = crear_baraja()
                                st.session_state.mano_jugador = [st.session_state.baraja.pop(), st.session_state.baraja.pop()]
                                st.session_state.mano_dealer = [st.session_state.baraja.pop(), st.session_state.baraja.pop()]
                                st.session_state.estado_juego = 'jugando'
                                st.rerun()
                    else:
                        st.warning("Debes dibujar un número primero.")
                except RuntimeError:
                    pass

elif st.session_state.estado_juego == 'jugando':
    st.markdown(f"### 💵 Apuesta en la mesa: **${st.session_state.apuesta}**")
    st.divider()
    
    col_dealer, col_jugador = st.columns(2)
    
    with col_dealer:
        st.subheader("🎩 Mano del Croupier")
        st.markdown(f"## {mostrar_mano(st.session_state.mano_dealer, ocultar_segunda=True)}")
        st.write("Puntaje: ?")
        
    with col_jugador:
        st.subheader("👤 Tu Mano")
        st.markdown(f"## {mostrar_mano(st.session_state.mano_jugador)}")
        puntaje_actual = calcular_puntaje(st.session_state.mano_jugador)
        st.write(f"Puntaje: **{puntaje_actual}**")
        
        st.write("---")
        btn_pedir, btn_plantar = st.columns(2)
        
        with btn_pedir:
            if st.button("👇 Pedir Carta", use_container_width=True):
                st.session_state.mano_jugador.append(st.session_state.baraja.pop())
                if calcular_puntaje(st.session_state.mano_jugador) > 21:
                    st.session_state.estado_juego = 'fin'
                    st.session_state.mensaje_final = "💥 ¡Te pasaste de 21! La casa gana."
                    # Restamos la apuesta de la billetera
                    st.session_state.dinero -= st.session_state.apuesta
                st.rerun()
                
        with btn_plantar:
            if st.button("✋ Plantarse", type="primary", use_container_width=True):
                while calcular_puntaje(st.session_state.mano_dealer) < 17:
                    st.session_state.mano_dealer.append(st.session_state.baraja.pop())
                
                p_jugador = calcular_puntaje(st.session_state.mano_jugador)
                p_dealer = calcular_puntaje(st.session_state.mano_dealer)
                
                if p_dealer > 21:
                    st.session_state.mensaje_final = f"🎉 ¡El Croupier se pasó con {p_dealer}! GANASTE **${st.session_state.apuesta}**."
                    st.session_state.dinero += st.session_state.apuesta
                elif p_jugador > p_dealer:
                    st.session_state.mensaje_final = f"🏆 ¡Tu puntaje es mayor! GANASTE **${st.session_state.apuesta}**."
                    st.session_state.dinero += st.session_state.apuesta
                elif p_dealer > p_jugador:
                    st.session_state.mensaje_final = "💸 La casa gana esta vez."
                    st.session_state.dinero -= st.session_state.apuesta
                else:
                    st.session_state.mensaje_final = "🤝 Empate. Recuperas tu apuesta."
                    # En empate, el saldo se mantiene igual
                    
                st.session_state.estado_juego = 'fin'
                st.rerun()

elif st.session_state.estado_juego == 'fin':
    st.markdown("### 🏁 Resultado de la Partida")
    st.success(st.session_state.mensaje_final)
    st.divider()
    
    col_dealer, col_jugador = st.columns(2)
    
    with col_dealer:
        st.subheader("🎩 Mano del Croupier")
        st.markdown(f"## {mostrar_mano(st.session_state.mano_dealer)}")
        st.write(f"Puntaje final: **{calcular_puntaje(st.session_state.mano_dealer)}**")
        
    with col_jugador:
        st.subheader("👤 Tu Mano")
        st.markdown(f"## {mostrar_mano(st.session_state.mano_jugador)}")
        st.write(f"Puntaje final: **{calcular_puntaje(st.session_state.mano_jugador)}**")
        
    st.write("---")
    if st.button("🔄 Jugar otra mano", type="primary"):
        st.session_state.estado_juego = 'apostar'
        st.rerun()
