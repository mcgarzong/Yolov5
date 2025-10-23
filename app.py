import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys
from PIL import Image

# 🌌 Configuración de la página
st.set_page_config(
    page_title="Visor del Mundo Invisible",
    page_icon="👁️",
    layout="wide"
)

# 🪄 Imagen o logo del visor
image = Image.open('hablar.jpeg')
st.image(image, width=300, caption="👁️ Activando el visor...")

# 🔧 Cargar modelo YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception:
                st.warning("🧠 Método alternativo de carga en proceso...")
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    except Exception as e:
        st.error(f"⚠️ Error al iniciar el visor: {str(e)}")
        return None

# 🪐 Título y descripción narrativa
st.title("👁️ Visor del Mundo Invisible")
st.markdown("""
Explora lo que los ojos no pueden ver.  
Este **visor inteligente** analiza el entorno y revela patrones ocultos a simple vista.  
Ajusta los parámetros en el panel lateral para calibrar tu dispositivo de percepción.
""")

# 🧠 Cargar el modelo
with st.spinner("🔮 Iniciando sistemas de percepción..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("⚙️ Panel del Explorador")
    
    with st.sidebar:
        st.subheader("🎚️ Calibración del Visor")
        model.conf = st.slider('Sensibilidad del visor', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Rango de percepción', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Sensibilidad: {model.conf:.2f} | Rango: {model.iou:.2f}")
        
        st.subheader("🧩 Modo experimental")
        try:
            model.agnostic = st.checkbox('Modo sin clasificación', False)
            model.multi_label = st.checkbox('Detección múltiple por área', False)
            model.max_det = st.number_input('Límites de hallazgos simultáneos', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas funciones experimentales no están disponibles.")

    main_container = st.container()
    
    with main_container:
        st.subheader("📷 Captura de Exploración")
        picture = st.camera_input("Activar Visor y Capturar Entorno", key="camera")
        
        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("🔎 Analizando energía visual..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante el escaneo: {str(e)}")
                    st.stop()
            
            try:
                predictions = results.pred[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🌠 Resultados del Escaneo")
                    results.render()
                    st.image(cv2_img, channels='BGR', use_container_width=True, caption="Energías detectadas")
                
                with col2:
                    st.subheader("📋 Elementos Descubiertos")
                    label_names = model.names
                    category_count = {}
                    
                    for category in categories:
                        idx = int(category.item()) if hasattr(category, 'item') else int(category)
                        category_count[idx] = category_count.get(idx, 0) + 1
                    
                    data = []
                    for category, count in category_count.items():
                        label = label_names[category]
                        confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                        data.append({
                            "Elemento Detectado": label,
                            "Cantidad": count,
                            "Precisión Media": f"{confidence:.2f}"
                        })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                        st.bar_chart(df.set_index('Elemento Detectado')['Cantidad'])
                    else:
                        st.info("🔍 No se detectaron patrones visibles con los parámetros actuales.")
                        st.caption("💡 Intenta aumentar la sensibilidad o el rango de percepción.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")
else:
    st.error("❌ No se pudo activar el visor. Verifica las dependencias o el modelo.")

# 🌟 Pie de página narrativo
st.markdown("---")
st.caption("""
🧬 **Visor del Mundo Invisible**  
Desarrollado para revelar lo que se oculta a simple vista.  
Basado en visión artificial (YOLOv5) y potenciado por Streamlit.  
💫 Creador: Explorador de lo oculto 👁️✨
""")
