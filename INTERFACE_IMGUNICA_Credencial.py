import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#

# 1. Título
st.set_page_config(page_title="Predições de Lesões de Pele")
st.image("campusHumanitas2.PNG", use_container_width=True)
st.title("Predições de Lesões de Pele")

# 2. Imagem de cabeçalho (placeholder)
# st.image("/path/to/header_image.png", use_container_width=True)

# 3. Head de texto com as classes da rede e breve explicação
st.header("Classes e Descrição")
st.write("Projeto de Inciação Científica da FCMSJC- HUMANITAS\n")
st.write("Autores: Gabriel Barcelos, Felipe Segreto, Mariana Ferreira.\n")
st.write("Orientadores: Pedro Euphrásio, Felipe Pacheco\n")

# 4. Carregar modelo YOLOv11
@st.cache_resource
def load_model(path: str):
    return YOLO(path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "YOLOv08n.pt")
model = load_model(MODEL_PATH)

# Função auxiliar para converter resultados para DataFrame
def results_to_df(results, image_name):
    if results.boxes is not None and results.boxes.xyxy is not None:
        xyxy = results.boxes.xyxy.cpu().numpy()
        conf = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names
        data = []
        for i in range(len(xyxy)):
            row = {
                "image_name": image_name,
                "name": names[cls[i]],
                "confidence": conf[i],
                "x1": xyxy[i][0],
                "y1": xyxy[i][1],
                "x2": xyxy[i][2],
                "y2": xyxy[i][3]
            }
            data.append(row)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

# 5. Upload de imagens (vários)
st.subheader("Upload de Imagens")
uploaded_files = st.file_uploader("Escolha uma ou mais imagens", type=["jpeg", "jpg", "png"], accept_multiple_files=True)

images = []
image_names = []
valid_images = False

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            images.append(img)
            image_names.append(uploaded_file.name)
            valid_images = True
        except Exception as e:
            st.error(f"Erro ao abrir {uploaded_file.name}: {e}")
    if valid_images:
        # mostra miniaturas
        cols = st.columns(len(images))
        for col, img, name in zip(cols, images, image_names):
            col.image(img, caption=name, use_container_width=True)
else:
    pass
    #st.info("Faça o upload de uma ou mais imagens para liberar a predição.")

# 6. Predição e apresentação para múltiplas imagens
if valid_images:
    if st.button("Fazer Previsão"):
        all_dfs = []
        for image, image_name in zip(images, image_names):
            img_array = np.array(image)
            results = model(img_array)
            r = results[0]

            # gera imagem anotada e exibe
            annotated = r.plot()
            st.subheader(f"Predição: {image_name}")
            st.image(annotated, use_container_width=True)

            # extrai DataFrame completo e de exibição
            df_full = results_to_df(r, image_name)
            df_display = df_full[["image_name", "name", "confidence"]]
            if df_full.empty:
                st.warning(f"Nenhuma detecção em {image_name}.")
            else:
                st.dataframe(df_display)
            all_dfs.append(df_full)

        # 8. Salvar todas as predições
        save_path = "predictions.csv"
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            if os.path.exists(save_path):
                old = pd.read_csv(save_path)
                combined = pd.concat([old, combined], ignore_index=True)
            combined.to_csv(save_path, index=False)
            st.success(f"Todas as predições salvas em {save_path}")

        # 9. Gráfico quantitativo a partir de predictions.csv
        st.subheader("Gráfico Quantitativo de Todas as Predições")
        if os.path.exists(save_path):
            all_preds = pd.read_csv(save_path)
            counts = all_preds['name'].value_counts().sort_index()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Classe")
            ax.set_ylabel("Quantidade")
            ax.set_title("Contagem de Classes em predictions.csv")
            st.pyplot(fig)
        else:
            st.warning("Arquivo predictions.csv não encontrado para plotagem.")
else:
    st.info("Faça o upload de uma imagem para liberar a predição.")
