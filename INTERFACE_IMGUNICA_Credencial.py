import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

try:
    import torch
except Exception:
    pass
# ================== LOGIN E CONTROLE DE SESSÃO ==================

# Função para carregar credenciais
def carregar_credenciais():
    path = os.path.join(os.path.dirname(__file__), "credenciais.json")
    with open(path, "r") as f:
        return json.load(f)

# Tela de login
def tela_login():
    st.title("🔐 Acesso Restrito")
    st.markdown("<h3>Entre com suas credenciais</h3>", unsafe_allow_html=True)

    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")

    if st.button("Entrar"):
        dados = carregar_credenciais()
        usuarios = {u["usuario"]: u["senha"] for u in dados["usuarios"]}

        if usuario in usuarios and senha == usuarios[usuario]:
            st.session_state["logado"] = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

# ===================================================================

# Se o usuário não estiver logado, mostra a tela de login
if "logado" not in st.session_state:
    tela_login()
else:
    # Se estiver logado, executa o app principal

    # 1. Configuração da página e cabeçalho
    st.set_page_config(page_title="Predições de Lesões de Pele")
    st.image("campusHumanitas2.PNG", use_container_width=True)

    st.markdown(
        "<h1 style='text-align: center;'>Projeto de Iniciação Científica da FCMSJC - HUMANITAS</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align: center;'>Predições de Lesões de Pele</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='font-size:18px; font-weight:bold;'>Autores: Gabriel Barcellos, Felipe Segreto, Mariana Ferreira.</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:18px; font-weight:bold;'>Orientadores: Pedro Euphrásio, Felipe Pacheco.</p>",
        unsafe_allow_html=True
    )

    # 2. Carregar modelo YOLO
    @st.cache_resource
    def load_model(path: str):
        return YOLO(path)

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "YOLOv08n.pt")
    model = load_model(MODEL_PATH)

    # 3. Criação de abas
    tab1, tab2 = st.tabs(["📷 Upload & Predições", "📊 Gráfico Quantitativo"])

    # 4. Tab de upload e predição
    with tab1:
        st.subheader("Classes e Descrição")
        st.markdown(
            "<p style='font-size:24px; font-weight:bold;'>Upload de Imagens.</p>",
            unsafe_allow_html=True
        )

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
                st.subheader("Imagens Carregadas")
                cols = st.columns(len(images))
                for col, img, name in zip(cols, images, image_names):
                    col.image(img, caption=name, use_container_width=True)
        else:
            st.info("Faça o upload de uma ou mais imagens para liberar a predição.")

        # Função já existente
        def results_to_df(r, image_name):
            if r.boxes is None or len(r.boxes) == 0:
                return pd.DataFrame(columns=["image_name","x1","y1","x2","y2","confidence","class_id","name"])
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            names = [r.names[c] for c in cls]
            df = pd.DataFrame({
                "image_name": image_name,
                "x1": xyxy[:,0], "y1": xyxy[:,1], "x2": xyxy[:,2], "y2": xyxy[:,3],
                "confidence": conf,
                "class_id": cls,
                "name": names
            })
            return df

        if valid_images:
            if st.button("Fazer Previsão"):
                all_dfs = []

                for image, image_name in zip(images, image_names):
                    img_array = np.array(image)
                    results = model(img_array)
                    r = results[0]

                    annotated = r.plot()
                    st.subheader(f"Predição: {image_name}")
                    st.image(annotated, use_container_width=True)

                    df_full = results_to_df(r, image_name)
                    df_display = df_full[["image_name", "name", "confidence"]]

                    if df_full.empty:
                        st.warning(f"Nenhuma detecção em {image_name}.")
                    else:
                        st.dataframe(df_display)

                    all_dfs.append(df_full)

                # Salvar resultados
                save_path = "predictions.csv"
                if all_dfs:
                    combined = pd.concat(all_dfs, ignore_index=True)
                    if os.path.exists(save_path):
                        old = pd.read_csv(save_path)
                        combined = pd.concat([old, combined], ignore_index=True)
                    combined.to_csv(save_path, index=False)
                    st.success(f"Todas as predições salvas em {save_path}")

    # 5. Tab do gráfico e botão de limpar
    with tab2:
        st.subheader("Gráfico Quantitativo de Todas as Predições")

        save_path = "predictions.csv"
        if os.path.exists(save_path):
            all_preds = pd.read_csv(save_path)
            counts = all_preds['name'].value_counts().sort_index()
            fig, ax = plt.subplots()
            counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Classe")
            ax.set_ylabel("Quantidade")
            ax.set_title("Contagem de Classes em predictions.csv")
            st.pyplot(fig)

            # Botão para limpar
            if st.button("🗑️ Limpar Predições"):
                os.remove(save_path)
                st.success("Arquivo predictions.csv apagado com sucesso!")
        else:
            st.warning("Arquivo predictions.csv não encontrado para plotagem.")
