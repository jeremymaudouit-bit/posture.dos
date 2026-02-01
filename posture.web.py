import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image
from fpdf import FPDF
from datetime import datetime
import io

# ===================== CONFIGURATION MEDIAPIPE =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Analyseur Postural Pro", layout="wide")

# Couleurs personnalisées
PRIMARY_COLOR = "#318CE7"

# ===================== LOGIQUE DE CALCUL =====================

def calculate_angle(p1, p2, p3):
    if not all([p1, p2, p3]): return 0.0
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0: return 0.0
    return abs(math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag1 * mag2))))))

def generate_pdf(res, img_annotated):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, f"BILAN POSTURAL : {res['nom']}", ln=True, align='C')
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 10, f"Date de l'examen : {datetime.now().strftime('%d/%m/%Y')}", ln=True)
    
    # Sauvegarde temporaire de l'image pour le PDF
    img_pil = Image.fromarray(img_annotated)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    with open("temp_report_img.jpg", "wb") as f:
        f.write(img_byte_arr.getvalue())
        
    pdf.image("temp_report_img.jpg", x=10, y=40, w=110)
    
    # Résultats
    pdf.set_xy(125, 45)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Mesures de Bascule :")
    pdf.set_font("Arial", '', 11)
    pdf.set_xy(125, 55)
    pdf.cell(0, 10, f"- Epaules : {res['ep_deg']:.1f} deg ({res['ep_cm']:.1f} cm)")
    pdf.set_xy(125, 62)
    pdf.cell(0, 10, f"- Bassin : {res['ba_deg']:.1f} deg ({res['ba_cm']:.1f} cm)")
    
    pdf.set_xy(125, 80)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Angles Articulaires :")
    pdf.set_font("Arial", '', 11)
    pdf.set_xy(125, 90)
    pdf.cell(0, 10, f"- Genou G : {res['ge_l']:.1f} deg")
    pdf.set_xy(125, 97)
    pdf.cell(0, 10, f"- Genou D : {res['ge_r']:.1f} deg")
    
    return pdf.output(dest='S').encode('latin-1')

# ===================== INTERFACE UTILISATEUR =====================

st.title("⚖️ Analyseur Postural Pro - Photo & Live")

# Sidebar pour les entrées
with st.sidebar:
    st.header("👤 Information Patient")
    name = st.text_input("Nom Complet du Patient", placeholder="Jean Dupont")
    user_h = st.number_input("Taille réelle (cm)", min_value=50.0, max_value=250.0, value=175.0)
    
    st.divider()
    st.header("📸 Source de l'image")
    source = st.radio("Choisir la source", ["Caméra en direct", "Importer une image"])

# Zone principale
col_img, col_res = st.columns([2, 1])

img_input = None
if source == "Caméra en direct":
    img_input = st.camera_input("Prendre une photo")
else:
    img_input = st.file_uploader("Charger une photo (.jpg, .png)", type=['jpg', 'jpeg', 'png'])

if img_input:
    # Traitement de l'image
    file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Analyse
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # Calibration Pixel -> CM
        heel_y = (lm[29].y + lm[30].y) / 2
        px_h = abs(heel_y - lm[0].y) * h
        ratio = user_h / px_h if px_h != 0 else 0
        
        # Calculs identiques à votre code original
        sh_angle = math.degrees(math.atan2(lm[11].y - lm[12].y, lm[11].x - lm[12].x))
        sh_cm = abs(lm[11].y - lm[12].y) * h * ratio
        
        hi_angle = math.degrees(math.atan2(lm[23].y - lm[24].y, lm[23].x - lm[24].x))
        hi_cm = abs(lm[23].y - lm[24].y) * h * ratio
        
        res_data = {
            "nom": name if name else "Anonyme",
            "ep_deg": sh_angle, "ep_cm": sh_cm,
            "ba_deg": hi_angle, "ba_cm": hi_cm,
            "ge_l": calculate_angle(lm[23], lm[25], lm[27]),
            "ge_r": calculate_angle(lm[24], lm[26], lm[28]),
            "pi_l": calculate_angle(lm[25], lm[27], lm[29]),
            "pi_r": calculate_angle(lm[26], lm[28], lm[30])
        }
        
        # Dessin des repères
        annotated_frame = img_rgb.copy()
        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Affichage
        with col_img:
            st.image(annotated_frame, caption="Analyse en cours...", use_container_width=True)
            
        with col_res:
            st.subheader("📊 Résultats")
            st.markdown(f"**Patient :** {res_data['nom']}")
            
            st.info(f"**Épaules**\n\n{sh_angle:.1f}° | {sh_cm:.1f} cm")
            st.info(f"**Bassin**\n\n{hi_angle:.1f}° | {hi_cm:.1f} cm")
            
            st.write("**Angles Articulaires**")
            st.write(f"- Genou G : {res_data['ge_l']:.1f}°")
            st.write(f"- Genou D : {res_data['ge_r']:.1f}°")
            st.write(f"- Pied G : {res_data['pi_l']:.1f}°")
            st.write(f"- Pied D : {res_data['pi_r']:.1f}°")
            
            # Bouton PDF
            pdf_bytes = generate_pdf(res_data, annotated_frame)
            st.download_button(
                label="📄 Télécharger le Rapport PDF",
                data=pdf_bytes,
                file_name=f"Bilan_{res_data['nom']}.pdf",
                mime="application/pdf"
            )
    else:
        st.error(" détection impossible. Assurez-vous d'être bien visible de face.")

else:
    st.info("Veuillez prendre une photo ou en importer une pour démarrer l'analyse.")