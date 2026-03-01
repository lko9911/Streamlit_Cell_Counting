import streamlit as st
import cv2
import numpy as np
from cellpose import models 
from streamlit_image_coordinates import streamlit_image_coordinates

# 페이지 설정 
st.set_page_config(layout="wide", page_title="Malaria Diagnostic AI")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .title {
        font-size:32px;
        font-weight:800;
        color:#0B3C5D;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size:16px;
        color:#666;
        margin-bottom: 20px;
    }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
    }
</style>
""", unsafe_allow_html=True)

# 제목 출력
st.markdown('<div class="title">🧪 Malaria Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI 기반 적혈구 감염 분석 시스템 (Cell Counting)</div>', unsafe_allow_html=True)

# ===============================
# 📦 모델 및 로직 함수
# ===============================
@st.cache_resource
def load_model():
    # CellposeModel 대신 정식 클래스인 Cellpose 사용, 모델은 cyto3 지정
    return models.Cellpose(gpu=True, model_type='cyto3')

def process_analysis(img_bgr, model):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # cyto3 모델로 세포 분할 실행
    masks, _, _, _ = model.eval(img_rgb, diameter=None, channels=[0,0])
    
    total_cells = int(np.max(masks))
    valid_cells = set(range(1, total_cells + 1))
    infected_cells = set()
    
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 말라리아 기생충 특유의 보라색 영역
    lower_purple = np.array([120, 50, 50]) 
    upper_purple = np.array([170, 255, 255])
    
    for cell_id in range(1, total_cells + 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        if np.sum(cell_mask) < 100:
            valid_cells.discard(cell_id)
            continue
            
        y_idx, x_idx = np.where(cell_mask == 1)
        roi_hsv = hsv_img[np.min(y_idx):np.max(y_idx)+1, np.min(x_idx):np.max(x_idx)+1]
        roi_mask = cell_mask[np.min(y_idx):np.max(y_idx)+1, np.min(x_idx):np.max(x_idx)+1]
        
        parasite_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
        parasite_in_cell = cv2.bitwise_and(parasite_mask, parasite_mask, mask=roi_mask)
        
        if cv2.countNonZero(parasite_in_cell) > 5:
            infected_cells.add(cell_id)
            
    return masks, valid_cells, infected_cells

# ===============================
# 📌 Sidebar & Session State
# ===============================
if "state" not in st.session_state:
    st.session_state.state = {"masks": None, "valid": set(), "infected": set(), "orig": None, "analyzed": False}

st.sidebar.header("📁 Image Management")
uploaded_file = st.sidebar.file_uploader("현미경 이미지 업로드", type=["jpg","png","jpeg","tif"])

if uploaded_file and st.sidebar.button("🔍 AI 분석 실행"):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    
    with st.spinner("Cyto3 모델이 세포를 분석 중입니다..."):
        model = load_model()
        masks, valid, infected = process_analysis(img_bgr, model)
        st.session_state.state.update({
            "masks": masks, "valid": valid, "infected": infected, "orig": img_bgr, "analyzed": True
        })

edit_mode = st.sidebar.radio("🛠 수동 보정 모드 선택", ["보기 전용", "감염 토글", "유효 RBC 토글"])
st.sidebar.info("💡 이미지를 클릭하면 해당 세포의 상태가 변경됩니다.")

# ===============================
# 🧪 결과 렌더링 및 인터랙션
# ===============================
if st.session_state.state["analyzed"]:
    s = st.session_state.state
    output = s["orig"].copy()
    
    # 그리기 최적화
    for cell_id in range(1, int(np.max(s["masks"])) + 1):
        # 보기 전용 모드에서 제외된 세포는 안 그림
        if cell_id not in s["valid"] and edit_mode == "보기 전용": continue
        
        if cell_id in s["valid"]:
            color = (0, 0, 255) if cell_id in s["infected"] else (0, 255, 0)
        else:
            color = (150, 150, 150) # 제외된 세포 (회색)
        
        cell_mask = (s["masks"] == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)

    st.subheader("📊 분석 결과 (이미지를 클릭하여 수정)")

    display_img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    # 🖱 클릭 좌표 획득
    coords = streamlit_image_coordinates(display_img, key="click", use_column_width=True)
    
    if coords and edit_mode != "보기 전용":
        orig_h, orig_w = s["masks"].shape[:2]
        x_ratio = orig_w / coords["width"]
        y_ratio = orig_h / coords["height"]
        
        real_x = int(coords["x"] * x_ratio)
        real_y = int(coords["y"] * y_ratio)
    
        if 0 <= real_y < orig_h and 0 <= real_x < orig_w:
            cell_id = s["masks"][real_y, real_x]
            
            if cell_id != 0:
                if edit_mode == "감염 토글":
                    if cell_id in s["infected"]:
                        s["infected"].remove(cell_id)
                    else:
                        s["infected"].add(cell_id)
                elif edit_mode == "유효 RBC 토글":
                    if cell_id in s["valid"]:
                        s["valid"].remove(cell_id)
                    else:
                        s["valid"].add(cell_id)
                
                st.rerun()

    # 🏥 진단 통계 패널
    tot_valid = len(s["valid"])
    tot_inf = len(s.get("infected", set()).intersection(s["valid"]))
    parasitemia = (tot_inf / tot_valid * 100) if tot_valid > 0 else 0

    st.markdown("### 🏥 Diagnostic Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid RBC", f"{tot_valid} 개")
    c2.metric("Infected RBC", f"{tot_inf} 개")
    c3.metric("Parasitemia (%)", f"{parasitemia:.2f} %")

    if parasitemia < 1:
        st.success("Low Infection Risk")
    elif parasitemia < 5:
        st.warning("Moderate Infection Risk")
    else:
        st.error("High Infection Risk – Immediate Attention Required")
