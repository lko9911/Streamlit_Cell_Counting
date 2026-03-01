import streamlit as st
import cv2
import numpy as np
from cellpose import models
from streamlit_image_coordinates import streamlit_image_coordinates

# 페이지 설정
st.set_page_config(layout="wide", page_title="Malaria Diagnostic AI (Cyto3)")

# ===============================
# 📦 모델 및 분석 로직
# ===============================
@st.cache_resource
def load_model():
    # cyto3 모델 로드 (GPU가 없다면 gpu=False로 자동 전환)
    try:
        return models.CellposeModel(gpu=True, model_type='cyto3')
    except:
        return models.CellposeModel(gpu=False, model_type='cyto3')

def process_analysis(img_bgr, model):
    # 1. 전처리: CLAHE로 대비 강화 (세포 경계 명확화)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # 2. Cyto3 분석 실행
    with st.spinner("Cyto3 모델이 세포를 정밀 분석 중입니다..."):
        masks, _, _ = model.eval(img_rgb, diameter=None, channels=[0,0])
    
    total_cells = int(np.max(masks))
    cell_areas = np.bincount(masks.flatten())
    
    # 3. 가장자리 및 크기 필터링 (로컬 코드 로직 반영)
    MARGIN = 30
    h, w = masks.shape
    edge_ids = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
               set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
    
    inner_areas = [cell_areas[i] for i in range(1, total_cells + 1) if i not in edge_ids]
    std_area = np.median(inner_areas) if inner_areas else 1000

    valid_cells = set()
    infected_cells = set()
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 80, 50]) # 보라색 감염체 기준
    upper_purple = np.array([170, 255, 255])

    for cell_id in range(1, total_cells + 1):
        # 크기 기반 노이즈 제거
        if cell_areas[cell_id] < (std_area * 0.4): continue
        # 가장자리 절단 세포 제거
        if cell_id in edge_ids and cell_areas[cell_id] < (std_area * 0.8): continue

        valid_cells.add(cell_id)
        
        # 감염 판별
        cell_mask = (masks == cell_id).astype(np.uint8)
        y_idx, x_idx = np.where(cell_mask == 1)
        y1, y2, x1, x2 = np.min(y_idx), np.max(y_idx), np.min(x_idx), np.max(x_idx)
        roi_hsv = hsv_img[y1:y2+1, x1:x2+1]
        roi_mask = cell_mask[y1:y2+1, x1:x2+1]
        
        parasite_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
        if cv2.countNonZero(cv2.bitwise_and(parasite_mask, parasite_mask, mask=roi_mask)) > 8:
            infected_cells.add(cell_id)
            
    return masks, valid_cells, infected_cells

# ===============================
# 🖥️ UI 구성
# ===============================
st.title("🧪 Cyto3 Malaria Diagnostic AI")

if "state" not in st.session_state:
    st.session_state.state = {"masks": None, "valid": set(), "infected": set(), "orig": None, "analyzed": False}

with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader("이미지 선택", type=["jpg","png","tif"])
    if uploaded_file and st.button("🚀 분석 시작"):
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        model = load_model()
        masks, valid, infected = process_analysis(img, model)
        st.session_state.state.update({"masks": masks, "valid": valid, "infected": infected, "orig": img, "analyzed": True})
    
    st.divider()
    edit_mode = st.radio("🛠 편집 모드", ["보기 전용", "감염 토글 (좌클릭)", "유효성 토글 (우클릭 대신 선택)"])

# ===============================
# 📊 결과 시각화 및 인터랙션
# ===============================
s = st.session_state.state
if s["analyzed"]:
    output = s["orig"].copy()
    
    # 시각화 루프
    for cell_id in range(1, int(np.max(s["masks"])) + 1):
        contours, _ = cv2.findContours((s["masks"] == cell_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cell_id in s["valid"]:
            color = (0, 0, 255) if cell_id in s["infected"] else (0, 255, 0) # Red or Green
        else:
            color = (128, 128, 128) # Gray (Excluded)
        cv2.drawContours(output, contours, -1, color, 2)

    st.subheader("분석 이미지 (클릭하여 보정)")
    # 이미지 클릭 이벤트 핸들러
    coords = streamlit_image_coordinates(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), key="cell_click")

    if coords:
        h_orig, w_orig = s["masks"].shape[:2]
        rx = int(coords["x"] * (w_orig / coords["width"]))
        ry = int(coords["y"] * (h_orig / coords["height"]))
        
        if 0 <= ry < h_orig and 0 <= rx < w_orig:
            cid = s["masks"][ry, rx]
            if cid != 0:
                if "감염" in edit_mode:
                    if cid in s["infected"]: s["infected"].remove(cid)
                    else: s["infected"].add(cid); s["valid"].add(cid)
                elif "유효성" in edit_mode:
                    if cid in s["valid"]: s["valid"].remove(cid)
                    else: s["valid"].add(cid)
                st.rerun()

    # 결과 요약 패널
    v_count = len(s["valid"])
    i_count = len(s["infected"].intersection(s["valid"]))
    rate = (i_count / v_count * 100) if v_count > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("검출된 적혈구", f"{v_count} 개")
    c2.metric("감염된 세포", f"{i_count} 개")
    c3.metric("감염률(Parasitemia)", f"{rate:.2f} %")
