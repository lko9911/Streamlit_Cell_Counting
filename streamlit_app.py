import streamlit as st
import cv2
import numpy as np
from cellpose import models
from PIL import Image
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="Malaria AI Analyzer", layout="wide")

st.title("🔬 말라리아 세포 분석 AI 도구")
st.sidebar.header("설정")

# ===============================
# 1. 세션 상태 초기화 (데이터 유지)
# ===============================
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.masks = None
    st.session_state.valid_cells = set()
    st.session_state.infected_cells = set()
    st.session_state.cell_contours = {}
    st.session_state.orig_img = None

# ===============================
# 2. 이미지 업로드 및 분석 함수
# ===============================
uploaded_file = st.sidebar.file_uploader("현미경 이미지 업로드", type=["jpg", "jpeg", "png", "tif", "tiff"])

@st.cache_resource # 모델 로딩 속도 최적화
def load_model():
    return models.CellposeModel(gpu=True, model_type='cyto3')

def process_image(img_bgr):
    model = load_model()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Cellpose 실행
    masks, flows, styles = model.eval(img_rgb, diameter=None, channels=[0, 0])
    total_cells = np.max(masks)
    
    # 필터링 로직 (기존 코드와 동일)
    height, width = masks.shape
    cell_areas = np.bincount(masks.flatten())
    MARGIN = 30
    edge_cells = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
                 set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
    edge_cells.discard(0)
    
    inner_areas = [cell_areas[cid] for cid in range(1, total_cells + 1) if cid not in edge_cells]
    std_area = np.median(inner_areas) if inner_areas else np.median(cell_areas[1:])
    
    valid_cells = set()
    infected_cells = set()
    contours_dict = {}
    
    lower_purple = np.array([120, 100, 50])
    upper_purple = np.array([170, 255, 255])
    
    for cell_id in range(1, total_cells + 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_dict[cell_id] = cnts
        
        # 유효 세포 판단
        if cell_id in edge_cells and cell_areas[cell_id] < (std_area * 0.9): continue
        if cell_areas[cell_id] < (std_area * 0.4): continue
        valid_cells.add(cell_id)
        
        # 감염 판단
        y, x = np.where(cell_mask == 1)
        if len(y) > 0:
            roi_hsv = hsv_img[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
            roi_m = cell_mask[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
            p_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
            if cv2.countNonZero(cv2.bitwise_and(p_mask, p_mask, mask=roi_m)) > 10:
                infected_cells.add(cell_id)
                
    return masks, valid_cells, infected_cells, contours_dict

# ===============================
# 3. 메인 실행 로직
# ===============================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.session_state.orig_img = img

    if st.sidebar.button("AI 분석 시작"):
        with st.spinner("AI가 세포를 분석 중입니다..."):
            m, v, i, c = process_image(img)
            st.session_state.masks = m
            st.session_state.valid_cells = v
            st.session_state.infected_cells = i
            st.session_state.cell_contours = c
            st.session_state.analyzed = True

if st.session_state.analyzed:
    # 통계 요약
    v_list = st.session_state.valid_cells
    i_list = st.session_state.infected_cells.intersection(v_list)
    parasitemia = (len(i_list) / len(v_list) * 100) if v_list else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 유효 세포(RBC)", len(v_list))
    col2.metric("감염된 세포", len(i_list))
    col3.metric("감염률 (Parasitemia)", f"{parasitemia:.2f}%")

    # 결과 이미지 그리기
    display_img = st.session_state.orig_img.copy()
    for cid, cnts in st.session_state.cell_contours.items():
        if cid in st.session_state.valid_cells:
            color = (0, 0, 255) if cid in st.session_state.infected_cells else (0, 255, 0)
        else:
            color = (128, 128, 128)
        cv2.drawContours(display_img, cnts, -1, color, 2)
    
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_column_width=True, caption="분석 결과 (Red: 감염, Green: 정상, Gray: 제외)")

    # 수정 기능 (Streamlit은 클릭 상호작용이 제한적이므로 리스트 선택 방식 권장)
    with st.expander("데이터 수동 수정"):
        cell_to_toggle = st.number_input("수정할 세포 ID 입력", min_value=1, max_value=int(np.max(st.session_state.masks)), step=1)
        c1, c2 = st.columns(2)
        if c1.button("감염 상태 반전"):
            if cell_to_toggle in st.session_state.infected_cells:
                st.session_state.infected_cells.remove(cell_to_toggle)
            else:
                st.session_state.infected_cells.add(cell_to_toggle)
            st.rerun()
            
        if c2.button("유효 상태(Gray) 반전"):
            if cell_to_toggle in st.session_state.valid_cells:
                st.session_state.valid_cells.remove(cell_to_toggle)
            else:
                st.session_state.valid_cells.add(cell_to_toggle)
            st.rerun()
else:
    st.info("왼쪽 사이드바에서 이미지를 업로드하고 'AI 분석 시작'을 눌러주세요.")
