import streamlit as st
import cv2
import numpy as np
from cellpose import models
from PIL import Image
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="Malaria AI Analyzer", layout="wide")

# ===============================
# 0. 유틸리티 함수 (최적화 관련)
# ===============================
def resize_for_ai(img, target_width=1000):
    """이미지가 너무 크면 연산 속도가 느려지므로 리사이징합니다."""
    h, w = img.shape[:2]
    if w <= target_width:
        return img, 1.0
    ratio = target_width / w
    target_height = int(h * ratio)
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio

@st.cache_resource
def load_cellpose_model():
    # Streamlit Cloud 무료티어는 GPU가 없으므로 gpu=False 권장
    return models.CellposeModel(gpu=False, model_type='cyto3')

# ===============================
# 1. 세션 상태 초기화
# ===============================
if 'analyzed' not in st.session_state:
    st.session_state.update({
        'analyzed': False,
        'masks': None,
        'valid_cells': set(),
        'infected_cells': set(),
        'cell_contours': {},
        'orig_img': None,
        'ratio': 1.0
    })

# ===============================
# 2. 사이드바 설정 및 파일 업로드
# ===============================
st.sidebar.header("🔬 설정 및 업로드")
uploaded_file = st.sidebar.file_uploader("현미경 이미지 업로드", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    st.session_state.orig_img = raw_img

# 분석 버튼
if st.sidebar.button("🚀 AI 분석 시작 (최적화 모드)"):
    if st.session_state.orig_img is not None:
        with st.spinner("이미지 최적화 및 세포 분석 중... (약 10~20초 소요)"):
            # 1. 리사이징 (속도 향상의 핵심)
            working_img, ratio = resize_for_ai(st.session_state.orig_img)
            st.session_state.ratio = ratio
            
            img_rgb = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
            
            # 2. 모델 로드 및 실행 (파라미터 최적화)
            model = load_cellpose_model()
            masks, flows, styles = model.eval(
                img_rgb, 
                diameter=30,      # 세포 평균 크기 고정 시 속도 향상
                channels=[0, 0],
                net_avg=False,    # 속도 최적화 1
                fast_mode=True,   # 속도 최적화 2
                resample=False    # 속도 최적화 3
            )
            
            # 3. 데이터 처리 및 필터링
            total_cells = np.max(masks)
            cell_areas = np.bincount(masks.flatten())
            MARGIN = 20
            edge_cells = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
                         set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
            edge_cells.discard(0)
            
            inner_areas = [cell_areas[cid] for cid in range(1, total_cells + 1) if cid not in edge_cells]
            std_area = np.median(inner_areas) if inner_areas else 500
            
            valid_cells = set()
            infected_cells = set()
            contours_dict = {}
            
            lower_purple = np.array([120, 100, 50])
            upper_purple = np.array([170, 255, 255])
            
            for cell_id in range(1, total_cells + 1):
                cell_mask = (masks == cell_id).astype(np.uint8)
                cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_dict[cell_id] = cnts
                
                # 필터링 로직
                if cell_id in edge_cells and cell_areas[cell_id] < (std_area * 0.8): continue
                if cell_areas[cell_id] < (std_area * 0.3): continue
                valid_cells.add(cell_id)
                
                # 감염 판별 (HSV 색상 분석)
                y, x = np.where(cell_mask == 1)
                if len(y) > 0:
                    roi_hsv = hsv_img[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    roi_m = cell_mask[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    p_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
                    if cv2.countNonZero(cv2.bitwise_and(p_mask, p_mask, mask=roi_m)) > 8:
                        infected_cells.add(cell_id)
            
            # 세션 업데이트
            st.session_state.update({
                'masks': masks,
                'valid_cells': valid_cells,
                'infected_cells': infected_cells,
                'cell_contours': contours_dict,
                'analyzed': True
            })
            st.success("분석 완료!")

# ===============================
# 3. 결과 대시보드 표시
# ===============================
if st.session_state.analyzed:
    v_list = st.session_state.valid_cells
    i_list = st.session_state.infected_cells.intersection(v_list)
    parasitemia = (len(i_list) / len(v_list) * 100) if v_list else 0
    
    st.subheader("📊 분석 결과 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 유효 RBC", f"{len(v_list)} 개")
    c2.metric("감염된 RBC", f"{len(i_list)} 개")
    c3.metric("감염률(Parasitemia)", f"{parasitemia:.2f}%")

    # 결과 이미지 생성 (리사이징된 크기에 맞게 드로잉)
    working_img, _ = resize_for_ai(st.session_state.orig_img)
    display_img = working_img.copy()
    
    for cid, cnts in st.session_state.cell_contours.items():
        if cid in st.session_state.valid_cells:
            color = (0, 0, 255) if cid in st.session_state.infected_cells else (0, 255, 0)
        else:
            color = (120, 120, 120) # 유효하지 않은 세포
        cv2.drawContours(display_img, cnts, -1, color, 1)
    
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 데이터 수정 섹션
    with st.expander("📝 수동 데이터 수정 (ID 기준)"):
        col_id, col_btn1, col_btn2 = st.columns([2, 1, 1])
        target_id = col_id.number_input("세포 ID 입력", min_value=1, step=1)
        if col_btn1.button("감염 상태 반전"):
            if target_id in st.session_state.infected_cells:
                st.session_state.infected_cells.remove(target_id)
            else:
                st.session_state.infected_cells.add(target_id)
            st.rerun()
        if col_btn2.button("유효 상태 반전"):
            if target_id in st.session_state.valid_cells:
                st.session_state.valid_cells.remove(target_id)
            else:
                st.session_state.valid_cells.add(target_id)
            st.rerun()

else:
    st.info("사이드바에서 이미지를 업로드하고 분석 시작 버튼을 눌러주세요.")
