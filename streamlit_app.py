import streamlit as st
import cv2
import numpy as np
from cellpose import models
from PIL import Image
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="Malaria AI Analyzer", layout="wide")

# ===============================
# 0. 유틸리티 함수 및 모델 로드
# ===============================
def resize_for_ai(img, target_width=1000):
    """연산 속도를 위해 이미지 크기를 조정합니다."""
    h, w = img.shape[:2]
    if w <= target_width:
        return img, 1.0
    ratio = target_width / w
    target_height = int(h * ratio)
    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, ratio

@st.cache_resource
def load_cellpose_model():
    # CPU 환경에서 가장 범용적인 cyto3 모델 로드 (gpu=False)
    return models.CellposeModel(gpu=False, model_type='cyto3')

# ===============================
# 1. 세션 상태 초기화 (데이터 보존)
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
# 2. 사이드바 UI
# ===============================
st.sidebar.header("🔬 설정 및 업로드")
uploaded_file = st.sidebar.file_uploader("현미경 이미지 업로드", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    st.session_state.orig_img = raw_img

# 분석 실행 버튼
if st.sidebar.button("🚀 AI 분석 시작"):
    if st.session_state.orig_img is not None:
        with st.spinner("AI 모델이 세포를 분석 중입니다... (약 15-30초 소요)"):
            # 이미지 리사이징 (속도 향상 핵심)
            working_img, ratio = resize_for_ai(st.session_state.orig_img)
            st.session_state.ratio = ratio
            
            img_rgb = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
            
            model = load_cellpose_model()
            
            # TypeError 방지를 위해 최소한의 필수 인자만 사용
            # 최신 버전 호환성을 위해 dict 형태로 인자 관리
            eval_kwargs = {
                "diameter": 30,
                "channels": [0, 0],
                "augment": False,
                "resample": False
            }
            
            try:
                masks, flows, styles = model.eval(img_rgb, **eval_kwargs)
            except TypeError:
                # 인자 오류 시 가장 기본적인 호출로 재시도
                masks, flows, styles = model.eval(img_rgb, diameter=30, channels=[0, 0])
            
            # 데이터 후처리
            total_cells = np.max(masks)
            cell_areas = np.bincount(masks.flatten())
            
            # 가장자리 제외 (Margin 20px)
            MARGIN = 20
            edge_cells = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
                         set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
            edge_cells.discard(0)
            
            inner_areas = [cell_areas[cid] for cid in range(1, total_cells + 1) if cid not in edge_cells]
            std_area = np.median(inner_areas) if inner_areas else 500
            
            valid_cells = set()
            infected_cells = set()
            contours_dict = {}
            
            # 기생충 감염 판별 색상 (보라색 계열)
            lower_purple = np.array([120, 70, 40])
            upper_purple = np.array([175, 255, 255])
            
            for cell_id in range(1, total_cells + 1):
                cell_mask = (masks == cell_id).astype(np.uint8)
                cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_dict[cell_id] = cnts
                
                # 세포 필터링 (너무 작거나 깨진 세포 제외)
                if cell_areas[cell_id] < (std_area * 0.3): continue
                if cell_id in edge_cells and cell_areas[cell_id] < (std_area * 0.8): continue
                
                valid_cells.add(cell_id)
                
                # 감염 여부 분석
                y, x = np.where(cell_mask == 1)
                if len(y) > 0:
                    roi_hsv = hsv_img[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    roi_m = cell_mask[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    p_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
                    if cv2.countNonZero(cv2.bitwise_and(p_mask, p_mask, mask=roi_m)) > 8:
                        infected_cells.add(cell_id)
            
            st.session_state.update({
                'masks': masks,
                'valid_cells': valid_cells,
                'infected_cells': infected_cells,
                'cell_contours': contours_dict,
                'analyzed': True
            })
            st.success("분석 완료!")

# ===============================
# 3. 결과 대시보드 및 시각화
# ===============================
if st.session_state.analyzed:
    v_set = st.session_state.valid_cells
    i_set = st.session_state.infected_cells.intersection(v_set)
    parasitemia = (len(i_set) / len(v_set) * 100) if v_set else 0
    
    st.divider()
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("유효 적혈구(RBC)", f"{len(v_set)} 개")
    col_stat2.metric("감염된 세포", f"{len(i_set)} 개")
    col_stat3.metric("감염률(Parasitemia)", f"{parasitemia:.2f} %")

    # 결과 이미지 그리기
    working_img, _ = resize_for_ai(st.session_state.orig_img)
    display_img = working_img.copy()
    
    for cid, cnts in st.session_state.cell_contours.items():
        if cid in st.session_state.valid_cells:
            color = (0, 0, 255) if cid in st.session_state.infected_cells else (0, 255, 0)
            thickness = 2
            # 세포 중앙에 ID 번호 표시
            M = cv2.moments(cnts[0])
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(display_img, str(cid), (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            color = (80, 80, 80) # 제외된 세포
            thickness = 1
            
        cv2.drawContours(display_img, cnts, -1, color, thickness)
    
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 수동 수정 섹션
    with st.expander("📝 데이터 수동 수정 (ID 입력)"):
        st.write("이미지에 표시된 숫자를 입력하여 상태를 반전시킵니다.")
        edit_id = st.number_input("수정할 세포 ID", min_value=1, step=1)
        btn_col1, btn_col2 = st.columns(2)
        
        if btn_col1.button("🔴 감염 상태 반전"):
            if edit_id in st.session_state.infected_cells:
                st.session_state.infected_cells.remove(edit_id)
            else:
                st.session_state.infected_cells.add(edit_id)
            st.rerun()
            
        if btn_col2.button("⚪ 유효 상태 반전"):
            if edit_id in st.session_state.valid_cells:
                st.session_state.valid_cells.remove(edit_id)
            else:
                st.session_state.valid_cells.add(edit_id)
            st.rerun()
else:
    st.info("왼쪽 사이드바에서 이미지를 업로드하고 분석 시작 버튼을 눌러주세요.")
