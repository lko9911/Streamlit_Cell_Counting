import streamlit as st
import cv2
import numpy as np
from cellpose import models
from PIL import Image
import pandas as pd

# 페이지 설정
st.set_page_config(page_title="Malaria AI Analyzer", layout="wide")

# ===============================
# 0. 유틸리티 함수
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
    # CPU 환경에서 가장 안정적인 cyto3 모델 로드
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
# 2. 사이드바 설정
# ===============================
st.sidebar.header("🔬 설정 및 업로드")
uploaded_file = st.sidebar.file_uploader("이미지 업로드 (jpg, png, tif)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    st.session_state.orig_img = raw_img

# 분석 버튼
if st.sidebar.button("🚀 AI 분석 시작"):
    if st.session_state.orig_img is not None:
        with st.spinner("세포 분석 중... 약 20초 정도 소요됩니다."):
            # 1. 최적화 리사이징
            working_img, ratio = resize_for_ai(st.session_state.orig_img)
            st.session_state.ratio = ratio
            
            img_rgb = cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
            
            # 2. 모델 실행 (TypeError 방지를 위해 인자 최적화)
            model = load_cellpose_model()
            
            # cellpose v3.0 이상에서 안전한 인자들만 사용
            masks, flows, styles = model.eval(
                img_rgb, 
                diameter=30,      # 평균 세포 크기
                channels=[0, 0],  # [그레이스케일, 없음]
                net_avg=False,    # 속도 향상
                augment=False,    # 속도 향상 (TTA 미사용)
                resample=False    # 속도 향상
            )
            
            # 3. 결과 후처리
            total_cells = np.max(masks)
            cell_areas = np.bincount(masks.flatten())
            
            # 가장자리 필터링 (Margin 20px)
            MARGIN = 20
            edge_cells = set(masks[:MARGIN, :].flatten()) | set(masks[-MARGIN:, :].flatten()) | \
                         set(masks[:, :MARGIN].flatten()) | set(masks[:, -MARGIN:].flatten())
            edge_cells.discard(0)
            
            inner_areas = [cell_areas[cid] for cid in range(1, total_cells + 1) if cid not in edge_cells]
            std_area = np.median(inner_areas) if inner_areas else 500
            
            valid_cells = set()
            infected_cells = set()
            contours_dict = {}
            
            # 말라리아 기생충 색상 범위 (보라색 계열)
            lower_purple = np.array([120, 80, 50])
            upper_purple = np.array([175, 255, 255])
            
            for cell_id in range(1, total_cells + 1):
                cell_mask = (masks == cell_id).astype(np.uint8)
                cnts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_dict[cell_id] = cnts
                
                # 유효 세포 필터 (크기 기준)
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
            st.success("분석이 완료되었습니다!")

# ===============================
# 3. 결과 표시
# ===============================
if st.session_state.analyzed:
    v_set = st.session_state.valid_cells
    i_set = st.session_state.infected_cells.intersection(v_set)
    parasitemia = (len(i_set) / len(v_set) * 100) if v_set else 0
    
    st.subheader("📊 검사 결과 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("유효 적혈구 수", f"{len(v_set)} 개")
    c2.metric("감염된 세포 수", f"{len(i_set)} 개")
    c3.metric("감염률 (Parasitemia)", f"{parasitemia:.2f} %")

    # 결과 이미지 렌더링
    working_img, _ = resize_for_ai(st.session_state.orig_img)
    display_img = working_img.copy()
    
    for cid, cnts in st.session_state.cell_contours.items():
        if cid in st.session_state.valid_cells:
            # 감염: 빨강, 정상: 초록
            color = (0, 0, 255) if cid in st.session_state.infected_cells else (0, 255, 0)
            thickness = 2
            # 세포 ID 표시 (선택 사항)
            M = cv2.moments(cnts[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(display_img, str(cid), (cx-10, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # 제외된 세포: 회색
            color = (100, 100, 100)
            thickness = 1
            
        cv2.drawContours(display_img, cnts, -1, color, thickness)
    
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_column_width=True, caption="분석 결과 (Red: Infected, Green: Normal, Gray: Excluded)")

    # 수동 수정 기능
    with st.expander("📝 데이터 수동 수정"):
        target_id = st.number_input("수정할 세포 ID 입력", min_value=1, step=1)
        col1, col2 = st.columns(2)
        if col1.button("감염 상태 반전"):
            if target_id in st.session_state.infected_cells:
                st.session_state.infected_cells.remove(target_id)
            else:
                st.session_state.infected_cells.add(target_id)
            st.rerun()
        if col2.button("유효 상태 반전"):
            if target_id in st.session_state.valid_cells:
                st.session_state.valid_cells.remove(target_id)
            else:
                st.session_state.valid_cells.add(target_id)
            st.rerun()
else:
    st.info("이미지를 업로드하고 사이드바의 분석 버튼을 눌러주세요.")
