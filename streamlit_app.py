import streamlit as st
import cv2
import numpy as np
from cellpose.models import Cellpose
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Malaria Diagnostic AI")

# ===============================
# 🎨 UI 스타일링
# ===============================
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .title { font-size:32px; font-weight:800; color:#1E3A8A; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧪 Malaria Diagnostic System</div>', unsafe_allow_html=True)

# ===============================
# 📦 모델 및 로직 함수
# ===============================
@st.cache_resource
def load_model():
    # GPU가 있다면 gpu=True로 변경 권장
    return Cellpose(gpu=False, model_type="cyto")

def process_analysis(img_bgr, model):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # diameter=30~40 정도가 일반적인 적혈구 크기에 적합할 수 있음
    masks, _, _, _ = model.eval(img_rgb, diameter=None, channels=[0,0])
    
    total_cells = int(np.max(masks))
    valid_cells = set(range(1, total_cells + 1))
    infected_cells = set()
    
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 말라리아 기생충(Giemsa stain) 특유의 보라색/진분홍색 영역
    lower_purple = np.array([130, 50, 50]) 
    upper_purple = np.array([170, 255, 255])
    
    for cell_id in range(1, total_cells + 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        if np.sum(cell_mask) < 100: # 너무 작은 노이즈 제거
            valid_cells.discard(cell_id)
            continue
            
        # ROI 추출 효율화
        y_idx, x_idx = np.where(cell_mask == 1)
        roi_hsv = hsv_img[np.min(y_idx):np.max(y_idx)+1, np.min(x_idx):np.max(x_idx)+1]
        roi_mask = cell_mask[np.min(y_idx):np.max(y_idx)+1, np.min(x_idx):np.max(x_idx)+1]
        
        parasite_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
        parasite_in_cell = cv2.bitwise_and(parasite_mask, parasite_mask, mask=roi_mask)
        
        if cv2.countNonZero(parasite_in_cell) > 5: # 임계값 조정
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
    
    with st.spinner("AI가 세포를 분석 중입니다..."):
        model = load_model()
        masks, valid, infected = process_analysis(img_bgr, model)
        st.session_state.state.update({
            "masks": masks, "valid": valid, "infected": infected, "orig": img_bgr, "analyzed": True
        })

edit_mode = st.sidebar.radio("🛠 보정 모드", ["보기 전용", "감염 토글", "유효 RBC 토글"])
st.sidebar.info("💡 이미지를 클릭하면 해당 세포의 상태가 변경됩니다.")

# ===============================
# 🧪 결과 렌더링 및 인터랙션
# ===============================
if st.session_state.state["analyzed"]:
    s = st.session_state.state
    output = s["orig"].copy()
    
    # 그리기 최적화: 한 번에 컨투어 그리기
    for cell_id in range(1, int(np.max(s["masks"])) + 1):
        if cell_id not in s["valid"] and edit_mode == "보기 전용": continue
        
        color = (150, 150, 150) # 기본 (제외된 세포)
        if cell_id in s["valid"]:
            color = (0, 0, 255) if cell_id in s["infected"] else (0, 255, 0)
        
        # 실제로는 컨투어보다 외곽선만 가볍게 그리는 것이 빠름
        cell_mask = (s["masks"] == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)

    st.subheader("📊 분석 결과 시각화")

    # 🔎 표시용 이미지 설정
    display_img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    coords = streamlit_image_coordinates(
        display_img,
        key="click",
        use_column_width=True
    )
    
    # -----------------------
    # 🖱 클릭 처리
    # -----------------------
    if coords and edit_mode != "보기 전용":
        
        # 1. 원본 이미지 크기
        orig_h, orig_w = masks.shape[:2]
        
        # 2. 현재 화면에 표시된 이미지 크기 (컴포넌트에서 반환해줌)
        disp_w = coords["width"]
        disp_h = coords["height"]
        
        # 3. 비율 계산 및 좌표 변환
        x_ratio = orig_w / disp_w
        y_ratio = orig_h / disp_h
        
        real_x = int(coords["x"] * x_ratio)
        real_y = int(coords["y"] * y_ratio)
    
        # 🔥 변환된 좌표로 안전 범위 체크
        if 0 <= real_y < orig_h and 0 <= real_x < orig_w:
            cell_id = masks[real_y, real_x]
            
            if cell_id != 0:
                if edit_mode == "감염 토글":
                    if cell_id in infected_cells:
                        infected_cells.remove(cell_id)
                    else:
                        infected_cells.add(cell_id)
    
                elif edit_mode == "유효 RBC 토글":
                    if cell_id in valid_cells:
                        valid_cells.remove(cell_id)
                    else:
                        valid_cells.add(cell_id)
                
                # 세션 상태 강제 업데이트 (필요 시)
                st.session_state.valid = valid_cells
                st.session_state.infected = infected_cells
                
                st.rerun()

    # 🏥 진단 통계
    tot_valid = len(s["valid"])
    tot_inf = len(s["infected"].intersection(s["valid"]))
    parasitemia = (tot_inf / tot_valid * 100) if tot_valid > 0 else 0

    st.divider()
    cols = st.columns(3)
    cols[0].metric("Total Valid RBC", f"{tot_valid} 개")
    cols[1].metric("Infected RBC", f"{tot_inf} 개")
    cols[2].metric("Parasitemia", f"{parasitemia:.2f} %")

    if parasitemia > 5:
        st.error(f"⚠️ 고위험: Parasitemia가 {parasitemia:.2f}%로 매우 높습니다.")
    elif parasitemia > 0:
        st.warning("🟡 주의: 감염된 세포가 발견되었습니다.")
    else:
        st.success("✅ 정상: 감염된 세포가 발견되지 않았습니다.")
