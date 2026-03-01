import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cellpose.models import Cellpose
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="Malaria Diagnostic System (Cell Counting)",
    layout="wide"
)

# ===============================
# 🏥 병원용 상단 UI
# ===============================
st.markdown("""
<style>
.main-title {
    font-size:32px;
    font-weight:700;
    color:#0E4C92;
}
.metric-box {
    background-color:#F4F8FB;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧪 Automated Malaria Diagnostic System</div>', unsafe_allow_html=True)
st.write("AI 기반 적혈구 분석 및 기생충 감염 판별 시스템")

# ===============================
# 모델 캐싱
# ===============================
@st.cache_resource
def load_model():
    return Cellpose(gpu=False, model_type="cyto")

model = load_model()

# ===============================
# 세션 상태 초기화
# ===============================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ===============================
# 이미지 업로드
# ===============================
uploaded_file = st.file_uploader("현미경 이미지 업로드", type=["jpg","png","jpeg","tif","tiff"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, use_container_width=True)

    if st.button("🔍 AI 분석 시작"):

        with st.spinner("AI 분석 중..."):

            masks, flows, styles, diams = model.eval(
                img_rgb,
                diameter=None,
                channels=[0,0]
            )

            total_cells = int(np.max(masks))
            cell_areas = np.bincount(masks.flatten())

            valid_cells = set(range(1, total_cells+1))
            infected_cells = set()

            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            lower_purple = np.array([120,100,50])
            upper_purple = np.array([170,255,255])

            for cell_id in range(1, total_cells+1):
                cell_mask = (masks == cell_id).astype(np.uint8)
                y_idx, x_idx = np.where(cell_mask==1)
                if len(y_idx)==0:
                    continue

                y_min,y_max = np.min(y_idx), np.max(y_idx)
                x_min,x_max = np.min(x_idx), np.max(x_idx)

                roi_hsv = hsv_img[y_min:y_max+1, x_min:x_max+1]
                roi_mask = cell_mask[y_min:y_max+1, x_min:x_max+1]

                parasite_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
                parasite_in_cell = cv2.bitwise_and(parasite_mask, parasite_mask, mask=roi_mask)

                if cv2.countNonZero(parasite_in_cell) > 10:
                    infected_cells.add(cell_id)

            st.session_state.masks = masks
            st.session_state.valid_cells = valid_cells
            st.session_state.infected_cells = infected_cells
            st.session_state.orig = img_bgr
            st.session_state.analyzed = True

# ===============================
# 분석 결과 + 수동 수정
# ===============================
if st.session_state.analyzed:

    masks = st.session_state.masks
    valid_cells = st.session_state.valid_cells
    infected_cells = st.session_state.infected_cells
    orig_bgr = st.session_state.orig

    output = orig_bgr.copy()

    for cell_id in range(1, int(np.max(masks))+1):

        cell_mask = (masks==cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cell_id in valid_cells:
            if cell_id in infected_cells:
                color=(0,0,255)
            else:
                color=(0,255,0)
        else:
            color=(128,128,128)

        cv2.drawContours(output, contours, -1, color, 2)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.subheader("📊 분석 결과 (클릭하여 수동 수정 가능)")

    value = streamlit_image_coordinates(output_rgb, key="coords")

    # -------------------------
    # 클릭 이벤트 처리
    # -------------------------
    if value is not None:
        x = value["x"]
        y = value["y"]
        cell_id = masks[y, x]

        if cell_id != 0:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("감염 상태 토글"):
                    if cell_id in infected_cells:
                        infected_cells.remove(cell_id)
                    else:
                        infected_cells.add(cell_id)

            with col2:
                if st.button("유효 RBC 토글"):
                    if cell_id in valid_cells:
                        valid_cells.remove(cell_id)
                    else:
                        valid_cells.add(cell_id)

    # -------------------------
    # 병원용 결과 패널
    # -------------------------
    tot_valid = len(valid_cells)
    tot_inf = len(infected_cells.intersection(valid_cells))
    parasitemia = (tot_inf/tot_valid*100) if tot_valid>0 else 0

    st.markdown("### 🏥 Diagnostic Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Valid RBC Count", tot_valid)
    c2.metric("Infected RBC Count", tot_inf)
    c3.metric("Parasitemia (%)", f"{parasitemia:.2f}")

    if parasitemia < 1:
        st.success("Low Infection Level")
    elif parasitemia < 5:
        st.warning("Moderate Infection Level")
    else:
        st.error("High Infection Level – Immediate Clinical Attention Required")
