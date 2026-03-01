import streamlit as st
import cv2
import numpy as np
from cellpose.models import Cellpose
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide")

# ===============================
# 🎨 병원 스타일 UI
# ===============================
st.markdown("""
<style>
.title {
    font-size:28px;
    font-weight:700;
    color:#0B3C5D;
}
.sidebar .sidebar-content {
    background-color:#F2F6FA;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧪 Malaria Diagnostic System</div>', unsafe_allow_html=True)
st.write("AI 기반 적혈구 감염 분석 시스템")

# ===============================
# 📦 모델 캐싱
# ===============================
@st.cache_resource
def load_model():
    return Cellpose(gpu=False, model_type="cyto")

model = load_model()

# ===============================
# 📌 Sidebar
# ===============================
st.sidebar.header("📁 Image Management")

uploaded_file = st.sidebar.file_uploader(
    "현미경 이미지 업로드",
    type=["jpg","png","jpeg","tif","tiff"]
)

analysis_button = st.sidebar.button("🔍 AI 분석 실행")

edit_mode = st.sidebar.radio(
    "🛠 수동 보정 모드 선택",
    ["보기 전용", "감염 토글", "유효 RBC 토글"]
)

# ===============================
# 세션 상태 초기화
# ===============================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ===============================
# 이미지 로딩
# ===============================
if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    if analysis_button:

        with st.spinner("AI 분석 중..."):

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            masks, _, _, _ = model.eval(img_rgb, diameter=None, channels=[0,0])

            total_cells = int(np.max(masks))
            valid_cells = set(range(1, total_cells+1))
            infected_cells = set()

            hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            lower_purple = np.array([120,100,50])
            upper_purple = np.array([170,255,255])

            for cell_id in range(1, total_cells+1):
                cell_mask = (masks==cell_id).astype(np.uint8)
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
            st.session_state.valid = valid_cells
            st.session_state.infected = infected_cells
            st.session_state.orig = img_bgr
            st.session_state.analyzed = True

# ===============================
# 🧪 분석 결과 표시
# ===============================
if st.session_state.analyzed:

    masks = st.session_state.masks
    valid_cells = st.session_state.valid
    infected_cells = st.session_state.infected
    orig_bgr = st.session_state.orig

    output = orig_bgr.copy()

    for cell_id in range(1, int(np.max(masks))+1):

        cell_mask = (masks==cell_id).astype(np.uint8)
        contours,_ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cell_id in valid_cells:
            if cell_id in infected_cells:
                color=(0,0,255)
            else:
                color=(0,255,0)
        else:
            color=(150,150,150)

        cv2.drawContours(output, contours, -1, color, 1)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    # -----------------------
    # 🔎 표시용 리사이즈 (원본은 유지)
    # -----------------------
    display_img = output_rgb

    st.subheader("📊 분석 결과 (이미지를 클릭하여 수정)")

    coords = streamlit_image_coordinates(
        display_img,
        key="click",
        use_column_width=True
    )

    # -----------------------
    # 🖱 클릭 처리
    # -----------------------
    if coords and edit_mode != "보기 전용":

        scale_x = orig_bgr.shape[1] / display_img.shape[1]
        scale_y = orig_bgr.shape[0] / display_img.shape[0]

        x = int(coords["x"] * scale_x)
        y = int(coords["y"] * scale_y)

        cell_id = masks[y, x]

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

            st.rerun()

    # -----------------------
    # 🏥 진단 패널
    # -----------------------
    tot_valid = len(valid_cells)
    tot_inf = len(infected_cells.intersection(valid_cells))
    parasitemia = (tot_inf/tot_valid*100) if tot_valid>0 else 0

    st.markdown("### 🏥 Diagnostic Summary")
    c1,c2,c3 = st.columns(3)
    c1.metric("Valid RBC", tot_valid)
    c2.metric("Infected RBC", tot_inf)
    c3.metric("Parasitemia (%)", f"{parasitemia:.2f}")

    if parasitemia < 1:
        st.success("Low Infection Risk")
    elif parasitemia < 5:
        st.warning("Moderate Infection Risk")
    else:
        st.error("High Infection Risk – Immediate Attention Required")
