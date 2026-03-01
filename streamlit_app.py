import streamlit as st
import cv2
import numpy as np
from cellpose import models

# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="Malaria Clinical Analyzer(Cell Counting)",
    layout="wide"
)

# ===============================
# CSS (병원 스타일)
# ===============================
st.markdown("""
<style>
.metric-card {
    background-color: #f8fbff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #d9e6f2;
}
.title-text {
    font-size: 28px;
    font-weight: 600;
    color: #1f4e79;
}
.subtitle-text {
    font-size: 14px;
    color: #6c757d;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 유틸 함수
# ===============================
def resize_for_ai(img, target_width=1000):
    h, w = img.shape[:2]
    if w <= target_width:
        return img
    ratio = target_width / w
    return cv2.resize(img, (target_width, int(h * ratio)))


@st.cache_resource
def load_model():
    return models.CellposeModel(gpu=False)


# ===============================
# 세션 초기화
# ===============================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.masks = None
    st.session_state.valid_cells = set()
    st.session_state.infected_cells = set()
    st.session_state.cell_contours = {}
    st.session_state.orig_img = None


# ===============================
# 헤더
# ===============================
st.markdown('<div class="title-text">🧬 Malaria Clinical Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI 기반 적혈구 감염 자동 분석 시스템</div>', unsafe_allow_html=True)
st.divider()

# ===============================
# 사이드바
# ===============================
st.sidebar.header("🔬 검사 설정")

uploaded_file = st.sidebar.file_uploader(
    "현미경 이미지 업로드",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    st.session_state.orig_img = cv2.imdecode(file_bytes, 1)

if st.sidebar.button("🚀 AI 자동 분석 시작"):
    if st.session_state.orig_img is None:
        st.warning("이미지를 업로드하세요.")
    else:
        with st.spinner("AI가 세포를 분석 중입니다..."):

            img = resize_for_ai(st.session_state.orig_img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            model = load_model()
            masks, flows, styles = model.eval(img_rgb, diameter=30)

            total_cells = np.max(masks)
            cell_areas = np.bincount(masks.flatten())

            valid = set()
            infected = set()
            contours_dict = {}

            lower_purple = np.array([120, 70, 40])
            upper_purple = np.array([175, 255, 255])

            for cell_id in range(1, total_cells + 1):
                mask = (masks == cell_id).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_dict[cell_id] = cnts

                if cell_areas[cell_id] < 200:
                    continue

                valid.add(cell_id)

                y, x = np.where(mask == 1)
                if len(y) > 0:
                    roi_hsv = hsv_img[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    roi_m = mask[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1]
                    p_mask = cv2.inRange(roi_hsv, lower_purple, upper_purple)
                    if cv2.countNonZero(cv2.bitwise_and(p_mask, p_mask, mask=roi_m)) > 8:
                        infected.add(cell_id)

            st.session_state.masks = masks
            st.session_state.valid_cells = valid
            st.session_state.infected_cells = infected
            st.session_state.cell_contours = contours_dict
            st.session_state.analyzed = True

        st.success("AI 분석 완료")


# ===============================
# 분석 결과 표시
# ===============================
if st.session_state.analyzed:

    v = st.session_state.valid_cells
    i = st.session_state.infected_cells.intersection(v)
    parasitemia = (len(i) / len(v) * 100) if v else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("유효 RBC 수", len(v))
    col2.metric("감염 세포 수", len(i))
    col3.metric("Parasitemia (%)", f"{parasitemia:.2f}")

    st.divider()

    # 결과 이미지
    img = resize_for_ai(st.session_state.orig_img)
    display = img.copy()

    for cid, cnts in st.session_state.cell_contours.items():
        if cid in v:
            color = (0, 0, 255) if cid in st.session_state.infected_cells else (0, 200, 0)
            thickness = 2
        else:
            color = (150, 150, 150)
            thickness = 1

        cv2.drawContours(display, cnts, -1, color, thickness)

    st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ===============================
    # 📝 수동 수정 (분석 후만 활성화)
    # ===============================
    st.divider()
    st.subheader("👨‍⚕️ 전문의 수동 수정")

    edit_id = st.number_input("세포 ID 입력", min_value=1, step=1)

    colA, colB = st.columns(2)

    if colA.button("🔴 감염 상태 토글"):
        if edit_id in st.session_state.infected_cells:
            st.session_state.infected_cells.remove(edit_id)
        else:
            st.session_state.infected_cells.add(edit_id)
        st.rerun()

    if colB.button("⚪ 유효 여부 토글"):
        if edit_id in st.session_state.valid_cells:
            st.session_state.valid_cells.remove(edit_id)
        else:
            st.session_state.valid_cells.add(edit_id)
        st.rerun()

else:
    st.info("좌측에서 이미지를 업로드하고 분석을 시작하세요.")
