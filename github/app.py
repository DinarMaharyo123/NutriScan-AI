from datetime import date, datetime
from pathlib import Path
import time
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# Direktori folder tempat app.py berada (di Streamlit Cloud: .../github)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"
NUTRITION_PATH = BASE_DIR / "nutrition_db.csv"
LOG_FILE = BASE_DIR / "calorie_log.csv"
CM_PATH = BASE_DIR / "confusion_matrix.png"

# 1. KONFIGURASI HALAMAN & CSS
st.set_page_config(
    page_title="NutriScan AI | Deteksi Makanan & Kalender Kalori",
    page_icon="🥗",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 55%, #43a047 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.3rem;
        box-shadow: 0 4px 14px rgba(27, 94, 32, 0.18);
    }
    .hero-banner h1 { color: white !important; font-size: 1.9rem !important; margin: 0 0 0.35rem 0 !important; }
    .hero-banner p { color: #e8f5e9 !important; font-size: 1rem !important; margin: 0 !important; }
    .step-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: 5px solid #2e7d32;
        border-radius: 14px;
        padding: 1.2rem 1.25rem;
        height: 100%;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .step-number {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .step-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.4rem;
    }
    .step-desc {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.5;
    }
    .result-card {
        background-color: #f1f8e9; border-left: 6px solid #2e7d32;
        padding: 1rem 1.25rem; border-radius: 10px; margin-bottom: 1rem;
    }
    .result-title { font-size: 0.82rem; text-transform: uppercase; color: #33691e; font-weight: 600; }
    .result-food { font-size: 1.6rem; font-weight: 700; color: #1b5e20; margin: 0.2rem 0 0 0; }
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        padding: 0.8rem 1rem; border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
    }
    .food-badge {
        display: inline-block; background-color: #e8f5e9; color: #1b5e20;
        border: 1px solid #c8e6c9; padding: 4px 10px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 500; margin: 3px 2px;
    }
    .akg-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #0284c7;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.8rem;
        font-size: 0.9rem;
        color: #1e293b;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 2. INISIALISASI STATUS TUTORIAL
if "tutorial_completed" not in st.session_state:
  st.session_state.tutorial_completed = False

# ==========================================
# HALAMAN AWAL: TUTORIAL PENGGUNAAN APLIKASI
# ==========================================
if not st.session_state.tutorial_completed:
  st.markdown(
      """
    <div class="hero-banner">
        <h1>👋 Selamat Datang di NutriScan AI</h1>
        <p>Pelajari panduan singkat berikut sebelum mulai memindai makanan dan memantau asupan kalorimu.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  st.subheader("📖 Panduan Cara Penggunaan (4 Langkah Mudah)")
  st.write("")

  t1, t2, t3, t4 = st.columns(4, gap="medium")

  with t1:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">LANGKAH 1</div>
            <div class="step-title">📸 Siapkan Foto Makanan</div>
            <div class="step-desc">
                Unggah gambar (JPG/PNG) atau ambil foto langsung lewat kamera. Pastikan pencahayaan terang dan fokus pada satu jenis makanan:
                <b>Apel, Gorengan, Kentang Goreng, Kopi,</b> atau <b>Roti</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with t2:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">LANGKAH 2</div>
            <div class="step-title">🚀 Analisis dengan AI</div>
            <div class="step-desc">
                Tekan tombol <b>Analisis Makanan Sekarang</b>. Sistem Computer Vision akan mengenali jenis makanan beserta skor tingkat keyakinan AI secara otomatis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with t3:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">LANGKAH 3</div>
            <div class="step-title">⚖️ Atur Porsi & Cek Gizi</div>
            <div class="step-desc">
                Geser pengatur berat porsi (gram) sesuai makanan yang dikonsumsi untuk melihat perkiraan <b>Kalori, Protein, Karbohidrat,</b> dan <b>Lemak</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  with t4:
    st.markdown(
        """
        <div class="step-card">
            <div class="step-number">LANGKAH 4</div>
            <div class="step-title">📅 Catat ke Kalender Kalori</div>
            <div class="step-desc">
                Simpan hasil analisis ke <b>Kalender Kalori</b> untuk memantau total asupan gizi harian dan membandingkannya dengan batas wajar harian.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.write("")
  st.info(
      "💡 **Tips Akurasi:** Hindari foto yang buram atau berisi banyak jenis"
      " makanan sekaligus dalam satu bingkai agar hasil identifikasi AI lebih"
      " tepat."
  )
  st.write("")

  _, btn_col, _ = st.columns([1, 1.4, 1])
  with btn_col:
    if st.button(
        "🚀 Mengerti & Masuk ke Dashboard",
        type="primary",
        use_container_width=True,
    ):
      st.session_state.tutorial_completed = True
      st.rerun()

  st.stop()

# ==========================================
# 3. MODEL, DATABASE GIZI & LOG KALENDER
# ==========================================
CLASS_NAMES = ["apel", "gorengan", "kentang_goreng", "kopi", "roti"]
DISPLAY_NAMES = {
    "apel": "Buah Apel 🍎",
    "gorengan": "Gorengan 🍢",
    "kentang_goreng": "Kentang Goreng 🍟",
    "kopi": "Minuman Kopi ☕",
    "roti": "Roti 🍞",
}
LOG_COLUMNS = [
    "id",
    "tanggal",
    "jam",
    "waktu_makan",
    "makanan",
    "porsi",
    "kalori",
    "protein",
    "karbo",
    "lemak",
]


@st.cache_resource
def load_model():
  if not MODEL_PATH.exists():
    return None
  m = models.mobilenet_v2(weights=None)
  m.classifier[1] = nn.Linear(m.last_channel, len(CLASS_NAMES))
  m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
  m.eval()
  return m


@st.cache_data
def load_nutrition():
  try:
    df = pd.read_csv(NUTRITION_PATH)
    df.columns = df.columns.str.strip().str.lower()
    return df
  except Exception:
    return pd.DataFrame({
        "makanan": ["apel", "gorengan", "kentang_goreng", "kopi", "roti"],
        "kalori": [52, 280, 312, 65, 265],
        "protein": [0.3, 3.5, 3.4, 1.5, 9.0],
        "karbo": [13.8, 25.0, 41.0, 9.0, 49.0],
        "lemak": [0.2, 18.0, 15.0, 2.5, 3.2],
    })


def load_calorie_log():
  if LOG_FILE.exists():
    try:
      return pd.read_csv(LOG_FILE, dtype={"id": str, "tanggal": str})
    except Exception:
      pass
  return pd.DataFrame(columns=LOG_COLUMNS)


def add_calorie_entry(entry: dict):
  df = load_calorie_log()
  df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
  df.to_csv(LOG_FILE, index=False)


def delete_calorie_entry(entry_id: str):
  df = load_calorie_log()
  df = df[df["id"] != entry_id]
  df.to_csv(LOG_FILE, index=False)


def evaluasi_status(nilai: float, batas_bawah: float, batas_atas: float) -> str:
  if nilai == 0:
    return "⚪ Belum ada asupan"
  elif nilai < batas_bawah:
    return "🟡 Masih di bawah kebutuhan"
  elif nilai <= batas_atas:
    return "🟢 Dalam batas wajar"
  else:
    return "🔴 Melebihi batas wajar"


model = load_model()
df_gizi = load_nutrition()

# ==========================================
# 4. SIDEBAR DASHBOARD
# ==========================================
with st.sidebar:
  st.markdown("### ⚙️ Panel Sistem")
  if model:
    st.success("✅ **Model AI Aktif** (`model.pth`)")
  else:
    st.error("❌ File `model.pth` tidak ditemukan.")

  if st.button("📖 Lihat Ulang Tutorial", use_container_width=True):
    st.session_state.tutorial_completed = False
    st.rerun()

  st.markdown("#### 🍽️ Kategori Dikenali AI")
  badges = "".join(
      [f'<span class="food-badge">{n}</span>' for n in DISPLAY_NAMES.values()]
  )
  st.markdown(badges, unsafe_allow_html=True)
  st.divider()

  st.markdown("#### 🎯 Sensitivitas Deteksi")
  threshold = st.slider("Batas minimal keyakinan (%):", 40, 95, 65, 5)

  st.markdown("#### 🔥 Target Kalori Harian")
  daily_target = st.number_input(
      "Target energi per hari (kkal):",
      min_value=1000,
      max_value=4500,
      value=2150,
      step=50,
      help=(
          "Batas wajar orang dewasa rata-rata: 1.800 – 2.500 kkal/hari (AKG"
          " Kemenkes RI: 2.150 kkal)."
      ),
  )

  if CM_PATH.exists():
    with st.expander("📊 Lihat Akurasi Model"):
      st.image(str(CM_PATH), caption="Confusion Matrix", use_container_width=True)
  st.divider()
  st.caption("🥗 **NutriScan AI** • Klasifikasi Makanan & Kalender Kalori")

# ==========================================
# 5. HEADER & SESSION STATE DASHBOARD
# ==========================================
st.markdown(
    """
<div class="hero-banner">
    <h1>🥗 NutriScan AI — Klasifikasi Makanan & Kalender Kalori</h1>
    <p>Deteksi makanan otomatis dengan Computer Vision, hitung kandungan gizinya, dan pantau asupan harianmu.</p>
</div>
""",
    unsafe_allow_html=True,
)

if "prediction_result" not in st.session_state:
  st.session_state.prediction_result = None
if "last_image_id" not in st.session_state:
  st.session_state.last_image_id = None

# ==========================================
# 6. NAVIGASI TAB UTAMA
# ==========================================
tab_scan, tab_calendar = st.tabs(
    ["🔍 Scan & Analisis Makanan", "📅 Kalender & Monitoring Kalori"]
)

# --- TAB 1: SCAN & ANALISIS MAKANAN ---
with tab_scan:
  col1, col2 = st.columns([1, 1.15], gap="large")

  with col1:
    with st.container(border=True):
      st.subheader("📸 Input Gambar Makanan")
      input_mode = st.radio(
          "Pilih Metode Input:",
          ["📁 Unggah File Gambar", "📷 Kamera Langsung"],
          horizontal=True,
          label_visibility="collapsed",
      )
      st.divider()
      image_data, current_id = None, None

      if input_mode == "📁 Unggah File Gambar":
        up_file = st.file_uploader(
            "Pilih foto makanan (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if up_file:
          image_data = Image.open(up_file).convert("RGB")
          current_id = f"up_{up_file.name}_{up_file.size}"
      else:
        camera_active = st.toggle("Nyalakan Kamera", value=True)
        if camera_active:
          cam_file = st.camera_input("Ambil foto menggunakan kamera")
          if cam_file:
            image_data = Image.open(cam_file).convert("RGB")
            current_id = f"cam_{cam_file.size}"
        else:
          st.info(
              "Kamera dimatikan. Aktifkan sakelar di atas untuk mengambil foto"
              " atau pindah ke mode **Unggah File Gambar**."
          )

      if current_id != st.session_state.last_image_id:
        st.session_state.prediction_result = None
        st.session_state.last_image_id = current_id

      if image_data:
        st.image(
            image_data,
            caption="Pratinjau Gambar Makanan",
            use_container_width=True,
        )
        if st.button(
            "🚀 Analisis Makanan Sekarang",
            type="primary",
            use_container_width=True,
        ):
          if model:
            with st.spinner("Menganalisis gambar..."):
              t0 = time.time()
              tfm = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  transforms.Normalize(
                      [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                  ),
              ])
              tensor = tfm(image_data).unsqueeze(0)
              with torch.no_grad():
                probs = torch.nn.functional.softmax(model(tensor)[0], dim=0)
                conf, idx = torch.max(probs, 0)

              st.session_state.prediction_result = {
                  "class": CLASS_NAMES[idx.item()],
                  "confidence": conf.item() * 100,
                  "time": (time.time() - t0) * 1000,
                  "all_probs": {
                      DISPLAY_NAMES[CLASS_NAMES[i]]: round(
                          probs[i].item() * 100, 1
                      )
                      for i in range(len(CLASS_NAMES))
                  },
              }
          else:
            st.error("Model AI belum terhubung.")
      else:
        st.caption(
            "💡 Tips: Gunakan foto dengan pencahayaan terang dan fokus pada"
            " satu jenis makanan."
        )

  with col2:
    with st.container(border=True):
      st.subheader("📊 Hasil Analisis & Estimasi Gizi")
      res = st.session_state.prediction_result

      if res and image_data:
        conf_score = res["confidence"]
        if conf_score < threshold:
          st.warning(
              f"⚠️ **Objek Tidak Dikenali ({conf_score:.1f}%)** — Dugaan"
              f" terdekat: **{DISPLAY_NAMES[res['class']]}**."
          )
        else:
          food_label = DISPLAY_NAMES[res["class"]]
          st.markdown(
              f"""
                    <div class="result-card">
                        <div class="result-title">Hasil Identifikasi Utama • ⚡ {res['time']:.0f} ms</div>
                        <p class="result-food">{food_label}</p>
                    </div>
                    """,
              unsafe_allow_html=True,
          )

          c1, c2 = st.columns([3, 1])
          c1.write("**Tingkat Keyakinan AI (*Confidence*)**")
          c2.markdown(
              f"<div style='text-align:right; font-weight:700;"
              f" color:#2e7d32;'>{conf_score:.1f}%</div>",
              unsafe_allow_html=True,
          )
          st.progress(min(int(conf_score), 100))
          st.divider()

          porsi = st.slider(
              "⚖️ Sesuaikan Berat Porsi (gram):",
              10,
              500,
              100,
              10,
              format="%d gram",
          )
          row = df_gizi[df_gizi["makanan"] == res["class"]]

          if not row.empty:
            kal = round(float(row["kalori"].values[0]) * porsi / 100, 1)
            prot = round(float(row["protein"].values[0]) * porsi / 100, 1)
            karbo = round(float(row["karbo"].values[0]) * porsi / 100, 1)
            lemak = round(float(row["lemak"].values[0]) * porsi / 100, 1)

            st.markdown(f"#### 🥗 Perkiraan Kandungan Gizi *(per {porsi}g)*")
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("🔥 Energi Total", f"{kal} kkal")
            m2.metric("🥩 Protein", f"{prot} g")
            m3.metric("🍞 Karbohidrat", f"{karbo} g")
            m4.metric("🥑 Lemak", f"{lemak} g")

            st.divider()
            st.markdown("#### 📅 Simpan ke Kalender Kalori?")
            d_col, w_col = st.columns(2)
            log_date = d_col.date_input(
                "Tanggal Konsumsi:", value=date.today(), key="scan_log_date"
            )
            meal_time = w_col.selectbox(
                "Waktu Makan:",
                ["Sarapan", "Makan Siang", "Makan Malam", "Camilan"],
                key="scan_meal_time",
            )

            if st.button(
                "➕ Masukkan ke Kalender Kalori",
                type="primary",
                use_container_width=True,
            ):
              now = datetime.now()
              add_calorie_entry({
                  "id": now.strftime("%Y%m%d%H%M%S%f"),
                  "tanggal": str(log_date),
                  "jam": now.strftime("%H:%M"),
                  "waktu_makan": meal_time,
                  "makanan": food_label,
                  "porsi": porsi,
                  "kalori": kal,
                  "protein": prot,
                  "karbo": karbo,
                  "lemak": lemak,
              })
              st.success(
                  f"✅ **{food_label} ({porsi}g — {kal} kkal)** ditambahkan ke"
                  f" Kalender Kalori tanggal **{log_date}**!"
              )
          else:
            st.error("Data gizi tidak ditemukan.")

        st.write("")
        with st.expander("📈 Lihat Detail Probabilitas Semua Kelas"):
          prob_df = pd.DataFrame({
              "Kategori": list(res["all_probs"].keys()),
              "Probabilitas (%)": list(res["all_probs"].values()),
          }).sort_values(by="Probabilitas (%)", ascending=False)
          st.bar_chart(
              prob_df.set_index("Kategori"), color="#2e7d32", height=220
          )
      else:
        st.info(
            "👈 **Belum ada gambar yang dianalisis.** Unggah foto atau gunakan"
            " kamera di sebelah kiri, lalu klik **🚀 Analisis Makanan"
            " Sekarang**."
        )

# --- TAB 2: KALENDER & MONITORING KALORI ---
with tab_calendar:
  df_log = load_calorie_log()

  with st.container(border=True):
    st.subheader("📆 Monitoring Asupan Gizi Harian")
    top_c1, top_c2 = st.columns([1, 2])
    selected_date = top_c1.date_input(
        "Pilih Tanggal Monitoring:", value=date.today(), key="cal_view_date"
    )

    df_day = df_log[df_log["tanggal"] == str(selected_date)].copy()

    tot_kal = round(df_day["kalori"].sum(), 1) if not df_day.empty else 0.0
    tot_prot = round(df_day["protein"].sum(), 1) if not df_day.empty else 0.0
    tot_karbo = round(df_day["karbo"].sum(), 1) if not df_day.empty else 0.0
    tot_lemak = round(df_day["lemak"].sum(), 1) if not df_day.empty else 0.0

    sisa_kal = round(daily_target - tot_kal, 1)
    persen_target = min(tot_kal / daily_target, 1.0) if daily_target > 0 else 0.0

    with top_c2:
      st.write(
          f"**Progres Kalori Tanggal {selected_date}** ({tot_kal} /"
          f" {daily_target} kkal)"
      )
      st.progress(persen_target)
      if sisa_kal >= 0:
        st.caption(
            f"🟢 Sisa kuota kalori hari ini: **{sisa_kal} kkal** dari target"
            " harian (Batas wajar dewasa: **1.800 – 2.500 kkal/hari**)."
        )
      else:
        st.caption(
            f"🔴 Asupan melebihi target harian sebesar **{abs(sisa_kal)}"
            " kkal**."
        )

    st.write("")
    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric(
        "🔥 Energi Terkonsumsi",
        f"{tot_kal} kkal",
        "Wajar: 1.800–2.500 kkal",
        delta_color="off",
    )
    dm2.metric(
        "🥩 Total Protein",
        f"{tot_prot} g",
        "Wajar: 50–65 g/hari",
        delta_color="off",
    )
    dm3.metric(
        "🍞 Total Karbohidrat",
        f"{tot_karbo} g",
        "Wajar: 275–325 g/hari",
        delta_color="off",
    )
    dm4.metric(
        "🥑 Total Lemak",
        f"{tot_lemak} g",
        "Maksimal: 67 g/hari",
        delta_color="off",
    )

    st.markdown(
        f"""
        <div class="akg-box">
            <b>📏 Status & Acuan Batas Wajar Asupan Harian Manusia (Standar AKG Kemenkes RI):</b><br>
- <b>Energi Total:</b> {tot_kal} kkal <i>(Acuan: 2.150 kkal | Rentang wajar: 1.800–2.500 kkal)</i> — <b>{evaluasi_status(tot_kal, 1800, 2500)}</b><br>
- <b>Protein:</b> {tot_prot} g <i>(Acuan: 60 g | Rentang wajar: 50–65 g)</i> — <b>{evaluasi_status(tot_prot, 50, 65)}</b><br>
- <b>Karbohidrat:</b> {tot_karbo} g <i>(Acuan: 300 g | Rentang wajar: 275–325 g)</i> — <b>{evaluasi_status(tot_karbo, 275, 325)}</b><br>
- <b>Lemak Total:</b> {tot_lemak} g <i>(Batas maksimal anjuran: 67 g/hari)</i> — <b>{"⚪ Belum ada asupan" if tot_lemak == 0 else ("🟢 Dalam batas wajar" if tot_lemak <= 67 else "🔴 Melebihi batas wajar")}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(f"#### 🍽️ Daftar Makanan Tercatat ({selected_date})")

    if not df_day.empty:
      tampil_df = df_day[[
          "jam",
          "waktu_makan",
          "makanan",
          "porsi",
          "kalori",
          "protein",
          "karbo",
          "lemak",
      ]].rename(
          columns={
              "jam": "Jam",
              "waktu_makan": "Waktu",
              "makanan": "Jenis Makanan",
              "porsi": "Porsi (g)",
              "kalori": "Kalori (kkal)",
              "protein": "Protein (g)",
              "karbo": "Karbo (g)",
              "lemak": "Lemak (g)",
          }
      )
      st.dataframe(tampil_df, use_container_width=True, hide_index=True)

      with st.expander("🗑️ Hapus Salah Satu Catatan di Tanggal Ini"):
        opsi_hapus = {
            (
                f"[{r['jam']}] {r['waktu_makan']} — {r['makanan']} ({r['porsi']}g,"
                f" {r['kalori']} kkal)"
            ): r["id"]
            for _, r in df_day.iterrows()
        }
        pilihan_hapus = st.selectbox(
            "Pilih catatan yang ingin dihapus:", list(opsi_hapus.keys())
        )
        if st.button("Hapus Catatan Terpilih", type="secondary"):
          delete_calorie_entry(opsi_hapus[pilihan_hapus])
          st.rerun()
    else:
      st.info(
          f"Belum ada makanan yang dicatat pada tanggal **{selected_date}**."
          " Pindai makanan di tab sebelah kiri lalu klik **➕ Masukkan ke"
          " Kalender Kalori**."
      )

  st.write("")
  with st.container(border=True):
    st.subheader("📈 Grafik Monitoring Kalori Antar Hari")
    if not df_log.empty:
      rekap_harian = (
          df_log.groupby("tanggal", as_index=False)[
              ["kalori", "protein", "karbo", "lemak"]
          ]
          .sum()
          .sort_values(by="tanggal")
      )
      st.bar_chart(
          rekap_harian.set_index("tanggal")["kalori"],
          color="#2e7d32",
          height=260,
      )

      with st.expander("📋 Lihat Tabel Rekap Semua Hari"):
        rekap_tampil = rekap_harian.rename(
            columns={
                "tanggal": "Tanggal",
                "kalori": "Total Kalori (kkal)",
                "protein": "Total Protein (g)",
                "karbo": "Total Karbo (g)",
                "lemak": "Total Lemak (g)",
            }
        )
        st.dataframe(rekap_tampil, use_container_width=True, hide_index=True)
    else:
      st.caption(
          "Grafik perbandingan harian akan otomatis muncul setelah kamu"
          " mencatat makanan pertamamu di Kalender Kalori."
      )

# === SELESAI ===