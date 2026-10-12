import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "medilens_model.joblib"
METRICS_PATH = ROOT / "metrics.json"

st.set_page_config(page_title="MediLens AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #14243f 0, #0a1020 38%, #080d18 100%); color:#e7eefc;}
section[data-testid="stSidebar"] {background:#0b1425; border-right:1px solid #233650;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#101f35,#0b1526); padding:18px; border:1px solid #263d5b; border-radius:16px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#263d5b!important;}
h1,h2,h3 {color:#f4f8ff;}
.small-muted {color:#9fb3d1; font-size:0.92rem;}
.hero {padding:22px 24px; border:1px solid #284362; border-radius:20px; background:linear-gradient(120deg,rgba(24,54,88,.88),rgba(13,24,43,.8)); margin-bottom:18px;}
.pill {display:inline-block; padding:5px 10px; border-radius:20px; background:#143b43; color:#8df2db; border:1px solid #23615e; font-size:12px; margin-bottom:8px;}
.warning {padding:14px 16px; background:#382718; border:1px solid #8d602d; border-radius:12px; color:#ffdfb1;}
.info {padding:14px 16px; background:#102d3b; border:1px solid #286176; border-radius:12px; color:#c6efff;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)

metrics = load_metrics()
model = load_model()

with st.sidebar:
    st.markdown("## 🧬 MediLens AI")
    st.caption("Учебная лаборатория ИИ")
    page = st.radio("Навигация", ["Анализ показателей", "О модели", "Данные и ограничения"])
    st.divider()
    lang = st.selectbox("Язык интерфейса / Тіл", ["Русский", "Қазақша"])
    st.caption("Версия 1.0 · прототип")

if lang == "Қазақша":
    title = "MediLens AI — қан көрсеткіштерін оқу талдауы"
    subtitle = "Қан талдауы көрсеткіштерін түсіндіруге арналған оқу прототипі."
    input_title = "Зертханалық көрсеткіштер"
    run_label = "Демо-модельді іске қосу"
    result_title = "Модель нәтижесі"
else:
    title = "MediLens AI"
    subtitle = "Учебный прототип для анализа лабораторных показателей, связанных с анемией."
    input_title = "Лабораторные показатели"
    run_label = "Запустить учебную модель"
    result_title = "Результат модели"

st.markdown('<div class="hero"><span class="pill">AI · HEMATOLOGY · EDUCATION</span>', unsafe_allow_html=True)
st.title(title)
st.markdown(f"<div class='small-muted'>{subtitle}</div></div>", unsafe_allow_html=True)
st.markdown('<div class="warning"><b>Важно:</b> это демонстрационный учебный проект. Модель обучена на синтетических, искусственно созданных данных, а не на реальных пациентах. Она не диагностирует анемию, не оценивает личный риск заболевания и не предназначена для медицинских решений.</div>', unsafe_allow_html=True)
st.write("")

if page == "Анализ показателей":
    st.subheader(input_title)
    st.markdown('<div class="small-muted">Введите примерные значения или нажмите «Загрузить демо-пример». Все поля используются только для демонстрации работы модели.</div>', unsafe_allow_html=True)
    if "demo_loaded" not in st.session_state:
        st.session_state.demo_loaded = False
    col_demo, col_space = st.columns([1,4])
    with col_demo:
        if st.button("Загрузить демо-пример", use_container_width=True):
            st.session_state.demo_loaded = True
    defaults = {"age_years": 35, "sex_code_demo": 0, "hemoglobin_g_dl": 10.8, "rbc_million_uL": 3.8, "mcv_fL": 76.0, "mch_pg": 25.5, "hematocrit_percent": 33.0}
    with st.form("lab_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            age = st.number_input("Возраст (лет)", min_value=18, max_value=100, value=defaults["age_years"] if st.session_state.demo_loaded else 30)
            sex_label = st.selectbox("Демо-код пола модели", ["0 — условная категория A", "1 — условная категория B"])
            sex_code = int(sex_label[0])
            hb = st.number_input("Гемоглобин (г/дл)", min_value=3.0, max_value=25.0, value=defaults["hemoglobin_g_dl"] if st.session_state.demo_loaded else 13.5, step=0.1)
        with c2:
            rbc = st.number_input("Эритроциты (млн/мкл)", min_value=1.0, max_value=8.0, value=defaults["rbc_million_uL"] if st.session_state.demo_loaded else 4.5, step=0.1)
            mcv = st.number_input("MCV (фл)", min_value=40.0, max_value=130.0, value=defaults["mcv_fL"] if st.session_state.demo_loaded else 88.0, step=0.5)
        with c3:
            mch = st.number_input("MCH (пг)", min_value=10.0, max_value=45.0, value=defaults["mch_pg"] if st.session_state.demo_loaded else 29.0, step=0.5)
            hct = st.number_input("Гематокрит (%)", min_value=5.0, max_value=75.0, value=defaults["hematocrit_percent"] if st.session_state.demo_loaded else 40.0, step=0.5)
        submitted = st.form_submit_button(run_label, type="primary", use_container_width=True)
    if submitted:
        sample = pd.DataFrame([{
            "age_years": age, "sex_code_demo": sex_code, "hemoglobin_g_dl": hb,
            "rbc_million_uL": rbc, "mcv_fL": mcv, "mch_pg": mch, "hematocrit_percent": hct
        }])
        prediction = int(model.predict(sample)[0])
        proba = model.predict_proba(sample)[0]
        st.subheader(result_title)
        r1,r2 = st.columns([1,2])
        with r1:
            st.metric("Класс модели", "Паттерн класса 1" if prediction == 1 else "Паттерн класса 0")
            st.metric("Оценка модели для класса 1", f"{proba[1]*100:.1f}%")
        with r2:
            st.markdown('<div class="info"><b>Как читать результат</b><br>Класс 0/1 — это метки, созданные для учебного синтетического набора данных. Процент — оценка модели, а не вероятность наличия анемии у человека. Не используйте результат для принятия медицинских решений.</div>', unsafe_allow_html=True)
        st.markdown("#### Введённые данные")
        st.dataframe(sample.rename(columns={
            "age_years":"Возраст", "sex_code_demo":"Демо-код пола", "hemoglobin_g_dl":"Гемоглобин (г/дл)",
            "rbc_million_uL":"Эритроциты (млн/мкл)", "mcv_fL":"MCV (фл)", "mch_pg":"MCH (пг)", "hematocrit_percent":"Гематокрит (%)"
        }), use_container_width=True, hide_index=True)
        st.caption("Показатели не сравниваются с клиническими референсами и не являются медицинской интерпретацией.")
elif page == "О модели":
    st.subheader("Модель машинного обучения")
    st.write("Используется Random Forest из scikit-learn в составе Pipeline с заполнением пропусков медианой и масштабированием. Модель действительно обучена скриптом `train_model.py` на приложенном синтетическом наборе данных.")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Строк в наборе", f"{metrics['rows_total']:,}")
    m2.metric("Обучение", f"{metrics['train_rows']:,}")
    m3.metric("Тест", f"{metrics['test_rows']:,}")
    m4.metric("Accuracy на синтетике", f"{metrics['accuracy']:.3f}")
    st.markdown("#### Фактические метрики на отложенной выборке")
    metric_df = pd.DataFrame([
        {"Метрика":"Accuracy", "Значение":metrics["accuracy"]},
        {"Метрика":"Precision", "Значение":metrics["precision"]},
        {"Метрика":"Recall", "Значение":metrics["recall"]},
        {"Метрика":"F1-score", "Значение":metrics["f1"]},
    ])
    st.dataframe(metric_df, hide_index=True, use_container_width=True)
    st.markdown("#### Матрица ошибок")
    cm = np.array(metrics["confusion_matrix_labels_0_1"])
    st.dataframe(pd.DataFrame(cm, index=["Факт: класс 0","Факт: класс 1"], columns=["Прогноз: класс 0","Прогноз: класс 1"]), use_container_width=True)
    st.markdown('<div class="warning">Эти числа показывают только то, насколько модель воспроизводит метки синтетического набора. Они не измеряют точность выявления анемии в реальной клинической практике.</div>', unsafe_allow_html=True)
    st.markdown("#### Признаки, используемые моделью")
    st.write(", ".join(metrics["features"]))
elif page == "Данные и ограничения":
    st.subheader("Данные, ограничения и этика")
    st.markdown("""
**Что находится в проекте**
- `data/demo_synthetic_blood_data.csv` — 1 800 искусственно сгенерированных строк.
- `models/medilens_model.joblib` — сохранённая обученная модель.
- `models/metrics.json` — фактические метрики запуска обучения и тестирования.

**Как созданы метки**
Метка формируется упрощённым учебным правилом на основе смоделированного гемоглобина и некоторых эритроцитарных индексов, после чего небольшая доля меток случайно изменяется. Это не клинический стандарт и не способ постановки диагноза.

**Ограничения**
1. Данные синтетические, не получены от реальных пациентов.
2. Метрики могут быть высокими из-за искусственного правила формирования меток.
3. Модель не валидирована врачами и не предназначена для диагностики, скрининга или лечения.
4. Значения не учитывают все факторы, влияющие на лабораторные показатели.
5. Вероятность модели не равна вероятности заболевания.

**Безопасное использование**
Используйте только для изучения программирования, обработки данных и оценки классификаторов. При вопросах о реальных анализах обращайтесь к квалифицированному медицинскому специалисту.
""")
    st.download_button("Скачать демонстрационный CSV", data=(ROOT/"demo_synthetic_blood_data.csv").read_bytes(), file_name="demo_synthetic_blood_data.csv", mime="text/csv")

st.divider()
st.caption("MediLens AI · Учебный прототип · Не медицинское изделие · Не использовать для диагностики")
