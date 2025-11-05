import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import font_manager as fm
from pathlib import Path

# =====================================================
# 0) PAGE CONFIG
# =====================================================
st.set_page_config(page_title="📊 Commission Dashboard", layout="wide")
st.title("📊 รายงานยอดขายและค่าคอมมิชชั่น")

# =====================================================
# 0B) THAI FONT (force + debug)
# =====================================================
def set_thai_font():
    """
    Find and register a Thai-capable TTF in ./fonts (any subfolder),
    prefer NotoSansThai-Regular.ttf, then fall back to other Thai families.
    Shows a caption with the font actually used.
    """
    here = Path(__file__).resolve().parent
    font_dir = here / "fonts"

    # Search patterns (recursive)
    patterns = [
        "**/NotoSansThai-Regular.ttf",  # preferred static regular
        "**/NotoSansThai-*.ttf",        # any Noto Sans Thai variant
        "**/Sarabun-Regular.ttf",
        "**/Sarabun-*.ttf",
        "**/THSarabunNew*.ttf",
    ]

    candidates = []
    if font_dir.exists():
        for pat in patterns:
            candidates += sorted(font_dir.glob(pat))

    picked_family, picked_path = None, None

    # Try files in ./fonts first
    for path in candidates:
        try:
            fm.fontManager.addfont(str(path))
            fam = fm.FontProperties(fname=str(path)).get_name()
            picked_family, picked_path = fam, path
            break
        except Exception:
            pass

    # Last-resort: try system font families by name
    if not picked_family:
        for fam in ["Noto Sans Thai", "Sarabun", "TH Sarabun New", "Tahoma", "Arial Unicode MS"]:
            try:
                _ = fm.findfont(fam, fallback_to_default=False)
                picked_family = fam
                break
            except Exception:
                pass

    if not picked_family:
        picked_family = "DejaVu Sans"  # not Thai-capable, but avoids crash

    # Apply globally
    matplotlib.rcParams["font.family"] = picked_family
    matplotlib.rcParams["font.sans-serif"] = [
        picked_family, "Sarabun", "TH Sarabun New", "Tahoma", "Arial Unicode MS", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["text.usetex"] = False
    plt.rcParams.update({"font.family": picked_family})

    msg = f"🆗 ใช้ฟอนต์สำหรับกราฟ: **{picked_family}**"
    if picked_path:
        msg += f" (ไฟล์: `{picked_path.name}`)"
    st.caption(msg)

    # Optional: quick visibility into what Streamlit sees in ./fonts
    with st.expander("🔧 Font debug (คลิกเพื่อเปิด)"):
        st.write("โฟลเดอร์ fonts:", str(font_dir))
        if font_dir.exists():
            st.write([str(p) for p in font_dir.rglob("*.ttf")][:40])
        else:
            st.write("ไม่พบโฟลเดอร์ fonts/")

set_thai_font()

# =====================================================
# 1) LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    sheet_name = "Comission_dashboard"
    sheet_id = "1GdOUIMfTOODsmBIo7Djf3RiG3kSGuVmq5xWbAvGHGcc"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df = pd.read_csv(url)

    # Numeric conversions
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Year"]  = pd.to_numeric(df["Year"],  errors="coerce")

    # Money column
    df["เป็นเงิน"] = df["เป็นเงิน"].astype(str).str.replace(",", "", regex=False).astype(float)

    # Ensure required columns exist
    for col in ["Nvat", "Paid", "Status", "Sales_CO_Combine"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown")

    df["Month"] = df["Month"].astype("Int64")
    df["MonthLabel"] = df["Month"].apply(lambda m: f"{int(m):02d}" if pd.notnull(m) else "NA")
    df = df.dropna(subset=["Year", "Month"])
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    return df

df = load_data()

# =====================================================
# 2) SIDEBAR FILTERS
# =====================================================
st.sidebar.header("ตัวกรองข้อมูล")

year_options = sorted(df["Year"].unique().tolist())
selected_year = st.sidebar.selectbox("เลือกปี (Year)", year_options, index=len(year_options) - 1)

sales_options = ["All"] + sorted(df["Sales_CO_Combine"].dropna().unique().tolist())
selected_sales = st.sidebar.selectbox("เลือก Sales_CO_Combine", sales_options)

# Filtered data for top section
filtered = df[df["Year"] == selected_year].copy()
if selected_sales != "All":
    filtered = filtered[filtered["Sales_CO_Combine"] == selected_sales].copy()

if filtered.empty:
    st.warning("ไม่มีข้อมูลตามเงื่อนไขนี้")
    st.stop()

# =====================================================
# 3) KPI SUMMARY
# =====================================================
total_sales_val = filtered["เป็นเงิน"].sum()
row_count = filtered.shape[0]
c1, c2 = st.columns(2)
c1.metric("ยอดขายรวม (บาท)", f"{total_sales_val:,.2f}")
c2.metric("จำนวนเอกสาร / รายการ", f"{row_count:,}")
st.markdown("---")

# =====================================================
# 4) HELPERS
# =====================================================
def prep_stacked(df_in: pd.DataFrame, stack_col: str) -> pd.DataFrame:
    tmp = df_in.groupby(["Month", "MonthLabel", stack_col], as_index=False)["เป็นเงิน"].sum()
    return tmp.pivot(index="MonthLabel", columns=stack_col, values="เป็นเงิน").fillna(0)

def sort_month_index(pivot_df: pd.DataFrame) -> pd.DataFrame:
    idx_as_int = [int(x) if str(x).isdigit() else 999 for x in pivot_df.index]
    order = np.argsort(idx_as_int)
    return pivot_df.iloc[order]

def plot_stacked(pivot_df: pd.DataFrame, title_main: str, legend_title: str):
    totals = pivot_df.sum(axis=1)
    palette = sns.color_palette("Set2", n_colors=len(pivot_df.columns))
    color_map = {cat: palette[i] for i, cat in enumerate(pivot_df.columns)}

    fig, ax = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(pivot_df.index))
    bottoms = np.zeros(len(pivot_df.index))

    for cat in pivot_df.columns:
        heights = pivot_df[cat].values
        ax.bar(x_positions, heights, bottom=bottoms, label=cat,
               color=color_map[cat], edgecolor="white", linewidth=0.4)
        for xi, h, btm, total in zip(x_positions, heights, bottoms, totals.values):
            if total > 0 and h > 0:
                ax.text(xi, btm + h / 2, f"{(h/total)*100:.1f}%",
                        ha="center", va="center", fontsize=7)
        bottoms += heights

    for xi, total in zip(x_positions, totals.values):
        ax.text(xi, total, f"{total:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index, fontsize=9)
    ax.set_xlabel("เดือน")
    ax.set_ylabel("ยอดขาย (บาท)")
    ax.set_title(title_main, fontweight="bold")
    ax.legend(title=legend_title, fontsize=8, title_fontsize=9, frameon=True)
    fig.tight_layout()
    return fig

# =====================================================
# 5) CHARTS: NVAT / PAID / STATUS
# =====================================================
sections = [
    ("1) ยอดขายรายเดือน แยกตาม NVAT",   "Nvat",   "ยอดขายรวมรายเดือน (Stacked by Nvat)",           "กลุ่มภาษี (Nvat)"),
    ("2) ยอดขายรายเดือน แยกตามการชำระเงิน", "Paid",   "ยอดขายรวมรายเดือน (ชำระแล้ว / ค้างชำระ)",        "สถานะการชำระเงิน"),
    ("3) ยอดขายรายเดือน แยกตามสถานะเอกสาร", "Status", "ยอดขายรวมรายเดือน (ตามสถานะเอกสารขาย)",         "สถานะเอกสาร"),
]

for title, col, chart_title, legend_title in sections:
    st.subheader(title)
    pvt = sort_month_index(prep_stacked(filtered, col))
    st.pyplot(plot_stacked(pvt, f"{chart_title} - {selected_year}", legend_title), use_container_width=True)
    st.markdown("---")

# =====================================================
# 6) MONTHLY YEAR-vs-YEAR (existing)
# =====================================================
st.subheader("📈 เปรียบเทียบยอดขายระหว่าง 2 ปี")
colA, colB = st.columns(2)
with colA:
    year_A = st.selectbox("เลือกปีที่ 1 (Year A)", year_options, index=max(0, len(year_options) - 2))
with colB:
    year_B = st.selectbox("เลือกปีที่ 2 (Year B)", year_options, index=len(year_options) - 1)

compare_df = df[df["Year"].isin([year_A, year_B])].copy()
if selected_sales != "All":
    compare_df = compare_df[compare_df["Sales_CO_Combine"] == selected_sales].copy()

compare_monthly = (
    compare_df.groupby(["Year", "Month"], as_index=False)["เป็นเงิน"].sum().sort_values(["Year", "Month"])
)
compare_monthly["MonthLabel"] = compare_monthly["Month"].apply(lambda x: f"{int(x):02d}")
pivot_compare = compare_monthly.pivot(index="MonthLabel", columns="Year", values="เป็นเงิน").fillna(0)

fig_compare, ax = plt.subplots(figsize=(8, 4))
width = 0.35
x = np.arange(len(pivot_compare.index))
years = pivot_compare.columns.tolist()
colors = sns.color_palette("Set2", n_colors=2)

for i, y in enumerate(years):
    vals = pivot_compare[y].values
    ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[i], label=str(y))
    for xi, v in zip(x, vals):
        ax.text(xi + (i - 0.5) * width, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(pivot_compare.index)
ax.set_xlabel("เดือน")
ax.set_ylabel("ยอดขาย (บาท)")
ax.set_title(f"ยอดขายรายเดือนเปรียบเทียบ {year_A} vs {year_B}", fontweight="bold")
ax.legend(title="ปี", fontsize=8, title_fontsize=9)
fig_compare.tight_layout()
st.pyplot(fig_compare, use_container_width=True)

# =====================================================
# 7) YEARLY TOTAL (existing)
# =====================================================
st.markdown("---")
st.subheader("📊 ยอดขายรวมรายปี")

yearly = df.groupby("Year", as_index=False)["เป็นเงิน"].sum().sort_values("Year")
if selected_sales != "All":
    yearly = (
        df[df["Sales_CO_Combine"] == selected_sales]
        .groupby("Year", as_index=False)["เป็นเงิน"].sum().sort_values("Year")
    )

fig_yearly, ax = plt.subplots(figsize=(6, 3))
sns.barplot(data=yearly, x="Year", y="เป็นเงิน", palette="Blues", ax=ax)
for i, row in yearly.iterrows():
    ax.text(i, row["เป็นเงิน"], f"{row['เป็นเงิน']:,.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xlabel("ปี")
ax.set_ylabel("ยอดขายรวม (บาท)")
ax.set_title("ยอดขายรวมรายปี (Yearly Performance)", fontweight="bold")
fig_yearly.tight_layout()
st.pyplot(fig_yearly, use_container_width=True)

# =====================================================
# 8) NEW: QUARTERLY COMPARISON BY Sales_CO_Combine
# =====================================================
st.markdown("---")
st.subheader("📈 เปรียบเทียบรายไตรมาสตาม Sales_CO_Combine (ในปีที่เลือก)")

sales_choices = sorted(df["Sales_CO_Combine"].dropna().unique().tolist())
cA, cB, cY = st.columns(3)
sales_A = cA.selectbox("เลือก Sales A", sales_choices, index=0, key="q_sales_A")
sales_B = cB.selectbox("เลือก Sales B", sales_choices, index=min(1, len(sales_choices) - 1), key="q_sales_B")
year_q = cY.selectbox("เลือกปี (ไตรมาส)", year_options, index=year_options.index(selected_year), key="q_year")

if sales_A == sales_B:
    st.info("กรุณาเลือก Sales A และ Sales B ให้ต่างกัน")
else:
    df_q = df[(df["Year"] == year_q) & (df["Sales_CO_Combine"].isin([sales_A, sales_B]))].copy()
    if df_q.empty:
        st.warning("ไม่มีข้อมูลสำหรับการเปรียบเทียบรายไตรมาส")
    else:
        df_q["Quarter"] = ((df_q["Month"] - 1) // 3 + 1).astype(int)
        df_q["QuarterLabel"] = df_q["Quarter"].apply(lambda q: f"Q{int(q)}")
        q_data = df_q.groupby(["QuarterLabel", "Sales_CO_Combine"], as_index=False)["เป็นเงิน"].sum()
        pivot_q = q_data.pivot(index="QuarterLabel", columns="Sales_CO_Combine", values="เป็นเงิน").fillna(0)
        pivot_q = pivot_q.reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0)

        fig_q, ax = plt.subplots(figsize=(8, 4))
        width = 0.35
        x = np.arange(len(pivot_q.index))
        groups = [sales_A, sales_B]
        colors = sns.color_palette("Set2", n_colors=2)
        for i, grp in enumerate(groups):
            vals = pivot_q[grp].values if grp in pivot_q.columns else np.zeros(len(pivot_q))
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=grp, color=colors[i])
            for xi, v in zip(x, vals):
                ax.text(xi + (i - 0.5) * width, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_q.index)
        ax.set_xlabel("ไตรมาส")
        ax.set_ylabel("ยอดขาย (บาท)")
        ax.set_title(f"ยอดขายรายไตรมาส: {sales_A} vs {sales_B} ({year_q})", fontweight="bold")
        ax.legend(title="Sales_CO_Combine")
        fig_q.tight_layout()
        st.pyplot(fig_q, use_container_width=True)

# =====================================================
# 9) NEW: YEARLY COMPARISON BY Sales_CO_Combine
# =====================================================
st.markdown("---")
st.subheader("📊 เปรียบเทียบยอดขายรายปีตาม Sales_CO_Combine (ทุกปี)")

sales_YA = st.selectbox("เลือก Sales A (รายปี)", sales_choices, index=0, key="y_sales_A")
sales_YB = st.selectbox("เลือก Sales B (รายปี)", sales_choices, index=min(1, len(sales_choices) - 1), key="y_sales_B")

if sales_YA == sales_YB:
    st.info("กรุณาเลือก Sales A และ Sales B ให้ต่างกัน")
else:
    df_y = df[df["Sales_CO_Combine"].isin([sales_YA, sales_YB])]
    y_data = df_y.groupby(["Year", "Sales_CO_Combine"], as_index=False)["เป็นเงิน"].sum()
    pivot_y = y_data.pivot(index="Year", columns="Sales_CO_Combine", values="เป็นเงิน").fillna(0)
    pivot_y = pivot_y.reindex(columns=[sales_YA, sales_YB]).fillna(0)

    fig_y, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    years_idx = np.arange(len(pivot_y.index))
    for i, s in enumerate([sales_YA, sales_YB]):
        vals = pivot_y[s].values if s in pivot_y.columns else np.zeros(len(pivot_y))
        ax.bar(years_idx + (i - 0.5) * width, vals, width=width, label=s)
        for xi, v in zip(years_idx, vals):
            ax.text(xi + (i - 0.5) * width, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(years_idx)
    ax.set_xticklabels(pivot_y.index.astype(int))
    ax.set_xlabel("ปี")
    ax.set_ylabel("ยอดขายรวม (บาท)")
    ax.set_title(f"ยอดขายรวมรายปี: {sales_YA} vs {sales_YB}", fontweight="bold")
    ax.legend(title="Sales_CO_Combine")
    fig_y.tight_layout()
    st.pyplot(fig_y, use_container_width=True)

# =====================================================
# 10) RAW TABLE PREVIEW
# =====================================================
st.markdown("---")
st.markdown("### ข้อมูลดิบหลังกรอง (ตัวอย่างแสดงผล)")

preview_cols = [
    "Year", "Month", "MonthLabel", "Sales_CO_Combine",
    "เป็นเงิน", "Nvat", "Paid", "Status", "ลูกค้า", "พนักงาน"
]
existing_cols = [c for c in preview_cols if c in filtered.columns]
st.dataframe(
    filtered[existing_cols].sort_values(["Month", "ลูกค้า"], na_position="last").reset_index(drop=True),
    use_container_width=True, height=400
)
