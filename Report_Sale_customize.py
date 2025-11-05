import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import font_manager as fm
from pathlib import Path
import os

# =====================================================
# 0. PAGE CONFIG & GLOBAL STYLE
# =====================================================
st.set_page_config(page_title="📊 Commission Dashboard", layout="wide")
st.title("📊 รายงานยอดขายและค่าคอมมิชชั่น")

# ---- Thai font setup (robust: searches recursively in ./fonts) ----
def set_thai_font():
    """
    Find and register a Thai-capable TTF in ./fonts (any subfolder),
    including VariableFont files from Google Fonts. Then set rcParams.
    """
    here = Path(__file__).resolve().parent
    font_dir = here / "fonts"

    # Gather candidates recursively (Noto Sans Thai & Sarabun families)
    # Works with paths like fonts/Noto_Sans_Thai/static/NotoSansThai-Regular.ttf
    patterns = [
        "**/NotoSansThai-*.ttf",
        "**/NotoSansThai*.ttf",
        "**/Sarabun-*.ttf",
        "**/Sarabun*.ttf",
        "**/THSarabunNew*.ttf",
    ]
    candidates = []
    if font_dir.exists():
        for pat in patterns:
            candidates.extend(sorted(font_dir.glob(pat)))

    picked_family = None
    picked_path = None

    for path in candidates:
        try:
            fm.fontManager.addfont(str(path))
            # Get the real family name from the file (don’t assume)
            family = fm.FontProperties(fname=str(path)).get_name()
            picked_family = family
            picked_path = path
            break
        except Exception as e:
            # Try next candidate
            continue

    # Final fallback by name if the system already has a Thai font installed
    if not picked_family:
        # Try by family name (may exist on the host)
        for fam in ["Noto Sans Thai", "Sarabun", "TH Sarabun New", "Tahoma", "Arial Unicode MS"]:
            try:
                # Check if Matplotlib can find a font with this family
                _ = fm.findfont(fam, fallback_to_default=False)
                picked_family = fam
                break
            except Exception:
                continue

    # If still nothing, just keep defaults (won’t render Thai correctly)
    if not picked_family:
        st.warning("ไม่พบฟอนต์ภาษาไทยในโฟลเดอร์ fonts/ และระบบไม่มีฟอนต์ที่รองรับไทย")
        picked_family = "DejaVu Sans"  # non-Thai fallback

    # Apply font settings
    matplotlib.rcParams["font.family"] = picked_family
    matplotlib.rcParams["font.sans-serif"] = [
        picked_family, "Sarabun", "TH Sarabun New", "Tahoma", "Arial Unicode MS", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # Tiny debug line so you can confirm which font is used
    msg = f"🆗 ใช้ฟอนต์สำหรับกราฟ: **{picked_family}**"
    if picked_path:
        msg += f"  (จากไฟล์: `{picked_path.name}`)"
    st.caption(msg)

set_thai_font()

# =====================================================
# 1. LOAD DATA FROM GOOGLE SHEET
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    sheet_name = 'Comission_dashboard'
    sheet_id = '1GdOUIMfTOODsmBIo7Djf3RiG3kSGuVmq5xWbAvGHGcc'
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    df = pd.read_csv(url)

    # Convert Month / Year to numeric
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Clean money column 'เป็นเงิน'
    df["เป็นเงิน"] = (
        df["เป็นเงิน"].astype(str).str.replace(",", "", regex=False).astype(float)
    )

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
# 2. SIDEBAR FILTERS
# =====================================================
st.sidebar.header("ตัวกรองข้อมูล")

year_options = sorted(df["Year"].unique().tolist())
selected_year = st.sidebar.selectbox(
    "เลือกปี (Year)",
    year_options,
    index=len(year_options)-1
)

sales_options = ["All"] + sorted(df["Sales_CO_Combine"].dropna().unique().tolist())
selected_sales = st.sidebar.selectbox(
    "เลือก Sales_CO_Combine",
    sales_options
)

# Filter data
filtered = df[df["Year"] == selected_year].copy()
if selected_sales != "All":
    filtered = filtered[filtered["Sales_CO_Combine"] == selected_sales].copy()

if filtered.empty:
    st.warning("ไม่มีข้อมูลตามเงื่อนไขนี้")
    st.stop()

# =====================================================
# 3. KPI SUMMARY
# =====================================================
total_sales_val = filtered["เป็นเงิน"].sum()
row_count = filtered.shape[0]

col1, col2 = st.columns(2)
with col1:
    st.metric("ยอดขายรวม (บาท)", f"{total_sales_val:,.2f}")
with col2:
    st.metric("จำนวนเอกสาร / รายการ", f"{row_count:,}")

st.markdown("---")

# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================
def prep_stacked(df_in: pd.DataFrame, stack_col: str):
    tmp = (
        df_in
        .groupby(["Month", "MonthLabel", stack_col], as_index=False)["เป็นเงิน"]
        .sum()
    )
    pivot_df = tmp.pivot(index="MonthLabel", columns=stack_col, values="เป็นเงิน").fillna(0)
    return pivot_df

def sort_month_index(pivot_df: pd.DataFrame):
    idx_as_int = []
    for x in pivot_df.index:
        try:
            idx_as_int.append(int(x))
        except:
            idx_as_int.append(999)
    order = sorted(range(len(idx_as_int)), key=lambda i: idx_as_int[i])
    return pivot_df.iloc[order]

def plot_stacked(pivot_df: pd.DataFrame, title_main: str, legend_title: str):
    totals = pivot_df.sum(axis=1)
    palette = sns.color_palette("Set2", n_colors=len(pivot_df.columns))
    color_map = {cat: palette[i] for i, cat in enumerate(pivot_df.columns)}

    fig, ax = plt.subplots(figsize=(8,4))
    x_positions = np.arange(len(pivot_df.index))
    bottoms = np.zeros(len(pivot_df.index))

    for cat in pivot_df.columns:
        heights = pivot_df[cat].values
        ax.bar(
            x_positions,
            heights,
            bottom=bottoms,
            label=cat,
            color=color_map[cat],
            edgecolor="white",
            linewidth=0.4
        )
        for xi, h, btm, total in zip(x_positions, heights, bottoms, totals.values):
            if total > 0 and h > 0:
                pct = (h / total) * 100
                ax.text(
                    xi, btm + h/2, f"{pct:.1f}%",
                    ha="center", va="center",
                    fontsize=6.5, fontweight="medium", color="black"
                )
        bottoms += heights

    for xi, total in zip(x_positions, totals.values):
        ax.text(xi, total, f"{total:,.0f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="black")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pivot_df.index, fontsize=8)
    ax.set_xlabel("เดือน", fontsize=8)
    ax.set_ylabel("ยอดขาย (บาท)", fontsize=8)
    ax.set_title(title_main, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis='y', labelsize=8)
    leg = ax.legend(title=legend_title, fontsize=7.5, title_fontsize=8, frameon=True)
    leg.get_frame().set_edgecolor("gray")
    leg.get_frame().set_linewidth(0.4)
    fig.tight_layout()
    return fig

# =====================================================
# 5. CHART 1 - Nvat
# =====================================================
st.subheader("1) ยอดขายรายเดือน แยกตาม NVAT")
nvat_pivot = prep_stacked(filtered, "Nvat")
nvat_pivot = sort_month_index(nvat_pivot)
fig1 = plot_stacked(nvat_pivot, f"ยอดขายรวมรายเดือน (Stacked by Nvat) - {selected_year}", "กลุ่มภาษี (Nvat)")
st.pyplot(fig1, use_container_width=True)
st.markdown("---")

# =====================================================
# 6. CHART 2 - Paid
# =====================================================
st.subheader("2) ยอดขายรายเดือน แยกตามสถานะการชำระเงิน")
paid_pivot = prep_stacked(filtered, "Paid")
paid_pivot = sort_month_index(paid_pivot)
fig2 = plot_stacked(paid_pivot, f"ยอดขายรวมรายเดือน (ชำระแล้ว / ค้างชำระ) - {selected_year}", "สถานะการชำระเงิน")
st.pyplot(fig2, use_container_width=True)
st.markdown("---")

# =====================================================
# 7. CHART 3 - Status
# =====================================================
st.subheader("3) ยอดขายรายเดือน แยกตามสถานะเอกสารขาย")
status_pivot = prep_stacked(filtered, "Status")
status_pivot = sort_month_index(status_pivot)
fig3 = plot_stacked(status_pivot, f"ยอดขายรวมรายเดือน (ตามสถานะเอกสารขาย) - {selected_year}", "สถานะเอกสาร")
st.pyplot(fig3, use_container_width=True)

# =====================================================
# 7B. COMPARISON CHARTS (Monthly & Yearly Performance)
# =====================================================
st.markdown("---")
st.subheader("📈 เปรียบเทียบยอดขายระหว่าง 2 ปี")

colA, colB = st.columns(2)
with colA:
    compare_year_1 = st.selectbox("เลือกปีที่ 1 (Year A)", year_options, index=max(0, len(year_options)-2))
with colB:
    compare_year_2 = st.selectbox("เลือกปีที่ 2 (Year B)", year_options, index=len(year_options)-1)

compare_df = df[df["Year"].isin([compare_year_1, compare_year_2])].copy()
if selected_sales != "All":
    compare_df = compare_df[compare_df["Sales_CO_Combine"] == selected_sales].copy()

compare_monthly = (
    compare_df.groupby(["Year", "Month"], as_index=False)["เป็นเงิน"]
    .sum()
    .sort_values(["Year", "Month"])
)
compare_monthly["MonthLabel"] = compare_monthly["Month"].apply(lambda x: f"{int(x):02d}")
pivot_compare = compare_monthly.pivot(index="MonthLabel", columns="Year", values="เป็นเงิน").fillna(0)

fig_compare, ax = plt.subplots(figsize=(8,4))
width = 0.35
x = np.arange(len(pivot_compare.index))
years = pivot_compare.columns.tolist()
colors = sns.color_palette("Set2", n_colors=2)

for i, year in enumerate(years):
    ax.bar(x + (i - 0.5) * width, pivot_compare[year], width=width, label=str(year), color=colors[i])
    for xi, val in zip(x, pivot_compare[year]):
        ax.text(xi + (i - 0.5) * width, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(pivot_compare.index, fontsize=8)
ax.set_xlabel("เดือน", fontsize=9)
ax.set_ylabel("ยอดขาย (บาท)", fontsize=9)
ax.set_title(f"ยอดขายรายเดือนเปรียบเทียบ {compare_year_1} vs {compare_year_2}", fontsize=11, fontweight="bold")
ax.legend(title="ปี", fontsize=8, title_fontsize=9)
ax.tick_params(axis='y', labelsize=8)
fig_compare.tight_layout()
st.pyplot(fig_compare, use_container_width=True)

# =====================================================
# 7C. YEARLY TOTAL PERFORMANCE
# =====================================================
st.markdown("---")
st.subheader("📊 ยอดขายรวมรายปี")

yearly = (
    df.groupby("Year", as_index=False)["เป็นเงิน"]
    .sum()
    .sort_values("Year")
)
if selected_sales != "All":
    yearly = (
        df[df["Sales_CO_Combine"] == selected_sales]
        .groupby("Year", as_index=False)["เป็นเงิน"]
        .sum()
        .sort_values("Year")
    )

fig_yearly, ax = plt.subplots(figsize=(6,3))
sns.barplot(data=yearly, x="Year", y="เป็นเงิน", palette="Blues", ax=ax)
for i, row in yearly.iterrows():
    ax.text(i, row["เป็นเงิน"], f"{row['เป็นเงิน']:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("ปี", fontsize=9)
ax.set_ylabel("ยอดขายรวม (บาท)", fontsize=9)
ax.set_title("ยอดขายรวมรายปี (Yearly Performance)", fontsize=11, fontweight="bold")
ax.tick_params(axis='y', labelsize=8)
fig_yearly.tight_layout()
st.pyplot(fig_yearly, use_container_width=True)

# =====================================================
# 7D. QUARTERLY COMPARISON BY Sales_CO_Combine (within selected year)
# =====================================================
st.markdown("---")
st.subheader("📈 เปรียบเทียบรายไตรมาสตาม Sales_CO_Combine (ในปีที่เลือก)")

sales_choices_only = sorted([s for s in df["Sales_CO_Combine"].dropna().unique().tolist() if s != "All"])

colSA, colSB, colSY = st.columns([1,1,1])
with colSA:
    sales_A = st.selectbox("เลือก Sales A", sales_choices_only, index=0, key="q_sales_A")
with colSB:
    default_idx = 1 if len(sales_choices_only) > 1 else 0
    sales_B = st.selectbox("เลือก Sales B", sales_choices_only, index=default_idx, key="q_sales_B")
with colSY:
    year_for_quarter = st.selectbox("เลือกปี (สำหรับรายไตรมาส)", year_options, index=year_options.index(selected_year), key="q_year")

if sales_A == sales_B:
    st.info("กรุณาเลือก Sales A และ Sales B ให้ต่างกัน")
else:
    df_q = df[(df["Year"] == year_for_quarter) & (df["Sales_CO_Combine"].isin([sales_A, sales_B]))].copy()
    if df_q.empty:
        st.warning("ไม่มีข้อมูลสำหรับการเปรียบเทียบรายไตรมาสตามเงื่อนไขนี้")
    else:
        df_q["Quarter"] = ((df_q["Month"] - 1) // 3 + 1).astype(int)
        df_q["QuarterLabel"] = df_q["Quarter"].apply(lambda q: f"Q{int(q)}")

        cmp_q = (
            df_q.groupby(["QuarterLabel", "Sales_CO_Combine"], as_index=False)["เป็นเงิน"]
                .sum()
                .sort_values(["QuarterLabel", "Sales_CO_Combine"])
        )
        pivot_q = cmp_q.pivot(index="QuarterLabel", columns="Sales_CO_Combine", values="เป็นเงิน").fillna(0)
        pivot_q = pivot_q.reindex(["Q1","Q2","Q3","Q4"]).fillna(0)

        fig_q, ax = plt.subplots(figsize=(8,4))
        width = 0.35
        x = np.arange(len(pivot_q.index))
        groups = [sales_A, sales_B]
        colors = sns.color_palette("Set2", n_colors=2)

        for i, grp in enumerate(groups):
            vals = pivot_q[grp].values if grp in pivot_q.columns else np.zeros(len(pivot_q.index))
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=str(grp), color=colors[i])
            for xi, val in zip(x, vals):
                ax.text(xi + (i - 0.5) * width, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot_q.index, fontsize=8)
        ax.set_xlabel("ไตรมาส", fontsize=9)
        ax.set_ylabel("ยอดขาย (บาท)", fontsize=9)
        ax.set_title(f"ยอดขายรายไตรมาส: {sales_A} vs {sales_B} ในปี {year_for_quarter}", fontsize=11, fontweight="bold")
        ax.legend(title="Sales_CO_Combine", fontsize=8, title_fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        fig_q.tight_layout()
        st.pyplot(fig_q, use_container_width=True)

# =====================================================
# 7E. YEARLY COMPARISON BY Sales_CO_Combine (across years)
# =====================================================
st.markdown("---")
st.subheader("📊 เปรียบเทียบยอดขายรายปีตาม Sales_CO_Combine (ทุกปี)")

colYA, colYB = st.columns(2)
with colYA:
    y_sales_A = st.selectbox("เลือก Sales A (รายปี)", sales_choices_only, index=0, key="y_sales_A")
with colYB:
    default_idx_y = 1 if len(sales_choices_only) > 1 else 0
    y_sales_B = st.selectbox("เลือก Sales B (รายปี)", sales_choices_only, index=default_idx_y, key="y_sales_B")

if y_sales_A == y_sales_B:
    st.info("กรุณาเลือก Sales A และ Sales B ให้ต่างกัน")
else:
    df_y = df[df["Sales_CO_Combine"].isin([y_sales_A, y_sales_B])].copy()
    if df_y.empty:
        st.warning("ไม่มีข้อมูลสำหรับการเปรียบเทียบรายปีตามเงื่อนไขนี้")
    else:
        yearly_sales_cmp = (
            df_y.groupby(["Year", "Sales_CO_Combine"], as_index=False)["เป็นเงิน"]
                .sum()
                .sort_values(["Year", "Sales_CO_Combine"])
        )
        pivot_y = yearly_sales_cmp.pivot(index="Year", columns="Sales_CO_Combine", values="เป็นเงิน").fillna(0)
        pivot_y = pivot_y.reindex(columns=[y_sales_A, y_sales_B]).fillna(0)

        fig_ycmp, ax = plt.subplots(figsize=(8,4))
        width = 0.35
        years_idx = np.arange(len(pivot_y.index))
        groups = [y_sales_A, y_sales_B]
        colors = sns.color_palette("Set2", n_colors=2)

        for i, grp in enumerate(groups):
            vals = pivot_y[grp].values if grp in pivot_y.columns else np.zeros(len(pivot_y.index))
            ax.bar(years_idx + (i - 0.5) * width, vals, width=width, label=str(grp), color=colors[i])
            for xi, val in zip(years_idx, vals):
                ax.text(xi + (i - 0.5) * width, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(years_idx)
        ax.set_xticklabels(pivot_y.index.astype(int), fontsize=8)
        ax.set_xlabel("ปี", fontsize=9)
        ax.set_ylabel("ยอดขายรวม (บาท)", fontsize=9)
        ax.set_title(f"ยอดขายรวมรายปี: {y_sales_A} vs {y_sales_B}", fontsize=11, fontweight="bold")
        ax.legend(title="Sales_CO_Combine", fontsize=8, title_fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        fig_ycmp.tight_layout()
        st.pyplot(fig_ycmp, use_container_width=True)

# =====================================================
# 8. RAW TABLE PREVIEW
# =====================================================
st.markdown("---")
st.markdown("### ข้อมูลดิบหลังกรอง (ตัวอย่างแสดงผล)")

preview_cols = [
    "Year", "Month", "MonthLabel", "Sales_CO_Combine",
    "เป็นเงิน", "Nvat", "Paid", "Status", "ลูกค้า", "พนักงาน"
]
existing_cols = [c for c in preview_cols if c in filtered.columns]

st.dataframe(
    filtered[existing_cols]
    .sort_values(["Month", "ลูกค้า"], na_position="last")
    .reset_index(drop=True),
    use_container_width=True,
    height=400
)
