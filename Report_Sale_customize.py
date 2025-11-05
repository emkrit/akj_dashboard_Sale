import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 0. CONFIG
# =========================
SHEET_NAME = 'PKPTRACT_Auto'
SHEET_ID = '1qB6P35sNZTd8uGt6vWXuNVH_H58fNd9_lcTlaHWhVnQ'
URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

TOP_N_MENUS = 5  # change this if you want top 6 etc.

# =========================
# 1. LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(URL)

    # Make sure Month exists
    if "Month" not in df.columns:
        df["_parsed_date"] = pd.to_datetime(df["วันที่"], errors="coerce", dayfirst=True)
        df["Month"] = df["_parsed_date"].dt.month
        df.drop(columns=["_parsed_date"], inplace=True)

    # Clean numeric cols
    df["เพิ่ม"] = pd.to_numeric(df["เพิ่ม"], errors="coerce").fillna(0)
    df["แก้ไข"] = pd.to_numeric(df["แก้ไข"], errors="coerce").fillna(0)

    # Keep all menus now (no whitelist filter)
    # Ensure they are strings (sometimes numbers come in)
    df["รหัสเมนู"] = df["รหัสเมนู"].astype(str)

    return df

df = load_data()

# =========================
# 2. MENU LABELS (CODE + DESCRIPTION)
# =========================
# Build label dict for ALL menus present in df
code_to_label = {}
for code in df["รหัสเมนู"].dropna().unique().tolist():
    desc_series = df.loc[df["รหัสเมนู"] == code, "ชื่อเมนู"].dropna()
    if len(desc_series) > 0:
        desc_text = str(desc_series.iloc[0]).strip()
        if desc_text:
            code_to_label[code] = f"{code} ({desc_text})"
        else:
            code_to_label[code] = code
    else:
        code_to_label[code] = code

# =========================
# 3. STREAMLIT FILTERS (ONLY USER NOW)
# =========================
st.set_page_config(page_title="User Activity Dashboard", layout="wide")
st.title("User Activity Dashboard 📊")

st.sidebar.header("Filters")

all_names = ["All"] + sorted(df["Name lookup"].dropna().unique().tolist())
selected_name = st.sidebar.selectbox("Select User", all_names)

# =========================
# 4. FILTER DATA FOR THAT USER
# =========================
if selected_name == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Name lookup"] == selected_name].copy()

if filtered_df.empty:
    st.warning("No data available for this selection.")
    st.stop()

# =========================
# 5. FIND TOP MENUS FOR THIS USER (DYNAMIC NOW)
# =========================
# We compute top menus by total additions in the filtered data
menu_usage = (
    filtered_df.groupby("รหัสเมนู", as_index=False)["เพิ่ม"]
    .sum()
    .rename(columns={"เพิ่ม": "TotalAdditions"})
    .sort_values("TotalAdditions", ascending=False)
)

top_menus = (
    menu_usage["รหัสเมนู"]
    .head(TOP_N_MENUS)
    .tolist()
)

# safety: keep only menus that exist in filtered_df just in case
top_menus = [m for m in top_menus if m in filtered_df["รหัสเมนู"].unique()]

if not top_menus:
    st.warning("No activity found for this user.")
    st.stop()

# =========================
# 6. KPI SUMMARY (for THIS user's filtered_df)
# =========================
total_add = int(filtered_df["เพิ่ม"].sum())
total_edit = int(filtered_df["แก้ไข"].sum())

col_kpi1, col_kpi2 = st.columns(2)
with col_kpi1:
    st.metric("Total Additions", f"{total_add}")
with col_kpi2:
    st.metric("Total Edits", f"{total_edit}")

# =========================
# 7. PER-MENU GRAPHS USING SEABORN (COMPACT)
# =========================
st.markdown("---")
if selected_name == "All":
    st.subheader(f"Top {TOP_N_MENUS} Menus (All Users)")
else:
    st.subheader(f"Top {TOP_N_MENUS} Menus for {selected_name}")

sns.set_style("whitegrid")

FIG_SIZE = (3, 2)
TITLE_SIZE = 10
LABEL_SIZE = 8
VALUE_SIZE = 7

for code in top_menus:
    menu_label = code_to_label.get(code, code)
    df_menu = filtered_df[filtered_df["รหัสเมนู"] == code].copy()

    # Aggregate per month for this specific menu and this user (or all users if "All")
    monthly_add = (
        df_menu.groupby("Month", as_index=False)["เพิ่ม"]
        .sum()
        .rename(columns={"เพิ่ม": "Additions"})
    )
    monthly_edit = (
        df_menu.groupby("Month", as_index=False)["แก้ไข"]
        .sum()
        .rename(columns={"แก้ไข": "Edits"})
    )

    st.markdown(f"### {menu_label}")

    col_a, col_b = st.columns(2)

    # Additions chart
    with col_a:
        st.markdown("**Monthly Additions**")
        fig_a, ax_a = plt.subplots(figsize=FIG_SIZE)
        sns.barplot(
            data=monthly_add,
            x="Month",
            y="Additions",
            hue="Month",
            dodge=False,
            legend=False,
            ax=ax_a
        )
        ax_a.set_title(f"Additions - {code}", fontsize=TITLE_SIZE, fontweight="bold")
        ax_a.set_xlabel("Month", fontsize=LABEL_SIZE)
        ax_a.set_ylabel("Additions", fontsize=LABEL_SIZE)
        for i, row in monthly_add.iterrows():
            ax_a.text(
                i,
                row["Additions"] + 0.05,
                f"{row['Additions']:.0f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=False)

    # Edits chart
    with col_b:
        st.markdown("**Monthly Edits**")
        fig_b, ax_b = plt.subplots(figsize=FIG_SIZE)
        sns.barplot(
            data=monthly_edit,
            x="Month",
            y="Edits",
            hue="Month",
            dodge=False,
            legend=False,
            ax=ax_b
        )
        ax_b.set_title(f"Edits - {code}", fontsize=TITLE_SIZE, fontweight="bold")
        ax_b.set_xlabel("Month", fontsize=LABEL_SIZE)
        ax_b.set_ylabel("Edits", fontsize=LABEL_SIZE)
        for i, row in monthly_edit.iterrows():
            ax_b.text(
                i,
                row["Edits"] + 0.05,
                f"{row['Edits']:.0f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold"
            )
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=False)

    st.markdown("---")

# =========================
# 8. FOOTER / CONTEXT
# =========================
pretty_menu_labels = [code_to_label.get(code, code) for code in top_menus]

st.markdown(
    f"""
**Current view:**  
• User = `{selected_name}`  
• Top {TOP_N_MENUS} Menus = `{', '.join(pretty_menu_labels)}`  
(Based on highest total 'เพิ่ม' in this filtered view)
"""
)
