import streamlit as st
import pandas as pd
import json

# === PAGE TITLE ===
st.set_page_config(page_title="Excel to JSON Converter", layout="wide")
st.title("Excel to JSON Converter (Policies)")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # === READ EXCEL ===
    df = pd.read_excel(uploaded_file)

    # === SCHEMA EXTRACTION ===
    SCHEMA = df.columns.tolist()
    st.subheader("Extracted Columns")
    st.write(SCHEMA)

    # === CONVERT TO JSON RECORDS ===
    policies = df.to_dict(orient="records")

    # === ADD POLICY_ID IF MISSING ===
    for idx, policy in enumerate(policies, start=1):
        if "Policy_ID" in SCHEMA:
            if pd.isna(policy.get("Policy_ID")) or not policy.get("Policy_ID"):
                policy["Policy_ID"] = f"POL{idx:03d}"
        else:
            policy["Policy_ID"] = f"POL{idx:03d}"

    # === FINAL JSON STRUCTURE ===
    output = {"policies": policies}

    # === TABLE PREVIEW OPTIONS ===
    st.subheader("Table Preview")
    if st.checkbox("Show full table"):
        st.dataframe(df)
    else:
        row_limit = st.slider("Select number of rows to preview", min_value=1, max_value=len(df), value=5)
        st.dataframe(df.head(row_limit))

    # === JSON PREVIEW ===
    st.subheader("JSON Preview (All Rows)")
    st.json({"policies": policies[:]})

    # === DOWNLOAD BUTTON ===
    json_str = json.dumps(output, indent=4, ensure_ascii=False, default=str)
    st.download_button(
        label="Download Full JSON",
        data=json_str,
        file_name="policies.json",
        mime="application/json"
    )
else:
    st.info("Please upload an Excel file to begin conversion.")