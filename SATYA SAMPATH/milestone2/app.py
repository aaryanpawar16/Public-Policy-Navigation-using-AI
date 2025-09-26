import streamlit as st
import pdfplumber
import pandas as pd
import json
uploaded_file = st.file_uploader("Upload a Dataset PDF", type=["pdf"])
if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        table = pdf.pages[0].extract_table()
        if table:
            df = pd.DataFrame(table[1:], columns=table[0])
            df = df.dropna(axis=1, how="all")
            df = df.loc[:, df.columns.notna()]
            df = df.loc[:, df.columns.str.strip() != ""]
            df = df.loc[:, ~df.columns.duplicated()]
            st.write(df)
            json_string = json.dumps(df.to_dict(orient="records"), indent=4)
            st.code(json_string, language="json")
            with open("output.json", "w", encoding="utf-8") as f:
                f.write(json_string)
