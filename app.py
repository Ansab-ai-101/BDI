import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import json

# --- Load GROQ API key from secrets ---
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# --- Load CSV from GitHub (use your raw URL) ---
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/Ansab-ai-101/BDI/main/DEI_dataset_llama_clean.csv"
    return pd.read_csv(csv_url)

df = load_data()

st.set_page_config(page_title="BDI Racial Inclusivity Dashboard", layout="wide")
st.title("📊 BDI Racial Inclusivity Score Dashboard")

company_name = st.text_input("🔍 Enter a company name to analyze:", "Nike")

def get_llama_analysis(company_data):
    input_json = company_data.to_dict()
    prompt = f"""
You are a DEI expert. Based on the following data for {input_json.get("Company Name")}, analyze its racial inclusivity with focus on Black representation, leadership, pay equity, and supplier diversity.

Data:
{json.dumps(input_json, indent=2)}

Output a JSON like:
{{
  "Inclusivity Score": 87,
  "Summary": "Nike has strong Black representation at the board level but lacks transparency in pay equity.",
  "Recommendations": [
    "Improve racial pay equity reporting.",
    "Increase Black executives in leadership roles.",
    "Strengthen supplier diversity programs."
  ]
}}
Return ONLY the JSON object. No explanation.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_completion_tokens=512,
        top_p=1,
        stream=False
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return None

if company_name:
    matches = df[df["Company Name"].str.lower() == company_name.lower()]
    if matches.empty:
        st.error("❌ Company not found in the dataset.")
    else:
        company_data = matches.iloc[0]
        st.subheader(f"📄 DEI Data for {company_name}")
        st.dataframe(pd.DataFrame(company_data).rename(columns={0: 'Value'}))

        with st.spinner("🤖 Analyzing with LLaMA..."):
            analysis = get_llama_analysis(company_data)

        if analysis:
            st.metric("Inclusivity Score", f"{analysis['Inclusivity Score']} / 100")
            st.subheader("📌 Summary")
            st.success(analysis["Summary"])

            st.subheader("🛠️ Recommendations")
            for rec in analysis["Recommendations"]:
                st.write(f"✅ {rec}")

            st.subheader("📈 DEI Radar Chart (Binary Indicators)")
            indicators = {
                "Black CEO": company_data["Black CEO"],
                "Supplier Diversity Program": company_data["Supplier Diversity Program"],
                "Racial Pay Equity": company_data["Racial Pay Equity"],
                "Black Employee Resource Group": company_data["Black Employee Resource Group"],
                "Executive Compensation Tied to DEI Metrics": company_data["Executive Compensation Tied to DEI Metrics"],
                "Skills-First Hiring (OneTen Coalition)": company_data["Skills-First Hiring (OneTen Coalition)"]
            }

            labels = list(indicators.keys())
            values = [1 if str(v).strip().lower() == "yes" else 0 for v in indicators.values()]
            values += values[:1]
            labels += labels[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.plot(range(len(labels)), values, marker='o')
            ax.fill(range(len(labels)), values, alpha=0.25)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, wrap=True)
            ax.set_yticklabels([])
            st.pyplot(fig)
        else:
            st.error("⚠️ Failed to parse LLaMA analysis.")
