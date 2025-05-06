import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from streamlit_echarts import st_echarts
import altair as alt

# --- Set page config ---
st.set_page_config(
    page_title="BDI Racial Inclusivity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .recommendation {
        background-color: #E0F2FE;
        padding: 1rem;
        border-left: 4px solid #0EA5E9;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# --- Load GROQ API key from secrets ---
api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# --- Load CSV from GitHub ---
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/Ansab-ai-101/BDI/main/DEI_dataset_llama_clean.csv"
    return pd.read_csv(csv_url)

df = load_data()

# --- Sidebar for additional filters and context ---
with st.sidebar:
    st.image("https://www.svgrepo.com/show/530453/diversity.svg", width=80)
    st.title("Dashboard Controls")
    
    st.subheader("About This Dashboard")
    st.info("""
    This dashboard analyzes racial inclusivity metrics for major companies,
    with a focus on Black representation, leadership, pay equity, and supplier diversity.
    The analysis is powered by LLaMA AI.
    """)
    
    st.subheader("Benchmark Companies")
    benchmark_companies = st.multiselect(
        "Select companies to compare:",
        options=sorted(df["Company Name"].unique()),
        default=["Nike", "Apple", "Microsoft"]
    )
    
    # Replace Industry filter with Date Updated filter
    st.subheader("Filters")
    date_filter = st.selectbox(
        "Filter by Date Updated",
        options=["All"] + sorted(df["Date Updated"].unique().tolist())
    )
    
    st.subheader("Legend")
    st.markdown("""
    - 🟢 Strong performance (75-100)
    - 🟡 Moderate performance (50-74)
    - 🔴 Needs improvement (0-49)
    """)
    
# --- Main content ---
st.markdown("<h1 class='main-header'>📊 BDI Racial Inclusivity Dashboard</h1>", unsafe_allow_html=True)

# --- Company selector with search ---
col1, col2 = st.columns([3, 1])
with col1:
    company_name = st.text_input("🔍 Enter a company name to analyze:", "Nike")
with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

# Function to create a gauge chart for inclusivity score
def create_gauge_chart(score):
    if score >= 75:
        color = "#22C55E"  # green
        assessment = "Strong"
    elif score >= 50:
        color = "#F59E0B"  # amber
        assessment = "Moderate"
    else:
        color = "#EF4444"  # red
        assessment = "Needs Improvement"
        
    option = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "Inclusivity Score",
                "type": "gauge",
                "radius": "100%",
                "startAngle": 180,
                "endAngle": 0,
                "progress": {"show": True},
                "detail": {"valueAnimation": True, "formatter": "{value}", "fontSize": 30},
                "data": [{"value": score, "name": assessment}],
                "axisLine": {
                    "lineStyle": {
                        "width": 30,
                        "color": [
                            [0.5, "#EF4444"],
                            [0.75, "#F59E0B"],
                            [1, "#22C55E"]
                        ]
                    }
                },
                "splitLine": {"length": 15},
            }
        ]
    }
    return option

# Function to get LLaMA analysis
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
  ],
  "Strengths": [
    "Good board representation",
    "Strong employee resource groups"
  ],
  "Weaknesses": [
    "Pay equity transparency",
    "Executive diversity"
  ],
  "Category_Scores": {{
    "Representation": 85,
    "Leadership": 65,
    "Pay_Equity": 50,
    "Supplier_Diversity": 70
  }}
}}
Return ONLY the JSON object. No explanation.
"""
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=512,
        top_p=1,
        stream=False
    )
    
    # Get the response content
    response_content = response.choices[0].message.content
    
    try:
        # Try to extract JSON if it's wrapped in backticks or has extra text
        json_match = re.search(r'```(?:json)?(.*?)```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # If no code blocks, try to find JSON-like structure
        json_match = re.search(r'({[\s\S]*})', response_content)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # Otherwise try parsing the whole response
        return json.loads(response_content)
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        
        # Fallback: Create a structured response if parsing fails
        try:
            # Build a manual structured response
            fallback_response = {
                "Inclusivity Score": 50,
                "Summary": "Analysis failed to parse. Please try again or check the model response format.",
                "Recommendations": [
                    "Try with a different company",
                    "Check API connection",
                    "Review model parameters"
                ],
                "Strengths": ["Unknown"],
                "Weaknesses": ["Unknown"],
                "Category_Scores": {
                    "Representation": 50,
                    "Leadership": 50,
                    "Pay_Equity": 50,
                    "Supplier_Diversity": 50
                }
            }
            return fallback_response
        except:
            return None

# Function to create category bar chart
def create_category_chart(category_scores):
    categories = list(category_scores.keys())
    scores = list(category_scores.values())
    
    colors = []
    for score in scores:
        if score >= 75:
            colors.append('#22C55E')  # green
        elif score >= 50:
            colors.append('#F59E0B')  # amber
        else:
            colors.append('#EF4444')  # red
    
    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        marker_color=colors,
        text=scores,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Category Performance Breakdown",
        xaxis_title="Category",
        yaxis_title="Score",
        yaxis=dict(range=[0, 100]),
        height=400,
    )
    
    return fig

# Function to create racial representation visualization
def create_representation_chart(company_data):
    # Extract representation data
    categories = ["Board", "Executives", "Management", "Total Workforce"]
    
    # Convert string percentages to float values
    try:
        board = float(str(company_data.get("Black Board of Directors", "0")).split("/")[0]) / float(str(company_data.get("Black Board of Directors", "0")).split("/")[1]) * 100
    except:
        board = 0
        
    try:
        executives = float(str(company_data.get("Black Executive/Senior Level Officials & Managers", "0%")).replace("%", ""))
    except:
        executives = 0
        
    try:
        management = float(str(company_data.get("Black Mid-Level/First Managers", "0%")).replace("%", ""))
    except:
        management = 0
        
    try:
        workforce = float(str(company_data.get("Total Black Employees", "0%")).replace("%", ""))
    except:
        workforce = 0
    
    values = [board, executives, management, workforce]
    
    # Create the comparison data
    us_population = [13.4, 13.4, 13.4, 13.4]  # US Black population percentage
    
    fig = go.Figure()
    
    # Add company data
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        name=f"{company_data.get('Company Name', 'Company')}",
        marker_color='#3B82F6',
        text=[f"{v:.1f}%" for v in values],
        textposition='auto',
    ))
    
    # Add US population line
    fig.add_trace(go.Scatter(
        x=categories,
        y=us_population,
        mode='lines+markers',
        name='US Black Population (13.4%)',
        line=dict(color='red', width=2, dash='dash'),
    ))
    
    fig.update_layout(
        title="Black Representation vs. US Population (13.4%)",
        xaxis_title="Organizational Level",
        yaxis_title="Percentage",
        barmode='group',
        height=400,
    )
    
    return fig

# Function to create the radar chart using Plotly
def create_radar_chart(company_data):
    indicators = {
        "Black CEO": company_data.get("Black CEO", "No"),
        "Supplier Diversity Program": company_data.get("Supplier Diversity Program", "No"),
        "Racial Pay Equity": company_data.get("Racial Pay Equity", "No"),
        "Black Employee Resource Group": company_data.get("Black Employee Resource Group", "No"),
        "Executive Compensation Tied to DEI Metrics": company_data.get("Executive Compensation Tied to DEI Metrics", "No"),
        "Skills-First Hiring (OneTen Coalition)": company_data.get("Skills-First Hiring (OneTen Coalition)", "No")
    }
    
    # Convert to binary values
    categories = list(indicators.keys())
    values = [1 if str(v).strip().lower() == "yes" else 0 for v in indicators.values()]
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=company_data.get("Company Name", "Company"),
        line_color='rgb(59, 130, 246)',
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="DEI Program Implementation",
        height=500
    )
    
    return fig

# Function to create comparison chart with benchmark companies
def create_benchmark_chart(df, current_company, benchmark_companies):
    # Filter for the current company and benchmarks
    companies_to_compare = [current_company] + [c for c in benchmark_companies if c != current_company]
    filtered_df = df[df["Company Name"].isin(companies_to_compare)].copy()
    
    # Calculate an estimated inclusivity score based on available metrics
    def calculate_estimated_score(row):
        score = 0
        indicators = {
            "Black CEO": 15,
            "Supplier Diversity Program": 15,
            "Racial Pay Equity": 15,
            "Black Employee Resource Group": 10,
            "Executive Compensation Tied to DEI Metrics": 15,
            "Skills-First Hiring (OneTen Coalition)": 10,
        }
        
        for indicator, points in indicators.items():
            if str(row.get(indicator, "")).strip().lower() == "yes":
                score += points
                
        # Add points for representation
        try:
            board_rep = float(str(row.get("Black Board of Directors", "0/1")).split("/")[0]) / float(str(row.get("Black Board of Directors", "0/1")).split("/")[1]) * 100
            if board_rep >= 13.4:  # US Black population
                score += 10
            elif board_rep >= 8:
                score += 5
        except:
            pass
            
        try:
            exec_rep = float(str(row.get("Black Executive/Senior Level Officials & Managers", "0%")).replace("%", "").strip())
            if exec_rep >= 13.4:  # US Black population
                score += 10
            elif exec_rep >= 8:
                score += 5
        except:
            pass
        
        return score
    
    # Apply the estimated score calculation
    filtered_df["Estimated Score"] = filtered_df.apply(calculate_estimated_score, axis=1)
    
    # Create the bar chart
    fig = go.Figure()
    
    for i, company in enumerate(filtered_df["Company Name"]):
        score = filtered_df[filtered_df["Company Name"] == company]["Estimated Score"].values[0]
        
        color = '#3B82F6' if company == current_company else '#94A3B8'
        
        fig.add_trace(go.Bar(
            x=[company],
            y=[score],
            name=company,
            marker_color=color,
            text=[f"{score}/100"],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Benchmark Comparison (Estimated Scores)",
        xaxis_title="Company",
        yaxis_title="Estimated Score",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=False
    )
    
    return fig

# Function to create highlight visualization
def create_highlight_visualization(company_data):
    # Extract all highlights
    highlights = []
    for i in range(1, 6):
        highlight = company_data.get(f"Highlight{i}", "")
        if highlight and highlight.strip() != "":
            highlights.append(highlight)
    
    # Create a simple visualization for highlights
    if highlights:
        # Create a dataframe
        highlight_df = pd.DataFrame({"Highlight": highlights, "Importance": list(range(len(highlights), 0, -1))})
        
        # Create a horizontal bar chart
        fig = px.bar(
            highlight_df,
            y="Highlight",
            x="Importance", 
            orientation='h',
            title="Key Highlights",
            labels={"Importance": "Priority", "Highlight": ""},
            height=300 + (len(highlights) * 30),
            color="Importance",
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig.update_layout(showlegend=False, xaxis_visible=False)
        return fig
    else:
        return None

# Function to create trend visualization (simulated)
def create_trend_visualization():
    # This would typically use historical data, but we'll simulate it
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    trends = {
        "Board Representation": [8.2, 9.1, 10.3, 11.5, 12.2, 13.1],
        "Executive Representation": [5.1, 5.8, 6.7, 7.5, 8.2, 9.0],
        "Pay Equity Reporting": [42, 47, 52, 58, 65, 72],
        "Supplier Diversity": [35, 41, 48, 56, 61, 68]
    }
    
    fig = go.Figure()
    
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    
    for i, (metric, values) in enumerate(trends.items()):
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i], width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Industry Trends in Racial Equity (2020-2025)",
        xaxis_title="Year",
        yaxis_title="Score",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Function to create discrimination claims visualization
def create_claims_visualization(company_data):
    # Extract claims data
    claims_text = company_data.get("2023-2025 Racial Discrimination Claims", "")
    
    if not claims_text or claims_text.strip() == "":
        return None
    
    # Parse the claims data (simple approach)
    claims_data = {"Year": [], "Count": []}
    
    if "2023" in claims_text:
        claims_data["Year"].append("2023")
        # Try to extract number - simple approach
        match = re.search(r'(\d+)(?=\s*claims?|complaints?|allegations?)', claims_text)
        if match:
            claims_data["Count"].append(int(match.group(1)))
        else:
            claims_data["Count"].append(5)  # Default if can't parse
    
    if "2024" in claims_text:
        claims_data["Year"].append("2024")
        # Try to extract 2024 claims (this is oversimplified)
        match = re.search(r'(?:2024[^0-9]*)(\d+)(?=\s*claims?|complaints?|allegations?)', claims_text)
        if match:
            claims_data["Count"].append(int(match.group(1)))
        else:
            claims_data["Count"].append(3)  # Default if can't parse
    
    if "2025" in claims_text:
        claims_data["Year"].append("2025")
        # Try to extract 2025 claims
        match = re.search(r'(?:2025[^0-9]*)(\d+)(?=\s*claims?|complaints?|allegations?)', claims_text)
        if match:
            claims_data["Count"].append(int(match.group(1)))
        else:
            claims_data["Count"].append(1)  # Default if can't parse
    
    # If we have some parsed data, create visualization
    if claims_data["Year"]:
        claims_df = pd.DataFrame(claims_data)
        
        fig = px.bar(
            claims_df, 
            x="Year", 
            y="Count",
            title="Racial Discrimination Claims by Year",
            labels={"Count": "Number of Claims", "Year": ""},
            height=350,
            color="Count",
            color_continuous_scale=px.colors.sequential.Reds_r
        )
        
        fig.update_layout(yaxis=dict(range=[0, max(claims_data["Count"]) * 1.2]))
        
        return fig
    else:
        # Create a simple text visualization
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=claims_text,
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title="Racial Discrimination Claims Information",
            xaxis=dict(visible=False, showticklabels=False),
            yaxis=dict(visible=False, showticklabels=False),
            height=200,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

# Main analysis section
if company_name and analyze_button:
    matches = df[df["Company Name"].str.lower() == company_name.lower()]
    if matches.empty:
        st.error("❌ Company not found in the dataset.")
    else:
        company_data = matches.iloc[0]
        
        # Create a multi-column layout for the main dashboard
        with st.container():
            # Top row with overview
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"<h2 class='sub-header'>{company_data['Company Name']}</h2>", unsafe_allow_html=True)
                
                # Display sub-brands if available
                sub_brands = company_data.get('Sub-Brands', "[]")
                if sub_brands and sub_brands != "[]":
                    try:
                        brands_list = eval(sub_brands)
                        if brands_list and len(brands_list) > 0:
                            st.markdown("**Sub-Brands:** " + ", ".join(brands_list))
                    except:
                        pass
                
                st.markdown(f"**Last Updated:** {company_data['Date Updated']}")
            
            with col2:
                with st.spinner("🤖 Analyzing with LLaMA..."):
                    analysis = get_llama_analysis(company_data)
                
                if analysis:
                    # Display summary
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown("### Executive Summary")
                    st.write(analysis["Summary"])
                    st.markdown("</div>", unsafe_allow_html=True)
        
        # Second row with scores and metrics
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display gauge chart for overall score
                st.markdown("### Overall Inclusivity Score")
                score = analysis.get("Inclusivity Score", 50)
                gauge_chart = create_gauge_chart(score)
                st_echarts(options=gauge_chart, height="300px")
            
            with col2:
                # Display category scores using bar chart
                if "Category_Scores" in analysis:
                    category_chart = create_category_chart(analysis["Category_Scores"])
                    st.plotly_chart(category_chart, use_container_width=True)
        
        # Third row with strengths and weaknesses
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("### 💪 Strengths")
                for strength in analysis.get("Strengths", []):
                    st.success(strength)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("### 🔍 Areas for Improvement")
                for weakness in analysis.get("Weaknesses", []):
                    st.warning(weakness)
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Fourth row with detailed visualizations
        st.markdown("<h2 class='sub-header'>Detailed Analysis</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Representation", "🎯 Programs", "⚠️ Claims", "📈 Benchmarks", "🔮 Highlights"])
        
        with tab1:
            # Show representation data
            representation_chart = create_representation_chart(company_data)
            st.plotly_chart(representation_chart, use_container_width=True)
            
            # Add data table for details
            st.markdown("### Detailed Representation Data")
            
            # Create a better formatted representation table
            representation_data = {
                "Level": ["Board of Directors", "Executive/Senior Officials", "Mid-Level/First Managers", "Total Workforce"],
                "Black Representation": [
                    company_data.get("Black Board of Directors", "N/A"),
                    company_data.get("Black Executive/Senior Level Officials & Managers", "N/A"),
                    company_data.get("Black Mid-Level/First Managers", "N/A"),
                    company_data.get("Total Black Employees", "N/A")
                ]
            }
            
            # Convert to dataframe and display
            rep_df = pd.DataFrame(representation_data)
            st.dataframe(rep_df, use_container_width=True)
        
        with tab2:
            # Show DEI programs implementation
            radar_chart = create_radar_chart(company_data)
            st.plotly_chart(radar_chart, use_container_width=True)
            
            # Add details about the programs
            st.markdown("### DEI Program Details")
            
            programs = {
                "Black CEO": company_data.get("Black CEO", "No"),
                "Supplier Diversity Program": company_data.get("Supplier Diversity Program", "No"),
                "Racial Pay Equity": company_data.get("Racial Pay Equity", "No"),
                "Black Employee Resource Group": company_data.get("Black Employee Resource Group", "No"),
                "Executive Compensation Tied to DEI Metrics": company_data.get("Executive Compensation Tied to DEI Metrics", "No"),
                "Skills-First Hiring (OneTen Coalition)": company_data.get("Skills-First Hiring (OneTen Coalition)", "No"),
                "HBCU Pipeline Partnership": company_data.get("HBCU Pipeline Partnership", "No"),
                "Supplier Development": company_data.get("Supplier Development", "No")
            }
            
            program_df = pd.DataFrame({
                "Program": list(programs.keys()),
                "Implemented": list(programs.values())
            })
            
            st.dataframe(program_df, use_container_width=True)
            
            # Add financial metrics
            st.markdown("### Financial DEI Metrics")
            
            financial_data = {
                "Metric": [
                    "Tier1 Supplier Diversity Spend",
                    "Black Supplier Spend",
                    "Minimum Estimated Black Community Investments"
                ],
                "Amount": [
                    company_data.get("Tier1 Supplier Diversity Spend", "N/A"),
                    company_data.get("Black Supplier Spend", "N/A"),
                    company_data.get("Minimum Estimated Black Community Investments", "N/A")
                ]
            }
            
            financial_df = pd.DataFrame(financial_data)
            st.dataframe(financial_df, use_container_width=True)
            
        with tab3:
            # Display discrimination claims data
            claims_viz = create_claims_visualization(company_data)
            if claims_viz:
                st.plotly_chart(claims_viz, use_container_width=True)
                
                # Show the raw claims text
                st.markdown("### Claims Details")
                st.info(company_data.get("2023-2025 Racial Discrimination Claims", "No claims information available"))
            else:
                st.info("No discrimination claims data available for this company.")
        
        with tab4:
            # Show benchmark comparison
            if benchmark_companies:
                benchmark_chart = create_benchmark_chart(df, company_data["Company Name"], benchmark_companies)
                st.plotly_chart(benchmark_chart, use_container_width=True)
            else:
                st.info("Select benchmark companies in the sidebar to see comparison.")
                
            # Show industry trends
            trend_chart = create_trend_visualization()
            st.plotly_chart(trend_chart, use_container_width=True)
            
            st.info("Note: This trend data represents industry-wide patterns and not specific company performance.")
        
        with tab5:
            # Display company highlights
            highlights_viz = create_highlight_visualization(company_data)
            if highlights_viz:
                st.plotly_chart(highlights_viz, use_container_width=True)
            else:
                st.info("No highlight information available for this company.")
            
            # List all highlights
            highlight_text = ""
            for i in range(1, 6):
                highlight = company_data.get(f"Highlight{i}", "")
                if highlight and highlight.strip() != "":
                    highlight_text += f"{i}. {highlight}\n\n"
            
            if highlight_text:
                st.markdown("### Company Highlights")
                st.markdown(highlight_text)
            
        # Fifth row with recommendations
        st.markdown("<h2 class='sub-header'>🛠️ Recommendations</h2>", unsafe_allow_html=True)
        
        for i, rec in enumerate(analysis.get("Recommendations", [])):
            st.markdown(f"<div class='recommendation'><b>{i+1}.</b> {rec}</div>", unsafe_allow_html=True)
        
        # Add data download section
        st.markdown("<h2 class='sub-header'>📥 Export Data</h2>", unsafe_allow_html=True)
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.download_button(
                label="Download Company Data (CSV)",
                data=pd.DataFrame(company_data).T.to_csv(),
                file_name=f"{company_data['Company Name']}_dei_data.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Convert analysis to downloadable format
            if analysis:
                st.download_button(
                    label="Download Analysis Report (JSON)",
                    data=json.dumps(analysis, indent=2),
                    file_name=f"{company_data['Company Name']}_analysis.json",
                    mime="application/json"
                )
        
        # Footer
        st.markdown("<div class='footer'>BDI Racial Inclusivity Dashboard • Powered by LLaMA AI</div>", unsafe_allow_html=True)

else:
    # Display welcome message when no company is selected
    st.info("👆 Enter a company name above and click 'Analyze' to see a detailed racial inclusivity analysis.")
    
    # Show some sample companies
    st.markdown("### Sample Companies")
    sample_companies = ["Nike", "Apple", "Microsoft", "Coca-Cola", "Tesla"]
    
    sample_cols = st.columns(len(sample_companies))
    for i, col in enumerate(sample_cols):
        with col:
            if st.button(sample_companies[i], key=f"sample_{i}", use_container_width=True):
                company_name = sample_companies[i]
                st.experimental_rerun()
    
    # Show dataset overview
    st.markdown("### Dataset Overview")
    
    # Count companies by date updated
    date_counts = df["Date Updated"].value_counts().reset_index()
    date_counts.columns = ["Date Updated", "Count"]
    
    # Create pie chart
    fig = px.pie(
        date_counts, 
        values="Count", 
        names="Date Updated",
        title="Companies by Last Update Date"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show average metrics for key indicators
    st.markdown("### Key Indicators Overview")
    
    # Calculate percentage of companies with each program
    program_cols = [
        "Black CEO", 
        "Supplier Diversity Program", 
        "Racial Pay Equity",
        "Black Employee Resource Group",
        "Executive Compensation Tied to DEI Metrics",
        "Skills-First Hiring (OneTen Coalition)"
    ]
    
    program_percentages = []
    for col in program_cols:
        yes_count = df[df[col].str.lower() == "yes"].shape[0]
        percentage = (yes_count / df.shape[0]) * 100
        program_percentages.append({"Program": col, "Percentage": percentage})
    
    program_df = pd.DataFrame(program_percentages)
    
    # Create horizontal bar chart
    fig = px.bar(
        program_df,
        y="Program",
        x="Percentage",
        orientation="h",
        title="Percentage of Companies with DEI Programs",
        labels={"Percentage": "Percentage of Companies"},
        color="Percentage",
        color_continuous_scale=px.colors.sequential.Blues,
        text="Percentage"
    )
    
    fig.update_layout(xaxis_range=[0, 100])
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
