import streamlit as st
import requests
import plotly.graph_objects as go

# --- Config ---
st.set_page_config(page_title="LeadFlow Dashboard", layout="wide")
API_URL = "http://localhost:8000/predict"

# --- UI Styling ---
st.title("🚀 Project LeadFlow: Intelligent Scoring")
st.markdown("""
This dashboard connects to the LeadFlow MLOps API to score new leads in real-time.
""")

# --- Sidebar Inputs ---
st.sidebar.header("📝 New Lead Details")
lead_source = st.sidebar.selectbox("Lead Source", ["Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference"])
lead_origin = st.sidebar.selectbox("Lead Origin", ["Landing Page Submission", "API", "Lead Add Form", "Lead Import"])
last_activity = st.sidebar.selectbox("Last Activity", ["Email Opened", "SMS Sent", "Page Visited on Website", "Converted to Lead"])
time_spent = st.sidebar.slider("Time Spent on Website (sec)", 0, 3000, 600)
total_visits = st.sidebar.slider("Total Visits", 0, 50, 5)

if st.sidebar.button("Score Lead"):
    # Construct payload matching the API schema
    payload = {
        "Total_Time_Spent_on_Website": time_spent,
        "TotalVisits": total_visits,
        "Lead_Source": lead_source,
        "Lead_Origin": lead_origin,
        "Last_Activity": last_activity
    }
    
    try:
        with st.spinner("Analyzing lead data..."):
            response = requests.post(API_URL, json=payload)
            
        if response.status_code == 200:
            result = response.json()
            score = result['score']
            priority = result['priority']
            
            # --- Main Display Results ---
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Conversion Score")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probability %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccb"},
                            {'range': [50, 80], 'color': "#ffffcc"},
                            {'range': [80, 100], 'color': "#90ee90"}],
                    }))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Strategic Action")
                if priority == "High":
                    st.success(f"**PRIORITY: {priority}**")
                    st.info("✅ **Action:** Route to Senior Sales Rep immediately. Call within 5 mins.")
                elif priority == "Medium":
                    st.warning(f"**PRIORITY: {priority}**")
                    st.info("⚠️ **Action:** Add to Nurture Campaign. Follow up in 24 hours.")
                else:
                    st.error(f"**PRIORITY: {priority}**")
                    st.info("❄️ **Action:** Automated Email only. Do not spend sales time.")
                    
        else:
            st.error(f"Error from API: {response.text}")
            
    except Exception as e:
        st.error(f"Failed to connect to API at {API_URL}. Is the server running? Error: {e}")

else:
    st.info("👈 Enter lead details in the sidebar and click 'Score Lead' to start.")

# --- Analytics Section (Mock) ---
st.divider()
st.subheader("📊 Pipeline Analytics (Mock Data)")
c1, c2, c3 = st.columns(3)
c1.metric("Leads Processed Today", "142", "+12%")
c2.metric("Avg Conversion Score", "64", "+5%")
c3.metric("High Priority Ratio", "18%", "-2%")