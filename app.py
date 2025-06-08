import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai # NEW: Import genai
import os # NEW: Import os for API key
import json
import re # NEW: Import re
from dotenv import load_dotenv # NEW: Import load_dotenv
import subprocess
load_dotenv()
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SupplyScout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED STYLING WITH PROPER LOGO POSITIONING ---
st.markdown("""
    <style>
        /* Hide Streamlit default header and branding */
        header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Hide hamburger menu */
        .css-1rs6os.edgvbvh3 {
            display: none;
        }
        
        /* Custom fixed header with logo */
        .header-container {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(90deg, #16202e 0%, #1a2332 100%);
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 1000;
            border-bottom: 1px solid #22304a;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .logo-text {
            font-size: 1.4rem;
            font-weight: 600;
            color: #a3c9f9;
            font-family: 'Segoe UI', sans-serif;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Dark theme for entire app */
        .stApp {
            background-color: #131a26;
            color: #e0e6ed;
        }
        
        /* Main content padding to account for fixed header */
        .main .block-container {
            padding-top: 80px;
            max-width: 100%;
        }
        
        /* Sidebar styling */
        .stSidebar {
            background-color: #16202e !important;
            border-right: 1px solid #22304a;
        }
        
        .stSidebar .stMarkdown {
            color: #e0e6ed;
        }
        
        /* Button styling with hover effects */
        .stButton>button {
            background-color: #22304a;
            color: #e0e6ed;
            border-radius: 6px;
            border: none;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .stButton>button:hover {
            background-color: #2a3a5a;
            transform: translateY(-1px);
            box_shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Input field styling */
        .stTextInput>div>div>input {
            background-color: #22304a;
            color: #e0e6ed;
            border: 1px solid #2a3a5a;
            border-radius: 4px;
        }
        
        .stSelectbox>div>div>select {
            background-color: #22304a;
            color: #e0e6ed;
            border: 1px solid #2a3a5a;
        }
        
        .stNumberInput>div>div>input {
            background-color: #22304a;
            color: #e0e6ed;
            border: 1px solid #2a3a5a;
        }
        
        /* DataFrame styling */
        .stDataFrame {
            background-color: #1a2332;
        }
        
        /* Metric cards styling */
        .metric-card {
            background-color: #1a2332;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #22304a;
            margin: 10px 0;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #22304a;
            color: #e0e6ed;
            border-radius: 4px;
        }
        
        /* Info box styling */
        .stAlert {
            background-color: #1a2332;
            border: 1px solid #22304a;
            color: #e0e6ed;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #22304a;
            color: #e0e6ed;
        }
        
        /* Container styling */
        .element-container {
            background-color: transparent;
        }
    </style>
    <div class="header-container">
        <div class="logo-text">
            🔍 SupplyScout
        </div>
    </div>
""", unsafe_allow_html=True)


# --- FUNCTION TO EXTRACT PRODUCT SPECIFICATIONS (MOVED FROM MODAL) ---
def get_dictionary_from_prompt(user_prompt: str) -> dict:
    """
    Parses a user's request and converts it into a structured dictionary
    of product specifications using the Gemini API. This function runs directly in Streamlit.
    """
    try:
        # Configure Gemini API using the secret (환경 변수에서 읽어옴)
        # Ensure GEMINI_API_KEY is set in your environment before running Streamlit
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
    except KeyError:
        st.error("Gemini API Key not found. Please set the 'GEMINI_API_KEY' environment variable.")
        return {}
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        return {}

    extraction_prompt = f"""
        You are an expert data extraction AI. Your task is to parse a user's
        request and convert it into a structured JSON object. The JSON object
        should be a dictionary where each key is the product name (string) and
        its value is a dictionary of that product's specifications.

        EXAMPLE
        User Prompt: "Silicon wafers, purity 99.999%, diameter 6 inches"
        Correct Output:
        json
        {{
          "Silicon wafers": {{
            "purity": "99.999%",
            "diameter": "6 inches"
          }}
        }}
        
        YOUR TASK
        User Prompt: "{user_prompt}"

        Respond ONLY with the single, valid JSON object.
    """

    try:
        response = model.generate_content(extraction_prompt)
        
        # We can directly print to Streamlit's debug area or use st.info for visibility
        # st.info(f"Gemini API Raw Response: {response.text}") 
        
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
            return parsed_json
        else:
            st.warning("No valid JSON object found in the model's response. Raw response:")
            st.code(response.text) # Show raw response if no JSON found
            return {}
    except Exception as e:
        st.error(f"An error occurred during API call or parsing: {e}")
        return {}


######## INPUTS #########
# --- SIDEBAR: SIMPLIFIED SEARCH ---
with st.sidebar:
    st.markdown("### 🔍 Product Search")
    
    # Main search input - natural language prompt
    search_prompt = st.text_area(
        "What are you looking for?",
        placeholder="e.g., I need a quantum sensor with wavelength range 400-700nm, price under $5000, from a reliable supplier with fast delivery",
        height=100,
        help="Describe what you're looking for including product type, specifications, price range, and any other requirements"
    )
    
    st.markdown("---")
    
    # Country selection with checkboxes 
    st.markdown("#### Country/Region")
    st.markdown("Select countries to source from:")
    
    uk_selected = st.checkbox(" United Kingdom", value=True)
    us_selected = st.checkbox("United States", value=True)
    germany_selected = st.checkbox(" Germany", value=False)
    china_selected = st.checkbox("China", value=False)
    japan_selected = st.checkbox(" Japan", value=False)
    
    st.markdown("---")
    
    # Single search button
    search_btn = st.button("Search Products", use_container_width=True, type="primary")
    
    # --- Direct Gemini API call (no Modal or subprocess) ---
    if search_btn:
        if search_prompt.strip():
            with st.spinner("Searching for products..."):
                response_data = get_dictionary_from_prompt(search_prompt)
                


            if response_data:
                #st.success("Search completed successfully!")
                st.session_state['search_results'] = response_data
                st.session_state['last_search'] = search_prompt
                
                for key in response_data.keys():
                    subprocess.run(["python","-m","modal","run","vendor_search_modal.py","--product",key])

                for key in response_data.keys():
                    #

                    output_string = f"{key}"

                    # Iterate through the items (key-value pairs) in response_data
                    for attr_key, attr_value in response_data[key].items():
                        output_string += f" {attr_value}"
                    print(output_string)
                    subprocess.run(["python","-m","modal","run","webscrapper.py","--json-file-path",f"vendors_{key.replace(" ", "_")}.json","--output-file-name", f"{key.replace(" ", "_")}_product_data.json", "--material-context", output_string])
                
                

                #st.markdown("#### Extracted Product Specifications:")
                #st.json(response_data) # Display the extracted dictionary
            else:
                st.error("No valid product specifications could be extracted.")
                
        else:
            st.warning("Please enter a search query first!")


########### OUTPUTS ###########
# --- MAIN AREA: GRAPH FROM SEPARATE FILE ---
st.markdown("### Product Comparison Overview")

if 'search_results' in st.session_state:
    st.info("Generating comparison graph based on extracted specifications...")
    
    extracted_products = list(st.session_state['search_results'].keys())
    if extracted_products:
        n_points = len(extracted_products)
        x = np.random.uniform(1, 10, n_points) 
        y = np.random.uniform(1, 10, n_points) 
        
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(x, y, c='#a3c9f9', s=120, edgecolors='#22304a', linewidth=2, alpha=0.8)
        
        for i, label in enumerate(extracted_products):
            ax.annotate(label, (x[i]+0.1, y[i]+0.1), color="#e0e6ed", fontsize=10, fontweight='bold')
        
        ax.set_facecolor("#1a2332")
        fig.patch.set_facecolor('#1a2332')
        
        for spine in ax.spines.values():
            spine.set_color('#a3c9f9')
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', colors='#a3c9f9', labelsize=10)
        ax.set_xlabel("Cost (simulated)", color='#a3c9f9', fontsize=12, fontweight='bold')
        ax.set_ylabel("Quality (simulated)", color='#a3c9f9', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, color='#a3c9f9')
        ax.set_title(f"Product Comparison for '{st.session_state.get('last_search', 'Your Search')}'", color='#e0e6ed', fontsize=14, fontweight='bold', pad=20)
        
        st.pyplot(fig)
    else:
        st.warning("No products extracted to display on the graph.")
        st.info("Displaying placeholder graph...")
        np.random.seed(42)
        n_points = 8
        x = np.random.uniform(1, 10, n_points)
        y = np.random.uniform(1, 10, n_points)
        labels = [f"Product {i+1}" for i in range(n_points)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(x, y, c='#a3c9f9', s=120, edgecolors='#22304a', linewidth=2, alpha=0.8)
        
        for i, label in enumerate(labels):
            ax.annotate(label, (x[i]+0.1, y[i]+0.1), color="#e0e6ed", fontsize=10, fontweight='bold')
        
        ax.set_facecolor("#1a2332")
        fig.patch.set_facecolor('#1a2332')
        
        for spine in ax.spines.values():
            spine.set_color('#a3c9f9')
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', colors='#a3c9f9', labelsize=10)
        ax.set_xlabel("Cost", color='#a3c9f9', fontsize=12, fontweight='bold')
        ax.set_ylabel("Quality", color='#a3c9f9', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, color='#a3c9f9')
        ax.set_title("Product Comparison", color='#e0e6ed', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        
else:
    st.info("Perform a search to see product comparison visualization")
    st.markdown("#### Sample Visualization")
    np.random.seed(42)
    n_points = 5
    x = np.random.uniform(1, 10, n_points)
    y = np.random.uniform(1, 10, n_points)
    labels = [f"Sample {i+1}" for i in range(n_points)]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(x, y, c='#a3c9f9', s=100, edgecolors='#22304a', linewidth=2, alpha=0.6)
    
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i]+0.1, y[i]+0.1), color="#e0e6ed", fontsize=9)
    
    ax.set_facecolor("#1a2332")
    fig.patch.set_facecolor('#1a2332')
    
    for spine in ax.spines.values():
        spine.set_color('#a3c9f9')
        spine.set_linewidth(1.5)
    
    ax.tick_params(axis='both', colors='#a3c9f9', labelsize=9)
    ax.set_xlabel("Cost ($)", color='#a3c9f9', fontsize=11, fontweight='bold')
    ax.set_ylabel("Quality Score", color='#a3c9f9', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, color='#a3c9f9')
    ax.set_title("Sample Product Comparison", color='#e0e6ed', fontsize=12, fontweight='bold', pad=15)
    
    st.pyplot(fig)

# --- QUICK STATS SIDEBAR (Moved to right of graph) ---
with st.sidebar:
    if 'search_results' in st.session_state:
        st.markdown("---")
        st.markdown("### Search Summary")
        
        total_products = len(st.session_state['search_results']) if st.session_state['search_results'] else 0
        
        st.metric("Products Identified", total_products) 
        st.metric("Last Search", st.session_state.get('last_search', 'None')[:20] + "...")
        
        if st.button("Export Graph", use_container_width=True):
            st.success("Graph exported!")
        
        if st.button("Refresh Data", use_container_width=True):
            st.info("Data refreshed!")

# --- SCROLLABLE TABLE SECTION ---
st.markdown("---")
st.markdown("### Detailed Search Results")

if 'search_results' in st.session_state:
    try:
        extracted_data = st.session_state['search_results']
        
        if extracted_data:
            table_rows = []
            for product_name, specs in extracted_data.items():
                row = {'Product': product_name}
                for spec_key, spec_value in specs.items():
                    row[spec_key.replace('_', ' ').title()] = spec_value 
                table_rows.append(row)
            
            df = pd.DataFrame(table_rows)
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                search_filter = st.text_input("Filter results", placeholder="Search in table...")
            with col2:
                sortable_columns = [col for col in df.columns if col != 'Product']
                sort_by = st.selectbox("Sort by", sortable_columns if sortable_columns else ["Product"])
            with col3:
                show_all = st.checkbox("Show all", value=True)
            
            if search_filter:
                df = df[df.astype(str).apply(lambda x: x.str.contains(search_filter, case=False, na=False)).any(axis=1)]
            
            if sort_by in df.columns:
                df = df.sort_values(by=sort_by)

            st.dataframe(
                df, 
                use_container_width=True,
                height=400 
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"supplyscout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                if st.button("Export Excel", use_container_width=True):
                    st.info("Excel export feature coming soon!")
            
            with col3:
                if st.button("Email Results", use_container_width=True):
                    st.info("Email sharing coming soon!")
        else:
            st.warning("No detailed product data received from extraction.")
            st.info("Displaying sample data...")
            
            sample_data = {
                'Vendor': ['ThorLabs', 'Newport', 'Edmund Optics', 'Coherent', 'Hamamatsu'],
                'Product': ['Quantum Sensor A', 'Sensor Pro B', 'OptiSense C', 'QuanTech D', 'PhotoSense E'],
                'Price ($)': [1200, 950, 1800, 750, 1400],
                'Lead Time': ['3-5 days', '5-7 days', '2-4 days', '7-10 days', '4-6 days'],
                'Rating': [4.8, 4.6, 4.7, 4.5, 4.9],
                'Location': ['UK', 'US', 'Germany', 'US', 'Japan']
            }
            
            df = pd.DataFrame(sample_data)
            st.dataframe(df, use_container_width=True, height=300)
            
    except Exception as e:
        st.error(f"Error generating table: {str(e)}")
        st.info("Sample data will be shown instead")

else:
    st.info("Perform a search to see detailed vendor results")
    st.markdown("*Results will include vendor information, pricing, lead times, and contact details*")

# --- FOOTER (KEPT AS REQUESTED) ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#4e6a89; margin-top:40px; padding: 20px;'>
        <p><strong>SupplyScout</strong> &copy; 2025 | Streamlining Scientific Procurement</p>
        <p style='font-size: 0.8em;'>Built with Streamlit | Version 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)