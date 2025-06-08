import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
    
    # Create checkboxes for each country - these are just for display
    uk_selected = st.checkbox(" United Kingdom", value=True)
    us_selected = st.checkbox("United States", value=True)
    germany_selected = st.checkbox(" Germany", value=False)
    china_selected = st.checkbox("China", value=False)
    japan_selected = st.checkbox(" Japan", value=False)
    
    st.markdown("---")
    
    # Single search button
    search_btn = st.button("Search Products", use_container_width=True, type="primary")
    
    ######### Gemini API integration
    if search_btn:
        if search_prompt.strip():
            st.success("Search initiated!")
            
            # Call Gemini API with the prompt
            try:
                import subprocess
                import sys
                import json
                
                # pass the user's search prompt to vendor_search_modal.py
                result = subprocess.run([
                    sys.executable, 
                    "vendor_search_modal.py",  ############## TO BE CHANGED TO GEMINI API######
                    search_prompt  # Pass only the prompt as a string
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Process successful results
                    response_data = json.loads(result.stdout)
                    st.success("Search completed successfully!")
                    
                    # Store results in session state for use in main area
                    st.session_state['search_results'] = response_data
                    st.session_state['last_search'] = search_prompt
                    
                else:
                    st.error(f"Search failed: {result.stderr}")
                    
            except Exception as e:
                st.error(f"Error executing search: {str(e)}")
            
        else:
            st.warning("Please enter a search query first!")






########### OUTPUTS ###########
# --- MAIN AREA: GRAPH FROM SEPARATE FILE ---
st.markdown("### Product Comparison Overview")

# Check if we have search results to display
if 'search_results' in st.session_state:
    # Call separate Python file to generate graph
    try:
        import subprocess
        import sys
        import json
        
        # Pass search results to graph generator
        graph_data = {
            "search_results": st.session_state['search_results'],
            "last_search": st.session_state.get('last_search', '')
        }
        
        # Call your graph generation file
        result = subprocess.run([
            sys.executable, 
            "graph_generator.py",  # Your graph generation file
            json.dumps(graph_data)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Graph file should be saved as 'comparison_graph.png'
            st.image("comparison_graph.png", use_column_width=True)
        else:
            st.error("Failed to generate graph")
            # Fallback to placeholder graph
            st.info("Displaying placeholder graph...")
            
            # Placeholder graph code (your existing code as fallback)
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
            
    except Exception as e:
        st.error(f"Error generating graph: {str(e)}")
        st.info("Please perform a search to see results")

else:
    # Show placeholder when no search results
    st.info("Perform a search to see product comparison visualization")
    
    # Optional: Show sample graph
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
        
        # Extract stats from search results
        results = st.session_state['search_results']
        total_vendors = results.get('total_vendors', 0)
        
        st.metric("Vendors Found", total_vendors)
        st.metric("Last Search", st.session_state.get('last_search', 'None')[:20] + "...")
        
        # Quick action buttons
        if st.button("Export Graph", use_container_width=True):
            st.success("Graph exported!")
        
        if st.button("Refresh Data", use_container_width=True):
            st.info("Data refreshed!")

# --- SCROLLABLE TABLE SECTION FROM SEPARATE FILE ---
st.markdown("---")
st.markdown("### Detailed Search Results")

if 'search_results' in st.session_state:
    # Call separate Python file to generate table
    try:
        # Pass search results to table generator
        table_data = {
            "search_results": st.session_state['search_results'],
            "last_search": st.session_state.get('last_search', '')
        }
        
        ######### Call table generation file
        result = subprocess.run([
            sys.executable, 
            "table_generator.py",  # Your table generation file
            json.dumps(table_data)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Table generator should output JSON with table data
            table_results = json.loads(result.stdout)
            
            # Display the generated table
            if 'dataframe' in table_results:
                df = pd.DataFrame(table_results['dataframe'])
                
                # Enhanced table display with filtering
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    search_filter = st.text_input("Filter results", placeholder="Search in table...")
                with col2:
                    sort_by = st.selectbox("Sort by", ["Vendor Name", "Price", "Rating", "Lead Time"])
                with col3:
                    show_all = st.checkbox("Show all", value=True)
                
                # Apply filters
                if search_filter:
                    df = df[df.astype(str).apply(lambda x: x.str.contains(search_filter, case=False, na=False)).any(axis=1)]
                
                # Display table with custom styling
                st.dataframe(
                    df, 
                    use_container_width=True,
                    height=400  # Fixed height for scrolling
                )
                
                # Download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Download CSV", use_container_width=True):
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Save CSV",
                            data=csv,
                            file_name=f"supplyscout_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Export Excel", use_container_width=True):
                        st.info("Excel export feature coming soon!")
                
                with col3:
                    if st.button("Email Results", use_container_width=True):
                        st.info("Email sharing coming soon!")
            
            else:
                st.warning("No table data received from generator")
                
        else:
            st.error("Failed to generate table")
            # Fallback table
            st.info("Displaying sample data...")
            
            # Sample fallback data
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

