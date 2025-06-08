import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai # NEW: Import genai
import os # NEW: Import os for API key
import json
import re # NEW: Import re
from dotenv import load_dotenv # NEW: Import load_dotenv
import base64


load_dotenv()
# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SupplyScout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

# --- LOGO ---
def get_image_as_base64(file):
    """ Reads a file and returns its content as a base64 encoded string. """
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# The path to your logo file
logo_path = "SupplyScout Logo.png" 
# Convert the logo to a base64 string
logo_base64 = get_image_as_base64(logo_path)

# Add the logo to the page with custom CSS for positioning only if logo exists
if logo_base64:
    st.markdown(
        f"""
        <style>
            .logo-container {{
                position: fixed;
                top: -10px;
                right: 10px;
                z-index: 9999;
                width: 200px;
                height: auto;
                margin: 0 !important;
                padding: 0 !important;
                line-height: 0 !important;
                font-size: 0 !important;
                overflow: hidden;
            }}
            .logo-img {{
                width: 100%;
                height: auto;
                display: block;
                margin: 0 !important;
                padding: 0 !important;
                line-height: 0 !important;
                border: none;
                outline: none;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }}
            /* Hide Streamlit's default header */
            header[data-testid="stHeader"] {{
                display: none;
            }}
        </style>
        <div class="logo-container">
            <img class="logo-img" src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True
    )

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
                st.success("Search completed successfully!")
                st.session_state['search_results'] = response_data
                st.session_state['last_search'] = search_prompt
                
                st.markdown("#### Extracted Product Specifications:")
                st.json(response_data) # Display the extracted dictionary
            else:
                st.error("No valid product specifications could be extracted.")
                
        else:
            st.warning("Please enter a search query first!")


########### OUTPUTS ###########
st.markdown("### Product Comparison & Vendor Ranking")

# 1) Parse the user’s search_prompt as JSON → query_specs
try:
    query_specs = json.loads(search_prompt)
    if not isinstance(query_specs, dict) or not all(isinstance(v, dict) for v in query_specs.values()):
        raise ValueError
except Exception:
    st.error(
        "❗ Please enter your query as JSON mapping each material to its price & match_score, e.g.:\n"
        "`{\"isopropanol\":{\"price\":10,\"match_score\":1.0},"
        "\"acetone\":{\"price\":20,\"match_score\":0.8}}`"
    )
    st.stop()

# 2) Load each material’s JSON file into supply_specs
#    Files must be named "<material>_product_data.json"
supply_specs = {}
for mat, specs in query_specs.items():
    filename = f"{mat}_product_data.json"
    try:
        data = json.load(open(filename))
    except FileNotFoundError:
        st.error(f"Could not find file: {filename}")
        st.stop()
    for vendor, entries in data.items():
        for info in entries.values():
            price = info.get('price', 0.0)
            score = info.get('match_score', 0.0)
            supply_specs[(mat, vendor)] = {'price': price, 'match_score': score}

# 3) Compute BiRank and minimal vendor subset
materials = list(query_specs.keys())
vendors   = sorted({v for (_, v) in supply_specs.keys()})

# build supply_map
supply_map = {v: set() for v in vendors}
for m, v in supply_specs:
    supply_map[v].add(m)

# build W
M, V = len(materials), len(vendors)
mat_index  = {m:i for i,m in enumerate(materials)}
vend_index = {v:j for j,v in enumerate(vendors)}
W = np.zeros((M, V))
for (m, v), specs in supply_specs.items():
    qdict = query_specs[m]
    keys  = sorted(set(qdict) | set(specs))
    qvec  = np.array([qdict.get(k,0.0) for k in keys]).reshape(1,-1)
    vvec  = np.array([specs.get(k,0.0) for k in keys]).reshape(1,-1)
    W[mat_index[m], vend_index[v]] = cosine_similarity(qvec, vvec)[0,0]

# run BiRank
alpha = 0.85
f = np.ones(M)/M
h = np.ones(V)/V
for _ in range(100):
    h = alpha * W.T.dot(f) + (1-alpha)*(np.ones(V)/V); h /= h.sum()
    f = alpha * W.dot(h)   + (1-alpha)*(np.ones(M)/M); f /= f.sum()

# find minimal covering subset
all_mat = set(materials)
best_size, best_score, best_subset = V+1, -np.inf, None
for r in range(1, V+1):
    for subset in itertools.combinations(vendors, r):
        covered = set().union(*(supply_map[v] for v in subset))
        if covered == all_mat:
            total = sum(h[vend_index[v]] for v in subset)
            if r < best_size or (r == best_size and total > best_score):
                best_size, best_score, best_subset = r, total, subset
    if best_subset: break

# 4) Display Vendor Scores with cost columns
st.subheader("🔢 Vendor BiRank Scores")
vendor_scores = pd.DataFrame({'vendor': vendors, 'score': h})
for mat in materials:
    vendor_scores[f"{mat}_cost"] = vendor_scores['vendor'].apply(
        lambda v: supply_specs.get((mat, v), {}).get('price', 'n/a')
    )
st.dataframe(vendor_scores.sort_values('score', ascending=False).reset_index(drop=True))

# 5) Plot the bipartite network
st.subheader("🌐 Materials–Vendors Network")
G = nx.Graph()
G.add_nodes_from(materials, bipartite=0)
G.add_nodes_from(vendors,   bipartite=1)
for (m, v) in supply_specs:
    G.add_edge(m, v)
pos = nx.bipartite_layout(G, materials)

fig, ax = plt.subplots(figsize=(8,4))
# draw edges
all_e = list(G.edges())
opt_e = [(m,v) for (m,v) in all_e if v in best_subset]
other_e = [e for e in all_e if e not in opt_e]
nx.draw_networkx_edges(G, pos, edgelist=other_e, edge_color='lightgray', width=1, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=opt_e,     edge_color='red',       width=2, ax=ax)
# draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=materials,
                       node_color='#4C72B0', node_shape='o', node_size=800, ax=ax)
base_vs = [v for v in vendors if v not in best_subset]
nx.draw_networkx_nodes(G, pos, nodelist=base_vs,
                       node_color='#55A868', node_shape='s', node_size=600, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=best_subset,
                       node_color='#C44E52', node_shape='s', node_size=1000,
                       edgecolors='black', linewidths=2, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
# legend
handles = [
    mpatches.Patch(color='#4C72B0', label='Materials'),
    mpatches.Patch(color='#55A868', label='Vendors'),
    mpatches.Patch(color='#C44E52', label='Optimal Vendors'),
    mpatches.Patch(color='red',       label='Selected Edges'),
]
ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
ax.axis('off')
st.pyplot(fig)

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
