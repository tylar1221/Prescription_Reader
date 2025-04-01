import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import time

# ---------------------- CONFIGURATION ----------------------
CSV_FILE = r"C:\Users\ASUS\OneDrive\Desktop\ISE11\A_Z_medicines_dataset_of_India.csv"  # CSV file path

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    """Load medicine data from the CSV file and preprocess it efficiently."""
    try:
        # Only load necessary columns to reduce memory usage
        df = pd.read_csv(CSV_FILE, encoding="utf-8")
        df.columns = df.columns.str.strip().str.lower()  # Standardize column names
        
        if "name" not in df.columns:
            st.error("❌ The column 'name' is missing in the CSV file. Please check the file format.")
            return None, None
        
        # Convert names to lowercase
        df["name"] = df["name"].astype(str).str.lower()
        
        # Create a unique set of medicine names for faster fuzzy matching
        unique_names = df["name"].unique()
        
        return df, unique_names
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None, None

# ---------------------- SEARCH FUNCTIONS ----------------------
def exact_match_search(df, user_inputs):
    """Perform vectorized exact match search."""
    user_inputs_set = {input_str.strip().lower() for input_str in user_inputs}
    exact_matches = df[df["name"].isin(user_inputs_set)]
    return exact_matches.groupby("name")

def search_medicine(user_inputs, df, unique_names):
    """Optimized search function with exact and fuzzy matching."""
    if df is None:
        return []
    
    start_time = time.time()
    results = []
    
    # Perform batch exact match first
    exact_match_groups = exact_match_search(df, user_inputs)
    
    for user_input in user_inputs:
        user_input = user_input.strip().lower()
        
        # Check if this input had an exact match
        if user_input in exact_match_groups.groups:
            results.append((user_input, exact_match_groups.get_group(user_input)))
            continue
        
        # Only do fuzzy matching for inputs without exact matches
        fuzzy_result = process.extractOne(user_input, unique_names) if len(unique_names) > 0 else None
        if fuzzy_result:
            best_match, score = fuzzy_result[:2]
            if score >= 90:
                fuzzy_match_df = df[df["name"] == best_match]
                results.append((f"{user_input} (matched to: {best_match})", fuzzy_match_df))
            elif score > 85:
                results.append((f"Did you mean: {best_match}?", None))
            else:
                results.append((f"No matches found for: {user_input}", None))
        else:
            results.append((f"No matches found for: {user_input}", None))
    
    search_time = time.time() - start_time
    return results, search_time

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="Medicine Search App", page_icon="💊", layout="wide")

st.title("💊 Medicine Search App (Optimized)")
st.write("Enter medicine names (comma-separated) to check availability and details.")

# Load data
with st.spinner("Loading medicine database..."):
    df, unique_names = load_data()
    if df is not None:
        st.success(f"✅ Database loaded successfully! ({len(df)} records)")

# User input field
user_input = st.text_input("🔍 Enter Medicine Names (comma-separated):", "")

# Advanced options
with st.expander("Advanced Search Options"):
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.slider("Maximum results per medicine", 1, 20, 5)
    with col2:
        fuzzy_threshold = st.slider("Fuzzy matching threshold", 80, 100, 85)

if st.button("Search"):
    if user_input:
        # Parse and clean user inputs
        user_inputs = [name.strip() for name in user_input.split(",") if name.strip()]
        
        # Perform search
        with st.spinner("Searching..."):
            search_results, search_time = search_medicine(user_inputs, df, unique_names)
        
        # Display results
        st.success(f"✅ Search completed in {search_time:.3f} seconds")
        
        if search_results:
            # Create tabs for each search term
            if len(search_results) > 1:
                tabs = st.tabs([name for name, _ in search_results])
                
                for i, (name, result) in enumerate(search_results):
                    with tabs[i]:
                        if result is not None:
                            # Show paginated results
                            total_results = len(result)
                            if total_results > 0:
                                st.success(f"Found {total_results} results for '{name}'")
                                
                                # Pagination controls
                                if total_results > max_results:
                                    page_num = st.number_input(f"Page (1-{(total_results + max_results - 1) // max_results})", 
                                                              min_value=1, 
                                                              max_value=(total_results + max_results - 1) // max_results,
                                                              key=f"page_{i}")
                                    start_idx = (page_num - 1) * max_results
                                    end_idx = min(start_idx + max_results, total_results)
                                    paginated_result = result.iloc[start_idx:end_idx]
                                    st.dataframe(paginated_result, use_container_width=True)
                                    st.info(f"Showing results {start_idx+1}-{end_idx} of {total_results}")
                                else:
                                    st.dataframe(result, use_container_width=True)
                            else:
                                st.warning("No results found")
                        else:
                            st.warning(name)
            else:
                # Single result case
                name, result = search_results[0]
                if result is not None:
                    total_results = len(result)
                    st.success(f"Found {total_results} results for '{name}'")
                    
                    if total_results > max_results:
                        page_num = st.number_input(f"Page (1-{(total_results + max_results - 1) // max_results})", 
                                                  min_value=1, 
                                                  max_value=(total_results + max_results - 1) // max_results)
                        start_idx = (page_num - 1) * max_results
                        end_idx = min(start_idx + max_results, total_results)
                        paginated_result = result.iloc[start_idx:end_idx]
                        st.dataframe(paginated_result, use_container_width=True)
                        st.info(f"Showing results {start_idx+1}-{end_idx} of {total_results}")
                    else:
                        st.dataframe(result, use_container_width=True)
                else:
                    st.warning(name)
        else:
            st.warning("No results found for any of the search terms.")
    else:
        st.warning("⚠️ Please enter at least one medicine name.")

# Add filters for additional search refinement
if 'result' in locals() and result is not None and len(result) > 0:
    with st.expander("Filter Results"):
        # Display filters for columns present in the data
        for col in result.columns:
            if col != "name":
                unique_values = result[col].dropna().unique()
                if len(unique_values) < 20:  # Only show filter for columns with few unique values
                    selected = st.multiselect(f"Filter by {col}", unique_values)
                    if selected:
                        result = result[result[col].isin(selected)]
                        st.dataframe(result)

# Add a footer with performance stats
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>Database: {len(df) if df is not None else 0} records | Unique medicines: {len(unique_names) if unique_names is not None else 0}</div>", unsafe_allow_html=True)
