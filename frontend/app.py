import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time
from streamlit_autorefresh import st_autorefresh
import threading

def plot_q_value_regression(df):
    """
    Plots regression graph for q-value calculation.
    """
    if "Log(D/D_max)" not in df.columns or "Log(%_CPFT)" not in df.columns:
        st.warning("⚠️ Required columns for regression are missing.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(df["Log(D/D_max)"], df["Log(%_CPFT)"], label="Data Points", color="blue")

    # ✅ Perform regression and plot line
    slope, intercept = np.polyfit(df["Log(D/D_max)"], df["Log(%_CPFT)"], 1)
    reg_line = slope * df["Log(D/D_max)"] + intercept
    plt.plot(df["Log(D/D_max)"], reg_line, label=f"Regression Line (q = {slope:.4f})", color="red")

    plt.xlabel("Log(D/D_max)")
    plt.ylabel("Log(% CPFT)")
    plt.title("q-Value Regression")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Background ping to refresh every 5 minutes
st_autorefresh(interval=300000, key="refresh")

# Track inactivity time
if "last_active" not in st.session_state:
    st.session_state.last_active = time.time()

inactive_threshold = 5*60  # 5 minutes

# Check if inactive
inactive = time.time() - st.session_state.last_active > inactive_threshold

# JavaScript & CSS for semi-transparent overlay
overlay_script = """
<script>
    function removeOverlay() {
        var overlay = document.getElementById("inactivityOverlay");
        if (overlay) {
            overlay.style.display = "none";
        }
    }
    document.addEventListener("click", removeOverlay);
</script>
<style>
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.3); /* Slightly dark transparent overlay */
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 1000;
    }
    .overlay-content {
        background: rgba(255, 221, 221, 0.9); /* Light red (error-like) */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        color: #800000; /* Deep red/maroon */
        font-size: 20px;
        font-weight: bold;
        backdrop-filter: blur(5px);
    }
</style>
"""

st.markdown(overlay_script, unsafe_allow_html=True)

if inactive:
    st.markdown("""
    <div id="inactivityOverlay" class="overlay">
        <div class="overlay-content">
            <p>Dashboard is inactive! Please reload to continue.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if not inactive:
    st.session_state.last_active = time.time()

st.markdown(
    """
    <h1 style='text-align: center;'>CUMI Dashboard</h1>
    <h2 style='text-align: center; color: grey;'>GBD & q-value Computation</h2>
    """,
    unsafe_allow_html=True
)

# Function to check server status
def is_backend_active():
    try:
        response = requests.get(f"{BASE_URL}/ping", timeout=3)  # Short timeout for quick check
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

BASE_URL = "https://prac-ic2k.onrender.com"

available_dates = None
selected_date = None
sample_data = None

if "default_proportions" not in st.session_state:
    st.session_state["default_proportions"] = [0.35, 0.20, 0.15, 0.10, 0.20]
if "proportions" not in st.session_state:
    st.session_state["proportions"] = st.session_state["default_proportions"].copy()


if uploaded_file:
    st.write("File uploaded successfully.")
    
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

    # Check backend status before making the request
    if not is_backend_active():
        with st.spinner("⏳ Waking up the server... This may take ~1 minute. Please wait..."):
            # Wait for the server to become active (poll every 10 seconds)
            max_wait_time = 60  # 1 minute
            start_time = time.time()
            
            while time.time() - start_time < max_wait_time:
                if is_backend_active():
                    break  # Exit loop if backend wakes up
                time.sleep(10)  # Check every 10 seconds

    #  # Now upload the file
    # with st.spinner("⏳ Uploading file... Please wait..."):
    response = requests.post(f"{BASE_URL}/upload/", files=files)


    if response.status_code == 200:
        result = response.json()
        date_range = result.get("date_range", [])

        if date_range and len(date_range) == 2:
            min_date = datetime.strptime(date_range[0], "%Y-%m-%d")
            max_date = datetime.strptime(date_range[1], "%Y-%m-%d")

            formatted_date_range = [min_date.strftime("%d-%m-%Y"), max_date.strftime("%d-%m-%Y")]
            st.write(f"Available date range: **{formatted_date_range[0]} to {formatted_date_range[1]}**")


            selected_date = st.date_input("Select a date:", value=None, format="DD-MM-YYYY")

            if selected_date:
                selected_date_dt = datetime.combine(selected_date, datetime.min.time())

                if selected_date_dt < min_date or selected_date_dt > max_date:
                    st.error(f"⚠️ Please select a date from the available range: {formatted_date_range[0]} to {formatted_date_range[1]}")
                else:
                    formatted_selected_date = selected_date_dt.strftime("%d-%m-%Y")

                    if st.button("🔍 Verify Sample Data"):
                        sample_response = requests.get(f"{BASE_URL}/get_sample_data/", params={"selected_date": formatted_selected_date})

                        if sample_response.status_code == 200:
                            sample_data = sample_response.json().get("sample_data", {})
                            if sample_data:
                                st.success(f"✅ Sample Data for {formatted_selected_date}:")
                                for sheet, df_data in sample_data.items():
                                    st.write(f"**📄 Sheet: {sheet}**")
                                    df = pd.DataFrame(df_data)

                                    if 'Received Date' in df.columns:
                                        df['Received Date'] = pd.to_datetime(df['Received Date']).dt.strftime("%d-%m-%Y")

                                    st.dataframe(df)
                            else:
                                st.warning(f"No sample data found for {formatted_selected_date}")
                        else:
                            st.error("Error retrieving sample data.")

                    

                    calculation_type = st.selectbox("Select Calculation Type:", ["Select", "GBD Values", "q-Values"])

                    calculate_button_label = None
                    packing_density = None
                    updated_proportions = None
                    

                    if calculation_type == "GBD Values":
                        st.session_state["proportions"] = [0.35, 0.20, 0.15, 0.10, 0.20]
                        st.session_state["table_width"] = "auto"  # Force reset width
                        st.write("### 📊 Edit Mixing Proportions")

                        # Define initial proportions as a DataFrame (for display & editing)
                        proportions_df = pd.DataFrame({
                            "Sheet": ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"],
                            "Proportion": st.session_state["proportions"]  # Default values (sum = 1)
                        })

                        # ✅ Display an editable DataFrame
                        updated_proportions_df = st.data_editor(
                            proportions_df,
                            num_rows="fixed",  # Prevent adding/removing rows
                            column_config={"Proportion": st.column_config.NumberColumn(format="%.4f")},
                            hide_index=True
                        )

                        
                        # ✅ Extract updated proportions
                        updated_proportions = updated_proportions_df["Proportion"].tolist()
                        total_proportion = sum(updated_proportions)

                        # ✅ Check if any proportion is negative
                        if any(p < 0 for p in updated_proportions):
                            st.error("❌ Proportion values cannot be negative. Please enter positive values.")
                            calculate_button_label = None  # Prevent calculations

                        # ✅ Ensure total proportion is exactly 1 before proceeding
                        elif total_proportion != 1:
                            st.error(f"⚠️ The sum of all proportions must be exactly 1. Currently: {total_proportion:.4f}")
                            calculate_button_label = None  # Prevent calculations
                        else:
                            # ✅ Store GBD proportions separately in session state
                            st.session_state["proportions"] = updated_proportions

                            porosity_input = st.text_input("Enter Porosity (value should be between 0-1):")
                        
                            if porosity_input:
                                try:
                                    porosity = float(porosity_input.strip())
                                    if 0 <= porosity <= 1:
                                        packing_density = 1 - porosity  # ✅ Convert to Packing Density
                                        calculate_button_label = f"Calculate {calculation_type} with Porosity: {porosity}"
                                    else:
                                        st.error("❌ Please enter a valid Porosity value between 0 and 1.")
                                except ValueError:
                                    st.error("❌ Please enter a numeric value for Porosity.")

                    elif calculation_type == "q-Values":
                        q_type = st.selectbox(
                            "Select q-Value Calculation Method:",
                            ["Select", "q-value using Andreasen Eq.", "q-value using Modified Andreasen Eq.", "q-value using Double Modified Andreasen Eq."],
                            key="q_type_select",
                            on_change=lambda: st.session_state.update({"proportions": st.session_state.default_proportions.copy()})
                        )

                        if q_type == "q-value using Andreasen Eq.":
                            st.session_state["table_width"] = "auto"  # Force reset width
                            st.write("### 📊 Edit Mixing Proportions")

                            proportions_df = pd.DataFrame({
                                "Sheet": ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"],
                                "Proportion": st.session_state["proportions"]  
                            })

                            updated_proportions_df = st.data_editor(
                                proportions_df,
                                num_rows="fixed",  
                                column_config={"Proportion": st.column_config.NumberColumn(format="%.4f")},
                                hide_index=True
                            )

                            updated_proportions = updated_proportions_df["Proportion"].tolist()
                            total_proportion = sum(updated_proportions)
                            print(total_proportion)


                            # ✅ Check if any proportion is negative
                            if any(p < 0 for p in updated_proportions):
                                st.error("❌ Proportion values cannot be negative. Please enter positive values.")
                                calculate_button_label = None  # Prevent calculations

                            # ✅ Ensure total proportion is exactly 1 before proceeding
                            elif total_proportion != 1:
                                st.error(f"⚠️ The sum of all proportions must be exactly 1. Currently: {total_proportion:.4f}")
                                calculate_button_label = None  # Prevent calculations 
                            else:
                                st.session_state["proportions"] = updated_proportions
                                calculate_button_label = f"Calculate {q_type}"

                        elif q_type == "q-value using Modified Andreasen Eq.":
                            st.session_state["table_width"] = "auto"  # Force reset width
                            # ✅ Editable Proportions Table
                            st.write("### 📊 Edit Mixing Proportions")
                            # Default proportions for Modified q-Values (reset when selecting this option)
                            proportions_df = pd.DataFrame({
                                "Sheet": ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"],
                                "Proportion": st.session_state["proportions"] 
                            })

                            updated_proportions_df = st.data_editor(
                                proportions_df,
                                num_rows="fixed",  
                                column_config={"Proportion": st.column_config.NumberColumn(format="%.4f")},
                                hide_index=True
                            )

                            updated_proportions = updated_proportions_df["Proportion"].tolist()
                            total_proportion = sum(updated_proportions)

                            # ✅ Check if any proportion is negative
                            if any(p < 0 for p in updated_proportions):
                                st.error("❌ Proportion values cannot be negative. Please enter positive values.")
                                calculate_button_label = None  # Prevent calculations

                            # ✅ Ensure total proportion is exactly 1 before proceeding
                            elif total_proportion != 1:
                                st.error(f"⚠️ The sum of all proportions must be exactly 1. Currently: {total_proportion:.4f}")
                                calculate_button_label = None  # Prevent calculations
                            else:
                                st.session_state["proportions"] = updated_proportions
                                porosity_input = st.text_input("Enter Porosity for Modified Andreasen Eq. (value should be between 0-1):")
                                packing_density = None
                                if porosity_input:
                                    try:
                                        porosity = float(porosity_input.strip())
                                        if 0 <= porosity <= 1:
                                            packing_density = 1 - porosity  # ✅ Convert to Packing Density
                                            calculate_button_label = f"Calculate {q_type} with Porosity: {porosity}"
                                        else:
                                            st.error("❌ Please enter a valid Porosity value between 0 and 1.")
                                    except ValueError:
                                        st.error("❌ Please enter a numeric value for Porosity.")
                        # else:
                        #     calculate_button_label = f"Calculate {q_type}"

                        elif q_type == "q-value using Double Modified Andreasen Eq.":
                            st.session_state["table_width"] = "auto"  # Force reset width
                            st.write("### 📊 Edit Mixing Proportions")
                            # ✅ Always reset proportions to default values
                            proportions_df = pd.DataFrame({
                                "Sheet": ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"],
                                "Proportion": st.session_state["proportions"]
                            })

                            # ✅ Display an editable DataFrame
                            updated_proportions_df = st.data_editor(
                                proportions_df,
                                num_rows="fixed",  # Prevent adding/removing rows
                                column_config={"Proportion": st.column_config.NumberColumn(format="%.4f")},
                                hide_index=True
                            )

                            # ✅ Extract updated proportions
                            updated_proportions = updated_proportions_df["Proportion"].tolist()
                            total_proportion = sum(updated_proportions)

                            # ✅ Check if any proportion is negative
                            if any(p < 0 for p in updated_proportions):
                                st.error("❌ Proportion values cannot be negative. Please enter positive values.")
                                calculate_button_label = None  # Prevent calculations

                            # ✅ Ensure total proportion is exactly 1 before proceeding
                            elif total_proportion != 1:
                                st.error(f"⚠️ The sum of all proportions must be exactly 1. (Currently: {total_proportion:.4f})")
                                calculate_button_label = None  # Prevent calculations
                            else:
                                st.session_state["proportions"] = updated_proportions
                                calculate_button_label = f"Calculate {q_type}"
 

                    # ✅ Button to trigger calculations
                     
                    if calculate_button_label and st.button(calculate_button_label):
                        st.info(f"Processing: {calculate_button_label}")

                        try:
                            payload = {
                                "selected_date": formatted_selected_date,
                                "packing_density": packing_density
                            }


                            if calculation_type == "GBD Values":
                                
                                payload["updated_proportions"] = ",".join(map(str, updated_proportions))  # ✅ Send only for GBD
                                
                                response = requests.get(f"{BASE_URL}/calculate_gbd/", params=payload)

                            elif q_type == "q-value using Andreasen Eq.":
                                # payload["updated_proportions"] = ",".join(map(str, updated_proportions))
                            
                                response = requests.get(f"{BASE_URL}/calculate_q_value/", params={"selected_date": formatted_selected_date, "updated_proportions": ",".join(map(str, updated_proportions))})
                                print(response.status_code)
                            elif q_type == "q-value using Modified Andreasen Eq.":
                                
                                payload["updated_proportions"] = ",".join(map(str, updated_proportions))

                                response = requests.get(f"{BASE_URL}/calculate_q_value_modified_andreason/", params=payload)

                            
                            elif q_type == "q-value using Double Modified Andreasen Eq.":
                                # payload["updated_proportions"] = ",".join(map(str, updated_proportions_dmod))

                                response = requests.get(f"{BASE_URL}/calculate_q_value_double_modified/", params = {"selected_date": formatted_selected_date, "updated_proportions": ",".join(map(str, updated_proportions))})

                            if response.status_code == 200:
                                result = response.json()

                                if calculation_type == "GBD Values":
                                    total_volume = result.get("total_volume")
                                    specific_gravity = result.get("specific_gravity")
                                    gbd_values = result.get("gbd_values")

                                    st.write("## Results")
                                    st.write(f"**🔹 Total Volume of the Mix:** `{total_volume:.4f}`")
                                    st.write(f"**🔹 Specific Gravity of the Mix:** `{specific_gravity:.4f} g/cc`")

                                    st.write("### **GBD Values**")
                                    for density, gbd in result["gbd_values"].items():
                                        formatted_density = int(float(density) * 100)
                                        porosity_value = 100 - formatted_density  # Convert packing density to porosity
                                        st.write(f"- **GBD for {porosity_value}% Porosity:** `{gbd:.4f} g/cc`")
                                        print("GBD Done!")
                                        # st.write(f"- **GBD for {formatted_density}% Packing Density:** `{gbd:.4f} g/cc`")

                                elif q_type == "q-value using Andreasen Eq.":
                                    
                                    st.write("## Results")
                                    
                                    # ✅ Display q-values **first**
                                    q_values = result.get("q_values", [])
                                    
                                    if q_values:
                                        
                                        for q_data in q_values:
                                            st.markdown(f"####  q-value on {q_data['Date']}: **`{q_data['q-value']:.4f}`**", unsafe_allow_html=True)


                                    # ✅ Collapse the intermediate table and regression plot for cleaner UI
                                    with st.expander("📊 Show Processed Data Table & Regression Graph", expanded=False):
                                        df_intermediate = pd.DataFrame(result.get("intermediate_table", []))
                                        if not df_intermediate.empty:
                                            # ✅ Rename columns for readability
                                            df_intermediate = df_intermediate.drop(columns=["sheet_constant", "pct_poros_CPFT"], errors="ignore").rename(columns={
                                                "sheet_proportion": "Sheet Proportion",
                                                "D_value": "Particle Size (μm)",
                                                "pct_CPFT" : "%_CPFT",
                                                "pct_CPFT_interpolated": "%_CPFT (Interpolated)",
                                                "Normalized_D": "D/D_max",
                                                "Log_D/Dmax_value": "Log(D/D_max)",
                                                "Log_pct_CPFT": "Log(%_CPFT)"
                                            })

                                            st.write("### 📄 **Processed Data Table**")
                                            st.dataframe(df_intermediate)
                                        else:
                                            st.warning("⚠️ No intermediate data available.")

                                        # ✅ Plot Regression Graph
                                        st.write("### 📈 **Regression Graph**")
                                        plot_q_value_regression(df_intermediate)

                                elif q_type == "q-value using Modified Andreasen Eq.":
                                    st.write("## Results")
                                    print("In mod q value")
                                    q_values = result.get("q_values", [])
                                    if q_values:
                                        for q_data in q_values:
                                           for density, q_value in q_data.items():
                                                if density != "Date":
                                                    formatted_density = density.replace("q_", "")  # ✅ Remove "q_" prefix only
                                                    porosity_value = 100 - int(formatted_density)  # Convert packing density to porosity
                                                    st.markdown(f"#### Modified q-value on {q_data['Date']} at {porosity_value}% Porosity: **`{q_value:.4f}`**", unsafe_allow_html=True)

                                                    
                                                    # st.markdown(f"####  q-value on {q_data['Date']} at {formatted_density}% Packing Density: **`{q_value:.4f}`**", unsafe_allow_html=True)
                                    # ✅ Display Intermediate CPFT Error Table
                                    cpft_error_table = result.get("cpft_error_table", [])
                                    if cpft_error_table:
                                        with st.expander("📊 Show Processed Data Table", expanded=False):
                                            st.write("### 📊 **Processed Data Table**")
                                            df_cpft_error = pd.DataFrame(cpft_error_table)

                                            # ✅ Rename columns for better readability
                                            df_cpft_error = df_cpft_error.rename(columns={
                                                "D_value": "Particle Size (μm)",
                                                "pct_CPFT_interpolation": "%_CPFT (Interpolated)",
                                                "pct_poros_CPFT": "Actual CPFT",
                                                "calculated_CPFT": "Calculated CPFT",
                                                "absolute_error": "Absolute Error"
                                            })
                                            # ✅ Rename columns for readability
                                            # df_cpft_error = df_cpft_error.rename(columns={
                                            #     "Sheet": "Sheet Name",
                                            #     "Mesh Size": "Mesh Size",
                                            #     "Particle Size (μm)": "Particle Size (μm)",
                                            # })
                                            # ✅ Dynamically rename packing density-related columns
                                            
                                            st.dataframe(df_cpft_error)

                                elif q_type == "q-value using Double Modified Andreasen Eq.":
                                    st.write("## Results")

                                    double_mod_q_values = result.get("double_modified_q_values", [])
                                    if double_mod_q_values:
                                        for q_data in double_mod_q_values:
                                            st.markdown(f"####  Double Modified q-value on {q_data['Date']}: **`{q_data['q_value']:.4f}`**", unsafe_allow_html=True)

                                    # ✅ Display Intermediate Table
                                    with st.expander("📊 Show Processed Data Table",expanded=False):
                                        intermediate_table = result.get("intermediate_table", [])
                                        
                                        if intermediate_table:
                                            st.write("### 📊 **Processed Data Table for Regression**")
                                            df_intermediate = pd.DataFrame(intermediate_table)
                                            # ✅ Rename columns for readability
                                            df_intermediate = df_intermediate.rename(columns={
                                                "pct_poros_CPFT" : "%_CPFT",
                                                "D_value" : "Particle Size (μm)",
                                                "x_value": "Log(D - D_min) - Log(D_max - D_min)",
                                                "log_pct_CPFT": "Log(%_CPFT)"
                                                
                                            })
                                            st.dataframe(df_intermediate)  # Show the table in Streamlit

                                            # ✅ Plot Regression Graph
                                            st.write("### 📈 **Regression Graph for Double Modified q-Value**")
                                            plt.figure(figsize=(6, 4))
                                            plt.scatter(df_intermediate["Log(D - D_min) - Log(D_max - D_min)"], df_intermediate["Log(%_CPFT)"], label="Data Points", color="blue")

                                            # ✅ Perform regression and plot line
                                            slope, intercept = np.polyfit(df_intermediate["Log(D - D_min) - Log(D_max - D_min)"], df_intermediate["Log(%_CPFT)"], 1)
                                            reg_line = slope * df_intermediate["Log(D - D_min) - Log(D_max - D_min)"] + intercept
                                            plt.plot(df_intermediate["Log(D - D_min) - Log(D_max - D_min)"], reg_line, label=f"Regression Line (q = {slope:.4f})", color="red")

                                            plt.xlabel("Log(D - D_min) - Log(D_max - D_min)")
                                            plt.ylabel("Log(% CPFT)")
                                            plt.title("Double Modified q-Value Regression")
                                            plt.legend()
                                            plt.grid()
                                            st.pyplot(plt)  # Show the plot in Streamlit
                                     # ✅ Plot Regression Graph (No Intermediate Table)
                            #         df_regression = pd.DataFrame(result.get("intermediate_table", []))
                            #         if not df_regression.empty and "Log_D/Dmax" in df_regression.columns and "Log_pct_cpft" in df_regression.columns:
                            #             with st.expander("📊 Show Processed Data Table & Regression Graph", expanded=False):
                            #                 st.write("### 📄 **Processed Data Table for Regression**")
                            #                 st.dataframe(df_regression)
                                            
                            #                 # ✅ Show Regression Graph
                            #                 st.write("### 📈 **Regression Graph**")
                            #                 plot_q_value_regression(df_regression)
                            #         else:
                            #             st.warning("⚠️ No data available for regression graph.")
                            else:
                                st.error(f"❌ Error calculating {calculation_type}. Backend response: {response.text}")

                        except Exception as e:
                            st.error(f"❌ Exception: {str(e)}")
        else:
            st.error("❌ Error: Could not retrieve date range. Please check your file.")

    else:
        st.error(f"❌ Error uploading file: {response.json().get('detail', 'Unknown error')}. \n\n⚠️ Please check your file format and ensure all required sheets/columns are included.")
                            
          
                            
                