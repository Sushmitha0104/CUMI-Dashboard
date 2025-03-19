import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.optimize import differential_evolution

# Step 1: Reading Excel files and defining sheets

def read_excel_file(file, required_sheets):
    """
    Reads the Excel file and returns the required sheets as DataFrames.

    Parameters:
        file_path (str): Path to the Excel file.
        required_sheets (list): List of sheet names to be read.

    Returns:
        dict: Dictionary of DataFrames for each required sheet.
    """
    xls = pd.ExcelFile(file)
    available_sheets = xls.sheet_names

    # Check if all required sheets are present
    missing_sheets = [sheet for sheet in required_sheets if sheet not in available_sheets]
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {', '.join(missing_sheets)}. Please upload a valid file.")

    sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in required_sheets}
    return sheets

# Step 2: Cleaning data

def clean_data(sheets, column_to_drop):
    """
    Cleans the data by removing headers, dropping unnecessary rows and columns.

    Parameters:
        sheets (dict): Dictionary of DataFrames for each sheet.
        column_to_drop (list): List of column names to be dropped if present.

    Returns:
        dict: Cleaned DataFrames for each sheet.
    """
    clean_data = {}
    for sheet_name, df in sheets.items():
        # Drop the first row (header)
        df = df.drop(index=0)

        # Set the column names using the first row (now the header)
        df.columns = df.iloc[0].astype(str)

        # Drop the first row again and reset the index
        df = df[1:].reset_index(drop=True)

        # Drop rows containing "Specification"
        df = df[~df.iloc[:, 0].astype(str).str.contains("Specification", na=False)]

        # Drop unnecessary columns
        df = df.drop(columns=[col for col in column_to_drop if col in df.columns], errors="ignore")

        clean_data[sheet_name] = df
        
    return clean_data

# Step 3: Standardizing column names and date conversion

def standardize_column_names_and_convert_dates(clean_data):
    """
    Standardizes column names and converts date strings to datetime objects.

    Parameters:
        clean_data (dict): Dictionary of cleaned DataFrames for each sheet.

    Returns:
        dict: DataFrames with standardized column names and converted date columns.
    """
    for sheet_name, df in clean_data.items():
        df.columns = df.columns.astype(str)  # Convert all column names to strings
        for col in df.columns:
            if "Received" in col and "Date" in col:
                df.rename(columns={col: "Received Date"}, inplace=True)
        if "Received Date" in df.columns:
            df["Received Date"] = pd.to_datetime(df["Received Date"], format="%d.%m.%y", errors="coerce")
    return clean_data

# Step 4: Get available date range

def get_available_date_range(standardized_sheets, required_sheets):
    """
    Returns the min and max available dates from the first required sheet.
    """
    main_dates = standardized_sheets[required_sheets[0]]["Received Date"].dropna().unique()
    main_dates = pd.to_datetime(main_dates)
    return main_dates.min(), main_dates.max()


# Step 5: Match the selected date

def get_sample_data_for_date(standardized_sheets, required_sheets, selected_date):
    """
    Finds the exact or nearest past date in each sheet based on user-selected date.
    """
    all_dates = {}
    for sheet_name in required_sheets:
        dates = standardized_sheets[sheet_name]["Received Date"].dropna().unique()
        all_dates[sheet_name] = pd.to_datetime(dates)

    matched_dates = {}
    selected_date = pd.to_datetime(selected_date, format="%d-%m-%Y", dayfirst=True, errors="coerce")

    for sheet_name, dates in all_dates.items():
        possible_dates = dates[dates <= selected_date]
        matched_dates[sheet_name] = possible_dates.max() if len(possible_dates) > 0 else None
    
    sample_data = {}
    for sheet, df in standardized_sheets.items():
        target_date = matched_dates.get(sheet, None)
        sample_data[sheet] = df[df["Received Date"] == target_date] if target_date is not None else None
    
    return sample_data




# Step 6: Convert Sample Data to Numeric & Compute Averages
def convert_to_numeric_and_calculate_average(sample_data):
    averages = {}
    for sheet_name, df in sample_data.items():
        if df is None or df.empty:
            averages[sheet_name] = None
            continue
        numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")
        averages[sheet_name] = numeric_df.mean()
    return averages

# Step 7: Compute Volume
def calculate_volume(average_data, proportions):
    volume_data = {}
    for sheet, df in average_data.items():
        if df is None:
            continue
        try:
            if isinstance(df, (pd.Series, pd.DataFrame)):
                sg_value = df.iloc[-1] if isinstance(df, pd.Series) else df.iloc[-1].values[0]
            else:
                sg_value = df  # Directly use the float value
            volume = proportions.get(sheet, 0) / sg_value if sg_value else None
            volume_data[sheet] = volume
        except (IndexError, ValueError, KeyError):
            volume_data[sheet] = None  # Handle missing values gracefully
    return volume_data

# Step 8: Compute Total Volume
def sum_volumes(volume_data):
    return sum([v for v in volume_data.values() if v is not None])

# Step 9: Compute Specific Gravity of the Mix
def calculate_sg_mix(total_volume):
    return round(1 / total_volume, 2) if total_volume else None

# Step 10: Compute GBD Values Dynamically
def calculate_gbd_values(sg_mix, packing_density_input):
    """
    Computes GBD values dynamically based on user-specified packing densities.
    The user can enter a single density or multiple densities (comma-separated).
    
    Parameters:
        sg_mix (float): Specific gravity of the mix.
        packing_density_input (str): Packing densities provided by the user (single value or list).
        
    Returns:
        pd.DataFrame: A table with computed GBD values for each provided density.
    """
    if sg_mix is None:
        return None  # Return None if SG mix calculation failed

    try:
        # ✅ If input is a **string**, convert it into a list of floats
        if isinstance(packing_density_input, str):
            packing_densities = [float(pd.strip()) for pd in packing_density_input.split(",")]

        # ✅ If input is already a **list**, ensure all elements are floats
        elif isinstance(packing_density_input, list):
            packing_densities = [float(pd) for pd in packing_density_input]

        # ✅ If input is a single float, wrap it in a list
        elif isinstance(packing_density_input, (int, float)):
            packing_densities = [float(packing_density_input)]

        else:
            raise ValueError("Invalid packing density input format.")

    except ValueError as e:
        raise ValueError(f"Invalid packing density input: {str(e)}. Please enter numeric values separated by commas.")

    # ✅ Calculate GBD for each density
    gbd_data = {"Packing Density": [], "GBD (g/cc)": []}
    for density in packing_densities:
        try:
            gbd_value = round(sg_mix * density, 4)
            gbd_data["Packing Density"].append(density)
            gbd_data["GBD (g/cc)"].append(gbd_value)
        except Exception as e:
            print(f"Error calculating GBD for density {density}: {str(e)}")

    return pd.DataFrame(gbd_data)

def convert_to_numeric_and_calculate_average_for_q_values(sample_data, selected_date):
    averages = {}
    averages[selected_date] = {}

    for sheet, df in sample_data.items():
        if df is None or df.empty:
            averages[selected_date][sheet] = None
            continue

        numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.dropna(axis=1, how="all")

        # ✅ Convert to DataFrame (Fix issue where it was a Series)
        avg_df = numeric_df.mean().to_frame(name="Mean Values")

        averages[selected_date][sheet] = avg_df  # ✅ Store DataFrame instead of Series

    return averages


def get_sheet_constants_from_proportions(proportions):
    """
    Computes sheet constants dynamically based on user-updated proportions.
    """
    proportion_values = list(proportions.values())
    sheet_names = ["7-12", "14-30", "36-70", "80-180", "220F"]

    sheet_constants = {}
    for i in range(len(proportion_values) - 1):  
        sheet_constant = round(sum(proportion_values[i+1:]) * 100)  
        sheet_constants[sheet_names[i]] = sheet_constant

    sheet_constants[sheet_names[-1]] = 0  
    return sheet_constants



# Step 11: Calculate reverse cumulative sum
def drop_last_3_and_reverse_cumsum(average_data, selected_date):
    weights = {}
    cum_sum = {}

    date = selected_date
    sheet_data = average_data.get(date, {})

    weights[date] = {}
    cum_sum[date] = {}

    for sheet, df in sheet_data.items():
        if df is None or not isinstance(df, pd.DataFrame):
            weights[date][sheet] = None
            cum_sum[date][sheet] = None
            continue

        # ✅ Ensure df is a DataFrame before dropping rows
        df = df.iloc[:-3] if df.shape[0] > 3 else df

        weights[date][sheet] = df

        df = df.to_frame() if isinstance(df, pd.Series) else df
        mesh_sizes = df.index.values
        weight_values = df.iloc[:, 0].values

        reverse_cumsum_weights = np.cumsum(weight_values[::-1])[::-1]
        reverse_cumsum_weights = reverse_cumsum_weights[1:]

        mesh_sizes = mesh_sizes[:-1]

        df_result = pd.DataFrame({
            'Mesh Size': mesh_sizes,
            'Cumulative Sum': reverse_cumsum_weights
        })

        cum_sum[date][sheet] = df_result

    return weights, cum_sum


# Step 12: Calculate cpft
def calculate_cpft(cum_sum, multipliers, selected_date):
    """
    Calculates CPFT values for the selected date.
    """
    date = selected_date  # ✅ Process only selected date
    sheet_data = cum_sum.get(date, {})

    cpft = {}
    cpft[date] = {}

    for sheet, df in sheet_data.items():
        if df is not None:
            multiplier = multipliers.get(sheet)
            if multiplier:
                df['cpft'] = df['Cumulative Sum'] * multiplier
                cpft[date][sheet] = df

    return cpft

# Step -13: Calculate pct_cpft values
def calculate_pct_cpft(cpft, sheet_constants, selected_date):
    """
    Calculates Percentage CPFT for selected date.
    """
    date = selected_date
    sheet_data = cpft.get(date, {})

    pct_cpft = {}
    pct_cpft[date] = {}

    for sheet, df in sheet_data.items():
        if df is not None:
            constant = sheet_constants.get(sheet, 0)
            df['pct_cpft'] = df['cpft'] + constant
            df['Sheet'] = sheet
            df_result = df[['Sheet', 'Mesh Size', 'cpft', 'pct_cpft']]
            pct_cpft[date][sheet] = df_result

    return pct_cpft


# step -14: Merge into single df
def merge_pct_cpft_into_df(mesh_size_to_particle_size, pct_cpft, valid_sheets):
    final_df = {}

    for date, sheet_data in pct_cpft.items():
        valid_dfs = [df for sheet, df in sheet_data.items() if df is not None and sheet in valid_sheets]


        if not valid_dfs:  # ✅ Prevent "No objects to concatenate"
            raise ValueError(f"No valid data for merging on {date}")

        final_df[date] = pd.concat(valid_dfs, ignore_index=True)
        final_df[date]['Particle Size'] = final_df[date]['Mesh Size'].map(mesh_size_to_particle_size)

    return final_df


# Step-15 : Interpolation
def calculate_interpolated_values(final_df, valid_sheets):
    """
    Calculates interpolated pct_cpft values for specified rows using linear interpolation.
    """
    for date, df in final_df.items():
        
         # ✅ Filter df to only include valid sheets
        df = df[df["Sheet"].isin(valid_sheets)]

        # Initialize with original values
        df['pct_cpft_interpolated'] = df['pct_cpft'] 

        # ✅ Identify rows where `pct_cpft` is missing (instead of using fixed indices)
        missing_rows = df[df['pct_cpft'].isna()].index.tolist() 

        # ✅ Remove missing rows from valid data
        valid_rows = df.drop(index=missing_rows)

        for i in missing_rows:
            if i not in df.index:  # Ensure index is within bounds
                continue  

            current_particle_size = df.loc[i, 'Particle Size']

            # Select nearest smaller and larger values from valid_rows
            small_rows = valid_rows[valid_rows['Particle Size'] < current_particle_size]
            large_rows = valid_rows[valid_rows['Particle Size'] > current_particle_size]

            if not small_rows.empty and not large_rows.empty:
                # Find nearest smaller and larger indices
                nearest_small_index = small_rows['Particle Size'].sub(current_particle_size).abs().idxmin()
                nearest_large_index = large_rows['Particle Size'].sub(current_particle_size).abs().idxmin()

                # Extract data for interpolation
                x_vals = [valid_rows.loc[nearest_small_index, 'Particle Size'], 
                          valid_rows.loc[nearest_large_index, 'Particle Size']]
                y_vals = [valid_rows.loc[nearest_small_index, 'pct_cpft'], 
                          valid_rows.loc[nearest_large_index, 'pct_cpft']]

                
                # Create interpolation function
                interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate')

                # Get interpolated value
                df.at[i, 'pct_cpft_interpolated'] = np.round(interp_func(current_particle_size), 3)
    return final_df

#  Step-16: Dropping duplicate rows
def drop_and_reset_indices(final_df, valid_sheets):
    """
    Drops specified rows and resets index for each date's DataFrame in final_df.
    """
    for date, df in final_df.items():

        # ✅ Filter df to only include valid sheets
        df = df[df["Sheet"].isin(valid_sheets)]

         # ✅ Identify duplicate rows dynamically
        duplicate_rows = df[df.duplicated(subset=['Particle Size'], keep='first')].index

        # ✅ Drop duplicate rows dynamically
        df = df.drop(duplicate_rows, axis=0)

        # ✅ Reset index
        final_df[date] = df.reset_index(drop=True)
    
    return final_df

# Step-17: Normalize particle size
def normalize_particle_size(final_df):
    """
    Normalizes the Particle Size by dividing by the maximum Particle Size for each date.
    """
    for date, df in final_df.items():
      
        
        d_max = df["Particle Size"].max()
        df['Normalized_D'] = df['Particle Size'] / d_max
      
    
    return final_df

# Step-18: q_value prediction

def q_value_prediction(final_df):
    log_q_values_data = []

    for date, df in final_df.items():
        if "Normalized_D" not in df.columns or "pct_cpft_interpolated" not in df.columns:
            continue

        df["Log_D/Dmax"] = np.log(df["Normalized_D"])
        df["Log_pct_cpft"] = np.log(df["pct_cpft_interpolated"])

        slope, intercept, r_value, p_value, std_err = linregress(df["Log_D/Dmax"], df["Log_pct_cpft"])

        log_q_values_data.append({
            "Date": date,
            "q-value": round(slope, 4),
            "r-squared": round(r_value**2, 4)  # ✅ Added R² value for accuracy check
        })

    return pd.DataFrame(log_q_values_data)

def prepare_mod_q_values(final_df, selected_date, packing_densities):
    """
    Extracts relevant columns for Modified Andreasen q-value calculations,
    but only for the selected date. Also calculates CPFT for user-defined packing densities.
    """
    if selected_date not in final_df:
        raise ValueError(f"No valid data found for {selected_date} in final_df.")

    df = final_df[selected_date]

    if df.empty:
        raise ValueError(f"final_df for {selected_date} is empty.")

    # ✅ Extract the required columns
    mod_q = {
        selected_date: df[["Sheet", "Mesh Size", "pct_cpft_interpolated", "Particle Size"]].rename(
            columns={
                "Sheet": "Sheet",
                "Mesh Size": "Mesh Size",
                "pct_cpft_interpolated": "pct_CPFT",
                "Particle Size": "Particle Size (μm)"
            }
        )
    }

    # ✅ Compute CPFT values dynamically based on user-provided packing densities
    for density in packing_densities:
        density_col = f"pct_{int(density * 100)}_poros_CPFT"
        mod_q[selected_date][density_col] = mod_q[selected_date]["pct_CPFT"] * density  # 🔥 Compute dynamically


    return mod_q



# Corrected Modified Andreason Equation
def modified_andreason_eq(q, D, D_min, D_max):
    """
    Computes the cumulative percent finer than (CPFT) based on the modified Andreasen equation.
    """
    return ((D ** q - D_min ** q) / (D_max ** q - D_min ** q)) * 100

# Objective Function: Squared Differences for Better Penalization
def objective_diff(q, particle_sizes, target_cpft):
    """
    Objective function for optimization: minimizes squared difference 
    between calculated and target CPFT values.
    """
    calculated_cpft = modified_andreason_eq(q, particle_sizes, particle_sizes.min(), particle_sizes.max())
    return np.sum((calculated_cpft - target_cpft) ** 2)  # Squared differences

# Differential Evolution Optimization
def optimize_q_de(particle_sizes, cpft_values):
    """
    Uses differential evolution to find the optimal q-value by minimizing 
    the squared difference between calculated and target CPFT values.
    """
    result = differential_evolution(
        objective_diff,
        bounds=[(0.20, 0.60)],  # Wider bounds for more flexibility
        args=(particle_sizes, cpft_values),
        strategy='best1bin',
        maxiter=2000,  # Increased iterations for better convergence
        tol=1e-10  # Tighter tolerance for more precision
    )
    return result.x[0]  # Extract optimal q-value

# Function to Predict q-values for User-Defined Packing Densities
def predict_mod_q_values(mod_q, packing_densities):
    """
    Computes q-values for dynamically entered packing densities using Modified Andreasen Eq.
    
    Parameters:
        mod_q (dict): Dictionary containing data for each date.
        packing_densities (list): List of user-defined packing densities.

    Returns:
        pd.DataFrame: DataFrame containing optimized q-values for each density.
    """
    optimized_q_list = []  # Collect results as a list of dictionaries

    for date, df in mod_q.items():
        # Extract relevant data
        particle_sizes = df['Particle Size (μm)'].values
        q_results = {'Date': date}

        for density in packing_densities:
            cpft_values = df['pct_CPFT'] * density  # Scale CPFT values by density
            q_value = optimize_q_de(particle_sizes, cpft_values)
            q_results[f'q_{int(density * 100)}'] = np.round(q_value, 4)  # Store q-value
        
        optimized_q_list.append(q_results)

    # Convert the list of dictionaries into a DataFrame
    optimized_q_df = pd.DataFrame(optimized_q_list)
    
    return optimized_q_df


# Function to Calculate CPFT, Absolute Error, and Create Dictionary of DataFrames
def calculate_cpft_error_dict(mod_q, q_values, packing_densities):
    """
    Computes CPFT, Absolute Error, and returns a dictionary of DataFrames.
    Adjusted to work with **dynamic user-defined packing densities.**
    
    Parameters:
        mod_q (dict): Processed data for each date.
        q_values (dict): Predicted q-values for each date.
        packing_densities (list): User-defined packing densities.
    
    Returns:
        dict: A dictionary where each date maps to a DataFrame of CPFT errors.
    """
    result_dict = {}

    q_values_dict = {entry["Date"]: entry for entry in q_values}  # Convert list to dict


    for date, df in mod_q.items():
        date_str = str(date)  
        particle_sizes = df['Particle Size (μm)'].values
        
        # Use the correct column name 'Mesh Size' (ensure no spaces)
        mesh_size = df['Mesh Size'].values if 'Mesh Size' in df.columns else ['N/A'] * len(particle_sizes)

        # Prepare dictionary to store computed values
        calculated_cpft = {}
        absolute_error = {}

        for density in packing_densities:
            density_col = f"pct_{int(density * 100)}_poros_CPFT"

            if density_col not in df.columns:
                raise ValueError(f"Missing column: {density_col} in DataFrame for {date}")

            # Get actual CPFT values
            actual_cpft = df[density_col].values

             # ✅ Lookup q-value from dictionary
            if date_str not in q_values_dict:
                raise ValueError(f"Missing q-values for {date_str}")
            
            q_value_key = f"q_{int(density * 100)}"
            if q_value_key not in q_values_dict[date_str]:
                raise ValueError(f"Missing {q_value_key} for {date_str}")

            q_value = q_values_dict[date_str][q_value_key]

            
            # Calculate CPFT using predicted q-value
            calculated_cpft[density] = modified_andreason_eq(q_value, particle_sizes, particle_sizes.min(), particle_sizes.max())

            # Calculate Absolute Error
            absolute_error[density] = np.abs(actual_cpft - calculated_cpft[density])

        # Create DataFrame for this date
        date_df = pd.DataFrame({
            'Sheet': df["Sheet"].values,  # ✅ Corrected: Use actual sheet names
            'Mesh Size': mesh_size,
            'Particle Size (μm)': particle_sizes,
        })

        for density in packing_densities:
            date_df[f'pct_{int(density * 100)}_poros_CPFT'] = df[f'pct_{int(density * 100)}_poros_CPFT']
            date_df[f'calculated_CPFT_{int(density * 100)}'] = calculated_cpft[density]
            date_df[f'absolute_error_{int(density * 100)}'] = absolute_error[density]

        # ✅ Store using **string format** for `date`
        result_dict[date_str] = date_df

    return result_dict



def prepare_mod_q_values_double_modified(final_df, selected_date):
    """
    Extracts relevant columns for **Double Modified Andreasen q-value calculations**,
    ensuring that the required columns exist **without applying packing density**.
    """
    if selected_date not in final_df:
        raise ValueError(f"No valid data found for {selected_date} in final_df.")

    # ✅ Convert dictionary of DataFrames into a single DataFrame
    df = final_df[selected_date]
    if isinstance(df, dict):  
        df = pd.concat(df.values(), ignore_index=True)  # Merge all sheets into one DataFrame

    if df.empty:
        raise ValueError(f"final_df for {selected_date} is empty.")

    # ✅ Ensure 'pct_cpft_interpolated' exists
    if "pct_cpft_interpolated" not in df.columns:
        raise ValueError(f"Missing column: 'pct_cpft_interpolated' in final_df[{selected_date}]")

    # ✅ Extract the required columns **without packing density modifications**
    mod_q = {
        selected_date: df[["Sheet", "Mesh Size", "pct_cpft_interpolated", "Particle Size"]].rename(
            columns={
                "Sheet": "Sheet",
                "Mesh Size": "Mesh Size",
                "pct_cpft_interpolated": "pct_cpft_interpolated",  # ✅ No packing density applied
                "Particle Size": "Particle Size (μm)"
            }
        )
    }

    return mod_q





def double_modified_q_values(mod_q, selected_date):
    """
    Computes Double Modified Andreasen q-values **only for the selected date**.
    """
    if selected_date not in mod_q:
        raise ValueError(f"No valid data found for {selected_date} in mod_q.")

    df = mod_q[selected_date]

    if df.empty:
        raise ValueError(f"df is empty for {selected_date} in mode_q.")

    D_min = df["Particle Size (μm)"].min()
    D_max = df["Particle Size (μm)"].max()

    # ✅ Use interpolated CPFT values instead of applying a packing density
    df_new = df[df["Particle Size (μm)"] > D_min].copy()

    # **Fix 1️⃣:** Ensure these columns exist before transformation
    if "pct_cpft_interpolated" not in df_new.columns:
        raise ValueError(f"Missing required column 'pct_cpft_interpolated' for {selected_date}")

    df_new["Log_D/Dmax"] = np.log(df_new["Particle Size (μm)"] - D_min) - np.log(D_max - D_min)
    df_new["Log_pct_cpft"] = np.log(df_new["pct_cpft_interpolated"])

    # **Fix 2️⃣:** Check for NaN or Infinite values before regression
    if df_new[["Log_D/Dmax", "Log_pct_cpft"]].isnull().values.any():
        raise ValueError(f"NaN values found in Log_D/Dmax or Log_pct_cpft for {selected_date}")

      # ✅ Debugging print statements
    print(f"✅ Debugging: df_new after transformation for {selected_date}:")
    print(df_new.head())

    regression_result = linregress(df_new["Log_D/Dmax"], df_new["Log_pct_cpft"])

    # ✅ Store q-value for the selected date
    double_modified_q_df = pd.DataFrame([{
        "Date": selected_date,
        "Double_modified_q": np.round(regression_result.slope, 4)
    }])

    # ✅ Ensure df_new is not empty before returning
    if df_new.empty:
        raise ValueError(f"df_new is empty after processing for {selected_date}")

    # ✅ Store intermediate table for the selected date
    intermediate_table = df_new[[
        "Sheet", "Mesh Size", "Particle Size (μm)", "pct_cpft_interpolated", "Log_D/Dmax", "Log_pct_cpft"
    ]]

    print(f"✅ Debugging: Intermediate table generated for {selected_date}:")
    print(intermediate_table.head())

    return double_modified_q_df, intermediate_table   # ✅ Return both q-value and intermediate table