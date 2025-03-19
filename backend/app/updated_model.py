import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Function to read the excel file 

import pandas as pd

def read_excel_file(file, required_sheets):
    """
    Reads the Excel file and returns the required sheets as DataFrames.

    Parameters:
        file (str): Path to the Excel file.
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

    sheets = {sheet: pd.read_excel(xls, sheet_name=sheet, header=2) for sheet in required_sheets}

    return sheets


# Function to clean the data

def clean_data(sheets):
    clean_data = {}

    for sheet_name, df in sheets.items():
        df.columns = df.columns.str.strip()

        # Standardize column names
        df.columns = df.columns.astype(str)  # Ensure all column names are strings
        df.rename(columns=lambda x: x.replace("+", ""), inplace=True)  # Remove "+" symbols if present
        for col in df.columns:
            if "Received" in col and "Date" in col:
                df.rename(columns={col: "Received Date"}, inplace=True)
                df['Received Date'] = pd.to_datetime(df['Received Date'], format='%d.%m.%y', errors='coerce')

            if "Total" in col or "total" in col:
                df.rename(columns={col: "Total"}, inplace=True)

        

        # Drop 'Samples No.' column if it exists
        if 'Samples No.' in df.columns:
            df = df.drop(columns=['Samples No.'])

        # Drop the first row (assuming it's a header or irrelevant data)
        df = df.drop(index=0)

        # Process the 'Received Date' column if it exists
        # if 'Received Date' in df.columns:
        #     df['Received Date'] = pd.to_datetime(df['Received Date'], format='%d.%m.%y', errors='coerce')

        if (sheet_name == "220F"):
            sheet_name = "220"

        new_sheet_name = f"H({sheet_name})"
        clean_data[new_sheet_name] = df

        # clean_data[sheet_name] = df
    return clean_data

# Function to view the sheets of the data

def view_sheets(sheets_data):
  """
  Prints the names of the sheets and displays the first few rows of each DataFrame.

  Args:
    sheets_data: A dictionary where keys are sheet names and values are their corresponding DataFrames.
  """

  for sheet_name, df in sheets_data.items():
    print(f"Sheet Name: {sheet_name}")
    print(df.head())  # Display the first few rows of the DataFrame
    print("="*100)

# Function to get available date range

def get_available_date_range(cleaned_sheets, required_sheets):
    """
    Returns the min and max available dates from the first required sheet.
    """
    main_dates = cleaned_sheets[required_sheets[0]]["Received Date"].dropna().unique()
    main_dates = pd.to_datetime(main_dates)
    return main_dates.min(), main_dates.max()



# Step 5: Match the selected date

def get_sample_data_for_date(cleaned_sheets, required_sheets, selected_date):
    """
    Finds the exact or nearest past date in each sheet based on user-selected date.
    """
    all_dates = {}
    for sheet_name in required_sheets:
        dates = cleaned_sheets[sheet_name]["Received Date"].dropna().unique()
        all_dates[sheet_name] = pd.to_datetime(dates)

    matched_dates = {}
    selected_date = pd.to_datetime(selected_date, format="%d-%m-%Y", dayfirst=True, errors="coerce")

    for sheet_name, dates in all_dates.items():
        possible_dates = dates[dates <= selected_date]
        matched_dates[sheet_name] = possible_dates.max() if len(possible_dates) > 0 else None
    
    sample_data = {}
    for sheet, df in cleaned_sheets.items():
        target_date = matched_dates.get(sheet, None)
        sample_data[sheet] = df[df["Received Date"] == target_date] if target_date is not None else None
    
    return sample_data


# Function to get average samples per date

def average_samples_per_date(sheets_data):
    """
    Averages samples per date for each sheet in the provided dictionary and returns the processed DataFrames.

    Args:
      sheets_data: A dictionary where keys are sheet names and values are their corresponding DataFrames.

    Returns:
      processed_dataframes: A dictionary where keys are sheet names and values are the processed DataFrames.
    """

    processed_dataframes = {}

    for sheet_name, PSD_df in sheets_data.items():
        # Make a copy of the DataFrame to avoid modifying the original one
        PSD_df_processed = PSD_df.copy()

       
        # Convert columns to numeric where applicable
        numeric_cols = PSD_df_processed.select_dtypes(include=['object']).columns
        PSD_df_processed[numeric_cols] = PSD_df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Group by 'Received Date' and calculate the average if 'Received Date' exists
        if 'Received Date' in PSD_df_processed.columns:
            avg_samples = PSD_df_processed.groupby('Received Date').mean()

            # Format the date to 'dd.mm.yy'
            avg_samples.index = avg_samples.index.strftime('%d.%m.%y')

            # **Remove columns with all NaN values**
            avg_samples = avg_samples.dropna(axis=1, how="all")

            # Store the processed DataFrame in the dictionary
            processed_dataframes[sheet_name] = avg_samples
        else:
            print(f"Warning: 'Received Date' column missing in sheet: {sheet_name}. Skipping this sheet.")

    return processed_dataframes

# Functionto calculate total volume and specific gravity, this returns total volume and the density (specific gravity)

def process_sheets_and_calculate_gbd(processed_dataframes, density_water, received_date, proportions):
    """
    Calculates the Green Bulk Density (GBD) based on the provided received_date or the nearest past date,
    using the processed dataframes and the given proportions.

    Args:
    processed_dataframes: Dictionary of DataFrames corresponding to the sheets.
    density_water: Density of water.
    received_date: If provided, the function uses the specified date or the nearest past date.
    proportions: Dictionary of proportions for each sheet.

    Returns:
    The calculated GBD for the entire sample.
    """

    total_volume_of_sample = 0  # Initialize total volume of the sample

    # Ensure that received_date is a pandas Timestamp (already passed as string in dd.mm.yy format)
    received_date = pd.to_datetime(received_date, format='%d.%m.%y', errors='coerce')

    for sheet_name, df in processed_dataframes.items():
        # Handle received_date or nearest past date
        df.index = pd.to_datetime(df.index, format='%d.%m.%y', errors='coerce') # Ensure index is Timestamps

        if received_date in df.index:
            nearest_date = received_date
        else:
            # Get the nearest past date (latest date before received_date)
            nearest_date = df.index[df.index <= received_date].max()

        # If there is no valid nearest date, raise an error
        if pd.isna(nearest_date):
            raise ValueError(f"No valid date found before or equal to {received_date} in sheet '{sheet_name}'")

        # Select the row corresponding to the nearest date
        row = df.loc[nearest_date]

        # Sushmitha : added a debugging statement 
        # print(f"Sheet Name: {sheet_name} Proportions : {proportions.get(sheet_name)}")
       
        # Get the value of 'Total' and multiply it with the proportion for this sheet
        weight_of_fraction = row['Total'] * proportions.get(sheet_name)  # Default to 1 if sheet name not found

        # Calculate the volume of the fraction
        volume_of_fraction = weight_of_fraction / row['Sp. gravity']

        # Add the volume of the fraction to the total volume
        total_volume_of_sample += volume_of_fraction
        # print(total_volume_of_sample)

    # Calculate GBD as 100 / total_volume_of_sample
    density = 100 / total_volume_of_sample  # Multiply density with packing density to obtain actual GBD

    return total_volume_of_sample, density

# ======================================================================================================================================================================

# Q values

# Function to calculate cumulative weights 

def calculate_cumulative_weights(sheets_data, excluded_columns):
    """
    This function calculates the cumulative weights for each sheet in the provided dictionary of DataFrames.

    Args:
    sheets_data: A dictionary where keys are sheet names and values are DataFrames.
    excluded_columns: List of columns to exclude from numeric calculations.

    Returns:
    cumulative_sheets: A dictionary with sheet names as keys and their corresponding cumulative DataFrames as values.
    """
    cumulative_sheets = {}

    for sheet_name, df in sheets_data.items():
        # Copy the DataFrame to avoid altering the original
        df.index = pd.to_datetime(df.index, format='%d.%m.%y', errors='coerce') # Ensure index is Timestamps

        df_processed = df.copy()

        # Identify numeric columns excluding specific ones
        numeric_columns = [col for col in df_processed.select_dtypes(include=['number']).columns if col not in excluded_columns]

        # Create a DataFrame for cumulative weights
        cumulative_df = pd.DataFrame()

        # Calculate cumulative weights
        for i in range(len(numeric_columns) - 1, 0, -1):
            cumulative_df[numeric_columns[i - 1]] = df_processed[numeric_columns[i:]].sum(axis=1)

        # Reverse the order of columns in the output
        cumulative_df = cumulative_df.iloc[:, ::-1]

        # Adjust column names to indicate cumulative sum
        cumulative_df.columns = [f"cumsum_{col}" for col in cumulative_df.columns]

        # Insert 'Received Date' column at the beginning if it exists
        if 'Received Date' in df_processed.columns:
            cumulative_df.insert(0, 'Received Date', df_processed['Received Date'])

        # Store the cumulative weights DataFrame in the dictionary
        cumulative_sheets[sheet_name] = cumulative_df

    return cumulative_sheets

# Function to calculate sheet constants from proportions

def get_sheet_constants_from_proportions(proportions):
    """
    This function computes the sheet constants from the given proportions using the cumulative sum approach,
    excluding sheets with zero proportions.
    It sums the remaining proportions from right to left and multiplies by 100 to convert to percentages.
    """
    # List of proportions from the get_proportions function (in decimal)
    proportion_values = list(proportions.values())

    # Sheet names (assuming you know the sheet names in advance)
    sheet_names = ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"]

    sheet_constants = {}
    
    # Calculate the sheet constants by summing from the current index to the end of the list
    for i in range(len(proportion_values) - 1):  # Only 4 sheet constants
        sheet_constant = round(sum(proportion_values[i+1:]) * 100)  # Sum from i+1 to the end, convert to percentage, and round
        sheet_constants[sheet_names[i]] = sheet_constant
    
    # The fifth sheet constant is always 0%
    sheet_constants[sheet_names[-1]] = 0
    # print("Sheet constants:", sheet_constants)
    return sheet_constants

# Function to calculate sheet cpft

def Calculate_Sheet_CPFT(cumulative_sheets, target_date, sheet_proportions, d_values):
    """
    Process the given dictionary of DataFrames to calculate weighted values for a specific target row
    or the nearest past date based on the sheet proportions, for all sheets, and include the D_value.

    Args:
        cumulative_sheets (dict): A dictionary where keys are sheet names and values are DataFrames.
        target_date (datetime): The target date for selecting the row.
        sheet_proportions (dict): A dictionary where keys are sheet names and values are proportions (already in decimal form).
        d_values (list): A list of D_value values to be added to the result DataFrame.

    Returns:
        pd.DataFrame: Consolidated DataFrame with weighted values for each sheet and column.
    """

    consolidated_data = []

    # Loop through each sheet in the cumulative_sheets dictionary
    for sheet_name, sheet_df in cumulative_sheets.items():
        sheet_df.index = pd.to_datetime(sheet_df.index, format='%d.%m.%y', errors='coerce')

        # print(f"Processing sheet: {sheet_name}")

        # Print out the first few rows of the index ('Received Date') for inspection
        # print(f"First few rows in 'Received Date' (index) for sheet {sheet_name}:")
        # print(sheet_df.index[:5])  # Access the first few values of the index

        # Ensure that the index (Received Date) is already in datetime64 format, as confirmed
        # No need to reformat, assuming the index is already datetime64[ns] after initial processing

        # Check if the target_date exists in the index
        if target_date in sheet_df.index:
            print(f"Target date {target_date} found in sheet {sheet_name}.")
            # Select the row for the target date
            selected_row = sheet_df.loc[target_date]
        else:
            print(f"Target date {target_date} not found in sheet {sheet_name}. Searching for the nearest past date.")
            # Find the nearest past date (before target_date)
            past_dates = sheet_df[sheet_df.index < target_date]

            # Check if there are any past dates
            if not past_dates.empty:
                # Sort the past dates and get the most recent one
                nearest_past_date = past_dates.index.max()
                print(f"Nearest past date for target {target_date} is {nearest_past_date}.")
                # Select the row for the nearest past date
                selected_row = sheet_df.loc[nearest_past_date]
            else:
                print(f"No past dates available for target date {target_date} in sheet {sheet_name}.")
                continue  # Skip this sheet if no past date is found

        # Get the proportion for the current sheet from the dictionary
        sheet_proportion = sheet_proportions.get(sheet_name, 1)  # Default to 1 if proportion is not found

        # Store Sheet Name, Column Name, and the weighted value (Sheet CPFT) in the consolidated data
        for column in selected_row.index:
            if column != 'Received Date':  # Skip 'Received Date' column (which is the index now)
                # Multiply by the proportion
                sheet_cpft_value = selected_row[column] * sheet_proportion
                consolidated_data.append({
                    'Sheet Name': sheet_name,
                    'Column Name': column,
                    'Sheet CPFT': sheet_cpft_value,
                    'sheet_proportion': sheet_proportion
                })

    # Create a DataFrame from the consolidated data
    result_df = pd.DataFrame(consolidated_data)
    
    # Add D_value column from the d_values list
    result_df['D_value'] = d_values[:len(result_df)]  # Add the corresponding D_values based on the DataFrame length
 
    # Filter out rows where proportion is zero
    # result_df = result_df[result_df['sheet_proportion'] != 0]

    return result_df

import pandas as pd

# Function to update dataframe based on particle size

def rearrange_mess_sizes(df):
    """
    This function adds a 'd_values' column to the DataFrame, sorts the DataFrame by 'D_value',
    removes duplicates based on specific indices, and rearranges the 'Sheet Name' column
    based on particle size ranges. Finally, it returns the updated DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to rearrange.
        
    Returns:
        pd.DataFrame: Updated DataFrame with 'Sheet Name' column rearranged based on particle size.
    """

    # Step 1: Sort the dataframe by 'D_value'
    df = df.sort_values(by='D_value', ascending=False)

    # =======Print the dataframe before removing duplicates=======
    # print("DataFrame before removing duplicates:")
    # print(df)
    
    # Remove duplicates at specific indices (e.g., 9, 13, 14, 17)
    df = df.drop(index=[9, 13, 14, 17]).reset_index(drop=True)

    # =======Print the dataframe after removing duplicates=======
    # print("\nDataFrame after removing duplicates:")
    # print(df)

    # Step 2: Define the function to assign Sheet Name based on 'D_value' (or particle size)
    def assign_sheet_name(d_value):
        if 1680 <= d_value <= 3360:
            return 'H(7-12)'
        elif 595 <= d_value <= 1410:
            return 'H(14-30)'
        elif 210 <= d_value <= 420:
            return 'H(36-70)'
        elif 105 <= d_value <= 177:
            return 'H(80-180)'
        elif d_value < 105:
            return 'H(220)'
        else:
            return 'Unknown'  # In case d_value is outside expected range

    # Step 3: Apply the function to the 'D_value' column to assign 'Sheet Name'
    df['Sheet Name'] = df['D_value'].apply(assign_sheet_name)

    # =======Print the final dataframe for verification=======
    # print("\nFinal DataFrame after assigning 'Sheet Name' based on 'D_value':")
    # print(df)

    return df



def add_columns(df, proportions, sheet_constants, packing_density):
    """
    Adds 'pct_CPFT', 'D_value', 'Normalized_D', and 'pct_poros_CPFT' columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the 'Sheet_CPFT' column.
        proportions (dict): Dictionary with sheet names and corresponding proportions.
        sheet_constants (dict): Dictionary with sheet names and corresponding constants.
        packing_density (float): A constant used to calculate the 'pct_poros_CPFT'.
        
    Returns:
        pd.DataFrame: The DataFrame with the new columns.
    """
    # Add 'proportion' column based on 'Sheet Name' using the proportions dictionary
    df['proportion'] = df['Sheet Name'].apply(lambda x: proportions.get(x, 0))  # Adding the 'proportion' column

    # Add 'sheet_constant' column based on 'Sheet Name' using the sheet_constants dictionary
    df['sheet_constant'] = df['Sheet Name'].apply(lambda x: sheet_constants.get(x, 0))

    # Initialize previous pct_CPFT to 100 (default value)
    prev_pct_CPFT = 100

    # Function to calculate pct_CPFT with exceptions
    def calculate_pct_CPFT(row):
        nonlocal prev_pct_CPFT
        
        # If proportion is 0 and Sheet CPFT is not zero, apply exceptions
        if row['proportion'] == 0 and row['Sheet CPFT'] != 0:
            if row['Sheet Name'] == 'H(7/12)':
                row['pct_CPFT'] = 100
            else:
                row['pct_CPFT'] = prev_pct_CPFT
        else:
            # Default case: Calculate pct_CPFT as Sheet CPFT + sheet constant
            row['pct_CPFT'] = row['Sheet CPFT'] + sheet_constants.get(row['Sheet Name'], 0)
        
        # Update prev_pct_CPFT if the current pct_CPFT is non-zero
        if row['pct_CPFT'] != 0:
            prev_pct_CPFT = row['pct_CPFT']
        
        return row

    # Apply the function to calculate 'pct_CPFT'
    df = df.apply(calculate_pct_CPFT, axis=1)

    # Create a new column 'pct_CPFT_interpolation' initialized to 'pct_CPFT'
    df['pct_CPFT_interpolation'] = df['pct_CPFT']

    # Define extreme points for interpolation
    x2, y2 = df.iloc[2]['pct_CPFT_interpolation'], df.iloc[2]['D_value']
    x6, y6 = df.iloc[6]['pct_CPFT_interpolation'], df.iloc[6]['D_value']
    x7, y7 = df.iloc[7]['pct_CPFT_interpolation'], df.iloc[7]['D_value']
    x9, y9 = df.iloc[9]['pct_CPFT_interpolation'], df.iloc[9]['D_value']
    x11, y11 = df.iloc[11]['pct_CPFT_interpolation'], df.iloc[11]['D_value']
    x13, y13 = df.iloc[13]['pct_CPFT_interpolation'], df.iloc[13]['D_value']
    x15, y15 = df.iloc[15]['pct_CPFT_interpolation'], df.iloc[15]['D_value']

    def interpolate_x(a, b, c, d, e):
        # Apply the linear interpolation formula
        x = a + (e - b) * (c - a) / (d - b)
        return x

    # Interpolate the values for specific rows
    for index, row in df.iterrows():
        if row['proportion'] > 0:  # Only apply interpolation when proportion is non-zero
            if index == 3:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x2, y2, x6, y6, row['D_value'])
            elif index == 4:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x2, y2, x6, y6, row['D_value'])
            elif index == 5:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x2, y2, x6, y6, row['D_value'])
            elif index == 8:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x7, y7, x9, y9, row['D_value'])
            elif index == 12:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x11, y11, x13, y13, row['D_value'])
            elif index == 14:
                df.at[index, 'pct_CPFT_interpolation'] = interpolate_x(x13, y13, x15, y15, row['D_value'])

    # Only create and insert new sample if the proportion for 'H(7/12)' is non-zero
    # Only create and insert new sample if the proportion for 'H(7/12)' is non-zero
    print("For 7/12", proportions.get('H(7/12)', 0))
    print("For 7-12", proportions.get('H(7-12)', 0))
    if proportions.get('H(7-12)', 0) != 0:
        new_sample = pd.DataFrame({
            'Sheet Name': ['H(7-12)'],
            'Column Name': ['cumsum_1'],
            'Sheet CPFT': [100],
            'sheet_proportion':[proportions.get('H(7-12)', 0)],
            'D_value': [3500],
            'proportion': [proportions.get('H(7-12)', 0)],
            'sheet_constant': [100],
            'pct_CPFT': [100],
            'pct_CPFT_interpolation': [100]
            
                # Proportion for the new sample
        })
        print(new_sample)
        # Insert the new sample at the beginning (index 0)
        df = pd.concat([new_sample, df], ignore_index=True)
        print(df)

    if packing_density is None:
        packing_density = 0.85
    # Add 'pct_poros_CPFT' column by multiplying 'pct_CPFT_interpolation' with 'packing_density'
    df['pct_poros_CPFT'] = df['pct_CPFT_interpolation'] * packing_density
        
    # Add 'Normalized_D' column
    df['Normalized_D'] = df['D_value'] / df['D_value'].max()

    # Filter the DataFrame to return only rows where both 'Sheet CPFT' and 'proportion' are non-zero
    df_filtered = df[df['sheet_proportion'] != 0]

    # Reset the index of the filtered DataFrame
    df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered


# Function to predict q value

def q_value_prediction(sorted_df, selected_date):
    log_q_values_data = []

    # Extract the relevant columns
    x = sorted_df['Log_D/Dmax_value'].values  # Independent variable (x-axis)
    y = sorted_df['Log_pct_CPFT'].values  # Dependent variable for the regression curve

    # Perform linear regression using scipy
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    log_q_values_data.append({
            "Date": selected_date,
            "q-value": round(slope, 4),
            "r-squared": round(r_value**2, 4)  # ✅ Added R² value for accuracy check
        })


    return pd.DataFrame(log_q_values_data)


# ======================================================================================================================================================================

# Modified Q values

# Function to optimize q value for modified andreasen equation

def optimize_q(df, D_col, pct_CPFT_col):
    """
    Optimize a single q-value using the Modified Andreasen equation.
    
    Parameters:
        df (DataFrame): Input DataFrame containing particle size and CPFT data.
        D_col (str): Column name for particle size (D_value).
        pct_CPFT_col (str): Column name for CPFT data (e.g., pct_poros_CPFT).    
        
    Returns:
        float: Optimal q-value for the specified packing density CPFT.
    """
    # Extract D_values and pct_CPFT as arrays
    D_values = df[D_col].values
    pct_CPFT = df[pct_CPFT_col].values / 100  # Convert to fractions

    # Define the Modified Andreasen equation
    def andreasen_eq(D, q, D_min, D_max):
        return (D**q - D_min**q) / (D_max**q - D_min**q)

    # Define D_min and D_max
    D_min = D_values.min()
    D_max = D_values.max()

    # Optimize q for the specified packing density (single column)
    params, _ = curve_fit(
        lambda D, q: andreasen_eq(D, q, D_min, D_max),
        D_values,
        pct_CPFT,
        bounds=(0.1, [0.5]),
        p0=[0.3]
    )

    # Return the optimal q-value
    return params[0]

# Function to predict CPFT and error.

def calculate_errors_and_mae(df, D_col, pct_CPFT_col, q):
    """
    Calculate predicted CPFT, absolute error, and mean absolute error for a single optimal q-value.
    
    Parameters:
        df (DataFrame): Input DataFrame containing particle size and CPFT data.
        D_col (str): Column name for particle size (D_value).
        pct_CPFT_col (str): Column name for actual CPFT (e.g., pct_poros_CPFT).
        q (float): Optimal q-value for the specified packing density.
    
    Returns:
        DataFrame: Updated DataFrame with predicted CPFT and absolute error.
        float: Mean Absolute Error (MAE) for the specified packing density.
    """
    # Define the Modified Andreasen equation
    def calculate_cpft(D, q, D_min, D_max):
        return (D**q - D_min**q) / (D_max**q - D_min**q)

    # Define D_min and D_max
    D_values = df[D_col].values
    D_min = D_values.min()
    D_max = D_values.max()

    # Calculate predicted CPFT for the specified q-value
    df['calculated_CPFT'] = df[D_col].apply(lambda D: calculate_cpft(D, q, D_min, D_max) * 100)

    # Calculate absolute error for the specified packing density
    df['absolute_error'] = abs(df[pct_CPFT_col] - df['calculated_CPFT'])

    # Calculate Mean Absolute Error (MAE)
    mae = df['absolute_error'].mean()

    return df, mae




# ======================================================================================================================================================================

# Double Modified Q values


def calculate_Q_value_and_plot(sorted_df, pct_CPFT_col='pct_poros_CPFT'):
    """
    Calculate the optimal Q-value for a single packing density and plot the results.
    
    Parameters:
        sorted_df (DataFrame): Input DataFrame containing particle size and CPFT data.
        pct_CPFT_col (str): Column name for CPFT data (e.g., pct_poros_CPFT).
    
    Returns:
        float: The optimal Q-value.
        DataFrame: Modified DataFrame with additional calculations.
    """
    # Step 1: Create the modified DataFrame with the necessary columns
    double_modified_df = sorted_df[['Sheet Name', 'Column Name', pct_CPFT_col, 'D_value']]
  


    # Step 2: Calculate Dmin and Dmax
    D_min = double_modified_df['D_value'].min()
    D_max = double_modified_df['D_value'].max()

    # Step 3: Remove the last row
    double_modified_df = double_modified_df.iloc[:-1]

    # Step 4: Calculate log(pct_CPFT)
    double_modified_df['log_pct_CPFT'] = np.log(double_modified_df[pct_CPFT_col])

    # Step 5: Compute x values
    double_modified_df['x_value'] = np.log(double_modified_df['D_value'] - D_min) - np.log(D_max - D_min)

    # Step 6: Define y values (log of CPFT)
    y = double_modified_df['log_pct_CPFT']
    x = double_modified_df['x_value']

    # Step 7: Perform linear regression
    slope, intercept, _, _, _ = linregress(x, y)

    # The Q-value is the slope of the regression line
    Q_value = slope

    # # Print the result
    # print(f"The optimal Q-value using the Modified Andreasen Equation is: {Q_value}")

    # # Step 8: Plotting the regression line
    # plt.figure(figsize=(10, 6))

    # # Plot the data points
    # plt.scatter(x, y, label="log(pct_CPFT)", color='blue', marker='o')

    # # Plot the regression line
    # plt.plot(x, slope * x + intercept, label=f"Regression line (Q={Q_value:.3f})", color='blue', linestyle='--')

    # # Customize the plot
    # plt.xlabel('log(D_value - D_min) - log(D_max - D_min)')
    # plt.ylabel('log(CPFT)')
    # plt.title('Linear Regression of log(pct_CPFT) vs log(D_value)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Return the modified DataFrame if needed
    return Q_value, double_modified_df




