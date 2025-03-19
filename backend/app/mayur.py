import pandas as pd
from datetime import datetime
import os

### Function to load input data from excel sheets

def load_excel_sheets(file_path):
    """
    Reads all sheets from an Excel file and stores them as DataFrames in a dictionary.

    Args:
    file_path (str): Path to the Excel file.

    Returns:
    dict: A dictionary where the keys are sheet names and the values are the corresponding DataFrames.
    """
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(file_path)
        
        # Create a dictionary to store DataFrames for each sheet
        sheets_data = {}

        # Loop through each sheet and read it into a DataFrame
        for sheet_name in excel_data.sheet_names:
            sheets_data[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)

        return sheets_data

    except Exception as e:
        print(f"An error occurred while loading the Excel sheets: {e}")
        return {}



### Function to calculate the average of samples per date



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

        # Drop 'Samples No.' column if it exists
        if 'Samples No.' in PSD_df_processed.columns:
            PSD_df_processed = PSD_df_processed.drop(columns=['Samples No.'])

        # Drop the first row (assuming it's a header or irrelevant data)
        PSD_df_processed = PSD_df_processed.drop(index=0)

        # Process the 'Received Date' column if it exists
        if 'Received Date' in PSD_df_processed.columns:
            PSD_df_processed['Received Date'] = pd.to_datetime(PSD_df_processed['Received Date'], format='%d.%m.%y', errors='coerce')

        # Convert columns to numeric where applicable
        numeric_cols = PSD_df_processed.select_dtypes(include=['object']).columns
        PSD_df_processed[numeric_cols] = PSD_df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Group by 'Received Date' and calculate the average if 'Received Date' exists
        if 'Received Date' in PSD_df_processed.columns:
            avg_samples = PSD_df_processed.groupby('Received Date').mean()

            # Format the date to 'dd.mm.yy'
            avg_samples.index = avg_samples.index.strftime('%d.%m.%y')

            # Store the processed DataFrame in the dictionary
            processed_dataframes[sheet_name] = avg_samples
        else:
            print(f"Warning: 'Received Date' column missing in sheet: {sheet_name}. Skipping this sheet.")
    
    return processed_dataframes



### Function to return the GBD of the mix on a specific date 

import pandas as pd


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

        # Get the value of 'Total' and multiply it with the proportion for this sheet
        weight_of_fraction = row['Total'] * proportions.get(sheet_name, 1)  # Default to 1 if sheet name not found

        # Calculate the volume of the fraction
        volume_of_fraction = weight_of_fraction / row['Sp. gravity']

        # Add the volume of the fraction to the total volume
        total_volume_of_sample += volume_of_fraction

    # Calculate GBD as 100 / total_volume_of_sample
    density = 100 / total_volume_of_sample  # Multiply density with packing density to obtain actual GBD

    return density



### Define the function to calculate the cumulative weights of the samples for all dates.























### View the input data

# Define file path
file_path = r"D:\Sushmitha\Gyan Data\Documents_for_Particle_Packing\GBD\DataSet.xlsx"

# Load the input data
sheets_data = load_excel_sheets(file_path)

# View the input data
print(sheets_data)

### Calculate and view the average of samples per date


processed_dataframes = average_samples_per_date(sheets_data)

processed_dataframes['H(7-12)']

### Specify the date for further calculations



import pandas as pd

# Function to convert string to datetime (date only)
def get_datetime_from_string(date_str):
    # Convert the string to datetime and extract only the date part
    return pd.to_datetime(date_str, format="%d.%m.%y").normalize()  # normalize to set time as 00:00:00

# Take input from the user for the date
date_str = input("Enter the date (dd.mm.yy): ")

# Convert the input date to datetime (date only)
target_date = get_datetime_from_string(date_str)

# Display the date in dd.mm.yy format
print(f"Target date (in datetime format): {target_date.strftime('%d.%m.%y')}")
print(f"Type of target date: {type(target_date)}")  # Shows <class 'pandas._libs.tslibs.timestamps.Timestamp'>


### Determine the value of GBD for a mix on the specified date

# Loop through each DataFrame in the processed_dataframes dictionary
for sheet_name, df in processed_dataframes.items():
    # Convert the index to datetime format if it's not already in datetime64[ns] format
    if df.index.dtype != 'datetime64[ns]':
        processed_dataframes[sheet_name].index = pd.to_datetime(df.index, format='%d.%m.%y', errors='coerce')
        print(f"Index for sheet '{sheet_name}' has been converted to datetime.")
    else:
        print(f"Index for sheet '{sheet_name}' is already in datetime64[ns].")
    
    # Print the head of the DataFrame to verify the change
    # print(f"Head of DataFrame for sheet '{sheet_name}':")
    # print(df.head())
    # print("\n")  # Add a newline for better readability between sheets


def get_proportions():
    """
    This function prompts the user to input the proportions for the different sheet ranges and 
    returns them as a dictionary. The user will input five different proportions.
    """
    proportions = {}
    print("Enter proportions for the following ranges (e.g., H(7-12): 0.35):")
    ranges = ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"]

    for sheet_range in ranges:
        while True:
            user_input = input(f"Enter proportion for {sheet_range}: ")
            try:
                value = float(user_input.strip())
                if 0 <= value <= 1:
                    proportions[sheet_range] = value
                    break
                else:
                    print("Please enter a proportion between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a valid numeric value.")
    
    return proportions

def get_porosity():
    """
    This function prompts the user to input the porosity value and returns it.
    Porosity is expected to be a float.
    """
    while True:
        try:
            porosity = float(input("Enter the porosity value (e.g., 0.25 for 25% porosity): "))
            if 0 <= porosity <= 1:
                return porosity
            else:
                print("Please enter a valid porosity value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a numeric value between 0 and 1.")

# Taking input for proportions (previously sheet constants)
proportions = get_proportions()

# Taking input for porosity
porosity = get_porosity()

# Calculate packing density
packing_density = 1 - porosity
print(f"The packing density is: {packing_density}")

#density of water
density_water = 1

# function call for GBD
density = process_sheets_and_calculate_gbd(processed_dataframes, density_water, target_date, proportions)

# Calculate GBD using packing density and proportions
GBD = density * packing_density  # Calculate GBD with the packing density

# Print result
print(f"The green bulk density (GBD) of the mix is: {GBD} g/cc")
