from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi.responses import JSONResponse 
from io import BytesIO
from app.updated_model import (
    read_excel_file, clean_data, view_sheets,
    get_available_date_range, get_sample_data_for_date,
    average_samples_per_date, process_sheets_and_calculate_gbd,
    calculate_cumulative_weights, get_sheet_constants_from_proportions, Calculate_Sheet_CPFT, rearrange_mess_sizes, add_columns, q_value_prediction,
    optimize_q, calculate_errors_and_mae,
    calculate_Q_value_and_plot
)

app = FastAPI()
# ✅ Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

@app.get("/ping")
async def ping():
    return {"status": "alive"}

# Define the required sheets and column drop list
required_sheets = ["7-12", "14-30", "36-70", "80-180", "220F"]
updated_sheets = ["H(7-12)", "H(14-30)", "H(36-70)", "H(80-180)", "H(220)"]
d_values = [
    3360, 2380, 2000, 1410, 1190, 1680, 1000, 595, 420, 595, 
    297, 210, 149, 297, 210, 177, 125, 63, 105, 
    74, 63, 53, 44
]

# Store uploaded file in memory for further processing
file_storage = {}

cached_final_df = {}
cached_q_values = {}
global_cache={}

# Endpoint to upload the Excel file

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_obj = BytesIO(contents)
        file_obj.seek(0)
        file_storage["file"] = file_obj

        # Check if it's an Excel file
        if file.filename.endswith(".xlsx"):
            sheets = read_excel_file(file_obj, required_sheets)

            # ✅ Validate that required sheets exist
            missing_sheets = [sheet for sheet in required_sheets if sheet not in sheets]
            if missing_sheets:
                raise HTTPException(status_code=400, detail=f"Missing required sheets: {missing_sheets}")
            cleaned_sheets = clean_data(sheets)

        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file_obj)

            # ✅ Validate that required columns exist
            expected_columns = ["Column1", "Column2", "Column3"]  # Replace with actual required columns
            missing_columns = [col for col in expected_columns if col not in df.columns]

            if missing_columns:
                raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
            cleaned_sheets = clean_data(sheets)

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV or Excel file.")
   
        
        min_date, max_date = get_available_date_range(cleaned_sheets, updated_sheets)
        
        return {"message": "File uploaded successfully", "date_range": [str(min_date.date()), str(max_date.date())]}
     
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Endpoint to get the sample data for a selected date

@app.get("/get_sample_data/")
async def get_sample_data(selected_date: str = Query(..., description="Selected date from user")):
    try:
        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)  # Reset file pointer before reading
        
        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets)
        

        # Get the available date range
        min_date, max_date = get_available_date_range(cleaned_sheets, updated_sheets)

        # ✅ Convert the user-selected date to `datetime`
        selected_date_obj = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")

        # ✅ Check if the selected date is within the range
        if selected_date_obj < min_date:
            return {"error": f"Please select a date within the range: {min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')}"}
        elif selected_date_obj > max_date:
            selected_date_obj = max_date  # Auto-select nearest past date

        sample_data = get_sample_data_for_date(cleaned_sheets, updated_sheets, selected_date)
        # sample_data = get_sample_data_for_date(standardized_sheets, required_sheets, selected_date)

        # ✅ Ensure sample data is not empty before proceeding
        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise ValueError(f"No sample data found for the selected date: {selected_date}")


        return {
            "message": "Sample data retrieved",
            "selected_date": selected_date_obj.strftime("%d-%m-%Y"),  # ✅ Return date in `dd-mm-yyyy` format
            "sample_data": {k: v.to_dict(orient="records") for k, v in sample_data.items() if v is not None}
        }

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Endpoint to calculate GBD Values

    
@app.get("/calculate_gbd/")
async def calculate_gbd( selected_date: str = Query(...), 
    packing_density: str = Query(...),
    updated_proportions: str = Query(...)):
    """
    Calculate GBD values dynamically for user-entered packing density values.
    """
    try:
        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)  # Reset file pointer before reading

        # Read and clean data
        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets)

        # ✅ Convert the user-selected date to `datetime`
        selected_date_obj = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")

        
        # Get sample data for selected date
        sample_data = get_sample_data_for_date(cleaned_sheets, updated_sheets, selected_date)

        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")

        # ✅ Convert user-input proportions to a dictionary
        proportions_list = [float(value.strip()) for value in updated_proportions.split(",")]
        proportions_dict = dict(zip(updated_sheets, proportions_list))  # Assign proportions to correct sheets
        # print(proportions_dict)

        # ✅ Ensure proportions sum up to 1
        if round(sum(proportions_dict.values()), 4) != 1.0:
            raise HTTPException(status_code=400, detail="Proportions must sum up to 1. Please check input values.")

        processed_data = average_samples_per_date(cleaned_sheets)

        # view_sheets(processed_data)


        # ✅ Convert packing density input (supports single or multiple values)
        try:
            packing_density = float(packing_density.strip())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid packing density input. Please enter valid numbers.")

        density_water = 1
      
        # Convert selected_date from "dd-mm-yyyy" to "dd.mm.yy"
        formatted_date = datetime.strptime(selected_date, "%d-%m-%Y").strftime("%d.%m.%y")

        # Convert to pandas Timestamp before passing
        received_date = pd.to_datetime(formatted_date, format='%d.%m.%y', errors='coerce')

        if pd.isna(received_date):
            return {"error": "Invalid date format conversion"}

        # print(f"Formatted Date: {formatted_date}, Timestamp: {received_date} (Type: {type(received_date)})")
        total_volume, density = process_sheets_and_calculate_gbd(processed_data, density_water, formatted_date, proportions_dict)
        print(total_volume, density)
        GBD = density * packing_density
        gbd_result = {str(packing_density): round(GBD, 4)}
        print(gbd_result)
        

        return {
            "message": f"GBD Calculation for {selected_date}",
            "total_volume": round(total_volume, 4),
            "specific_gravity": round(density, 4),
            "gbd_values": gbd_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Endpoint to calculate q values

@app.get("/calculate_q_value/")
async def calculate_q_value(
    selected_date: str = Query(...), 
    updated_proportions: str = Query(None)
):
    """
    Calculate q-value using Andreasen Equation for a given date.
    """
    global cached_final_df, cached_q_values  # Allow modification of global variable

    try:

        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)

        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets)
        
        # selected_date = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")
        sample_data = get_sample_data_for_date(cleaned_sheets, updated_sheets, selected_date)
        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")
        
        # ✅ Convert proportions from query string to dictionary
        proportions_list = [float(value.strip()) for value in updated_proportions.split(",")]
        proportions_dict = dict(zip(updated_sheets, proportions_list))
        
        print(proportions_dict)

        processed_data = average_samples_per_date(cleaned_sheets)
        print("Average calculation Done!")

        # Define the columns you want to exclude from numeric calculations
        excluded_columns = ['Total', 'Loose Bulk Density (gm/cc)', 'Sp. gravity']

        # Call the function to calculate cumulative weights
        cumulative_sheets = calculate_cumulative_weights(processed_data, excluded_columns)

        # ✅ Compute sheet constants dynamically
        sheet_multipliers = get_sheet_constants_from_proportions(proportions_dict)
        print(sheet_multipliers)
        
        # Process the data and consolidate the results from all sheets
        sheet_CPFT_df = Calculate_Sheet_CPFT(cumulative_sheets, selected_date, proportions_dict, d_values)

        # Call the function
        updated_df = rearrange_mess_sizes(sheet_CPFT_df)

        packing_density = None
        # Add columns to the DataFrame
        sorted_df = add_columns(updated_df, proportions_dict, sheet_multipliers, packing_density)

        sorted_df['Log_D/Dmax_value'] = np.log(sorted_df['Normalized_D'])
        sorted_df['Log_pct_CPFT'] = np.log(sorted_df['pct_CPFT_interpolation'])
        
        # Predict q values
        q_value = q_value_prediction(sorted_df, selected_date)  
        

        # ===================================Debug statements======================================
        # view_sheets(cumulative_sheets)

        # for sheet_name, df in cumulative_sheets.items():
        #     print(f"Index dtype for sheet '{sheet_name}': {df.index.dtype}")

        # print(cumulative_sheets['H(7-12)'].index) 

        # print(f"\n{sheet_multipliers}")
        # print(f"\n{proportions_dict}")

        # print("\nNow view the dataframe with CPFT values for each sheet:")
        # print(sheet_CPFT_df)

        # print("\nFinal DataFrame after assigning 'Sheet Name' based on 'D_value':")
        # print(updated_df)

        print("\nFinal dataframe")
        print(sorted_df)
    
        print(f"\nq value : {q_value}")

        q_values={}
        q_values[selected_date] = q_value
        # print(q_values)
        # print(type(q_values))

        final_df={}
        final_df[selected_date] = sorted_df

        return {
            "message": f"q-value Calculation for {selected_date}",
            "intermediate_table": final_df[selected_date].to_dict(orient="records"),
            "q_values": q_values[selected_date].to_dict(orient="records")
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value Error: {str(ve)}")

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Endpoint to calculate modified q values

@app.get("/calculate_q_value_modified_andreason/")
async def calculate_q_value_modified_andreason(
    selected_date: str = Query(...),
    packing_density: str = Query(...),
    updated_proportions: str = Query(None)
):
    """
    Calculate q-value using Andreasen Equation for a given date.
    """
    global cached_final_df, cached_q_values, sorted_df  # Allow modification of global variable

    try:

        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)

        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets)
        
        # selected_date = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")
        sample_data = get_sample_data_for_date(cleaned_sheets, updated_sheets, selected_date)
        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")
        
        # ✅ Convert proportions from query string to dictionary
        proportions_list = [float(value.strip()) for value in updated_proportions.split(",")]
        proportions_dict = dict(zip(updated_sheets, proportions_list))
        
        processed_data = average_samples_per_date(cleaned_sheets)
        print("Average calculation Done!")

        # Define the columns you want to exclude from numeric calculations
        excluded_columns = ['Total', 'Loose Bulk Density (gm/cc)', 'Sp. gravity']

        # Call the function to calculate cumulative weights
        cumulative_sheets = calculate_cumulative_weights(processed_data, excluded_columns)

        # ✅ Compute sheet constants dynamically
        sheet_multipliers = get_sheet_constants_from_proportions(proportions_dict)
        
        # Process the data and consolidate the results from all sheets
        sheet_CPFT_df = Calculate_Sheet_CPFT(cumulative_sheets, selected_date, proportions_dict, d_values)

        # Call the function
        updated_df = rearrange_mess_sizes(sheet_CPFT_df)

        # ✅ Convert packing density input (supports single or multiple values)
        try:
            packing_density = float(packing_density.strip())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid packing density input. Please enter valid numbers.")

        print(packing_density)
        # Add columns to the DataFrame
        sorted_df = add_columns(updated_df, proportions_dict, sheet_multipliers, packing_density)

        sorted_df['Log_D/Dmax_value'] = np.log(sorted_df['Normalized_D'])
        sorted_df['Log_pct_CPFT'] = np.log(sorted_df['pct_CPFT_interpolation'])

        print(sorted_df[["Column Name", 'pct_poros_CPFT']])

    
        # Create the modified DataFrame with specific columns
        modified_df = sorted_df[['Sheet Name', 'Column Name', 'D_value', 'pct_CPFT_interpolation', 'pct_poros_CPFT']]
        print(f"\nDF for modified andreason:{modified_df}")

        # Step 1: Optimize q-value for a single packing density
        optimal_q = optimize_q(modified_df, D_col='D_value', pct_CPFT_col='pct_poros_CPFT')
        print(f"Optimal q-value: q = {optimal_q}")

        q_results = {"Date": selected_date,
                     f'q_{int(packing_density * 100)}': np.round(optimal_q, 4)}

        # q_results[f'q_{int(packing_density * 100)}'] = np.round(optimal_q, 4)

        print(q_results)
        print(type(q_results))

        q_df = pd.DataFrame([q_results])
        print(q_df)
        print(type(q_df))


        # Step 2: Calculate errors and MAE for the single q-value
        modified_andreasen_df, mae = calculate_errors_and_mae(modified_df, D_col='D_value', pct_CPFT_col='pct_poros_CPFT', q=optimal_q)

        # Print the updated DataFrame and MAE
        print(f"Updated DataFrame:\n{modified_andreasen_df}")
        print(f"Mean Absolute Error (MAE): {mae}")

        cpft_error_dict={}
        cpft_error_dict[selected_date] = modified_andreasen_df

        print(cpft_error_dict)
        print(cpft_error_dict[selected_date])

        return {
            "message": f"q-value Calculation using Modified Andreasen Eq. for {selected_date}",
            "q_values": q_df.to_dict(orient="records"),
            "cpft_error_table": cpft_error_dict[selected_date].to_dict(orient="records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}   This is the error")
    
# Endpoint to calculate modified q values

# ✅ **New API Endpoint for Double Modified q-values**
@app.get("/calculate_q_value_double_modified/")
async def calculate_q_value_double_modified(
    selected_date: str = Query(...), 
    updated_proportions: str = Query(None)
):
    """
    Calculate q-values using the **Double Modified Andreasen Equation** for a given date.
    """
    global cached_final_df, sorted_df # Use cached final_df instead of recomputing

    try:


        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)

        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets)
        
        # selected_date = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")
        sample_data = get_sample_data_for_date(cleaned_sheets, updated_sheets, selected_date)
        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")
        
        # ✅ Convert proportions from query string to dictionary
        proportions_list = [float(value.strip()) for value in updated_proportions.split(",")]
        proportions_dict = dict(zip(updated_sheets, proportions_list))
        
        processed_data = average_samples_per_date(cleaned_sheets)
        print("Average calculation Done!")

        # Define the columns you want to exclude from numeric calculations
        excluded_columns = ['Total', 'Loose Bulk Density (gm/cc)', 'Sp. gravity']

        # Call the function to calculate cumulative weights
        cumulative_sheets = calculate_cumulative_weights(processed_data, excluded_columns)

        # ✅ Compute sheet constants dynamically
        sheet_multipliers = get_sheet_constants_from_proportions(proportions_dict)
        
        # Process the data and consolidate the results from all sheets
        sheet_CPFT_df = Calculate_Sheet_CPFT(cumulative_sheets, selected_date, proportions_dict, d_values)

        # Call the function
        updated_df = rearrange_mess_sizes(sheet_CPFT_df)

        # # ✅ Convert packing density input (supports single or multiple values)
        # try:
        #     packing_density = float(packing_density.strip())
        # except ValueError:
        #     raise HTTPException(status_code=400, detail="Invalid packing density input. Please enter valid numbers.")

        packing_density = None
        # Add columns to the DataFrame
        sorted_df = add_columns(updated_df, proportions_dict, sheet_multipliers, packing_density)

        sorted_df['Log_D/Dmax_value'] = np.log(sorted_df['Normalized_D'])
        sorted_df['Log_pct_CPFT'] = np.log(sorted_df['pct_CPFT_interpolation'])

        
        print("Before main function call")
        # Call the function with the sorted DataFrame
        Q_value, modified_df = calculate_Q_value_and_plot(sorted_df, pct_CPFT_col='pct_poros_CPFT')
        print("After main function call")
        print(f"The optimal Q-value using the Modified Andreasen Equation is: {Q_value}")

        # Print the updated DataFrame
        # print(f"Modified DataFrame:\n{modified_df}")

        print("\nq results:")
        q_results = {"Date": selected_date,
                     f'q_value': np.round(Q_value, 4)}

        print(q_results)
        print(type(q_results))

        q_df = pd.DataFrame([q_results])
        print(q_df)
        print(type(q_df))

        print("\nFinal_df")
        final_df={}
        final_df[selected_date] = modified_df

        print(final_df)
        print(final_df[selected_date])

        return {
            "message": f"Double Modified q-value Calculation for {selected_date}",
            "double_modified_q_values": q_df.to_dict(orient="records"),
            "intermediate_table": final_df[selected_date].to_dict(orient="records") # ✅ Pass the intermediate table for regression
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
