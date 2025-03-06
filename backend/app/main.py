from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse 
from io import BytesIO
from app.model import (
    read_excel_file, clean_data, standardize_column_names_and_convert_dates,
    get_available_date_range, get_sample_data_for_date, calculate_gbd_values,
    convert_to_numeric_and_calculate_average, calculate_volume, sum_volumes,
    calculate_sg_mix, 
    convert_to_numeric_and_calculate_average_for_q_values, drop_last_3_and_reverse_cumsum,
    calculate_cpft, calculate_pct_cpft, merge_pct_cpft_into_df,
    calculate_interpolated_values, drop_and_reset_indices,
    normalize_particle_size, q_value_prediction, predict_mod_q_values, 
    calculate_cpft_error_dict, prepare_mod_q_values, double_modified_q_values, prepare_mod_q_values_double_modified
)

app = FastAPI()

# Define the required sheets and column drop list
required_sheets = ["7-12", "14-30", "36-70", "80-180", "220F"]
column_to_drop = ["Samples No."]
proportions = {'7-12': 0.35, '14-30': 0.20, '36-70': 0.15, '80-180': 0.10, '220F': 0.20}
sheet_constants = {'7-12': 65, '14-30': 45, '36-70': 30, '80-180': 20, '220F': 0}
mesh_size_to_particle_size = {'+6': 3360, '+8': 2380, '+10': 2000, '+14': 1410, '+16': 1190, '+12': 1680, '+18': 1000, '+30': 595, '+40': 420, '+50': 297, '+70': 210, '+100': 149, '+80': 177, '+120': 125, '+140': 105, '+200': 74, '+230': 63, '+270': 53, '+325': 44}

# Store uploaded file in memory for further processing
file_storage = {}

cached_final_df = {}
cached_q_values = {}  # Store previously computed q-values to avoid redundant calculations


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

        elif file.filename.endswith(".csv"):
            df = pd.read_csv(file_obj)

            # ✅ Validate that required columns exist
            expected_columns = ["Column1", "Column2", "Column3"]  # Replace with actual required columns
            missing_columns = [col for col in expected_columns if col not in df.columns]

            if missing_columns:
                raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")

        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV or Excel file.")

        
        sheets = read_excel_file(file_obj, required_sheets)
        if not sheets:
            raise HTTPException(status_code=400, detail="Invalid file uploaded. Required sheets are missing.")
        
        cleaned_sheets = clean_data(sheets, column_to_drop)
        standardized_sheets = standardize_column_names_and_convert_dates(cleaned_sheets)
        min_date, max_date = get_available_date_range(standardized_sheets, required_sheets)
        
        return {"message": "File uploaded successfully", "date_range": [str(min_date.date()), str(max_date.date())]}
     
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get_sample_data/")
async def get_sample_data(selected_date: str = Query(..., description="Selected date from user")):
    try:
        if "file" not in file_storage:
            raise HTTPException(status_code=400, detail="No file uploaded. Please upload a file first.")

        file_obj = file_storage["file"]
        file_obj.seek(0)  # Reset file pointer before reading
        
        sheets = read_excel_file(file_obj, required_sheets)
        cleaned_sheets = clean_data(sheets, column_to_drop)
        standardized_sheets = standardize_column_names_and_convert_dates(cleaned_sheets)

        # Get the available date range
        min_date, max_date = get_available_date_range(standardized_sheets, required_sheets)

        # ✅ Convert the user-selected date to `datetime`
        selected_date_obj = pd.to_datetime(selected_date, format="%d-%m-%Y", errors="coerce")

        # ✅ Check if the selected date is within the range
        if selected_date_obj < min_date:
            return {"error": f"Please select a date within the range: {min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')}"}
        elif selected_date_obj > max_date:
            selected_date_obj = max_date  # Auto-select nearest past date

        sample_data = get_sample_data_for_date(standardized_sheets, required_sheets, selected_date)
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
    
@app.get("/calculate_gbd/")
async def calculate_gbd(selected_date: str = Query(...), packing_density: str = Query(...)):
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
        cleaned_sheets = clean_data(sheets, column_to_drop)
        standardized_sheets = standardize_column_names_and_convert_dates(cleaned_sheets)

        # Get sample data for selected date
        sample_data = get_sample_data_for_date(standardized_sheets, required_sheets, selected_date)

        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")

        # Compute intermediate values
        averages = convert_to_numeric_and_calculate_average(sample_data)
        volume_data = calculate_volume(averages, proportions)
        total_volume = sum_volumes(volume_data)
        sg_mix_data = calculate_sg_mix(total_volume)

        # ✅ Convert packing density input (supports single or multiple values)
        try:
            packing_densities = [float(density.strip()) for density in packing_density.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid packing density input. Please enter valid numbers.")

        # Compute GBD
        gbd_df = calculate_gbd_values(sg_mix_data, packing_densities)
        gbd_dict = {str(row["Packing Density"]): row["GBD (g/cc)"] for _, row in gbd_df.iterrows()}

        return {
            "message": f"GBD Calculation for {selected_date}",
            "total_volume": round(total_volume, 4),
            "specific_gravity": round(sg_mix_data, 4),
            "gbd_values": gbd_dict
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@app.get("/calculate_q_value/")
async def calculate_q_value(selected_date: str = Query(...)):
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
        cleaned_sheets = clean_data(sheets, column_to_drop)
        standardized_sheets = standardize_column_names_and_convert_dates(cleaned_sheets)

        sample_data = get_sample_data_for_date(standardized_sheets, required_sheets, selected_date)
        if not sample_data or all(df is None or df.empty for df in sample_data.values()):
            raise HTTPException(status_code=400, detail=f"No sample data found for {selected_date}")

        # ✅ Compute q-value averages safely
        averages = convert_to_numeric_and_calculate_average_for_q_values(sample_data, selected_date)
        if selected_date not in averages or not averages[selected_date]:
            raise HTTPException(status_code=400, detail=f"No valid averages computed for {selected_date}")

        weights, cum_sum = drop_last_3_and_reverse_cumsum(averages, selected_date)
        cpft = calculate_cpft(cum_sum, proportions, selected_date)
        pct_cpft = calculate_pct_cpft(cpft, sheet_constants, selected_date)

        # ✅ Merge into single DataFrame
        final_df = merge_pct_cpft_into_df(mesh_size_to_particle_size, pct_cpft)
        
        # ✅ 🔥 Insert the New Sample Here (Before Interpolation)
        new_sample = pd.DataFrame({
            'Sheet': ['7-12'],
            'Mesh Size': ['+1'],
            'cpft': [100],
            'pct_cpft': [100],
            'Particle Size': [3500]  # This column was added during merging
        })

        # ✅ Insert new sample at the beginning (index 0) for each date
        for date in final_df:
            final_df[date] = pd.concat([new_sample, final_df[date]], ignore_index=True)

        
        
        final_df = calculate_interpolated_values(final_df)
        final_df = drop_and_reset_indices(final_df)
        final_df = normalize_particle_size(final_df)

        # final_df = add_sample_for_q_values(final_df)  # ✅ Add new sample to dataframe

       
        # ✅ Ensure final_df is correctly formatted before storing
        if isinstance(final_df, dict):
            # ✅ If dictionary contains DataFrames, merge them
            if all(isinstance(v, pd.DataFrame) for v in final_df.values()):
                cached_final_df[selected_date] = pd.concat(final_df.values(), ignore_index=True)
            else:
                raise ValueError(f"Invalid final_df format for {selected_date}: Expected DataFrames, got {type(final_df)}")
        else:
            cached_final_df[selected_date] = final_df

        cached_q_values[selected_date] = q_value_prediction(final_df)  # Store q-values for later use
        # # Store q-values for later use
        # print(f"🛠️ Debug: final_df structure for {selected_date}: {type(final_df)}")
        # print(f"🛠️ Debug: final_df contents:\n{final_df}")
        
        # df_display = final_df[selected_date].drop(columns=["cpft", "pct_cpft"], errors="ignore").rename(columns={
        #     "Particle Size": "Particle Size (μm)",
        #     "pct_cpft_interpolated": "%_CPFT (interpolated)",
        #     "Normalized_D": "D/D_max",
        #     "Log_D/Dmax": "Log(D/D_max)",
        #     "Log_pct_cpft": "Log(%_CPFT)"
        # })


        return {
            "message": f"q-value Calculation for {selected_date}",
            "intermediate_table": final_df[selected_date].to_dict(orient="records"),
            "q_values": cached_q_values[selected_date].to_dict(orient="records")
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Value Error: {str(ve)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/calculate_q_value_modified_andreason/")
async def calculate_q_value_modified_andreason(
    selected_date: str = Query(...),
    packing_density: str = Query(...)
):
    """
    Calculate q-values using the Modified Andreasen Equation for user-defined packing densities.
    """
    global cached_final_df, cached_q_values  # Use the cached final_df instead of recomputing
    try:
        # ✅ Ensure q-values are calculated first
        if selected_date not in cached_q_values or selected_date not in cached_final_df:
            print(f"ℹ️ Calling calculate_q_value() first to store final_df for {selected_date}...")
            q_response = await calculate_q_value(selected_date)  # Compute normal q-values first

            # ✅ Properly store final_df after calling calculate_q_value
            if "intermediate_table" in q_response:
                cached_final_df[selected_date] = pd.DataFrame(q_response["intermediate_table"])

        # ✅ Now check final_df availability
        if (selected_date not in cached_final_df 
                or not isinstance(cached_final_df[selected_date], pd.DataFrame) 
                or cached_final_df[selected_date].empty):
            print(f"🚨 Debugging: No valid data found for {selected_date} in cached_final_df.")
            raise HTTPException(status_code=500, detail=f"No valid data for merging on {selected_date}")

        print(f"🛠️ Debug: final_df[{selected_date}] columns: {cached_final_df[selected_date].columns if isinstance(cached_final_df[selected_date], pd.DataFrame) else 'Not a DataFrame'}")
        print(f"🛠️ Debug: final_df[{selected_date}] data:\n{cached_final_df[selected_date]}")

        # ✅ Convert user input to a list of packing densities
        try:
            packing_densities = [float(density.strip()) for density in packing_density.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid packing density input. Please enter valid numbers.")
        
        # ✅ Prepare data for q-value optimization
        mod_q = prepare_mod_q_values(cached_final_df, selected_date, packing_densities)
  

        # 🔥 **Check if `mod_q` is empty**
        if not mod_q:
            raise HTTPException(status_code=500, detail=f"Unable to prepare data for optimization on {selected_date}")

        # ✅ Compute modified q-values dynamically for user-defined densities
        optimized_q_values_df = predict_mod_q_values(mod_q, packing_densities)

        # ✅ Now call error calculation after ensuring required columns exist
        cpft_error_dict = calculate_cpft_error_dict(mod_q, optimized_q_values_df.to_dict(orient="records"), packing_densities)

        # ✅ Rename columns for better readability before returning
        # for date in cpft_error_dict:
        #     cpft_error_dict[date] = cpft_error_dict[date].rename(columns={
        #         "Sheet": "Sheet Name",
        #         "Mesh Size": "Mesh Size",
        #         "Particle Size (μm)": "Particle Size (μm)"
        #     })
            
        #     for density in packing_densities:
        #         cpft_error_dict[date].rename(columns={
        #             f"pct_{int(density * 100)}_poros_CPFT": f"Actual CPFT ({int(density * 100)}% Packing Density)",
        #             f"calculated_CPFT_{int(density * 100)}": f"Predicted CPFT ({int(density * 100)}% Packing Density)",
        #             f"absolute_error_{int(density * 100)}": f"Absolute Error ({int(density * 100)}% Packing Density)"
        #         }, inplace=True)



        return {
            "message": f"q-value Calculation using Modified Andreasen Eq. for {selected_date}",
            "q_values": optimized_q_values_df.to_dict(orient="records"),
            "cpft_error_table": cpft_error_dict[selected_date].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    


# ✅ **New API Endpoint for Double Modified q-values**
@app.get("/calculate_q_value_double_modified/")
async def calculate_q_value_double_modified(selected_date: str = Query(...)):
    """
    Calculate q-values using the **Double Modified Andreasen Equation** for a given date.
    """
    global cached_final_df  # Use cached final_df instead of recomputing

    try:
        # ✅ Ensure q-values are calculated first
        if selected_date not in cached_q_values or selected_date not in cached_final_df:
            print(f"ℹ️ Calling calculate_q_value() first to store final_df for {selected_date}...")
            q_response = await calculate_q_value(selected_date)  # Compute normal q-values first

            # ✅ Properly store final_df after calling calculate_q_value
            if "intermediate_table" in q_response:
                cached_final_df[selected_date] = pd.DataFrame(q_response["intermediate_table"])

        # ✅ Now check final_df availability
        if (selected_date not in cached_final_df 
                or not isinstance(cached_final_df[selected_date], pd.DataFrame) 
                or cached_final_df[selected_date].empty):
            print(f"🚨 Debugging: No valid data found for {selected_date} in cached_final_df.")
            raise HTTPException(status_code=500, detail=f"No valid data for merging on {selected_date}")

         # ✅ Fix: Use `prepare_mod_q_values_double_modified()`
        mod_q = prepare_mod_q_values_double_modified(cached_final_df, selected_date)

         # 🔥 **Check if `mod_q` is empty**
        if not mod_q:
            raise HTTPException(status_code=500, detail=f"Unable to prepare data for optimization on {selected_date}")


         # ✅ Compute double modified q-values
        double_modified_q_df, df_regression = double_modified_q_values(mod_q, selected_date)
        
        df_regression_dict = df_regression.to_dict(orient="records") if not df_regression.empty else []

        print(f"🛠️ Debugging: df_regression for {selected_date}:\n{df_regression}")

        return {
            "message": f"Double Modified q-value Calculation for {selected_date}",
            "double_modified_q_values": double_modified_q_df.to_dict(orient="records"),
            "intermediate_table": df_regression_dict  # ✅ Pass the intermediate table for regression
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")