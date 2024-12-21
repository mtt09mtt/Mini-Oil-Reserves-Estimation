import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d # For calculating reserves of any given probability values

# Alias the Session State
ss = st.session_state

# Page configuration
st.set_page_config(page_title="Mini Oil Reserves Estimation - Version 1.0", page_icon="📈", layout="wide")

# Add main title
st.title("Probabilistic Oil Reserves Estimate Using Monte Carlo Simulations")

# Create some tabs
tab1, tab2, = st.tabs(["✍️ Main",  "✍️ Help/About"])

# Text out
text_out = "Upload your formatted Excel file. Refer to the Help page for the specific format"

def load_excel_file():
    
    # Upload the Excel file
    uploaded_file = st.file_uploader(text_out, type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Group the DataFrame by the "Reservoir" column
        grouped_df = df.groupby("Reservoir")

        # Create a list of DataFrames, each with a unique index based on the "Reservoir" value
        dataframes = []
        for reservoir, group_df in grouped_df:
            group_df = group_df.drop("Reservoir", axis=1)  # Remove the "Reservoir" column
            group_df.index = [reservoir] * len(group_df)
            dataframes.append(group_df)

        # Display the DataFrames
        # for df in dataframes:
        #     st.dataframe(df)    
        
        # Store dataframes and number of reservoir into session state (ss)     
        ss.dataframes = dataframes
        ss.numOfRes = len(dataframes)
        

# Generate input data function
def generate_data(min_val, mean_val, max_val, num_samples, distribution):    
    
    if distribution == 'Normal':
        # Calculate standard deviation based on min, mean, and max
        std_dev = (max_val - min_val) / 6  # Assuming 6 sigma range for normal distribution
        data = np.random.normal(loc=mean_val, scale=std_dev, size=num_samples)
        data = np.clip(data, min_val, max_val)  # Clip values to stay within min and max
        
    elif distribution == 'Triangle':
        mode_val = mean_val  # Assuming mode is equal to mean for triangle distribution
        data = np.random.triangular(left=min_val, mode=mode_val, right=max_val, size=num_samples)
        
    elif distribution == 'Uniform':
        data = np.random.uniform(low=min_val, high=max_val, size=num_samples)
        
    elif distribution == 'Lognormal':
        # Approximate parameters of the underlying normal distribution
        sigma = (np.log(max_val) - np.log(min_val)) / 6  # Assuming 6 sigma range
        mean_normal = np.log(mean_val) - 0.5 * sigma**2  # Adjust mean for lognormal
        data = np.random.lognormal(mean=mean_normal, sigma=sigma, size=num_samples)
        data = np.clip(data, min_val, max_val)  # Clip values to stay within min and max
        
    else:
        raise ValueError("Invalid distribution type. Choose 'Normal', 'Lognormal', 'Uniform' or 'Triangle'.")

    return data

# Get out reserves value from CCDF function
def get_reserves_with_equivalent_probability(x_data, y_data, probability):

    # Ensure the data is sorted correctly
    sorted_indices = np.argsort(y_data)
    x_data_sorted = np.array(x_data)[sorted_indices]
    y_data_sorted = np.array(y_data)[sorted_indices]
    
    # Create the interpolation function
    interpolation_func = interp1d(y_data_sorted, x_data_sorted, bounds_error=False, fill_value="extrapolate")
    if (probability > 0) and (probability < 1):
        out_value = interpolation_func(probability)
        return out_value
    else:
        return -999.25
    
# Plot reserves 
def myplot(oil_reserves, res_num):
    
    # Compute the ECDF
    oil_reserves_sorted = np.sort(oil_reserves)
    y_ecdf = np.arange(1, len(oil_reserves_sorted) + 1) / len(oil_reserves_sorted)

    # Compute the CCDF - Complementary Cumulative Distribution Function
    y_ccdf = 1 - y_ecdf
    
    # Create subtitle for plot
    if res_num < ss.numOfRes :
        plot_title = "Recoverable Oil of Reservoir " + str(res_num +1)
        # plot_title = "Recoverable Oil of " + xxx
    else:
        plot_title = "Aggregated Recoverable Oil Across All Reservoirs"
    
    # Create a side by side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(9, 4))
    ax1.hist(oil_reserves, bins=50, edgecolor='k', linewidth=0.2, color='blue', alpha=0.6)
    ax2.plot(oil_reserves_sorted, y_ccdf, marker='.', markersize=1, linestyle='none', label='CCDF', color='green')
    # ax2.grid(True, color='grey', alpha=0.2, linestyle='--')
    ax2.grid(True, color='grey', alpha=0.2)
    fig.suptitle(plot_title)

    ax1.set_xlabel('Recoverable Oil (MMBO)', labelpad=10, fontsize=6, color='blue', alpha=0.6)
    ax1.set_ylabel('Frequency', labelpad=10, fontsize=6, color='blue', alpha=0.6)
    
    # The labels below are followed the ROSE
    ax2.set_xlabel('Recoverable Oil (MMBO)', labelpad=10, fontsize=6, color='green')
    ax2.set_ylabel('Cumulative Probability', labelpad=10, fontsize=6, color='green')
    
    # Set font size for axis ticks
    ax1.tick_params(axis='both', labelsize=6)
    ax2.tick_params(axis='both', labelsize=6)
    
    fig.tight_layout()
    
    st.pyplot(fig, use_container_width=True)
    
    return oil_reserves_sorted, y_ccdf
    

def main_entry():
    
    try:
        if "dataframes" not in ss:
             load_excel_file()
             # Remove file uploader widget if needed
             
        all_reserves = []
        for res_num, reservoir_df in enumerate(ss.dataframes):
                                      
            # Generate data samples - Use Min, Max, Mid
            net_rock_volume = generate_data(reservoir_df.iloc[0]['Min'] * 1e6, reservoir_df.iloc[0]['Mid'] * 1e6, 
                                            reservoir_df.iloc[0]['Max'] * 1e6, 10000, reservoir_df.iloc[0]['Distribution'])
            porosity = generate_data(reservoir_df.iloc[1]['Min'], reservoir_df.iloc[1]['Mid'], reservoir_df.iloc[1]['Max'], 10000, reservoir_df.iloc[1]['Distribution'])
            oil_saturation = generate_data(reservoir_df.iloc[2]['Min'], reservoir_df.iloc[2]['Mid'], reservoir_df.iloc[2]['Max'], 10000, reservoir_df.iloc[2]['Distribution'])
            fvf = generate_data(reservoir_df.iloc[3]['Min'], reservoir_df.iloc[3]['Mid'], reservoir_df.iloc[3]['Max'], 10000, reservoir_df.iloc[3]['Distribution'])
            rf = generate_data(reservoir_df.iloc[4]['Min'], reservoir_df.iloc[4]['Mid'], reservoir_df.iloc[4]['Max'], 10000, reservoir_df.iloc[4]['Distribution'])
            
            # Calculating reserves for this reservoir
            oiip = (net_rock_volume * 6.2898 * porosity * oil_saturation) / (fvf * 1e6)

            oil_reserves = oiip * rf
            ss.oil_reserves = oil_reserves

            # Append to the aggregated reserves list
            all_reserves.append(oil_reserves)
            
            # Plot individual reservoir reserves
            myplot(ss.oil_reserves, res_num)
            
        # Aggregate all reserves
        total_reserves = np.sum(all_reserves, axis=0)
        
        # Plot the total_reserves and write out the reserves as a table - Use any number that greater than the actual mumber of reservoir. Here I used 7777
        x_data, y_data = myplot(total_reserves, 7777)
        p10 = get_reserves_with_equivalent_probability(x_data, y_data, 0.1)
        p50 = get_reserves_with_equivalent_probability(x_data, y_data, 0.5)
        p90 = get_reserves_with_equivalent_probability(x_data, y_data, 0.9)
        pmean = np.mean(total_reserves)
        
        st.subheader("Aggregated Recoverable Oil Across All Reservoirs")    
        
        data = {
            'P10': p10,
            'P50': p50,
            'P90': p90,
            'Pmean': pmean}
        
        # Convert the dictionary to a DataFrame
        df_out = pd.DataFrame(list(data.items()), columns=["Probability", "Recoverable Oil (MMBOE)"])
           
        # Ensure consistent types in the 'Recoverable Oil (MMBOE)' column
        df_out["Recoverable Oil (MMBOE)"] = df_out["Recoverable Oil (MMBOE)"].astype(float)
        
        # Format the 'Recoverable Oil (MMBOE)' column to two decimal places
        df_out["Recoverable Oil (MMBOE)"] = df_out["Recoverable Oil (MMBOE)"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
        # Reset the index to remove any explicit index column
        df_out.reset_index(drop=True, inplace=True)

        # Display the DataFrame as a table in Streamlit
        st.dataframe(df_out, hide_index=True)
        
    except Exception as e:
        pass    
    
    with tab2:
        st.markdown("""
        **Mini Oil Reserves Estimation - Version 1.0 - December 2024**
        
        - Accepts input for an unlimited number of reservoirs  
        - Each reservoir is treated as an independent entity with no interdependencies 
        - Supports triangular, uniform, lognormal, and normal probability distributions for input parameters 
        - Excludes gas-to-oil conversion and associated gas handling 
        - Runs 10,000 iterations by default for probabilistic assessments  
        - **Detailed Inputs**: 
            - An Excel file with the first sheet named "Input", has fixed column names: "Reservoir", "Parameter", "Max", "Mid", "Min" and "Distribution"
            - Pay attention to unit of each parameter
        - **Detailed Outputs**:  
            - Probability density function (histogram) and Cumulative probability distribution curve for each reservoir
            - And aggregated results across all reservoirs
            
        - This tool is free to use, but please use it at your own risk!
        - Feedback: mytienthang@gmail.com
        """)              
                
if __name__ == "__main__":
    main_entry()
