import matplotlib.pyplot as plt
import math
import statsmodels.api as sm

def no_grid_plot_raw_sensors(df,sensor, unit_id):
    
    df_unit = df[df["unit_number"] == unit_id]
    # print('hello')

    plt.figure(figsize=(4, 4))  # Create a new figure for each sensor
    plt.plot(df_unit["time_in_cycles"], df_unit[sensor], label=sensor)
    
    plt.xlabel("Time in Cycles")
    plt.ylabel("Sensor Readings")
    plt.title(f"Sensor Measurements Over Time for Unit {unit_id} - {sensor}")
    plt.legend()
    

    
    return plt

def auto_correlation(df, sensor, unit_id, lags=50):
    df_unit = df[df["unit_number"] == unit_id]
    
    plt.figure(figsize=(8, 5))
    sm.graphics.tsa.plot_acf(df_unit[sensor], lags=lags, alpha=0.05)
    plt.title(f"Autocorrelation of {sensor} for Unit {unit_id}")
    plt.xlabel("Lags")
    plt.ylabel("Autocorrelation")
    
    
    return plt

def grid_plot_same_sensor_all_units(df, sensor):
    # Define the number of rows and columns for the grid layout
    num_units = 24  # Total number of unit IDs
    cols = 4  # Adjust based on preference
    rows = math.ceil(num_units / cols)  # Calculate the required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Create a grid of subplots
    axes = axes.flatten()  # Flatten axes array to easily index them


    # Loop through each unit and plot on a grid
    for idx, unit_id in enumerate(range(1, num_units + 1)):
        df_unit = df[df["unit_number"] == unit_id]
        
        axes[idx].plot(df_unit["time_in_cycles"], df_unit[sensor], label=sensor)
        axes[idx].set_xlabel("Time in Cycles")
        axes[idx].set_ylabel("Sensor Readings")
        axes[idx].set_title(f"Unit {unit_id}")
        axes[idx].legend()

    # Hide any unused subplots (if num_units doesn't perfectly fit into the grid)
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout to prevent overlapping
    return plt

def boxplot(df):
    # Filter columns with "sensor" in their names
    sensor_cols = [col for col in df.columns if "sensor" in col]

    # Number of plots
    num_sensors = len(sensor_cols)
    rows = (num_sensors // 4) + (num_sensors % 4 > 0)  # 4 plots per row

    # Create a grid of box plots
    fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
    axes = axes.flatten()  # Flatten to iterate easily

    for i, col in enumerate(sensor_cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Hide empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return plt

def grid_plot_RUL(df):
    # Define the number of rows and columns for the grid layout
    num_units = 24  # Total number of unit IDs
    cols = 4  # Adjust based on preference
    rows = math.ceil(num_units / cols)  # Calculate the required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Create a grid of subplots
    axes = axes.flatten()  # Flatten axes array to easily index them


    # Loop through each unit and plot on a grid
    for idx, unit_id in enumerate(range(1, num_units + 1)):
        df_unit = df[df["unit_number"] == unit_id]
        
        axes[idx].plot(df_unit["time_in_cycles"], df_unit['RUL'], label='RUL')
        axes[idx].set_xlabel("Time in Cycles")
        axes[idx].set_ylabel("Sensor Readings")
        axes[idx].set_title(f"Unit {unit_id}")
        axes[idx].legend()

    # Hide any unused subplots (if num_units doesn't perfectly fit into the grid)
    for i in range(idx + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout to prevent overlapping
    return plt