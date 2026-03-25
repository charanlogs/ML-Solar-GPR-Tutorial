import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Loading the two datasets from our 'data' folder
print("Loading files...")
gen_df = pd.read_csv('data/Plant_1_Generation_Data.csv')
weather_df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')

# Step 2: Fixing the date formats so we can merge the files properly
# We use 'mixed' format to stop the computer from getting confused by different date styles
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='mixed')
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='mixed')

# Step 3: Merging the weather data with the power generation data
# This connects sunlight levels (Irradiation) to how much power was actually made
df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')

# To keep things simple for the tutorial, we will just look at one specific inverter
target_id = df['SOURCE_KEY_x'].unique()[0]
df_sample = df[df['SOURCE_KEY_x'] == target_id].copy()

# Step 4: Cleaning the data
# We only want daytime data (sunlight > 0). 
# This helps the model focus on the relationship that actually matters.
df_sample = df_sample[df_sample['IRRADIATION'] > 0.05]
print(f"Data fused and cleaned. Studying Inverter: {target_id}")

# Step 5: Preparing our input (Sunlight) and output (Power)
X = df_sample[['IRRADIATION']].values
y = df_sample['DC_POWER'].values

# We need to scale the data so the Gaussian Process can do its math correctly
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Step 6: Setting up our "Kernel"
# This is the most important part! We use an RBF kernel for the smooth trend
# and a WhiteKernel to handle the sensor noise.
kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 2.0)) \
         + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 0.5))

# Step 7: Training the model
# We set n_restarts high so the model finds the best possible fit
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25, random_state=42)
print("Training the Gaussian Process... this takes a moment.")
gp.fit(X_scaled, y_scaled)

# Step 8: Making predictions
# We create a smooth range of irradiation values to show the 'trend line'
x_plot = np.linspace(X_scaled.min(), X_scaled.max(), 200).reshape(-1, 1)
y_pred_scaled, sigma = gp.predict(x_plot, return_std=True)

# Step 9: Checking how good our model is (Evaluation)
# We calculate R-squared and MAE to prove our model works well
y_fit_scaled = gp.predict(X_scaled)
r2 = r2_score(y_scaled, y_fit_scaled)
mae = mean_absolute_error(y_scaled, y_fit_scaled)
print(f"Model Training Done! R2 Score: {r2:.4f} | Error: {mae:.4f}")

# Step 10: Rescaling back to real-world units (kW and W/m2)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
X_plot_orig = scaler_X.inverse_transform(x_plot).ravel()
X_data_orig = scaler_X.inverse_transform(X_scaled).ravel()
y_data_orig = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

# Step 11: Final Visualization (This is for our report!)
plt.figure(figsize=(12, 7), facecolor='white')

# Plot the real data points in gray
plt.scatter(X_data_orig, y_data_orig, color='gray', alpha=0.25, s=15, label='Actual Yield (Data)')

# Plot our model's smooth blue prediction line
plt.plot(X_plot_orig, y_pred_orig, color='#1f77b4', lw=3, label='GPR Prediction (Trend)')

# Add the shaded "Confidence Interval" (This shows the model's uncertainty)
plt.fill_between(X_plot_orig, 
                 y_pred_orig - 1.96 * (sigma * scaler_y.scale_), 
                 y_pred_orig + 1.96 * (sigma * scaler_y.scale_), 
                 color='#1f77b4', alpha=0.15, label='95% Confidence Interval')

# Professional styling
plt.title("Tutorial: Mapping Sunlight to Energy with Gaussian Processes", fontsize=16, pad=20)
plt.xlabel("Solar Irradiation (W/m²)", fontsize=13)
plt.ylabel("DC Power Output (kW)", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Save the final image to our folder
plt.savefig('gpr_plot.png', dpi=300)
print("Success! Visualization saved as gpr_plot.png")
plt.show()