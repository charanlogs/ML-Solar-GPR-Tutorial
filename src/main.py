import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings # We'll use this to hide the messy red warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, ExpSineSquared
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Making a folder to keep our results organized
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"I've created a folder for our results here: {output_dir}")

# Step 2: Loading our datasets
print("Getting the data ready...")
gen_df = pd.read_csv('data/Plant_1_Generation_Data.csv')
weather_df = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')

# Step 3: Fixing dates and merging the files
# We use 'mixed' so the computer doesn't get confused by different date styles
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='mixed')
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='mixed')

# This merges the two files into one big table
df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')

# We'll just study one specific inverter for this project
target_id = df['SOURCE_KEY_x'].unique()[0]
df_sample = df[df['SOURCE_KEY_x'] == target_id].copy()

# Step 4: Cleaning the data
# We only want daytime data (when the sun is actually out!)
df_sample = df_sample[df_sample['IRRADIATION'] > 0.05]
print(f"Data is cleaned. We're looking at Inverter: {target_id}")

X = df_sample[['IRRADIATION']].values
y = df_sample['DC_POWER'].values

# Step 5: Scaling the data
# Machine learning works better when numbers are small (around 0 and 1)
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Step 6: Testing different models (The Comparison Task)
print("Testing 3 different ways to model this data...")

# This part hides those annoying red 'Convergence' warnings for the bad models
warnings.filterwarnings('ignore') 

kernels = [
    ("Simple RBF", C(1.0) * RBF(length_scale=1.0)),
    ("Periodic", C(1.0) * ExpSineSquared(periodicity=1.0)),
    ("Best (Composite)", C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))
]

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
x_plot = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)

for i, (name, kernel) in enumerate(kernels):
    # We train each model on the same data
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X_scaled, y_scaled)
    y_pred, sigma = gp.predict(x_plot, return_std=True)
    
    # Plotting what the model 'learned'
    axes[i].scatter(X_scaled, y_scaled, color='gray', alpha=0.15, s=10)
    axes[i].plot(x_plot, y_pred, color='red', lw=2, label='Prediction')
    axes[i].fill_between(x_plot.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma, color='red', alpha=0.1)
    
    score = r2_score(y_scaled, gp.predict(X_scaled))
    axes[i].set_title(f"Model: {name}\nScore: {score:.3f}")
    axes[i].legend(loc='lower right')

# Saving the comparison to our output folder
comparison_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(comparison_path, dpi=300)
print(f"I've saved the comparison graph to: {comparison_path}")
plt.show()

# Step 7: Final Professional Model
print("Now training the best version of the model...")
final_kernel = C(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
gp_final = GaussianProcessRegressor(kernel=final_kernel, n_restarts_optimizer=25, random_state=42)
gp_final.fit(X_scaled, y_scaled)

# Measuring how well we did
y_pred_final, sigma_final = gp_final.predict(x_plot, return_std=True)
final_r2 = r2_score(y_scaled, gp_final.predict(X_scaled))
final_mae = mean_absolute_error(y_scaled, gp_final.predict(X_scaled))
print(f"Final Result -> Success Rate: {final_r2*100:.1f}% | Error: {final_mae:.3f}")

# Step 8: Final Plotting in real-world units
y_pred_orig = scaler_y.inverse_transform(y_pred_final.reshape(-1, 1)).ravel()
X_plot_orig = scaler_X.inverse_transform(x_plot).ravel()
X_data_orig = scaler_X.inverse_transform(X_scaled).ravel()
y_data_orig = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

plt.figure(figsize=(12, 7), facecolor='white')
plt.scatter(X_data_orig, y_data_orig, color='gray', alpha=0.25, s=15, label='Actual Sensor Data')
plt.plot(X_plot_orig, y_pred_orig, color='#1f77b4', lw=3, label='Final Forecast')

# Showing the uncertainty range (The 95% Confidence Interval)
plt.fill_between(X_plot_orig, 
                 y_pred_orig - 1.96*(sigma_final*scaler_y.scale_), 
                 y_pred_orig + 1.96*(sigma_final*scaler_y.scale_), 
                 color='#1f77b4', alpha=0.15, label='95% Confidence Interval')

plt.title("Final Project Result: Optimized Solar Power Forecast", fontsize=15, pad=20)
plt.xlabel("Sunlight Intensity (W/m2)", fontsize=12)
plt.ylabel("Power Output (kW)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Saving the final professional plot
final_plot_path = os.path.join(output_dir, 'final_forecast_plot.png')
plt.savefig(final_plot_path, dpi=300)
print(f"Final professional forecast saved to: {final_plot_path}")
plt.show()

print("\nEverything is finished! You can find the graphs in the /output folder.")