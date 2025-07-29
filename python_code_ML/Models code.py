
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from scipy.stats import spearmanr

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

# Load and preprocess data
# dataset = pd.read_excel("V3u-20230625-for-resit.xlsx")
dataset = pd.read_excel("V4o-20230625_data2.xlsx")
dataset = dataset.drop_duplicates(subset='instance', keep='first')
modified_dataset = dataset.iloc[:, 2:-2]
X = modified_dataset[['evts', 'backtrackEvts', 'pruneBacktrackEvts', 'pruneEvts', 'strengthenEvts']]
y = modified_dataset['expandEvts']

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    # "BayesianRidge": BayesianRidge(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(verbosity=0),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0),
    "MLP": MLPRegressor(max_iter=1000),
    "Stacking": StackingRegressor(
        estimators=[("rf", RandomForestRegressor()), ("xgb", XGBRegressor(verbosity=0))],
        final_estimator=Ridge()
    )
}

results = []
for name, model in models.items():
    r2_scores, adj_r2_scores, rmse_scores, mae_scores = [], [], [], []
    mape_scores, mre_scores, max_err_scores = [], [], []
    smape_scores, std_residuals, r2_train_scores = [], [], []
    within_10_acc, within_20_acc, spearman_corrs = [], [], []
    train_times = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_val)
        y_pred_train = model.predict(X_train)

        residuals = y_val - y_pred
        r2 = r2_score(y_val, y_pred)
        adj_r2 = 1 - ((1 - r2) * (len(y_val) - 1)) / (len(y_val) - X_val.shape[1] - 1)
        r2_train = r2_score(y_train, y_pred_train)

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        mape_scores.append(mean_absolute_percentage_error(y_val, y_pred) * 100)
        mre_scores.append(np.mean(np.abs((y_val - y_pred) / y_val)))
        max_err_scores.append(max_error(y_val, y_pred))
        smape_scores.append(smape(y_val.to_numpy(), y_pred))
        std_residuals.append(np.std(residuals))
        r2_train_scores.append(r2_train)
        within_10_acc.append(np.mean(np.abs((y_val - y_pred) / y_val) < 0.10) * 100)
        within_20_acc.append(np.mean(np.abs((y_val - y_pred) / y_val) < 0.20) * 100)
        spearman_corrs.append(spearmanr(y_val, y_pred).correlation)
        train_times.append(train_time)

    results.append({
        "Model": name,
        "R2": np.mean(r2_scores),
        "R2_Train": np.mean(r2_train_scores),
        "Adjusted_R2": np.mean(adj_r2_scores),
        "RMSE": np.mean(rmse_scores),
        "MAE": np.mean(mae_scores),
        "MAPE": np.mean(mape_scores),
        "MRE": np.mean(mre_scores),
        "SMAPE": np.mean(smape_scores),
        "MaxError": np.mean(max_err_scores),
        "StdResiduals": np.mean(std_residuals),
        "%Within10%": np.mean(within_10_acc),
        "%Within20%": np.mean(within_20_acc),
        "SpearmanCorr": np.mean(spearman_corrs),
        "TrainTime(s)": np.mean(train_times)
    })

# Final results
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\nModel Performance Comparison Across All Models:")
print(results_df)
results_df.to_excel('v4o_results/Model_results/results_df.xlsx', index=False)


for name, model in models.items():
    # Last fold setup
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        break  # Use only first fold for illustration

    # Scatter Plot: Prediction vs Actual
    plt.figure(figsize=(6, 5))

    # Sort by index or value to make lines meaningful
    sorted_idx = y_val.sort_index().index  # or use y_val.sort_values().index for value-based sort
    
    plt.plot(y_val.loc[sorted_idx].values, label='Actual', linestyle='-', marker='o')
    plt.plot(y_pred[sorted_idx.argsort()], label='Predicted', linestyle='--', marker='x')  # align predictions
    
    plt.title(f'{name}: Line Plot of Actual vs Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"V4o_results/Model_results/{name}_line_plot.jpg", dpi=900)
    plt.show()

    # Residual Plot
    # residuals = y_val - y_pred
    # plt.figure(figsize=(6, 5))
    # sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    # plt.axhline(0, linestyle='--', color='red')
    # plt.title(f'{name}: Residual Plot')
    # plt.xlabel('Predicted Values')
    # plt.ylabel('Residuals (Actual - Predicted)')
    # plt.tight_layout()
    # plt.savefig(f"v3u_results/Model_results/{name}_residuals.jpg", dpi=900)
    # plt.show()



# Enhanced plots
plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='R2', palette='viridis')
plt.title('Average R² Score by Model')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/r2_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='Adjusted_R2', palette='crest')
plt.title('Adjusted R² Score by Model')
plt.ylabel('Adjusted R²')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/adjusted_r2_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='RMSE', palette='rocket')
plt.title('Average RMSE by Model')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/rmse_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='MAE', palette='flare')
plt.title('Average MAE by Model')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/mae_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='MAPE', palette='mako')
plt.title('Average MAPE by Model (%)')
plt.ylabel('MAPE (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/mape_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='MRE', palette='coolwarm')
plt.title('Average MRE by Model')
plt.ylabel('MRE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/mre_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='MaxError', palette='ch:s=-.2,r=.6')
plt.title('Average Max Error by Model')
plt.ylabel('Max Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/max_error_comparison.jpg", dpi=900)
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=results_df, x='Model', y='TrainTime(s)', palette='light:#5A9')
plt.title('Average Training Time by Model (seconds)')
plt.ylabel('Train Time (s)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("V4o_results/Evaluation_results/train_time_comparison.jpg", dpi=900)
plt.show()