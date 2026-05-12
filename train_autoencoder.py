import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("labeled_dataset.csv")

feature_cols = ["roa", "roa_change", "leverage", "cash_to_assets", "asset_growth",
                "cash_profit_ratio", "income_change", "dsri", "rev_growth",
                "recv_growth", "rev_recv_divergence", "leverage_change", "accruals"]

X = df[feature_cols].values
y = df["is_fraud"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split: train autoencoder ONLY on clean data
clean_mask = y == 0
X_clean = X_scaled[clean_mask]
X_all = X_scaled
y_all = y

print(f"Training on {len(X_clean)} clean filings")
print(f"Will score {len(X_all)} total filings ({y_all.sum()} fraud)")

# Convert to tensors
X_clean_tensor = torch.FloatTensor(X_clean)
X_all_tensor = torch.FloatTensor(X_all)

# ============================================
# AUTOENCODER MODEL
# ============================================
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Encoder: compress 13 features down to 3
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 5),
            nn.ReLU(),
            nn.Linear(5, 3)  # bottleneck — forces compression
        )
        
        # Decoder: reconstruct back to 13
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================
# TRAINING
# ============================================
model = FraudAutoencoder(input_dim=len(feature_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining autoencoder...")
epochs = 100
batch_size = 64

for epoch in range(epochs):
    model.train()
    
    # Shuffle clean data each epoch
    indices = torch.randperm(len(X_clean_tensor))
    total_loss = 0
    batches = 0
    
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = X_clean_tensor[batch_idx]
        
        # Forward pass: try to reconstruct the input
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    avg_loss = total_loss / batches
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")

# ============================================
# SCORING: reconstruction error = anomaly score
# ============================================
print("\nScoring all filings...")
model.eval()
with torch.no_grad():
    reconstructed = model(X_all_tensor)
    # Per-sample reconstruction error
    errors = torch.mean((X_all_tensor - reconstructed) ** 2, dim=1).numpy()

df["anomaly_score"] = errors

# ============================================
# RESULTS
# ============================================
# PR-AUC
precision, recall, thresholds = precision_recall_curve(y_all, errors)
pr_auc = auc(recall, precision)
print(f"\nAutoencoder PR-AUC: {pr_auc:.4f}")

# Compare average reconstruction error: fraud vs clean
fraud_errors = errors[y_all == 1]
clean_errors = errors[y_all == 0]
print(f"\nAvg reconstruction error (clean):  {clean_errors.mean():.4f}")
print(f"Avg reconstruction error (fraud):  {fraud_errors.mean():.4f}")
print(f"Fraud error is {fraud_errors.mean()/clean_errors.mean():.1f}x higher than clean")

# Top 20 most anomalous filings
print("\n" + "="*50)
print("TOP 20 MOST ANOMALOUS FILINGS (autoencoder)")
print("="*50)
suspicious = df.nlargest(20, "anomaly_score")[["company", "ticker", "year", "anomaly_score", "is_fraud"]]
print(suspicious.to_string(index=False))

top20_hits = suspicious["is_fraud"].sum()
print(f"\nFraud cases in top 20: {top20_hits}/20 ({100*top20_hits/20:.0f}% hit rate)")

# ============================================
# COMPARE WITH XGBOOST
# ============================================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"XGBoost + SMOTE PR-AUC:  0.1193")
print(f"Autoencoder PR-AUC:      {pr_auc:.4f}")
print(f"\nKey difference: XGBoost needed fraud labels to learn.")
print(f"The autoencoder learned what 'normal' looks like with ZERO fraud examples.")
print(f"In production, this means it can catch NEW types of fraud it's never seen.")

# Save
df.to_csv("scored_dataset.csv", index=False)
print("\nSaved scores to scored_dataset.csv")

# Save the model
torch.save(model.state_dict(), "autoencoder_model.pth")
print("Saved model to autoencoder_model.pth")g