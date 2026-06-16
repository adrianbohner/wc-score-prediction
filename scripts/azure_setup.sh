#!/bin/bash
# =============================================================
# WC Score Prediction — Azure one-time setup
#
# Idempotent: safe to run multiple times.
# Prerequisites:
#   - Azure CLI >= 2.50  (az login already done)
#   - jq
#   - models/match_score_model.pkl already trained locally
#
# Usage:
#   bash scripts/azure_setup.sh
# =============================================================

# ---- EDIT THESE BEFORE RUNNING ------------------------------
GITHUB_ORG="adrianbohner"        # e.g. adrianbohner
GITHUB_REPO="wc-score-prediction"        # GitHub repo name
APP_NAME="wc-score-prediction"           # Must be globally unique (becomes <APP_NAME>.azurewebsites.net)
STORAGE_ACCOUNT_NAME="stwcprediction"    # Globally unique, 3-24 lowercase alphanumeric only
STORAGE_CONTAINER_NAME="models"
LOCATION="westeurope"
RESOURCE_GROUP="rg-sandbox-wc-westeu"
SP_NAME="sp-wc-prediction-cicd"
# -------------------------------------------------------------

set -euo pipefail

# Prevent Git Bash on Windows from converting /subscriptions/... paths to C:/Program Files/Git/...
export MSYS_NO_PATHCONV=1

# Resolve subscription and tenant from the active az login session
TENANT_ID=$(az account show --query tenantId -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "Tenant:       ${TENANT_ID}"
echo "Subscription: ${SUBSCRIPTION_ID}"
echo ""

# ------------------------------------------------------------------
# 1. Resource group
# ------------------------------------------------------------------
echo "[1/8] Creating resource group..."
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

# ------------------------------------------------------------------
# 2. Storage account + blob container
# ------------------------------------------------------------------
echo "[2/8] Creating storage account..."
if ! az storage account show \
    --name "${STORAGE_ACCOUNT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --output none 2>/dev/null; then
  az storage account create \
    --name "${STORAGE_ACCOUNT_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --sku Standard_LRS \
    --min-tls-version TLS1_2 \
    --allow-blob-public-access false \
    --output none
else
  echo "  Storage account already exists, skipping."
fi

ACCOUNT_KEY=$(az storage account keys list \
  --account-name "${STORAGE_ACCOUNT_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --query "[0].value" -o tsv)

az storage container create \
  --name "${STORAGE_CONTAINER_NAME}" \
  --account-name "${STORAGE_ACCOUNT_NAME}" \
  --account-key "${ACCOUNT_KEY}" \
  --output none

# ------------------------------------------------------------------
# 3. Upload initial artifacts
# ------------------------------------------------------------------
echo "[3/8] Uploading initial artifacts to blob storage..."
if [ -f "models/match_score_model.pkl" ]; then
  az storage blob upload \
    --account-name "${STORAGE_ACCOUNT_NAME}" \
    --account-key "${ACCOUNT_KEY}" \
    --container-name "${STORAGE_CONTAINER_NAME}" \
    --name match_score_model.pkl \
    --file models/match_score_model.pkl \
    --overwrite \
    --output none
  echo "  Uploaded models/match_score_model.pkl"
else
  echo "  WARNING: models/match_score_model.pkl not found locally."
  echo "  Train the model first (python -m wc_predictor.models.train_app_model),"
  echo "  then upload manually:"
  echo "    az storage blob upload --account-name ${STORAGE_ACCOUNT_NAME} \\"
  echo "      --account-key <key> --container-name ${STORAGE_CONTAINER_NAME} \\"
  echo "      --name match_score_model.pkl --file models/match_score_model.pkl --overwrite"
fi

if [ -f "data/raw/results.csv" ]; then
  az storage blob upload \
    --account-name "${STORAGE_ACCOUNT_NAME}" \
    --account-key "${ACCOUNT_KEY}" \
    --container-name "${STORAGE_CONTAINER_NAME}" \
    --name results.csv \
    --file data/raw/results.csv \
    --overwrite \
    --output none
  echo "  Uploaded data/raw/results.csv"
fi

# ------------------------------------------------------------------
# 4. Generate read-only SAS URL for model artifact (expires 2028-12-31)
#    This URL is set as an App Service env var — NOT stored in GitHub.
# ------------------------------------------------------------------
echo "[4/8] Generating SAS URL for model artifact..."
SAS_TOKEN=$(az storage blob generate-sas \
  --account-name "${STORAGE_ACCOUNT_NAME}" \
  --account-key "${ACCOUNT_KEY}" \
  --container-name "${STORAGE_CONTAINER_NAME}" \
  --name match_score_model.pkl \
  --permissions r \
  --expiry "2028-12-31T00:00:00Z" \
  --https-only \
  --output tsv)

MODEL_ARTIFACT_SAS_URL="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${STORAGE_CONTAINER_NAME}/match_score_model.pkl?${SAS_TOKEN}"

# ------------------------------------------------------------------
# 5. App Service Plan + Web App
# ------------------------------------------------------------------
echo "[5/8] Creating App Service Plan (B1 Linux)..."
if ! az appservice plan show \
    --name "asp-wc-prediction" \
    --resource-group "${RESOURCE_GROUP}" \
    --output none 2>/dev/null; then
  az appservice plan create \
    --name "asp-wc-prediction" \
    --resource-group "${RESOURCE_GROUP}" \
    --is-linux \
    --sku B1 \
    --output none
else
  echo "  App Service Plan already exists, skipping."
fi

echo "[5/8] Creating Web App (Python 3.11)..."
if ! az webapp show \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --output none 2>/dev/null; then
  az webapp create \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --plan "asp-wc-prediction" \
    --runtime "PYTHON:3.11" \
    --output none
else
  echo "  Web App already exists, skipping."
fi

# ------------------------------------------------------------------
# 6. App Service configuration
#    These are RUNTIME environment variables — live in Azure App Settings,
#    never in GitHub Secrets or the repository.
# ------------------------------------------------------------------
echo "[6/8] Configuring App Settings and startup command..."
az webapp config appsettings set \
  --name "${APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --settings \
    MODEL_ARTIFACT_SAS_URL="${MODEL_ARTIFACT_SAS_URL}" \
    SCM_DO_BUILD_DURING_DEPLOYMENT="true" \
    WEBSITES_PORT="8000" \
  --output none

az webapp config set \
  --name "${APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --startup-file "bash scripts/start.sh" \
  --always-on true \
  --output none

# ------------------------------------------------------------------
# 7. Service principal for GitHub Actions (OIDC — no client secret)
# ------------------------------------------------------------------
echo "[7/8] Creating service principal for GitHub Actions (OIDC)..."
EXISTING_CLIENT_ID=$(az ad sp list \
  --display-name "${SP_NAME}" \
  --query "[0].appId" -o tsv 2>/dev/null)

if [ -n "${EXISTING_CLIENT_ID}" ] && [ "${EXISTING_CLIENT_ID}" != "None" ]; then
  echo "  Service principal already exists, reusing."
  SP_CLIENT_ID="${EXISTING_CLIENT_ID}"
else
  SP_JSON=$(az ad sp create-for-rbac \
    --name "${SP_NAME}" \
    --role contributor \
    --scopes "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}" \
    --output json)
  SP_CLIENT_ID=$(echo "${SP_JSON}" | jq -r .appId)
fi

SP_OBJECT_ID=$(az ad sp show --id "${SP_CLIENT_ID}" --query id -o tsv)

# Grant Storage Blob Data Contributor so the retrain workflow can upload artifacts
EXISTING_ROLE=$(az role assignment list \
  --assignee "${SP_OBJECT_ID}" \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT_NAME}" \
  --query "[0].id" -o tsv 2>/dev/null)

if [ -z "${EXISTING_ROLE}" ] || [ "${EXISTING_ROLE}" = "None" ]; then
  az role assignment create \
    --assignee "${SP_OBJECT_ID}" \
    --role "Storage Blob Data Contributor" \
    --scope "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT_NAME}" \
    --output none
else
  echo "  Storage Blob Data Contributor already assigned, skipping."
fi

# OIDC federated credential — covers push to main AND workflow_dispatch from main
EXISTING_CRED=$(az ad app federated-credential list \
  --id "${SP_CLIENT_ID}" \
  --query "[?name=='github-main'].id" -o tsv 2>/dev/null)

if [ -z "${EXISTING_CRED}" ] || [ "${EXISTING_CRED}" = "None" ]; then
  az ad app federated-credential create \
    --id "${SP_CLIENT_ID}" \
    --parameters "{
      \"name\": \"github-main\",
      \"issuer\": \"https://token.actions.githubusercontent.com\",
      \"subject\": \"repo:${GITHUB_ORG}/${GITHUB_REPO}:ref:refs/heads/main\",
      \"audiences\": [\"api://AzureADTokenExchange\"]
    }" \
    --output none
else
  echo "  OIDC federated credential already exists, skipping."
fi

# ------------------------------------------------------------------
# 8. Print GitHub Secrets to add
# ------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Add these 7 values to GitHub Secrets:"
echo "  (Repo → Settings → Secrets and variables → Actions → New secret)"
echo "================================================================"
echo "  AZURE_TENANT_ID              = ${TENANT_ID}"
echo "  AZURE_SUBSCRIPTION_ID        = ${SUBSCRIPTION_ID}"
echo "  AZURE_RESOURCE_GROUP         = ${RESOURCE_GROUP}"
echo "  AZURE_WEBAPP_NAME            = ${APP_NAME}"
echo "  AZURE_CLIENT_ID              = ${SP_CLIENT_ID}"
echo "  AZURE_STORAGE_ACCOUNT_NAME   = ${STORAGE_ACCOUNT_NAME}"
echo "  AZURE_STORAGE_CONTAINER_NAME = ${STORAGE_CONTAINER_NAME}"
echo "================================================================"
echo ""
echo "  MODEL_ARTIFACT_SAS_URL has been written directly to App Service"
echo "  App Settings — it is a runtime secret, NOT a GitHub Secret."
echo "  It expires 2028-12-31. Regenerate it before that date."
echo ""
echo "  App URL: https://${APP_NAME}.azurewebsites.net"
echo "================================================================"
