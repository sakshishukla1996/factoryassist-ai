# FactoryAssist AI — Azure Deployment

This guide describes the **working cloud deployment** of FactoryAssist using Azure Container Apps, a private Azure Container Registry, Azure Blob Storage, and **Groq-hosted Qwen** inference. The API and UI run in Azure; model inference runs on **Groq, not Azure OpenAI**. The application does not require a developer's Mac to stay on.

For the local Ollama setup, see [Local Setup](local-setup.md).

## 1. Architecture and prerequisites

```text
Browser → Gradio (external Azure Container App)
        → FastAPI (internal Azure Container App)
        → LangGraph → FAISS index downloaded from Azure Blob
                    → Groq-hosted Qwen
        → Answer + citations
        ↘ Langfuse traces and scores
```

Prerequisites:

- Active Azure subscription with permissions to create resources and assign RBAC roles
- Azure CLI (`az`), Docker Desktop with Buildx, and the Container Apps CLI extension
- A Groq API key with access to `qwen/qwen3.8-27b`
- Langfuse credentials if observability is enabled
- A **trusted, locally built** FAISS index (`index.faiss` and `index.pkl`)
- Synthetic or otherwise approved documents only; Groq is an external inference provider

Check authentication and install the extension:

```bash
az login
az account show --query '{Name:name,State:state}' -o table
az extension add --name containerapp --upgrade --yes
```

Resource names in this guide reflect the example deployment; change them for your own subscription. Registry and storage account names must be globally unique.

```bash
RG=rg-factoryassist-dev
LOCATION=germanywestcentral
STORAGE=factoryassist5a797891
ACR=factoryassistacr5a797891
ENV=factoryassist-env
IMAGE_TAG=cloud-v2
```

## 2. Register Azure resource providers

```bash
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

Wait until relevant providers return `Registered`:

```bash
az provider show --namespace Microsoft.App --query registrationState -o tsv
az provider show --namespace Microsoft.ContainerRegistry --query registrationState -o tsv
```

## 3. Document storage and FAISS index

Create the resource group/storage account if they do not already exist:

```bash
az group create --name "$RG" --location "$LOCATION"

# Run only if the storage account does not exist:
# az storage account create --name "$STORAGE" --resource-group "$RG" \
#   --location "$LOCATION" --sku Standard_LRS --kind StorageV2
```

Create private Blob containers (requires appropriate Blob data-plane access):

```bash
az storage container create --name manufacturing-docs \
  --account-name "$STORAGE" --auth-mode login --public-access off
az storage container create --name faiss-index \
  --account-name "$STORAGE" --auth-mode login --public-access off
```

Upload synthetic documents using your existing ingestion/upload workflow. Build the FAISS index locally using the **same embedding model** used by the cloud API, then upload:

```bash
az storage blob upload-batch \
  --account-name "$STORAGE" \
  --destination faiss-index \
  --source artifacts/faiss \
  --auth-mode login --overwrite
```

Verify the private index container contains `index.faiss` and `index.pkl`:

```bash
az storage blob list --account-name "$STORAGE" \
  --container-name faiss-index --auth-mode login \
  --query '[].{Name:name,Bytes:properties.contentLength}' -o table
```

**Important:** `index.pkl` can execute code when deserialized. Keep the container private, restrict write permissions, and load only an index created by the trusted ingestion pipeline. The API downloads the index to ephemeral local storage and caches the loaded vector store in memory; uploads do **not** automatically refresh running replicas.

## 4. Azure Container Registry and the image

Create a private, Basic-tier registry if it does not already exist:

```bash
az acr create --resource-group "$RG" --name "$ACR" \
  --location "$LOCATION" --sku Basic --admin-enabled false
az acr login --name "$ACR"
```

The Dockerfile uses `uv` to install Python dependencies. Verify `.dockerignore` excludes `.env`, `.git/`, `.venv/`, and `artifacts/` before building.

Azure Container Apps requires a `linux/amd64` image. On an Apple Silicon machine, explicitly build for AMD64:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile \
  --tag "${ACR}.azurecr.io/factoryassist:${IMAGE_TAG}" \
  --provenance=false \
  --push \
  --progress=plain \
  .
```

Verify the tag:

```bash
az acr repository show-tags --name "$ACR" \
  --repository factoryassist -o table
```

**Why not `az acr build`?** Some subscriptions, including the example subscription used for this project, restrict ACR Tasks with `TasksOperationsNotAllowed`. Direct Docker Buildx push avoids that restriction. Builds on Apple Silicon can take time due to AMD64 emulation.

## 5. Create the Container Apps environment

```bash
az containerapp env create --name "$ENV" \
  --resource-group "$RG" --location "$LOCATION" \
  --environment-mode ConsumptionOnly \
  --logs-destination none
```

This example disables persistent Azure application logs to simplify the initial deployment. Enable Azure Monitor / Log Analytics for operational troubleshooting in a longer-lived deployment.

## 6. Create managed identities and least-privilege roles

Create separate identities for the API and UI:

```bash
az identity create --name factoryassist-api-identity \
  --resource-group "$RG" --location "$LOCATION" -o none
az identity create --name factoryassist-ui-identity \
  --resource-group "$RG" --location "$LOCATION" -o none

API_ID=$(az identity show --name factoryassist-api-identity \
  --resource-group "$RG" --query id -o tsv)
API_PRINCIPAL=$(az identity show --name factoryassist-api-identity \
  --resource-group "$RG" --query principalId -o tsv)
API_CLIENT_ID=$(az identity show --name factoryassist-api-identity \
  --resource-group "$RG" --query clientId -o tsv)
UI_ID=$(az identity show --name factoryassist-ui-identity \
  --resource-group "$RG" --query id -o tsv)
UI_PRINCIPAL=$(az identity show --name factoryassist-ui-identity \
  --resource-group "$RG" --query principalId -o tsv)
ACR_ID=$(az acr show --name "$ACR" --resource-group "$RG" --query id -o tsv)
STORAGE_ID=$(az storage account show --name "$STORAGE" --resource-group "$RG" --query id -o tsv)
```

Grant **AcrPull** to each identity (assumes the registry uses standard RBAC rather than repository ABAC):

```bash
for PRINCIPAL in "$API_PRINCIPAL" "$UI_PRINCIPAL"; do
  az role assignment create \
    --assignee-object-id "$PRINCIPAL" \
    --assignee-principal-type ServicePrincipal \
    --role AcrPull --scope "$ACR_ID" -o none
done
```

Only the API needs read access to the private FAISS container:

```bash
az role assignment create \
  --assignee-object-id "$API_PRINCIPAL" \
  --assignee-principal-type ServicePrincipal \
  --role 'Storage Blob Data Reader' \
  --scope "$STORAGE_ID/blobServices/default/containers/faiss-index" \
  -o none
```

RBAC permissions may take several minutes to propagate.

## 7. Deploy the private FastAPI backend

Export `GROQ_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` securely **in the terminal session** first. Never put real values in Git or committed deployment scripts. Check presence without printing them:

```bash
: "${GROQ_API_KEY:?GROQ_API_KEY is required}"
: "${LANGFUSE_PUBLIC_KEY:?LANGFUSE_PUBLIC_KEY is required}"
: "${LANGFUSE_SECRET_KEY:?LANGFUSE_SECRET_KEY is required}"
```

Deploy FastAPI with internal ingress. `--command uvicorn --args factoryassist.api.main:app` avoids Azure CLI parsing problems with shell `-c` arguments.

```bash
az containerapp create \
  --name factoryassist-api \
  --resource-group "$RG" --environment "$ENV" \
  --image "${ACR}.azurecr.io/factoryassist:${IMAGE_TAG}" \
  --registry-server "${ACR}.azurecr.io" \
  --user-assigned "$API_ID" --registry-identity "$API_ID" \
  --ingress internal --target-port 8000 --transport http \
  --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 1 \
  --command uvicorn --args factoryassist.api.main:app \
  --secrets \
    "groq-api-key=$GROQ_API_KEY" \
    "langfuse-secret-key=$LANGFUSE_SECRET_KEY" \
  --env-vars \
    MODEL_PROVIDER=groq \
    GROQ_MODEL=qwen/qwen3.8-27b \
    GROQ_API_KEY=secretref:groq-api-key \
    INDEX_SOURCE=azure \
    AZURE_STORAGE_ACCOUNT="$STORAGE" \
    AZURE_FAISS_CONTAINER=faiss-index \
    AZURE_CLIENT_ID="$API_CLIENT_ID" \
    LANGFUSE_PUBLIC_KEY="$LANGFUSE_PUBLIC_KEY" \
    LANGFUSE_SECRET_KEY=secretref:langfuse-secret-key \
    LANGFUSE_BASE_URL=https://cloud.langfuse.com \
    MAX_QUESTION_TOKENS=256 \
    MAX_CONTEXT_TOKENS=2000 \
    MAX_INPUT_TOKENS=3000 \
    MAX_OUTPUT_TOKENS=384 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    PYTHONUNBUFFERED=1 \
  --output none
```

The CLI receives secret values during this initial setup; for production, prefer Azure Key Vault references and avoid leaking secrets in terminal recordings or process inspection. Use `--min-replicas 1` initially for reliable debugging; this incurs charges while running.

Verify:

```bash
az containerapp revision list --name factoryassist-api \
  --resource-group "$RG" \
  --query '[].{Name:name,Running:properties.runningState,Health:properties.healthState}' \
  -o table
```

## 8. Deploy the Gradio frontend

Gradio needs image-pull permission but **does not need Groq keys or Blob permissions**. It calls the internal API by service name:

```bash
az containerapp create \
  --name factoryassist-ui \
  --resource-group "$RG" --environment "$ENV" \
  --image "${ACR}.azurecr.io/factoryassist:${IMAGE_TAG}" \
  --registry-server "${ACR}.azurecr.io" \
  --user-assigned "$UI_ID" --registry-identity "$UI_ID" \
  --ingress external --target-port 7860 --transport http \
  --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 \
  --command python --args /app/src/factoryassist/ui/gradio_app.py \
  --env-vars \
    FACTORYASSIST_API_URL=http://factoryassist-api/ask \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1 \
  --output none
```

**Important UI setting:** `gradio_app.py` must read `FACTORYASSIST_API_URL` using `os.getenv(...)`. Hardcoding `http://127.0.0.1:8000/ask` inside a cloud container will fail because it points back to the UI container, not the API. The Gradio HTTP client should allow sufficient time for retrieval and generation (e.g. a 120-second timeout).

Verify and obtain the URL:

```bash
az containerapp revision list --name factoryassist-ui \
  --resource-group "$RG" \
  --query '[].{Name:name,Running:properties.runningState,Health:properties.healthState}' \
  -o table

az containerapp show --name factoryassist-ui --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv
```

Open the returned hostname with `https://`.

## 9. Restrict and verify access

The Gradio endpoint has external ingress. Restrict the demo to your IP (or implement authenticated access before sharing broadly):

```bash
MY_IP=$(curl -4 -fsS https://api.ipify.org)
az containerapp ingress access-restriction set \
  --name factoryassist-ui --resource-group "$RG" \
  --rule-name AllowMyLaptop --ip-address "${MY_IP}/32" --action Allow
```

IP restrictions are **not user authentication** and your public IP may change.

Test the internal backend from the UI container:

```bash
az containerapp exec --name factoryassist-ui \
  --resource-group "$RG" --command /bin/sh
```

Inside the container:

```bash
python -c "import httpx; r=httpx.get('http://factoryassist-api/health',timeout=15); print(r.status_code,r.text)"
```

Expected: `200 {"status":"ok"}`. Then submit a sample question in Gradio:

> What causes error E-2047 in machine M-204, and how should maintenance personnel respond?

Check for an evidence-grounded answer, source citations, and a Langfuse trace. A healthy revision alone doesn't prove that the complete RAG workflow succeeds.

## 10. Updates and costs

For code changes, build a **new immutable image tag** and update only the affected service:

```bash
NEW_TAG=cloud-v3

docker buildx build --platform linux/amd64 --file Dockerfile \
  --tag "${ACR}.azurecr.io/factoryassist:${NEW_TAG}" \
  --provenance=false --push .

az containerapp update --name factoryassist-ui \
  --resource-group "$RG" \
  --image "${ACR}.azurecr.io/factoryassist:${NEW_TAG}"
```

Update `factoryassist-api` instead if its own code changed. The example uses one shared image for both services; a future optimisation is separate, smaller UI and API images.

Costs to monitor: Container Apps replicas, Basic ACR storage/service charges, Blob Storage, and Groq's separate rate limits or paid usage. **Azure credits do not pay Groq bills.** For an occasional demo, consider scale-to-zero after verifying cold-start behaviour:

```bash
az containerapp update --name factoryassist-api \
  --resource-group "$RG" --min-replicas 0 --max-replicas 1
az containerapp update --name factoryassist-ui \
  --resource-group "$RG" --min-replicas 0 --max-replicas 1
```

Scale-to-zero can introduce significant cold starts, especially when the API needs to download and load embedding models. Use `min-replicas 1` when demonstrating and monitor actual Azure spending. The Basic ACR and storage can continue accruing charges even if the apps scale to zero.

## 11. Known limitations and future work

- The fixed FAISS index is suitable for the current small, synthetic corpus; a production system would need index versioning, refresh, access control, and potentially a managed search service.
- Prototype retrieval requires explicit machine/error identifiers and is not a general enterprise retrieval solution.
- Groq and Langfuse are external services: review privacy, geographic processing, retention, and organisational policy before real customer data is used.
- Source citation presence and valid IDs do not guarantee factual grounding.
- Add authentication, operational logging, alerts, rate limiting, automated tests, and CI/CD before treating this as production-ready.
- Azure OpenAI was considered, but this demo uses Groq because the subscription had no available quota for the selected Azure OpenAI model/deployment SKU.