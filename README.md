# SVM Kernel Lab V2 — Vercel Native

互動式 SVM 教學儀表板，使用 Vercel Services 同時部署：

- `frontend/`：Next.js App Router + Plotly.js。
- `backend/`：FastAPI + NumPy + scikit-learn。
- `/api/model`：即時訓練 SVC 並回傳資料點、decision grid、support vectors 與模型指標。

## 功能

- Linear、Polynomial、RBF、Sigmoid 四種 kernel。
- 同心圓環、交錯月牙、XOR、線性可分四種資料形狀。
- 調整 C、Gamma、Degree、Coef0、雜訊、標籤錯置、類別比例、資料間距與 random seed。
- 2D decision boundary、margin、support vectors、3D decision function、floor projection、wireframe 與鏡頭控制。
- 即時計算 training accuracy、support-vector ratio、margin region 與 boundary complexity。

## 本機開發

### 分開啟動

```powershell
pip install -e .\backend
python -m uvicorn backend.main:app --reload --port 8000

cd frontend
npm install
npm run dev
```

分開啟動時，前端需自行代理 `/api` 到 `http://localhost:8000`。完整的同源環境建議使用 Vercel CLI。

### Vercel Services 同源環境

```powershell
npm install --global vercel
vercel dev -L
```

開啟 `http://localhost:3000`。健康檢查位於 `/api/health`。

## 部署到 Vercel

1. 將 GitHub repository 連接到 Vercel。
2. 專案根目錄維持 repository root，不要改成 `frontend/` 或 `backend/`。
3. Framework Preset 選擇 **Services**；`vercel.json` 會分別建立 Next.js 與 FastAPI service。
4. 重新部署。`/api/*` 會交由 FastAPI，其餘路徑交由 Next.js。

## 數學說明

3D 圖顯示的是模型 decision function `z = f(x, y)`，不是把 RBF 特徵空間畫成三維。RBF kernel 對應高維或無限維特徵映射；`z = x² + y²` 則是解釋 kernel trick 的教學用顯式映射。
