# SVM Kernel Lab V2

科技感、可互動的 SVM 教學儀表板。新版獨立放在 `V2 demo`，不會覆寫原始版本。

## 啟動

```powershell
cd "E:\Mume_AI\NCHU HW\20260618-SVM\V2 demo"
pip install -r requirements.txt
streamlit run app.py
```

## 本版可調項目

- 模型：Kernel、C、Gamma、Polynomial degree、Coef0、class weight。
- 資料：同心圓環、交錯月牙、XOR、線性可分、樣本數、feature noise、label noise、類別比例、間距、圓環半徑與 random seed。
- 視覺：decision heatmap、boundary、margin、support vectors、3D surface、wireframe、2D floor projection、f=0 plane、透明度、色盤、網格精度與 3D 鏡頭。
- 操作：Reset、Random data、CSV export、Plotly 旋轉／縮放／截圖。

## 數學說明

3D 圖顯示的是模型的 decision function `z = f(x, y)`，不是把 RBF 特徵空間畫成三維。RBF kernel 對應高維或無限維特徵映射；`z = x² + y²` 則是用來解釋 kernel trick 的教學用顯式映射。
