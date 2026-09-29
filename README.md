**Live Demo：<https://svm-kernel-2nhlfdtkz-newmume-s-projects.vercel.app/>**

# SVM Kernel Lab

這是一個互動式 SVM（Support Vector Machine）教學網頁。調整參數後，系統會用 scikit-learn 重新訓練模型，並同步更新 2D 決策邊界、3D decision function 與模型指標。

## 網頁使用說明

1. 在 **Dataset geometry** 選擇資料形狀：同心圓環、交錯月牙、XOR 四象限或線性可分。
2. 在 **Kernel function** 切換 Linear、Polynomial、RBF 或 Sigmoid，比較不同 kernel 能形成的決策邊界。
3. 調整模型參數：
   - **C**：錯分懲罰強度。
   - **Gamma γ**：單一樣本的影響範圍；適用於非線性 kernel。
   - **Degree**：Polynomial kernel 的多項式階數。
   - **Coef0**：Polynomial 與 Sigmoid kernel 的獨立項。
   - **Class weight · balanced**：依類別數量自動調整權重。
4. 使用 **Data Generator** 改變資料點數、特徵雜訊、標籤雜訊、類別比例、分離程度與 random seed；同心圓環資料還能調整內外半徑。
5. 展開 **View options**，控制 heatmap、決策邊界、margin、support vectors、3D surface、wireframe、floor projection、零平面、透明度及配色。
6. 展開 **3D camera** 調整方位角、仰角與縮放，也可以直接拖曳旋轉 3D 圖、滾輪縮放、雙擊重設視角。
7. 觀察四項即時計算結果：Training Accuracy、Support Vectors、Margin Region 與 Boundary Complexity。
8. 使用 **Reset** 回到預設值、**Random** 更換隨機種子，或按 **Export CSV** 匯出目前資料點、decision score 與 support-vector 標記。

建議先選擇 RBF，依序比較 γ = 0.1、1、10；再加入 label noise，比較 C = 0.3 與 100。最後把同心圓環切換成 Linear，即可直接觀察線性模型對非線性資料的限制。

## 數學說明

### Soft-margin SVM

SVM 在容許部分樣本違反 margin 的情況下，求解：

$$
\min_{w,b,\xi}\ \frac{1}{2}\lVert w\rVert^2 + C\sum_{i=1}^{n}\xi_i
$$

並滿足：

$$
y_i\bigl(w^T\phi(x_i)+b\bigr) \ge 1-\xi_i,\qquad \xi_i\ge0
$$

$C$ 越大，模型越重視訓練錯誤，邊界可能更貼近資料；$C$ 越小，模型容許更多錯分，通常可得到較寬、較平滑的 margin。

### Decision function 與分類規則

Kernel SVM 的 decision function 為：

$$
f(x)=\sum_{i\in SV}\alpha_i y_i K(x_i,x)+b
$$

- $f(x)>0$ 時預測為 Class $+1$；$f(x)<0$ 時預測為 Class $-1$。
- $f(x)=0$ 是決策邊界，$f(x)=\pm1$ 是 margin 線。
- Support vectors 是決定邊界位置的關鍵樣本。
- 網頁的 3D 高度是 $z=f(x_1,x_2)$，不是 kernel 映射後的真實特徵空間；RBF 對應的是高維、甚至無限維的特徵映射。

### Kernel functions

| Kernel | 公式 | 直觀解讀 |
| --- | --- | --- |
| Linear | $K(x,z)=x^Tz$ | 在輸入空間建立線性邊界。 |
| Polynomial | $K(x,z)=(\gamma x^Tz+r)^d$ | 用階數 $d$ 表示多項式交互作用。 |
| RBF / Gaussian | $K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$ | 依兩點距離衡量相似度，可建立局部且彎曲的邊界。 |
| Sigmoid | $K(x,z)=\tanh(\gamma x^Tz+r)$ | 形式類似神經網路的啟用函數。 |

對 RBF 而言，較小的 $\gamma$ 代表單一樣本影響範圍較廣，邊界較平滑；較大的 $\gamma$ 代表影響範圍較窄，邊界會更貼近個別資料點，也更容易過度擬合。
