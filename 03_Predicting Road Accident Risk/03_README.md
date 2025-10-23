# Predicting Road Accident Risk
![image](https://github.com/kaneda05/kaggle_Playground-Series/blob/main/png/003_playground.png)

[kaggleページ](https://www.kaggle.com/competitions/playground-series-s5e10/overview)]

# Kaggleコンペ概要: Playground Series S5E10 (Predicting Road Accident Risk)

---

## 概要：Kaggle Playground Series S5E10
このコンペティションは、KaggleのPlaygroundシリーズの一部であり、初心者から中級者が機械学習スキルを練習するためのもの。

### 🎯 目的 (Your Goal)

**道路の種類**に基づいて、**事故の可能性（`accident_risk`）**を予測。
予測値は0から1の間の数値（確率）である必要がある。

### 📈 評価指標 (Evaluation)
**RMSE (Root Mean Squared Error)**：予測値と実際の値との間の二乗誤差の平方根で評価される。

### 🗓️ タイムライン (Timeline)

* **開始日:** 2025年10月1日
* **最終提出期限:** 2025年10月31日

### ✨ 特別チャレンジ (Stack Overflow)

このコンペはStack Overflowとの2部構成チャレンジの第1部です。
両方のチャレンジを完了すると、KaggleとStack Overflowの両方で**「Code Scientist」バッジ**が付与される。

### 💾 データ (Data)
データセットは、実世界のデータから**合成的に生成**されたものが使用されている。

---

## データセット概要
### 💾 元データと生成方法
* このコンペのデータセット（`train.csv`, `test.csv`）は、「**Simulated Roads Accident**」というオリジナルのデータセットで訓練された深層学習（Deep Learning）モデルによって**合成的に生成**
* 特徴量の分布は、オリジナルデータと**近いですが、完全には一致しない**。

### 📂 ファイル一覧
* **`train.csv`**: 訓練用データセット。
    * 目的変数（ターゲット）: `accident_risk` （0から1の間の連続値）
* **`test.csv`**: テスト用データセット。
    * このデータの`accident_risk`を予測。
* **`sample_submission.csv`**: 提出ファイルの正しい形式を示したサンプル。

---

## 2025/10/23
**Score:0.05539**
[PS-s5e10 | simple vote](https://www.kaggle.com/code/masakane/ps-s5e10-simple-vote/edit)
kaggleでよく使われている複数の予測結果を組み合わせて、より精度の高い最終的な予測ファイルを作成する**アンサンブル（ブレンディング）**を実行してる。

[🚑 [Road Accident Risk] - AutoGluon 📈](https://www.kaggle.com/code/aliffaagnur/road-accident-risk-autogluon)