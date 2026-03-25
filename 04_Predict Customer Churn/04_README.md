# 顧客の解約（Churn）予測 ベースラインモデルの構築

## 概要
このプロジェクトは、Kaggleの「Predict Customer Churn（Playground Series）」コンペティションに参加し、顧客データからサービスの解約（Churn）を予測する二値分類モデルを構築した記録です。
データ分析の基本サイクル（データの確認 → 前処理 → モデリング → 評価 → 提出）を回すことを目的としています。

## 💡 全体を通して学んだこと（Key Learnings）
1. **EDA（探索的データ分析）の重要性**: いきなりAIに学習させるのではなく、人間がグラフを見て「解約しやすい人の特徴（アタリ）」をつけることが、後の予測の説得力に繋がる。
2. **機械学習のための「翻訳」**: モデルは文字列を理解できないため、カテゴリ変数は `0` と `1` に変換（One-Hotエンコーディング）する必要がある。
3. **評価指標の使い分け**: 離脱予測のような不均衡データ（解約者が約2割）では、単なる「正解率（Accuracy）」よりも、解約しそうな人を正しく上位にランク付けできているかを示す「ROC-AUC」が重要視される。
4. **AIの思考回路の確認**: ランダムフォレストの「特徴量重要度」を確認することで、AIの判断基準が人間の直感（EDAでの仮説）と一致しているか答え合わせができる。

---

## 🛠️ ステップバイステップの実行記録

### Step 1: データの読み込みと全体像の把握
まずはデータを読み込み、欠損値の有無やターゲット変数（Churn）の割合を確認しました。

* **学び**: 今回のデータは欠損値が0だったため、面倒な欠損値補完処理をスキップできた。ターゲット（解約者）の割合は約22.5%であった。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# データの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# データサイズと欠損値の確認
print(f"学習データのサイズ: {train.shape}")
print(train.isnull().sum())

# ターゲット（Churn）の割合を確認
train['Churn'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Churn (Exited) Proportion')
plt.show()
```

### Step 2: 探索的データ分析 (EDA)
解約（Churn）と強い関連がありそうな変数について、グラフ化して仮説を立てました。

* **学び（グラフからの洞察）**:
  1. `tenure`（利用期間）: 利用期間が短い（初期の）顧客ほど離脱しやすい。
  2. `MonthlyCharges`（月額料金）: 80〜100ドルの高価格帯で解約が集中している。
  3. `Contract`（契約形態）: Month-to-Month（単月契約）の解約が圧倒的に多い。

```python
sns.set_theme(style="whitegrid")

# 月額料金と解約の関係
plt.figure(figsize=(10, 5))
sns.histplot(data=train, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30, palette='Set2')
plt.title('Monthly Charges vs Churn')
plt.show()
```

### Step 3: データの前処理（Preprocessing）
機械学習モデルにデータを読み込ませるための準備を行いました。

* **学び**: モデルは数値しか計算できない。`object`型（文字列）の列（性別や契約形態など）を、`pd.get_dummies()` を使って `0/1` のフラグ（ダミー変数）に変換する「One-Hotエンコーディング」の手法を学んだ。

```python
# 予測に不要なIDを削除
train_processed = train.drop('id', axis=1)

# ターゲットを 0/1 に変換
train_processed['Churn'] = train_processed['Churn'].map({'Yes': 1, 'No': 0})

# カテゴリ変数をダミー変数化（One-Hot Encoding）
train_processed = pd.get_dummies(train_processed, drop_first=True)
```

### Step 4: モデルの学習と評価（Baseline Model）
最初の基準となるモデル（ベースライン）として、強力なアルゴリズムである Random Forest を使用しました。

* **学び**: データを学習用（Train）と評価用（Validation）に8:2で分割（ホールドアウト法）。解約者の割合が崩れないよう `stratify=y` を指定した。
* **結果**: ROC-AUC スコア: **0.8953** / 正解率 (Accuracy): **0.8431** （初回から実務レベルの高い精度が出た）

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

X = train_processed.drop('Churn', axis=1)
y = train_processed['Churn']

# データの分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# モデルの学習
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 予測と評価
y_pred_proba = model.predict_proba(X_valid)[:, 1]
print(f"ROC-AUC スコア: {roc_auc_score(y_valid, y_pred_proba):.4f}")
```

### Step 5: 特徴量重要度（Feature Importance）の確認
モデルがどのデータを重要視して「解約する」と判断したのかを確認しました。

* **学び**: `TotalCharges`（総支払額）と `MonthlyCharges`（月額料金）がトップ2になった。これは「料金の高さ」と「利用期間（総支払額に内包される）」という、EDAで人間が立てた仮説（アタリ）をAIも重要視しているという強力な裏付けになった。

```python
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Feature Importances')
plt.show()
```

### Step 6: テストデータの予測と提出（Submission）
未知のテストデータに対して予測を行い、Kaggle提出用のCSVファイルを作成しました。

* **学び**: テストデータに対しても、学習データと全く同じ「前処理（列の削除やダミー変数化）」を行う必要がある。

```python
test_ids = test['id']
test_processed = test.drop('id', axis=1)
test_processed = pd.get_dummies(test_processed, drop_first=True)

# 学習データと列を合わせる
test_processed = test_processed.reindex(columns=X.columns, fill_value=0)

# 確率の予測とCSV出力
test_preds = model.predict_proba(test_processed)[:, 1]
submission = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
submission.to_csv('submission.csv', index=False)
```

## 🚀 次のステップ (Next Steps)
精度のさらなる向上を目指し、以下の手法を試す予定です。
1. **アルゴリズムの変更**: Random Forest から Kaggleの王道である `LightGBM` への移行。
2. **特徴量エンジニアリング**: `TotalCharges / tenure` など、新しい列（変数）の作成。
3. **ハイパーパラメータチューニング**: モデルの設定値の最適化。

# Kaggle 顧客解約（Churn）予測：特徴量エンジニアリングと高度なアンサンブル

## 📌 アップデート概要
ベースラインモデル完成後、さらなる予測精度の向上とモデルの安定化を目指し、Kaggle上位陣の定石である「深掘りEDAからの特徴量生成」「K-Fold交差検証」「勾配ブースティング3種のアンサンブル」を実装した記録です。

## 💡 新たに学んだこと（Key Learnings）
1. **深掘りEDAとビジネス仮説の重要性**: 単純な集計だけでなく、「オプション契約数が多いほど解約しづらい（粘着度）」「計算上の支払額と実際の支払額のズレに不満が隠れている（違和感）」といったビジネスロジックに基づく仮説が、強力な特徴量を生み出す。
2. **K-Fold交差検証（Cross Validation）**: データを分割して複数回学習・評価（今回は5-Fold）を行うことで、データの「引き運（まぐれ当たり）」による過学習を防ぎ、モデルの真の実力（Out-of-Fold スコア）を正確に測ることができる。
3. **OOF予測値を使った高度なアンサンブル**: LightGBM、CatBoost、XGBoostという特性の異なる3つのモデルのOOF（評価用）予測値を使用し、総当たりで最適な加重平均（ブレンド比率）を探索することで、単体モデルの限界を突破できる。

---

## 🛠️ ステップバイステップの実行記録

### Step 1: 深掘りEDAと特徴量エンジニアリング (Feature Engineering)
初期のEDAから一歩踏み込み、顧客の「行動心理」を反映した2つの新しい特徴量を作成しました。

* **学び**: 新しく作成した `Num_Services`（オプション契約数）を積み上げ棒グラフで確認したところ、契約数が0〜1個の層は解約率が高く、4〜6個と増えるにつれて解約率が激減する明確な傾向（サービスの粘着度）を確認できた。

```python
# 1. サービスの粘着度 (Stickiness): 追加オプションの契約数をカウント
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies']
train['Num_Services'] = train[services].apply(lambda x: (x == 'Yes').sum(), axis=1)

# 2. 支払いの違和感 (Discrepancy): 実際の総支払額と計算上の総支払額の差額
train['Calculated_Total'] = train['MonthlyCharges'] * train['tenure']
train['Total_Difference'] = train['TotalCharges'] - train['Calculated_Total']
```

### Step 2: K-Fold交差検証による学習 (K-Fold CV)
新特徴量を追加したデータに対し、5分割の交差検証を行い、強力な3つの勾配ブースティングモデルを学習させました。

* **学び**: 各Fold間でスコアにばらつき（例: 0.914〜0.917）が出た。K-Foldを行うことで、このブレを吸収した安定した評価が可能になる。特に文字列（カテゴリ）データに強い `CatBoost` が単体で高いパフォーマンスを発揮した。

```python
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier

# 5-Foldの設定
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_cb = np.zeros(len(X))

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # CatBoostの学習（LightGBM, XGBoostも同様に実行）
    model_cb = CatBoostClassifier(random_state=42, verbose=0)
    model_cb.fit(X_train_fold, y_train_fold)
    
    # OOF（Out-of-Fold）予測値の保存
    oof_preds_cb[valid_idx] = model_cb.predict_proba(X_valid_fold)[:, 1]
```

### Step 3: OOFアンサンブルと最終予測 (Ensemble Blending)
3つのモデルのOOF予測値を使って0.00〜1.00の範囲で最適な重みを探索し、最終的な予測値を算出しました。

* **学び**: 最適な重みは `LightGBM: 0.15`, `CatBoost: 0.50`, `XGBoost: 0.35` となった。一番精度の高いモデル（CatBoost）を主軸にしつつ、他のモデルで予測のブレを補正する理想的なアンサンブルが完成し、スコアの限界突破に成功した。

```python
best_auc = 0
best_weights = []

# 総当たりで最適なブレンド比率を探索
for w_lgb in np.arange(0, 1.05, 0.05):
    for w_cb in np.arange(0, 1.05, 0.05):
        w_xgb = 1.0 - w_lgb - w_cb
        if w_xgb < -0.0001 or w_xgb > 1.0001: continue
            
        ensemble_oof = (w_lgb * oof_preds_lgb) + (w_cb * oof_preds_cb) + (w_xgb * oof_preds_xgb)
        auc = roc_auc_score(y, ensemble_oof)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = [w_lgb, w_cb, w_xgb]

print(f"🚀 Final Ensemble OOF AUC: {best_auc:.5f}")
```