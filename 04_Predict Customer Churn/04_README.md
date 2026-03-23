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