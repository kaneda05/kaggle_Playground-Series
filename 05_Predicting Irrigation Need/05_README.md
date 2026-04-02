# Kaggle Diary: Playground Series s6e4 (Predicting Irrigation Need)

## 🎯 コンペ概要
* **タスク:** 灌漑（水やり）の必要性の予測 (3クラス分類: `Low`, `Medium`, `High`)
* **評価指標:** Balanced Accuracy
* **データ:** 実世界データをベースにしたディープラーニングによる合成データ

## 🚀 実装プロセスとスコア推移

### Phase 1: EDA & ベースライン構築 (LB: 0.96692)
* **データの特性:** * 欠損値やTrain/Test間のデータ分布のズレ（データドリフト）は無し。
  * `High` クラスが約3.3%しかない**極端な不均衡データ**。
* **初期アプローチ:**
  * 特徴量重要度（Feature Importance）の上位である `Soil_Moisture`, `Crop_Growth_Stage`, `Temperature_C` を軸に、ドメイン知識ベースの特徴量（乾燥指数など）を作成。
  * `StratifiedKFold` による分割と、LightGBMの `class_weight='balanced'` でクラス不均衡に対処。

### Phase 2: スコアの壁と上位陣の分析 (LB: 0.96991)
オリジナルデータ（*Irrigation Water Requirement Prediction Dataset*）を結合するData Augmentationを試すもLBは微増。メダル圏内（0.977+）の解法を分析し、以下の戦略へのシフトを決定した。

* **カテゴリ変数キラーの投入:** `Crop_Type` などのカテゴリ変数を強力に処理できる `CatBoost` を主軸に追加。
* **相対的文脈の付与:** 「特定作物ごとの平均水分量との差」など、Groupbyを用いた集約特徴量（Aggregated Features）を追加。
* **最適化アルゴリズムの活用:** Nelder-Mead法を用いて、複数モデル（LGBM, XGBoost, CatBoost）の「最適なブレンド比率」と「各クラスの予測確率閾値」を算出。

### Phase 3: 最終パイプラインの構築
Phase 2の仮説をもとに、以下の要素を統合した高度なアンサンブルモデルを構築。

1. **Augmentation:** 元データ + 合成データの結合
2. **Feature Engineering:** ドメイン特徴量 + Groupby集約特徴量
3. **Model Blending:** LightGBM + XGBoost + CatBoost
4. **Threshold Tuning:** 評価指標（Balanced Accuracy）に特化した予測閾値の最適化

## 💡 学び・Tips
* **不均衡データの評価:** Balanced Accuracyが指標の場合、単純な確率最大値(`np.argmax`)でのクラス決定は悪手。OOF予測を用いた**Threshold Tuning（閾値最適化）**がスコアを絞り出す鍵になる。
* **モデル選定:** カテゴリ変数が多い表形式データでは、CatBoostが決定木アンサンブルの中で抜きん出た性能を発揮することが多い。