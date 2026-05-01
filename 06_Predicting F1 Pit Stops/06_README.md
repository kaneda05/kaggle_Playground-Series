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


---

# Kaggle Diary: Playground Series s6e4 (Update)

## 🎯 現在のステータス
* **Best LB Score:** `0.97081` (Previous: `0.96991` -> `0.97040`)
* **CV Score (OOF):** `0.97220`

## 🚀 実装プロセスと実験の軌跡 (Phase 4 - 6)

### Phase 4: 上位陣の手法の分析と導入
LB 0.977+のトップ層のソリューションを分析し、以下の強力なアプローチをパイプラインに組み込んだ。

1. **Pairwise Combinations (ペアワイズ特徴量):**
   * `Crop_Type` と `Season` など、関連性の高いカテゴリ変数を文字列結合し、決定木が条件分岐しやすい複合カテゴリ特徴量を生成。
2. **Target Encoding (TE) の厳格な適用:**
   * 生成したペアワイズ特徴量に対してTEを適用。
   * **[重要]** ターゲットリーク（過学習）を防ぐため、必ずKFoldのFold内部で計算を実行。さらに `smoothing=50.0`, `min_samples_leaf=20` という強めの正則化をかけ、希少なカテゴリの過信を防止した。
3. **モデルのハイブリッド化:**
   * アンサンブルの構成を見直し、TE済みの数値特徴量に強い **XGBoost** と、生のカテゴリ文字列の処理に秀でた **CatBoost** の2モデル構成（LightGBMを削除）へシフトした。

### Phase 5: 最適化の罠と「大暴落からのリカバリー」
* **失敗した仮説 (CV: 0.42042):**
  * 「学習時の `class_weight`（不均衡対策）」と「予測後の Nelder-Mead法（閾値最適化）」の二重がけ（Double-dipping）は過剰対策だと仮定し、学習時の重みを削除して純粋な確率を出力させた。
  * **結果:** 評価指標（Balanced Accuracy）が 0.42042 へ大暴落。
* **原因と学び (The Gradient Problem):**
  * 極端な不均衡データ（Highクラスが約3%）において生の確率を出力させると、Nelder-Mead法が「少し閾値を動かしても予測結果（Low/Medium）が変わらない」という勾配消失に陥り、最適化がストップすることが判明した。
  * **結論:** 予測確率に `class_weight` で「ゲタ」を履かせることが、後処理の最適化アルゴリズムを正常に駆動させるための必須条件であった。重みを復活させた結果、CVは `0.97220` へ劇的に回復。

### Phase 6: 究極の堅牢化 (Seed Averaging)
Private LBでのShake-downを防ぎ、スコアを限界まで絞り出すため、以下のパイプラインで最終提出物を構築した。

* **Seed Averaging:** 3つの異なる乱数シード（`42, 2024, 777`）を使用。
* **総モデル数:** 3シード × 5Fold × 2モデル = **計30モデル** の予測確率を平均化。
* **最適化ブレンディング:** Nelder-Mead法により、XGBoostとCatBoostの最適なアンサンブル比率（概ね `0.58 : 0.42`）を算出し、最終的なクラス予測閾値を決定。

## 💡 今後の課題とネクストステップ
CV `0.97220` に対して LB `0.97081` と、手元と本番のスコアは綺麗に連動しているが、トップ層の `0.977` にはまだ届いていない。
上位陣との残りの差分は、以下のいずれかに潜んでいる可能性が高い。
* Optunaを用いた数時間単位の徹底的なハイパーパラメータ探索
* Target Encoding以外のエンコーディング手法（CatBoostの内部処理への完全依存など）
* まだ見落としている強力なドメイン集約特徴量

---
# Kaggle Diary: Playground Series s6e4 (Update - Phase 7)

## 🎯 現在のステータス
* **Best LB Score:** `0.97081` (前回のSeed Averagingアンサンブル時のスコア)
* **Current Strategy:** Single CatBoost + Drift-Aware + Pairwise TE

## 🚀 実装プロセスと実験の軌跡 (Phase 7)

### Phase 7: 上位陣の「一点突破」戦略の模倣 (Single CatBoost & Drift-Aware)
LB 0.977〜0.979を叩き出している最上位層のアプローチを分析し、これまでの「メガ・アンサンブル」路線から一転して**CatBoost単体への特化**と**データドリフトへの適応**へと舵を切った。

#### 1. Single CatBoostへの特化 (XGBoostのパージ)
* **仮説:** ペアワイズ特徴量（`Crop_Season`など）とTarget Encodingが複雑に絡み合うデータ空間において、XGBoostの予測が逆にアンサンブルのノイズになっている可能性が浮上。
* **施策:** XGBoostをパイプラインから完全に削除。浮いた計算リソースをCatBoostに全振りし、イテレーション数（木の数）を1200から1500に増やし、学習率を下げて単一モデルの極限を目指した。

#### 2. Drift-Aware Weighting (データドリフト適応)
* **仮説:** Data Augmentationで追加した「Originalデータ（実データ）」は強力だが、予測対象であるTestデータはTrainデータと同じロジックで作られた「合成データ」である。Originalデータを等価に扱うと、Testデータの分布から微妙にズレる（Covariate Shift）。
* **施策:** 学習時のサンプルウェイトに「Drift-Aware（ドリフト考慮）」の概念を導入。
  * `合成Trainデータ : 1.0`
  * `Originalデータ : 0.5`
  * このドリフト比率と、不均衡対策のクラスウェイト（`class_weight='balanced'`）を掛け合わせた**「究極のカスタムウェイト (`sw_final`)」**を作成し、モデルの学習に適用した。

#### 3. ハイブリッド・カテゴリ処理
* Target Encodingを施した「数値特徴量」と、元の「生の文字列カテゴリ特徴量」を**両方ともCatBoostに投入**。TEの強力な確率情報と、CatBoost特有の高度なカテゴリ処理アルゴリズムのメリットを両取りする構造にした。

## 💡 学び・Tips
* **「引き算」の勇気:** アンサンブルはKaggleの鉄板だが、特定のデータセット（特にカテゴリ変数が支配的な場合）においては、最強の単一モデル（CatBoost）のノイズになることがある。
* **Drift-Aware (分布のシフト対策):** 合成データのコンペにおいて、外部データ（実データ）は強力な武器になるが、Testデータとの「性質の違い」をサンプルウェイトで補正するテクニックは非常に汎用性が高い。

## ⏩ ネクストステップ
現在の「Single CatBoost + Drift-Aware」パイプラインのLBスコアを確認する。
この一点突破戦略でスコアが跳ねた場合、最後の一手として**Optuna**を導入し、CatBoostのハイパーパラメータ（`depth`, `l2_leaf_reg`, `random_strength` 等）をこのデータセットに過学習スレスレまで徹底的にチューニングする。
---