# Predicting the Beats-per-Minute of Songs
![image](https://github.com/kaneda05/kaggle_Playground-Series/blob/main/png/002_playground.png)



[kaggleページ](https://www.kaggle.com/competitions/playground-series-s5e9)

---

## 0901

[ベースラインの作成]()

## データ加工
連続値の特徴料を区間ごとに離散化（ピニング）
```python
def bin_column(df, column, bins, bin_names=None):
    if bin_names is None:
        bin_names = [f'{b:.1f}_to_{b_next:.1f}' for b, b_next in zip(bins[:-1], bins[1:])]
    df[column + '_binned'] = pd.cut(df[column], bins=bins, labels=bin_names, include_lowest=True)
    return df
```
#### ピニングの目的
**数値をカテゴリに変換**
モデルによっては、連続値よりもカテゴリデータの方が扱いやすい場合がある
（例: 決定木系モデルでの分割の可視化や特徴量解釈）
**可視化や分析のため**
「低・中・高」といったラベル分けがあると分布比較が直感的に分かりやすくなる
**特徴量エンジニアリング**
微妙な数値差を丸めて、モデルに「区間ごとの傾向」を学習させやすくする

#### 各特徴量に対するピニング
- RhythmScore → 0.0–1.0 を 0.2 刻み
- VocalContent → 特殊な区切り [0.025, 0.1, 0.15, 0.2]
- AcousticQuality → 0.01–1.0 をいくつかの区間
- InstrumentalScore → 0.001–1.0 を区間化
- LivePerformanceLikelihood → [0.05, 0.2, 0.4]
- MoodScore → 0.0–1.0 を 0.2 刻み
- Energy → 0.0–1.0 を 0.2 刻み

ビニングは 「区間ごとに異なる意味を持つ数値特徴量」や「外れ値に弱い特徴量」に特に有効
ユーザーのケース（楽曲特徴量）だと、

音量・リズム・エネルギー → 感覚的に「低・中・高」と区切った方が人間にも解釈しやすい
時間・スコア系 → ある閾値を超えると印象が大きく変わる

といった性質があるので、ビニングが有効に働きやすい

---

## 0902

#### 単純アンサンブル（平均 / 重み付き平均）
XGBoost / LightGBM / CatBoost をそれぞれクロスバリデーションで学習させ、アンサンブル学習の実装を行う。

単純アンサンブル（平均 / 重み付き平均）
**📌 方法**
複数のモデル（例: XGBoost, LightGBM, CatBoost）の予測結果を平均するだけ。
重みを調整する場合もある（例: 0.5×XGB + 0.3×LGBM + 0.2×Cat）。

すでに色々なnotebookがあるため、それらを参考にしながら、
スタッキングやアンサンブルなどを行なっていきたい。

---

## 0911
[Tiny Nudges, Big Gains🎯| Public_Score: 26.38135](https://www.kaggle.com/code/princevegeta515/tiny-nudges-big-gains-public-score-26-38135)

上記のNotebookを要約すると

`データセットの特徴`: このコンペのデータはノイズが非常に多く、本質的なパターンが少ない

`RMSEという評価指標`: RMSEは外れ値に対してペナルティが大きい。ノイズの多いデータで複雑なモデルを使うと、一部のデータに過剰に適合してしまう。

`解決策`: 下手にノイズを予測しようとするよりも、全データの平均値という「最も外れにくい予測」に近づけた方が、結果的に大きな間違いが減り、スコアが安定・向上する。

今回の単純にモデルの改善や特徴量の作成も重要だとは思うが、なるべく外れ値を出さないような予測の作成は必要になってくる。そこで現状作成しているxgboostのベースラインコードを使って、モデルで予測した結果をどれだけ平均の影響を受けるのかの検証を実施。

[作成コード](https://www.kaggle.com/code/masakane/baseline-xgboost)

---

## 0912
[Beats per minute | Ensemble | S5E9](https://www.kaggle.com/code/anthonytherrien/beats-per-minute-ensemble-s5e9)

LB ⇨ 26.38137（30/1393）

アンサンブル学習を実施することで、精度の向上が見られそうということがわかる。
どうやらモデルによって学習が得意なモデルとそうで無いモデルがあるので、
例えばラグのみを学習するモデルとそれ以外などで分けてアンサンブルさせるなどを行なっても良いかも？