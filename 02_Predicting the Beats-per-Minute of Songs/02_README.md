# Predicting the Beats-per-Minute of Songs
![image](https://github.com/kaneda05/kaggle_Playground-Series/blob/main/png/02_Predicting%20the%20Beats-per-Minute%20of%20Songs.png)



[kaggleページ](https://www.kaggle.com/competitions/playground-series-s5e9)

---

#### 0901

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