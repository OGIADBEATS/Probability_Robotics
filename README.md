# 概要
このプログラムは, Q学習を用いてロボットが1次元の数直線上を移動し, ゴール地点に到達するための最適な行動を学習するデモンストレーションである.

# 動作説明
1. ロボットの動作
    * 数直線の長さ(`LINE_LENGTH`)とゴール位置(`GOAL_POSITION`)を設定.
    * ロボットは位置`0`からスタートし, 数直線上を右移動(R)または左移動(L)する.

2. Q学習のプロセス
    * 報酬設計
        * ゴールに到達した場合 : `+1`
        * その他の移動の場合 :  `-0.1`

    * 実装内容
        * ロボットが行動と報酬を基にQ値を更新.
        * 学習率 (`ALPHA`), 割引率 (`GAMMA`), 探索率 (`EPSILON`)などのパラメータを調整可能.

3. 出力
   * 学習後, ゴールまでの最適な行動がコンソールに表示される.

# 数直線上のイメージ
   * 以下のような数直線上で学習が行われる(初期設定の場合) : 
<img width="658" alt="number_line" src="https://github.com/user-attachments/assets/bf3b294e-fc93-4940-be29-b060d2a3cc97" />

   *S : スタート位置(0)
   *G : ゴール位置(9)

# パラメータ設定
各パラメータとデフォルト値は以下の通りである.
| パラメータ     | 説明               | デフォルト値    | 
| -------------- | ------------------ | --------------- | 
| `LINE_LENGTH`    | 数直線の長さ       | `10`              | 
| `START_POSITION` | ロボットの開始位置 | `0`               | 
| `GOAL_POSITION`  | ゴール位置         | `LINE_LENGTH - 1` | 
| `ALPHA`          | 学習率             | `0.1`             | 
| `GAMMA`          | 割引率             | `0.9`             | 
| `EPSILON`        | 探索率             | `0.2`             | 
| `EPISODES`       | 学習エピソード数   | `1000`            | 

# 出力例
プログラム実行後, 学習された結果が以下のように表示される(初期設定の場合) : 
```
Optimal Policy:
R R R R R R R R R G
```
ここで,
* R : 右に移動
* L : 左に移動

である.


