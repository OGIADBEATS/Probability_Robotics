# 概要
このプログラムは, Q学習を用いてロボットが1次元の数直線上を移動し, ゴール地点に到達するための最適な行動を学習するデモンストレーションである.

# 機能説明
* ロボットの動作
  * 設定した開始位置から数直線上を移動し, ゴールに到達する.
  * 移動は「左移動 (L)」または「右移動 (R)」である.

* Q学習の実装
  * ロボットが行動と報酬を基にQ値を更新.
  * 学習率、割引率、探索率などのパラメータを調整可能.

* 出力
  * 学習後, ゴールまでの最適な行動がコンソールに表示される.

# パラメータ設定
| パラメータ     | 説明               | デフォルト値    | 
| -------------- | ------------------ | --------------- | 
| LINE_LENGTH    | 数直線の長さ       | 10              | 
| START_POSITION | ロボットの開始位置 | 0               | 
| GOAL_POSITION  | ゴール位置         | LINE_LENGTH - 1 | 
| ALPHA          | 学習率             | 0.1             | 
| GAMMA          | 割引率             | 0.9             | 
| EPSILON        | 探索率             | 0.2             | 
| EPISODES       | 学習エピソード数   | 1000            | 

# 出力例
プログラム実行後, 学習された結果が以下のように表示される.
```
Optimal Policy:
R R R R R R R R R G
```
ここで,
* R : 右に移動
* L : 左に移動

である.

# 動作説明
1. ロボットの動作
    * 数直線の長さ(LINE_LENGTH)とゴール位置(GOAL_POSITION)を設定.
    * ロボットは位置0からスタートし, 数直線上を移動する.
2. Q学習のプロセス
    * ロボットの行動は右移動(R)または左移動(L)である.
    * 報酬設計
        * ゴールに到達した場合 : +1
        * その他の移動の場合 :  -0.1
     * Q値の更新式
     <img src="[https://latex.codecogs.com/gif.latex?\int_a^bf(x)dx](https://latex.codecogs.com/svg.image?&space;)](https://latex.codecogs.com/svg.image?Q(s,a)=Q(s,a)&plus;ALPHA*[reward&plus;GAMMA*max(Q(s',a'))-Q(s,a)](https://latex.codecogs.com/svg.image?Q(s,a)=Q(s,a)&plus;ALPHA*[reward&plus;GAMMA*max(Q(s',a'))-Q(s,a)])" />

最適政策の生成:

学習終了後、ゴールに到達するための最適な行動が出力されます。
