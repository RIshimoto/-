# 1.強化学習
## 1-1.要点まとめ
- **強化学習とは**

  長期的に報酬を最大化できるように環境のなかで行動を選択できるエージェントを作ることを目標とする機械学習の一分野。</br>
  行動の結果として与えられる利益（報酬）をもとに、行動を決定する原理を改善していく仕組み。</br>

- **強化学習の応用例**

  マーケティングの場合。
  * 環境：会社の販売促進部
  * エージェント：プロフィールと購入履歴に基づいてキャンペーンメールを送る顧客を決めるソフトウェア。
  * 行動：顧客ごとに送信、非送信の二つの行動を選ぶことになる。
  * 報酬：キャンペーンのコストという負の報酬とキャンペーンで生み出される推測される売り上げという正の報酬を受ける。

- **探索と利用とトレードオフ**

  環境について事前に完璧な知識があれば、最適な行動を予測し決定することが可能だが、</br>
  強化学習の場合、不完全な知識を元に行動しながらデータを収集し、最適な行動を見つけていく。</br>

  そのとき、過去のデータで、ベストとされる行動のみを常にとり続ければ、他にもっとベストな行動を見つけることができない。</br>
  また、未知の行動のみをとり続ければ、過去の経験が活かせない。</br>
  探索と利用のトレードオフという。</br>

- **強化学習のイメージ**

  <img width="251" alt="image" src="https://user-images.githubusercontent.com/57135683/148182204-52b6d8fe-9f2d-467a-bdee-0d38dd38a638.png"></br>

  <img width="230" alt="image" src="https://user-images.githubusercontent.com/57135683/148182386-12b03e41-5dce-4cb2-adcf-af412bd40741.png"></br>
  方策関数と行動価値関数の二つを学習させる。

- **強化学習の差分**

  教師なし、あり学習では、データに含まれるパターンを見つけ出す。およびそのデータから予測することが目標。</br>
  一方、強化学習では、優れた方策を見つけることが目標。</br>

- **強化学習の歴史**
  * Q学習
    行動価値関数を、行動するたびに更新することにより学習を進める方法。

  * 関数近似法
    価値関数や方策関数を関数近似する手法のこと。

- **価値関数**

  価値関数を表す関数としては、状態価値関数と行動価値関数の２種類がある。</br>

  ある状態の価値に注目する場合は、状態価値関数。</br>
  状態と価値を組み合わせた価値に注目する場合は、行動価値関数。</br>


- **方策関数**

  方策ベースの強化学習手法において、
  ある状態でどのような行動を採るのかの確率を与える関数のこと。</br>
  方策反復法でモデル化して最適化する。

- **方策勾配法**

  <img src="https://latex.codecogs.com/svg.image?\theta^{(t&plus;1)}=\theta^{(t)}&plus;\varepsilon&space;\nabla&space;J\left(\theta\right)" title="\theta^{(t+1)}=\theta^{(t)}+\varepsilon \nabla J\left(\theta\right)" /></br>

  Jとは方策の良さ。定義しなければならない。</br>
  定義の方法は、</br>
  * 平均報酬
  * 割引報酬

## 1-2.確認テスト
## 1-3.実装演習

# 2.AlphaGo
## 2-1.要点まとめ
- **AlphaGo Lee**
  PolicyNet</br>
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/57135683/148198176-3e1f7b33-595d-446e-8b2f-ba2a657b80a3.png"></br>
  CNNでできている方策関数。エージェントがどこに打つのが一番いいかの予測確率を出す。</br>
  
  ValueNet</br>
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/57135683/148198232-e49982e6-7197-4654-9676-9fdc29d9093f.png">
  CNNでできている価値関数。現局面の勝率を出力する。</br>

  それぞれのチャンネルの情報は、</br>
  <img width="388" alt="image" src="https://user-images.githubusercontent.com/57135683/148199028-a226ede0-70c4-44ef-8697-5de8c0f0fd04.png">

学習は以下のステップで行われる。</br>
1.教師あり学習によるRollOutPolicyとPolicyNetの学習</br>
2.強化学習によるPolicyNetの学習</br>
3.強化学習によるValueNetの学習</br>

- **AlphaGo Zero**
  AlphaGo Leeとの違いは、
  1.教師あり学習を一切行わず、強化学習のみで作成。
  2.特徴入力からヒューリスティックな要素を排除し、石の配置のみにした。
  3.PolicyNetとValueNetを１つのネットワークに統合した。
  4.ResidualNetを導入した。
  5.モンテカルロ木探索からRollOutシミュレーションをなくした。

  <img width="425" alt="image" src="https://user-images.githubusercontent.com/57135683/148202253-c313fe76-afc8-4a59-bd8b-c0b9fbd7e5b2.png">

  ResidualNetworkとは、</br>
  <img width="397" alt="image" src="https://user-images.githubusercontent.com/57135683/148202538-a31a6f7c-4f03-4b14-9295-126e6950bd78.png">
ネットワークにショートカット構造を追加して、勾配の爆発、消失を抑える効果を狙った。</br>

## 2-2.確認テスト
## 2-3.実装演習

# 3.軽量化・高速化技術
# 4.応用モデル
# 5.Transformer
# 6.物体検知・セグメンテーション
