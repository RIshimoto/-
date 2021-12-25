# 1.勾配消失問題  
<details><summary>クリックすると展開されます</summary>
  
## 1-1.要点まとめ
　誤差逆伝播法は階層が進んでいくにつれて、勾配がどんどん緩やかになっていく。  
　そのため、勾配降下法による、更新では下位パラメータはほとんど変わらず、訓練は最適値に収束しなくなる。  

  - なぜ起こるのか

　　微分値が0-1の範囲をとるものが多いため、</br>
　　層が深くなりより掛け合わせると、どんどん値が小さくなるから。</br>

　　（例）シグモイド関数</br>
 　　　 <img width="200" alt="image" src="https://user-images.githubusercontent.com/57135683/147320472-cc32beb3-9613-4bad-a930-18de05df26d9.png"></br>
　　　　微分すると、</br>
  　　　<img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/147320455-ecb32e8b-2090-405d-8782-ceee485026a6.png"></br>
 　　　　となり、最大0.25までしか値をとらない。</br>

</br>

  - どうやって解消するか
    
    * **活性化関数の選択**

      ReLU関数を使う。</br>
      <img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/147321909-506f8cb3-04cd-456b-adc4-5fc250d741fb.png"></br>
      微分が1になるので勾配消失が起きない。</br>
      </br>
    
    * **重みの初期値設定**
    
       **Xavier**：</br>
        　正規分布を前のレイヤーのノード数の平方根で割った値。</br>
        　活性化関数がReLU関数、シグモイド関数、双曲線正接関数に用いられる。</br>
        **He**：</br>
    　     重みの要素を、前の層のノード数の平方根で除算した値に対し、√2を掛け合わせた値。</br>
    　     活性化関数がReLU関数に用いられる。</br>
        </br>
        
    * **バッチ正規化**
    
       ミニバッチ単位で、入力値のデータの偏りを抑制する手法。</br>
       活性化関数に値を渡す前後に、バッチ正則化の処理は孕んだ層を加える。</br>
       数学的手順としては、</br>
       <img width="110" alt="image" src="https://user-images.githubusercontent.com/57135683/147325109-4dd32ba8-454e-44d5-a763-b994b237c773.png">
       <img width="353" alt="image" src="https://user-images.githubusercontent.com/57135683/147325127-a31a8d7f-eaf9-4212-a050-669a28c9ec92.png">



## 1-2.確認問題

> 連鎖律の原理を使い、dz/dxを求めよ。</br>　<img src="https://latex.codecogs.com/svg.image?\begin{align*}z&=t^2\\t&=x&plus;y\end{align*}&space;" title="\begin{align*}z&=t^2\\t&=x+y\end{align*} " />

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\mathrm{d}&space;z}{\mathrm{d}&space;x}&=\frac{\mathrm{d}z}{\mathrm{d}t}\frac{\mathrm{d}t}{\mathrm{d}x}\\\frac{\mathrm{d}&space;z}{\mathrm{d}&space;t}&=2t\\\frac{\mathrm{d}&space;x}{\mathrm{d}&space;t}&=1\end{align*}&space;" title="\begin{align*}\frac{\mathrm{d} z}{\mathrm{d} x}&=\frac{\mathrm{d}z}{\mathrm{d}t}\frac{\mathrm{d}t}{\mathrm{d}x}\\\frac{\mathrm{d} z}{\mathrm{d} t}&=2t\\\frac{\mathrm{d} x}{\mathrm{d} t}&=1\end{align*} " /></br>
 
　より、</br>
 
　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\mathrm{d}&space;z}{\mathrm{d}&space;x}&=2t\cdot1\\&space;&=2t\\&space;&=2\left(x&plus;y\right)\end{align*}&space;" title="\begin{align*}\frac{\mathrm{d} z}{\mathrm{d} x}&=2t\cdot1\\ &=2t\\ &=2\left(x+y\right)\end{align*} " />
 
</br>

> シグモイド関数を微分したとき、入力値が0の時に最大値をとる。</br>
> その値として正しいもの。

  0.25

</br>

> 重みの初期値に0を設定すると、どのような問題が発生するか。</br>

　すべての重みの値が均一に更新されるため、多数の重みをもつ意味がなくなる。
 
</br>

> 一般的に考えられるバッチ正規化の効果を２点あげよ。</br>

- 過学習が起きづらくなる。
- 学習が安定し、学習スピードが上がる。

</br>

## 1-3.実装演習

```code

```
</details>

# 2.学習率最適化手法
<details><summary>クリックすると展開されます</summary>
  
## 2-1.要点のまとめ
### 2-1-1.モメンタム
　誤差をパラメータで微分したものと学習率の積を減算した後、</br>
　現在の重みに前回の重みを減算した値と慣性の積を加算する。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}V_t&space;&=&space;\mu&space;V_{t-1}-\epsilon&space;\nabla&space;E\\\boldsymbol{w}^{(t&plus;1)}&=\boldsymbol{w}^{(t)}&plus;V_t\end{align*}&space;" title="\begin{align*}V_t &= \mu V_{t-1}-\epsilon \nabla E\\\boldsymbol{w}^{(t+1)}&=\boldsymbol{w}^{(t)}+V_t\end{align*} " /></br>
　　μ:慣性</br>

</br>

　**メリット**
   - 局所最適解にはならず、大域的最適解になる。
   - 谷間についてから最も低い位置（最適値）にいくまでの時間が早い。
</br>

### 2-1-2.AdaGrad
　誤差をパラメータで微分したものと再定義した学習率の積を減算する。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}h_0&=\theta&space;\\h_t&=h_{t-1}&plus;(\nabla&space;E)^2\\\boldsymbol{w}^{(t&plus;1)}&=\boldsymbol{w}^{t}-\epsilon&space;\frac{1}{\sqrt{h_t}&plus;\theta}\nabla&space;E\end{align*}&space;" title="\begin{align*}h_0&=\theta \\h_t&=h_{t-1}+(\nabla E)^2\\\boldsymbol{w}^{(t+1)}&=\boldsymbol{w}^{t}-\epsilon \frac{1}{\sqrt{h_t}+\theta}\nabla E\end{align*} " /><br>


　**メリット**  
 　　勾配の緩やかな斜面に対して、最適値に近づける。  

　**課題**  
 　　学習率が徐々に小さくなるので、**鞍点問題**を引き起こすことがあった。  
</br>

### 2-1-3.RMSProp
　誤差をパラメータで微分したものと再定義した学習率の積を減算する。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}h_t&=\alpha&space;h_{t-1}&plus;\left(1-\alpha\right)\left(\nabla&space;E\right)^2\\\boldsymbol{w}^{(t&plus;1)}&=\boldsymbol{w}^{(t)}-\epsilon&space;\frac{1}{\sqrt{h_t}&plus;\theta}\nabla&space;E\end{align*}&space;" title="\begin{align*}h_t&=\alpha h_{t-1}+\left(1-\alpha\right)\left(\nabla E\right)^2\\\boldsymbol{w}^{(t+1)}&=\boldsymbol{w}^{(t)}-\epsilon \frac{1}{\sqrt{h_t}+\theta}\nabla E\end{align*} " /></br>

</br>

　**メリット**
   - 局所最適解にはならず、大域的最適解になる。
   - ハイパーパラメータの調整が必要な場合が少ない。
</br>

### 2-1-4.Adam
  * モメンタムの、過去の勾配の指数関数的減衰平均。
  * RMSPropの、過去の勾配の２乗の指数関数的減数平均。  
上記をそれぞれ孕んだ最適化アルゴリズム。</br>

　**メリット**  
　　モメンタムおよびRMSPropのメリットを孕んでいる。</br>
</br>


## 2-2.確認問題

> モメンタム・AdaGrad・RMSPropの特徴をそれぞれ簡潔に説明せよ。

  - モメンタム：前回の学習量を用いて学習するため、加速がつくと一気に学習が進む。
  - AdaGrad：勾配がゆるやかなときにうまくいきやすいが、大域最適解にたどり着きづらい。
  - RMSProp：欠点を改良したAdaGrad

</br>

## 2-3.実装演習
```code
```

</details>

# 3.過学習
## 3-1.要点のまとめ
### 3-1-1.L1正則化、L2正則化
  - **過学習の原因**  
    学習を進めると、重みにバラつきが出る。
    重みが大きすぎる値をとることで、過学習が発生することがある。

  - **過学習の解決策**   
    誤差に対して、正則化項を加算することで、重みを制約する。</br>  
    誤差関数に、pノルムを加える。</br>
    
    
    　<img src="https://latex.codecogs.com/svg.image?\begin{align*}&E_n\left(\boldsymbol{w}\right)&plus;\frac{1}{p}\lambda\left\|\boldsymbol{x}\right\|_p\\&\left\|\boldsymbol{x}\right\|_p&space;=&space;\left(\left|x_1\right|^p&plus;...&plus;\left|x_n\right|^p\right)^{\frac{1}{p}}\end{align*}&space;" title="\begin{align*}&E_n\left(\boldsymbol{w}\right)+\frac{1}{p}\lambda\left\|\boldsymbol{x}\right\|_p\\&\left\|\boldsymbol{x}\right\|_p = \left(\left|x_1\right|^p+...+\left|x_n\right|^p\right)^{\frac{1}{p}}\end{align*} " /></br>
    　p=1の場合、L1正則化（ラッソ回帰）</br>
    　p=2の場合、L2正則化（リッジ回帰）</br>


### 3-1-2.ドロップアウト
　ランダムにノードを削除して学習させること。</br>
　<img width="155" alt="image" src="https://user-images.githubusercontent.com/57135683/147377460-c3851725-08f4-40fe-b6b2-e431acec7afe.png"></br>
　メリットとして、データ量を変化させずに、異なるモデルを学習させていると解釈できる。　　　　　


## 3-2.確認問題・例題
> <img width="395" alt="image" src="https://user-images.githubusercontent.com/57135683/147377385-30f6c5e6-17ff-45b7-ba4d-ff973217a071.png">

　d

> <img width="350" alt="image" src="https://user-images.githubusercontent.com/57135683/147377391-60c99a22-34b7-4799-8f79-01a73a40508b.png">

　右

## 3-3.実装演習
```code
```

# 4.畳み込みニューラルネットワークの概念

# 5.最新のCNN
<details><summary>クリックすると展開されます</summary>
  
## 5-1.要点のまとめ
### AlexNet
　５層の畳み込みそうおよびプーリング層など、それに続く３層の全結合層から構成される。</br>
　<img width="298" alt="image" src="https://user-images.githubusercontent.com/57135683/147377661-27e0269b-b21a-4c4c-b7d1-9cec37a4bb9d.png"></br>
　過学習を防ぐために、サイズ4096の全結合層の出力にドロップアウトを使用している。 
  
　CNNから全結合層へ移行する処理は以下の通り。
  - Fratten  
    すべての値を横一列に並び変える。
    
  - GolbalMaxPooling  
    各チャンネルの一番大きいものを使う。

  - GolbalAvgPooling  
    各チャンネルの一番平均を使う。

  </br>
  
## 5-2.確認問題
## 5-3.実装演習

</details>
