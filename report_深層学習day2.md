# 1.勾配消失問題
## 1-1.要点まとめ
誤差逆伝播法は階層が進んでいくにつれて、勾配がどんどん緩やかになっていく。  
そのため、勾配降下法による、更新では下位パラメータはほとんど変わらず、訓練は最適値に収束しなくなる。  

- なぜ起こるのか

  微分値が0-1の範囲をとるものが多いため、</br>層が深くなりより掛け合わせると、
どんどん値が小さくなるから。</br>

  （例）シグモイド関数</br>
        <img width="200" alt="image" src="https://user-images.githubusercontent.com/57135683/147320472-cc32beb3-9613-4bad-a930-18de05df26d9.png"></br>
　      微分すると、</br>
        <img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/147320455-ecb32e8b-2090-405d-8782-ceee485026a6.png"></br>
        となり、最大0.25までしか値をとらない。

- どうやって解消するか
  * **活性化関数の選択**

    ReLU関数を使う。</br>
    <img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/147321909-506f8cb3-04cd-456b-adc4-5fc250d741fb.png"></br>
    微分が1になるので勾配消失が起きない。</br>
    
  * **重みの初期値設定**
    
    **Xavier**：</br>
    　正規分布を前のレイヤーのノード数の平方根で割った値。</br>
      活性化関数がReLU関数、シグモイド関数、双曲線正接関数に用いられる。</br>
    **He**：</br>
    　重みの要素を、前の層のノード数の平方根で除算した値に対し、√2を掛け合わせた値。</br>
      活性化関数がReLU関数に用いられる。</br>
  
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
 
 
> シグモイド関数を微分したとき、入力値が0の時に最大値をとる。</br>
> その値として正しいもの。

  0.25


> 重みの初期値に0を設定すると、どのような問題が発生するか。</br>

　すべての重みの値が均一に更新されるため、多数の重みをもつ意味がなくなる。

> 一般的に考えられるバッチ正規化の効果を２点あげよ。</br>

- 過学習が起きづらくなる。
- 学習が安定し、学習スピードが上がる。

## 1-3.実装演習
```code

```
# 2.学習率最適化手法
# 3.過学習
# 4.畳み込みニューラルネットワークの概念
# 5.最新のCNN
