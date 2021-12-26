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
　重みの初期化、活性化関数の選択を行い勾配消失を防ぐコードは以下の通り。</br>
 
　__init_weightのweight_init_stdに用いたい初期化を指定する。</br>
```code
  def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
```

　ガウス分布に基づいて重みを初期化し、sigmoidを用いて学習した場合。</br>
 
　　<img width="298" alt="image" src="https://user-images.githubusercontent.com/57135683/147398136-49db452e-0211-464e-9fd3-50e447552fdf.png"></br>
  
</br>

　ガウス分布に基づいて重みを初期化し、reluを用いて学習した場合。</br>
 
　　<img width="302" alt="image" src="https://user-images.githubusercontent.com/57135683/147398246-1abb640f-4857-422e-a9a5-54b766122595.png"></br>
  
</br>

　Xavierに基づいて重みを初期化し、sigmoidを用いて学習した場合。</br>
 
　　<img width="298" alt="image" src="https://user-images.githubusercontent.com/57135683/147398260-5df95315-8c21-417d-928a-e6c0fa74018e.png"></br>
  
</br>

　Xavierに基づいて重みを初期化し、reluを用いて学習した場合。</br>
 
　　<img width="302" alt="image" src="https://user-images.githubusercontent.com/57135683/147398353-bc4f0040-8661-40c7-adb9-c0cddd3e6746.png"></br>

</br>

　Heに基づいて重みを初期化し、sigmoidを用いて学習した場合。</br>
 
　　<img width="309" alt="image" src="https://user-images.githubusercontent.com/57135683/147398342-ce931a0c-d6e8-4883-b7df-dc1a2e34a5c8.png"></br>

</br>

　Heに基づいて重みを初期化し、reluを用いて学習した場合。</br>
 
　　<img width="305" alt="image" src="https://user-images.githubusercontent.com/57135683/147398284-7837d805-741a-4810-b9c5-384f59fa1cb2.png"></br>
</br>
　また、中間層の数を2層から9層に変えてみると、</br>
　sigmoid + Xavierでは勾配消失を起こしたが、ReLU + Xavierでは勾配消失が起こらなかった。</br>
  
</br>

　次にバッチ正規化について。</br>
```code
  mu = x.mean(axis=0) # 平均
  xc = x - mu # xをセンタリング
  var = np.mean(xc**2, axis=0) # 分散
  std = np.sqrt(var + 10e-7) # スケーリング
  xn = xc / std
            
  self.batch_size = x.shape[0]
  self.xc = xc
  self.xn = xn
  self.std = std
  self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu # 平均値の加重平均
  self.running_var = self.momentum * self.running_var + (1-self.momentum) * var #分散値の加重平均
```

　sigmoid + ガウス分布にバッチ正規化の処理を加えると、勾配消失は起こらなかった。</br>
　
　<img width="326" alt="image" src="https://user-images.githubusercontent.com/57135683/147398618-26f112e8-daca-4a15-b3ce-60b33af7e5c1.png"></br>
 
 しかし、これは中間層を増やすと勾配消失が起きてしまう。</br>
 重みの初期値をガウス分布からXavierに変えても同様に勾配消失が起こり、学習がうまくいかない。</br>
 これは、simoid関数が原因だと思われ、可能ならsigmoid関数ではなくReLUを使うことが望ましいことがわかった。</br>

</br>

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

- Momentum</br>
 ```code
  v[key] = momentum * v[key] - learning_rate * grad[key]
  network.params[key] += v[key]
 ```
　<img width="325" alt="image" src="https://user-images.githubusercontent.com/57135683/147398786-ed9a3a40-eb2d-43af-80f6-c3ba06687b15.png"></br>

- AdaGrad</br>
 ```code
 ```
　<img width="316" alt="image" src="https://user-images.githubusercontent.com/57135683/147398798-76bd2eee-14f9-4631-a3ef-31f5c612bbd9.png"></br>

- RMSProp</br>
 ```code
  h[key] *= decay_rate
  h[key] += (1 - decay_rate) * np.square(grad[key])
  network.params[key] -= learning_rate * grad[key] / (np.sqrt(h[key]) + 1e-7)
```
　<img width="325" alt="image" src="https://user-images.githubusercontent.com/57135683/147398805-eb2947a0-97cc-434c-8cd4-c83e3b393f67.png"></br>
 

- Adam</br>
 ```code
 learning_rate_t  = learning_rate * np.sqrt(1.0 - beta2 ** (i + 1)) / (1.0 - beta1 ** (i + 1))    
    for key in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
        if i == 0:
            m[key] = np.zeros_like(network.params[key])
            v[key] = np.zeros_like(network.params[key])
            
        m[key] += (1 - beta1) * (grad[key] - m[key])
        v[key] += (1 - beta2) * (grad[key] ** 2 - v[key])            
        network.params[key] -= learning_rate_t * m[key] / (np.sqrt(v[key]) + 1e-7)                

```
　<img width="329" alt="image" src="https://user-images.githubusercontent.com/57135683/147398806-ab703d52-7769-4ef4-8e01-33a3436560c7.png"></br>

</br>

</details>

# 3.過学習
<details><summary>クリックすると展開されます</summary>
  
## 3-1.要点のまとめ
### 3-1-1.　L1正則化、L2正則化
  - **過学習の原因**  
    学習を進めると、重みにバラつきが出る。
    重みが大きすぎる値をとることで、過学習が発生することがある。

  - **過学習の解決策**   
    誤差に対して、正則化項を加算することで、重みを制約する。</br>  
    誤差関数に、pノルムを加える。</br>
    
    
    　<img src="https://latex.codecogs.com/svg.image?\begin{align*}&E_n\left(\boldsymbol{w}\right)&plus;\frac{1}{p}\lambda\left\|\boldsymbol{x}\right\|_p\\&\left\|\boldsymbol{x}\right\|_p&space;=&space;\left(\left|x_1\right|^p&plus;...&plus;\left|x_n\right|^p\right)^{\frac{1}{p}}\end{align*}&space;" title="\begin{align*}&E_n\left(\boldsymbol{w}\right)+\frac{1}{p}\lambda\left\|\boldsymbol{x}\right\|_p\\&\left\|\boldsymbol{x}\right\|_p = \left(\left|x_1\right|^p+...+\left|x_n\right|^p\right)^{\frac{1}{p}}\end{align*} " /></br>
    　p=1の場合、L1正則化（ラッソ回帰）</br>
    　p=2の場合、L2正則化（リッジ回帰）</br>
      
</br>

### 3-1-2.　ドロップアウト
　ランダムにノードを削除して学習させること。</br>
　<img width="155" alt="image" src="https://user-images.githubusercontent.com/57135683/147377460-c3851725-08f4-40fe-b6b2-e431acec7afe.png"></br>
　メリットとして、データ量を変化させずに、異なるモデルを学習させていると解釈できる。　　　　　

</br>

## 3-2.確認問題・例題
> <img width="395" alt="image" src="https://user-images.githubusercontent.com/57135683/147377385-30f6c5e6-17ff-45b7-ba4d-ff973217a071.png">

　d

</br>

> 下図のL1正則化を表しているグラフはどちらか  
> <img width="350" alt="image" src="https://user-images.githubusercontent.com/57135683/147377391-60c99a22-34b7-4799-8f79-01a73a40508b.png">

　右

</br>

> <img width="403" alt="image" src="https://user-images.githubusercontent.com/57135683/147382189-dcaae477-0c39-44f0-80c7-29006c702954.png">

　4

</br>

> <img width="410" alt="image" src="https://user-images.githubusercontent.com/57135683/147382200-93d3010c-8f2b-47b5-be57-8926e7445277.png">

　3

</br>

> <img width="349" alt="image" src="https://user-images.githubusercontent.com/57135683/147382210-9c827666-7332-4f63-83d9-dc258a0194a6.png">

　4

</br>

## 3-3.実装演習
- L2正則化</br>

  ```code
  weight_decay += 0.5 * weight_decay_lambda * np.sqrt(np.sum(network.params['W' + str(idx)] ** 2))
  ```
  <img width="321" alt="image" src="https://user-images.githubusercontent.com/57135683/147399396-c9e46ed4-5863-4190-8cf1-c0817eba5bf5.png"></br>
  正則化項を加えることで訓練データでoverfittingしないようになっている。</br>
  
</br>

- L1正則化</br>

  ```code
  weight_decay += weight_decay_lambda * np.sum(np.abs(network.params['W' + str(idx)]))
  ```
  <img width="319" alt="image" src="https://user-images.githubusercontent.com/57135683/147399413-dc2533af-4cfb-4afb-a052-cbcedadb28be.png"></br>
  一部の重みが消されるので、特徴的な形になった。</br>
  
</br>

- ドロップアウト</br>

  ```code
  class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
   ```
  <img width="299" alt="image" src="https://user-images.githubusercontent.com/57135683/147399428-1db689f9-3afc-43f1-a2e8-e1422754191d.png"></br>

    ドロップ+L1正則化</br>
    <img width="292" alt="image" src="https://user-images.githubusercontent.com/57135683/147399460-0b538695-bda4-4e61-94e4-51a3d79738cb.png"></br>

</br>

</details>

# 4.畳み込みニューラルネットワークの概念
<details><summary>クリックすると展開されます</summary>
  
## 4-1.要点のまとめ
全結合層は、カラー画像におけるRGBなど、各チャンネル間の関連性が学習に反映されない。</br>

そこで全結合の前に畳み込み処理をすることでその問題を解決できる。</br>
これにより、CNNで画像識別や音声など、次元間でつながりのあるデータを扱えるようになる。</br>

CNNの代表的なものの一つにLeNetがある。</br>

CNNの構成は以下の通り。</br>

<img width="114" alt="image" src="https://user-images.githubusercontent.com/57135683/147378818-35355fd0-5c57-417b-b2c1-f0665b076920.png"></br>

</br>

### 4-1-1.畳み込み層
　畳み込み層では畳み込み演算を行う。</br>

　畳み込み演算とは、フィルターを用いて入力画像の対象領域に演算を行い、バイアスを加え出力とする。</br>

　この際、畳み込み演算のフィルターの数を**チャンネル**、</br>
　フィルターをかける際に何マスずらすかを**ストライド**、</br>
　また、畳み込み演算を行うと、画像のサイズが小さくなってしまうため、</br>
　フィルターをかける前に、上下左右に画像を広げる。これを**パディング**という。</br>

　畳み込み層は、画像の場合、縦、横、チャンネルの3次元のデータをそのまま学習し、次に伝えることができる。</br>

</br>

### 4-1-2.プーリング層
　畳み込み層と組み合わせて使われる。</br>
 
　畳み込み層と同様にすこしずつずれながら画像を読み取る。その際行う処理は様々あり、</br>
　対象領域の最高値を使う**MaxPooling**や、平均値を使う**AvgPooling**がある。</br>

</br>

## 4-2.確認問題
> サイズ6x6の入力画像を、サイズ2x2のフィルタで畳み込んだ時の出力画像のサイズを答えよ。
> なおストライドとパディングは1とする。

　7x7

</br>

## 4-3.実装演習
- 畳み込み層
```code
class Convolution:
    # W: フィルター, b: バイアス
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # フィルター・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # FN: filter_number, C: channel, FH: filter_height, FW: filter_width
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 出力値のheight, width
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        
        # xを行列に変換
        col = im2col(x, FH, FW, self.stride, self.pad)
        # フィルターをxに合わせた行列に変換
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        # 計算のために変えた形式を戻す
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # dcolを画像データに変換
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
```

</br>

- プーリング層
```code
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        # xを行列に変換
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # プーリングのサイズに合わせてリサイズ
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        #maxプーリング
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
```

</br>
</details>

# 5.最新のCNN
<details><summary>クリックすると展開されます</summary>
  
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

</details>
