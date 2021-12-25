# 1.入力層～中間層
<details><summary>クリックすると展開されます</summary>
    
## 1-1.要点まとめ
<img width="407" alt="image" src="https://user-images.githubusercontent.com/57135683/147182866-6eee92d4-47b9-4799-9a53-bb3a0155c481.png"></br>
入力に重みを付けて混ぜ合わせた結果を、活性化関数で処理して出力に変換する。</br>
</br>
入力を</br>
　　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{x}&space;=&space;\begin{bmatrix}x_1&space;\\...&space;\\x_l\end{bmatrix}&space;" title="\boldsymbol{x} = \begin{bmatrix}x_1 \\... \\x_l\end{bmatrix} " /></br>
重みを</br>
　　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{W}&space;=&space;\begin{bmatrix}w_1&space;\\...&space;\\w_l\end{bmatrix}&space;" title="\boldsymbol{W} = \begin{bmatrix}w_1 \\... \\w_l\end{bmatrix} " /></br>
とし、それにバイアスを加えて</br>
</br>
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}u&=w_1x_1&plus;w_2x_2&plus;w_3x_3&plus;w_4x_4&plus;b\\&=\boldsymbol{W}\boldsymbol{x}&plus;b\end{align*}&space;" title="\begin{align*}u&=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b\\&=\boldsymbol{W}\boldsymbol{x}+b\end{align*} " /></br>
</br>
これを総入力といい、これに活性化関数を通すことで出力が得られる。</br>
</br>
この出力は、次のニューラルネットワークの入力として使われる。</br>
</br>
いわば、入力層と中間層が一つのパーツとみなせ、</br>
このパーツが何層も連なっているのが、深層ディープニューラルネットワークの仕組みになっている。</br>

## 1-2.確認テスト
> 図式に動物分類の実例を入れる。  

　<img width="361" alt="image" src="https://user-images.githubusercontent.com/57135683/147183843-b269c72f-7bbb-41b0-b0da-8544dfbb9b62.png"></br>
　動物の特徴として、体長、体重、ひげの本数、毛の平均長、耳の大きさ、眉間、足の長さとし、</br>
　入力層にそれぞれの動物の特徴を表す数字を入れ、重みとバイアスで束ねて、次の層へと伝播させる。</br>

</br>

> <img src="https://latex.codecogs.com/svg.image?\begin{align*}u&=w_1x_1&plus;w_2x_2&plus;w_3x_3&plus;w_4x_4&plus;b\\&=\boldsymbol{W}\boldsymbol{x}&plus;b\end{align*}&space;" title="\begin{align*}u&=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b\\&=\boldsymbol{W}\boldsymbol{x}+b\end{align*} " /></br>をPythonでかけ。
```code
u = np.dot(x, W) + b
```

</br>

> 中間層の出力を定義しているソースを抜き出せ
```code
z = functions.relu(u)
```

</br>

## 1-3.実装演習
入力xに重みを付けてバイアスを加え、活性化関数で変換したものを出力するコード。
```code
# 重み
W = np.array([
    [0.1, 0.2, 0.3,0], 
    [0.2, 0.3, 0.4, 0.5], 
    [0.3, 0.4, 0.5, 1],
])
# バイアス
b = np.array([0.1, 0.2, 0.3])

# 入力値
x = np.array([1.0, 5.0, 2.0, -1.0])

#  総入力
u = np.dot(W, x) + b

# 中間層出力
z = functions.sigmoid(u)
print_vec("中間層出力", z)
```
実行結果は</br>
<img width="187" alt="image" src="https://user-images.githubusercontent.com/57135683/147312365-96916fab-f62c-4a19-bf06-84dcf65b4d77.png">

</br>

</details>

# 2.活性化関数
<details><summary>クリックすると展開されます</summary>
    
## 2-1.要点まとめ
ニューラルネットワークにおいて、次の層への出力の大きさを決める**非線形の関数**。</br>
入力値の値によって、次の層への信号のON/OFFや強弱を定める働きを持つ。</br>
線形な処理を非線形な活性化関数を通すことで、よりバラエティのある出力を作り出すことができる。</br>
</br>
例として、中間層用の活性化関数は、</br>
- ReLu関数  
    <img width="209" alt="image" src="https://user-images.githubusercontent.com/57135683/147189057-c5b3b14a-e6df-423c-a820-a392b80aaf16.png"></br
    勾配消失問題とスパース化に貢献することで良い成果をもたらしている。</br>
    </br>
- シグモイド（ロジスティック）関数  
    <img width="224" alt="image" src="https://user-images.githubusercontent.com/57135683/147189024-f5233382-2373-497c-98a6-f28856fda14e.png">  
    0～1の間を緩やかに変化する関数で、信号の強弱を伝えられるようになり、予想ニューラルネットワークの普及のきっかけとなった。</br>
    層が深くなると勾配消失が問題となる。</br>
    </br>
- ステップ関数  
    <img width="223" alt="image" src="https://user-images.githubusercontent.com/57135683/147189005-1f1e0ecd-adcf-47f2-8c61-163930d8e203.png">  
    閾値を超えたら1を出力する。パーセプトロンで利用された関数だが、線形分離可能なものしか学習できず、今は使われていない。</br>
    </br>
</br>


## 2-2.確認テスト
> 線形と非線形の違いについて図を書いて簡易に説明せよ。

  - 線形  
      <img width="284" alt="image" src="https://user-images.githubusercontent.com/57135683/147187450-399e6f0e-a9a7-4f31-a0ed-74ed2ca8ab04.png"></br>
      線形は比例関係を満たす。</br>
      数学的な特徴では、
      - 加法性：f(x+y) = f(x) + f(y)
      - 斉次性：f(kx) = kf(x) 
      を満たす。</br>
    
  - 非線形  
      <img width="277" alt="image" src="https://user-images.githubusercontent.com/57135683/147187685-72e27ca0-fb63-4da3-91b9-514a01e99323.png"></br>
      非線形は比例関係を満たさない。</br>
      数学的な特徴では、加法性、斉次性を満たさない。</br>

</br>

> 配布されたソースコードより該当する箇所（z=f(u)）を抜き出せ。
```code
z1 = functions.sigmoid(u)
```

## 2-3.実装演習
```code
# 順伝播（単層・複数ユニット）

# 重み
W = np.array([
    [0.1, 0.2, 0.3,0], 
    [0.2, 0.3, 0.4, 0.5], 
    [0.3, 0.4, 0.5, 1],
])
print_vec("重み", W)

# バイアス
b = np.array([0.1, 0.2, 0.3])
print_vec("バイアス", b)

# 入力値
x = np.array([1.0, 5.0, 2.0, -1.0])
print_vec("入力", x)

#  総入力
u = np.dot(W, x) + b
print_vec("総入力", u)

# 中間層出力
z = functions.sigmoid(u)
print_vec("中間層出力", z)
```
実行結果は、</br>
<img width="215" alt="image" src="https://user-images.githubusercontent.com/57135683/147225417-ee5ebba0-cd9b-4d82-91be-11aa3834dcb0.png">

</details>

# 3.出力層
<details><summary>クリックすると展開されます</summary>
    
## 3-1.要点まとめ
### 3-1-1.誤差関数
　ニューラルネットワークを学習させるには、</br>
 
　1. まず、入力データとそれに対応した正解値を用意する。</br>
 
　　<img width="272" alt="image" src="https://user-images.githubusercontent.com/57135683/147194721-92e232d0-d6a1-4966-bfdd-db56c3e596ba.png"></br>
　
 </br>
 
　2. 入力データをニューラルネットワークに入力し、</br>
　ニューラルネットワークの出力層からそれぞれのクラスの確率が出力させる。</br>
 
　　<img width="400" alt="image" src="https://user-images.githubusercontent.com/57135683/147195470-b2e92de9-6df3-40ee-a4b1-cbcae80d14ac.png"></br>
　
</br>

　3. その出力と正解値と比べ、</br>
　どのくらい正解との誤差があるかを誤差関数によって出力する。</br>
 
　　<img width="388" alt="image" src="https://user-images.githubusercontent.com/57135683/147195090-77a5dd24-ae96-4309-b6cf-e92b7d77c2d9.png"></br>
 
</br>
　誤差関数の数値が小さければ、正しい答えに近いということを示し、</br>
　数値が大きければ、正しい答えからそれていることがわかる。</br>
</br>
　誤差関数には、二乗和誤差関数などがある。</br>
</br>

## 3-1-2.活性化関数
　中間層と出力層では用いる活性化関数が異なる。</br>
</br>
　中間層では閾値前後で信号の強弱を調整するために活性化関数を用いたが、</br>
　出力層では信号の大きさをそのままに我々が使いやすいように変換するのが目的。</br>
</br>
　また分類問題では出力層の出力の総和は1になる必要がある。</br>

　例として出力層用の活性化関数は、</br>
  - ソフトマックス関数
  - 恒等写像
  - シグモイド関数（ロジスティック関数）

　 があり、使い分けは以下の通り。</br>
　 <img width="395" alt="image" src="https://user-images.githubusercontent.com/57135683/147198517-b7de72b9-761a-4ead-b470-ab369e94dd3c.png"></br>
</br>

## 3-2.確認テスト
> 引き算ではなく二乗するかを述べよ</br>

　誤差が正負でうち消すと誤差が正しく表せないため、2乗して正の値にして計算する。

</br>

> 下式の1/2はどういう意味か  
><img src="https://latex.codecogs.com/svg.image?E_n(\boldsymbol{w})&space;=&space;&space;\frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2&space;=&space;\frac{1}{2}\left\|&space;\left(y&space;-&space;d\right)\right\|^2" title="E_n(\boldsymbol{w}) = \frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2 = \frac{1}{2}\left\| \left(y - d\right)\right\|^2" />

　ネットワークの学習を行うときに行う逆伝播法の計算で微分する際、式が簡単になるため。
 
</br>
 
> <img width="353" alt="image" src="https://user-images.githubusercontent.com/57135683/147199893-4c7e9462-5ed0-44f6-b5e0-8a42656c4671.png">
```code
# ソフトマックス関数
def softmax(x): # ①
    if x.ndim == 2: #ミニバッチとしてデータを取り扱う場合
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)　# オーバーフロー対策
    #本質的な部分はこの一行
    return np.exp(x) / np.sum(np.exp(x)) # return ②/③
```

</br>

> <img width="398" alt="image" src="https://user-images.githubusercontent.com/57135683/147201488-4fcd6147-3aee-4c40-aeb6-178a14647c09.png">
```code
# クロスエントロピー
def cross_entropy_error(d, y): # ①
    if y.ndim == 1:
        d = d.reshape(1, d.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if d.size == y.size:
        d = d.argmax(axis=1)
             
    batch_size = y.shape[0]
    
    # 本質的な実装は、
    # -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) の部分。
    # これが数式の②の部分にあたる。
    return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size
```

</br>

## 3-3.実装演習
　分類問題の場合はクロスエントロピー誤差を用い、</br>
　<img src="https://latex.codecogs.com/svg.image?E_n(\boldsymbol{w})&space;=&space;-\sum_{j=1}^{I}d_j&space;\log&space;y_j" title="E_n(\boldsymbol{w}) = -\sum_{j=1}^{I}d_j \log y_j" /></br>

　回帰の場合は二乗誤差を用いる。</br>
　<img src="https://latex.codecogs.com/svg.image?E_n(\boldsymbol{w})&space;=&space;&space;\frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2&space;" title="E_n(\boldsymbol{w}) = \frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2 " /></br>
 
　これらの式をコードにすると、</br>
```code
# 誤差関数
# 平均二乗誤差
def mean_squared_error(d, y):
    #squareは二乗の計算を行う。meanで平均をとる。
    return np.mean(np.square(d - y)) / 2 

# クロスエントロピー
def cross_entropy_error(d, y):
    if y.ndim == 1:
        d = d.reshape(1, d.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if d.size == y.size:
        d = d.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size
```

</details>

# 4.勾配降下法
<details><summary>クリックすると展開されます</summary>

## 4-1.要点まとめ
　深層学習の目的は、学習を通して誤差を最小にするネットワークを作成すること。</br>
　つまり、誤差E(w)を最小化するパラメータwを発見する。</br>
　この最適なパラメータは**勾配降下法**を利用して求める。</br>
</br>
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E&space;" title="\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\varepsilon \nabla E " /></br>
　　<img src="https://latex.codecogs.com/svg.image?\nabla&space;E=\frac{\partial&space;E}{\partial&space;\boldsymbol{w}}=[\frac{\partial&space;E}{\partial&space;w_1}...\frac{\partial&space;E}{\partial&space;w_M}]" title="\nabla E=\frac{\partial E}{\partial \boldsymbol{w}}=[\frac{\partial E}{\partial w_1}...\frac{\partial E}{\partial w_M}]" /></br>
　　εは学習率と呼ばれるものでこれで学習の効率が決まる。</br>
  
</br>

　ある重みの時の誤差があり、それを調整して誤差が最小にするwを見つける。</br>  
　　<img width="258" alt="image" src="https://user-images.githubusercontent.com/57135683/147204802-587d3c11-220c-4044-b3f2-41040e185ec7.png"></br>
 
</br>

　εが大きいと、局所解を通り過ぎ、発散してしまう。</br>  
　　<img width="303" alt="image" src="https://user-images.githubusercontent.com/57135683/147205150-2062d045-174d-4eba-9767-804023b20e4d.png"></br>
 
</br>

　逆にεが小さいと、収束するまでに時間がかかってしまう。</br>
　　<img width="311" alt="image" src="https://user-images.githubusercontent.com/57135683/147205175-1fa2dc08-55b1-4135-9b21-2b7d91ef9c27.png"></br>
 
</br>

　勾配降下法の学習率の決定、収束率向上のアリゴリズムは以下のものがある。</br>
  - Momentum
  - AdaGrad
  - Adadelta
  - Adam

　この勾配降下法を用いて、</br>
　誤差関数の値をより小さくする方向に重みとバイアスを更新し、次の周（エポック）反映させる。</br>

</br>

　また、勾配降下法の派生形として確率的勾配降下法とミニバッチ勾配降下法がある。</br>

 - **確率的勾配降下法**  
  
 　  　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E_n" title="\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\varepsilon \nabla E_n" /></br>
    
  　 　学習に使うデータの一部をランダムに抽出して学習する。</br>
  　 　これにより、無駄な計算を減らせること、局所極小解に収束するリスクが減る。</br>

  　 　また**オンライン学習**ができるのもメリットである。</br>
  　 　オンライン学習とは都度都度学習データを与えて学習させていく。</br>
  　 　これはリアルタイムでデータが集まってくるものなどに有効である。</br>

</br>

   - **ミニバッチ勾配降下法**  

  　 　<img src="https://latex.codecogs.com/svg.image?\begin{align*}&\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E_t\\&E_t&space;=&space;\frac{1}{N_t}\sum_{n\in&space;D_t}E_n\\&N_t&space;=&space;&space;\left\|&space;D_t\right\|&space;\end{align*}&space;" title="\begin{align*}&\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E_t\\&E_t&space;=&space;\frac{1}{N_t}\sum_{n\in&space;D_t}E_n\\&N_t&space;=&space;&space;\left\|&space;D_t\right\|&space;\end{align*} " /></br>
  　 　D_tはミニバッチ</br>
   
  　 　バッチ学習のやり方にオンライン学習の考え方を取り入れて、データを分割して学習させていく。</br>
  　 　データをランダムに分割して（ミニバッチ）、少しづつ学習をさせ、</br>
  　 　分割したデータのそれぞれのサンプルの平均誤差をとる。</br>

  　 　確率的勾配降下法のメリットを損なわず、またCPUを利用したスレッド並列化やGPUを利用したSIMD並列化が行える。</br>

</br>

## 4-2.確認テスト
> <img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E_n&space;" title="\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\varepsilon \nabla E_n " />の該当するソースコードを探す。

```code
network[key] -= learning_rate * grad[key]
```

</br>

> オンライン学習とは何か

　学習データが入ってくるたびに都度パラーメータを学習する。バッチ学習の逆。  
　バッチ学習はすべての学習データを一度に学習させるため、メモリ不足に陥ることがある。  

</br>

> <img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{(t&plus;1)}=\boldsymbol{w}^{(t)}-\varepsilon&space;\nabla&space;E_n" title="\boldsymbol{w}^{(t+1)}=\boldsymbol{w}^{(t)}-\varepsilon \nabla E_n" />の意味を図示して説明せよ。

　<img width="337" alt="image" src="https://user-images.githubusercontent.com/57135683/147212493-2f325688-7bd5-4e76-b61b-da0f436f6f92.png">    
　エポックとは一つのデータセット。</br>
　エポックを繰り返してそのデータセットで得られた誤差が最小になるようにパラメータを更新していく。  

</br>

## 4-3.実装演習
　確率的勾配降下法の実装演習を行う。</br>
　確率的勾配降下法はデータをランダムに抽出したのち勾配降下を行う。</br>
 
　以下のコードでは逆伝播法で勾配を計算し、</br>
　パラメータに勾配を適用させ、実際に誤差がどのように変化していくかを見る。</br>
```code
losses = []
# 学習率
learning_rate = 0.07

# 抽出数
epoch = 1000

# パラメータの初期化
network = init_network()
# データのランダム抽出
random_datasets = np.random.choice(data_sets, epoch)

# 勾配降下の繰り返し
for dataset in random_datasets:
    x, d = dataset['x'], dataset['d']
    z1, y = forward(network, x)
    grad = backward(x, d, z1, y)
    # パラメータに勾配適用
    for key in ('W1', 'W2', 'b1', 'b2'):
        network[key]  -= learning_rate * grad[key]

    # 誤差
    loss = functions.mean_squared_error(d, y)
    losses.append(loss)
```
　実行結果は、</br>
　<img width="343" alt="image" src="https://user-images.githubusercontent.com/57135683/147313163-03dd0475-e1e9-47a4-84e9-2e35aa341c3c.png"></br>
　となり、誤差が徐々に減っていくのがわかる。

</br>

</details>

# 5.誤差逆伝播法
<details><summary>クリックすると展開されます</summary>
	
## 5-1.要点のまとめ
　∇Eを求める際に数値微分を用いると、負荷が大きいため、誤差逆伝播法を利用する。</br>
 
　逆伝播法とは、算出された誤差を、出力側から順に微分し、前の層前の層へと伝播させる方法。</br>
　最小限の計算で各パラメータでの微分値を解析的に計算できる。</br>
　<img width="355" alt="image" src="https://user-images.githubusercontent.com/57135683/147216595-ae9427d5-d882-49f8-ad88-fd0d1bbf5c59.png"></br>


　計算結果から微分を逆算することで、不要な再帰的計算を避けて微分を算出できる。</br>
　<img width="420" alt="image" src="https://user-images.githubusercontent.com/57135683/147217598-01c16471-1f2e-4889-ab93-43d79c9dcdf0.png"></br>

</br>

　誤差関数のwでの微分した値を連鎖律を利用して求める。</br>
　二乗誤差関数は、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?E(\boldsymbol{y})=\frac{1}{2}\sum_{j=1}^{J}(y_j&space;-&space;d_j)^2&space;=&space;\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{d}\right\|^2" title="E(y)=\frac{1}{2}\sum_{j=1}^{J}(y_j&space;-&space;d_j)^2&space;=&space;\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{d}\right\|^2" /></br>
	
　より、yについて微分すると、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E(\boldsymbol{y})}{\partial&space;\boldsymbol{y}}=\frac{\partial&space;}{\partial&space;\boldsymbol{y}}\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{d}\right\|^2=\boldsymbol{y}-\boldsymbol{d}" title="\frac{\partial E(\boldsymbol{y})}{\partial \boldsymbol{y}}=\frac{\partial }{\partial \boldsymbol{y}}\frac{1}{2}\left\|\boldsymbol{y}-\boldsymbol{d}\right\|^2=\boldsymbol{y}-\boldsymbol{d}" /></br>

</br>

　出力層の活性化関数(恒等写像）の微分は、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{y}=\boldsymbol{u}^{(L)}" title="\boldsymbol{y}=\boldsymbol{u}^{(L)}" /></br>
	
　より、uについて微分すると、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;y(\boldsymbol{u})}{\partial&space;\boldsymbol{u}}=&space;\frac{\partial&space;u}{\partial&space;u}=1" title="\frac{\partial y(\boldsymbol{u})}{\partial \boldsymbol{u}}= \frac{\partial u}{\partial u}=1" /></br>

</br>

　総入力の計算は、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{u}^{(l)}=\boldsymbol{w}^{(l)}\boldsymbol{z}^{(l-1)}&plus;\boldsymbol{b}^{(l)}" title="\boldsymbol{u}^{(l)}=\boldsymbol{w}^{(l)}\boldsymbol{z}^{(l-1)}+\boldsymbol{b}^{(l)}" /></br>
	
　より、wについて微分すると、</br>
 
　　<img width="377" alt="image" src="https://user-images.githubusercontent.com/57135683/147222358-f71d068f-8ff0-45b0-b5b2-6400adcd7a3c.png"></br>
	
</br>

　よって、誤差関数のwでの微分は、</br>
 
　　<img width="118" alt="image" src="https://user-images.githubusercontent.com/57135683/147222935-3d36e6d3-c430-488d-8db5-42b4372e7095.png"></br>
	
　　<img width="274" alt="image" src="https://user-images.githubusercontent.com/57135683/147222856-f5419074-c99a-49b4-b073-40bde40231f5.png"></br>
	
　となる。</br>

</br>

## 5-2.確認問題
> 誤差逆伝播法では不要な再帰的処理を避けることができる。  
> 既に行った計算結果を保持しているソースコードを抽出せよ。  


```code
delta2 = functions.d_mean_squared_error(d, y)
delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)
delta1 = delta1[np.newaxis, :]
```

</br>

> それぞれに該当するソースコードを抜き出す。

 
　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E}{\partial&space;\boldsymbol{y}}" title="\frac{\partial E}{\partial \boldsymbol{y}}" /></br>
```code
delta2 = functions.d_mean_squared_error(d, y)
```  

</br>

　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E}{\partial&space;\boldsymbol{y}}\frac{\partial&space;\boldsymbol{y}}{\partial&space;\boldsymbol{u}}" title="\frac{\partial E}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{u}}" /></br>
```code
delta1 = np.dot(delta2, W2.T) * functions.d_sigmoid(z1)
```  

</br>

　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E}{\partial&space;\boldsymbol{y}}\frac{\partial&space;\boldsymbol{y}}{\partial&space;\boldsymbol{u}}\frac{\partial&space;\boldsymbol{u}}{\partial&space;\boldsymbol{w_{ji}^{2}}}" title="\frac{\partial E}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{u}}\frac{\partial \boldsymbol{u}}{\partial \boldsymbol{w_{ji}^{2}}}" /></br>
```code
grad['W1'] = np.dot(x.T, delta1)
```

</br>

## 5-3.実装演習
　実際に逆伝播の計算がどのように実装されているかを見る。</br>
```code
# 誤差逆伝播
def backward(x, d, z1, y):
    print("\n##### 誤差逆伝播開始 #####")

    grad = {}

    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']
    #  出力層でのデルタ
    delta2 = functions.d_sigmoid_with_loss(d, y)
    #  b2の勾配
    grad['b2'] = np.sum(delta2, axis=0)
    #  W2の勾配
    grad['W2'] = np.dot(z1.T, delta2)
    #  中間層でのデルタ
    delta1 = np.dot(delta2, W2.T) * functions.d_relu(z1)
    # b1の勾配
    grad['b1'] = np.sum(delta1, axis=0)
    #  W1の勾配
    grad['W1'] = np.dot(x.T, delta1)
        
    print_vec("偏微分_dE/du2", delta2)
    print_vec("偏微分_dE/du2", delta1)

    print_vec("偏微分_重み1", grad["W1"])
    print_vec("偏微分_重み2", grad["W2"])
    print_vec("偏微分_バイアス1", grad["b1"])
    print_vec("偏微分_バイアス2", grad["b2"])

    return grad
```
　実行結果は、</br>
　<img width="248" alt="image" src="https://user-images.githubusercontent.com/57135683/147314192-1ec0dcd9-8462-4c77-840b-ece47b6063b6.png">

</details>
