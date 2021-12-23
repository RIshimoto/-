# 入力層～中間層
## 要点まとめ
<img width="407" alt="image" src="https://user-images.githubusercontent.com/57135683/147182866-6eee92d4-47b9-4799-9a53-bb3a0155c481.png">
入力に重みを付けて混ぜ合わせた結果を、活性化関数で処理して出力に変換する</br>
それぞれの入力を</br>
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{x}&space;=&space;\begin{bmatrix}x_1&space;\\...&space;\\x_l\end{bmatrix}&space;" title="\boldsymbol{x} = \begin{bmatrix}x_1 \\... \\x_l\end{bmatrix} " />
重みを</br>
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{W}&space;=&space;\begin{bmatrix}w_1&space;\\...&space;\\w_l\end{bmatrix}&space;" title="\boldsymbol{W} = \begin{bmatrix}w_1 \\... \\w_l\end{bmatrix} " />
とし、それにバイアスを加えて</br>
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}u&=w_1x_1&plus;w_2x_2&plus;w_3x_3&plus;w_4x_4&plus;b\\&=\boldsymbol{W}\boldsymbol{x}&plus;b\end{align*}&space;" title="\begin{align*}u&=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b\\&=\boldsymbol{W}\boldsymbol{x}+b\end{align*} " /></br>
これを総入力といい、これに活性化関数を通すことで出力が得られる。</br>
その出力は、次のニューラルネットワークの入力として使われる。</br>
いわば、入力層と中間層が一つのパーツとみなせ、</br>
このパーツが何層も連なっているのが、深層ディープニューラルネットワークの仕組みになっている。</br>


## 確認テスト
> この図式に動物分類の実例を入れる。  

<img width="361" alt="image" src="https://user-images.githubusercontent.com/57135683/147183843-b269c72f-7bbb-41b0-b0da-8544dfbb9b62.png"></br>
動物の特徴として、体長、体重、ひげの本数、毛の平均長、耳の大きさ、眉間、足の長さとし、</br>
入力層にそれぞれの動物の特徴を表す数字を入れ、重みとバイアスで束ねて、次の層へと伝播させる。

> <img src="https://latex.codecogs.com/svg.image?\begin{align*}u&=w_1x_1&plus;w_2x_2&plus;w_3x_3&plus;w_4x_4&plus;b\\&=\boldsymbol{W}\boldsymbol{x}&plus;b\end{align*}&space;" title="\begin{align*}u&=w_1x_1+w_2x_2+w_3x_3+w_4x_4+b\\&=\boldsymbol{W}\boldsymbol{x}+b\end{align*} " /></br>をPythonでかけ。
```code
u = np.dot(x, W) + b
```

> 中間層の出力を定義しているソースを抜き出せ
```code
z = functions.relu(u)
```

## 実装演習
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

# 活性化関数
## 要点まとめ
ニューラルネットワークにおいて、次の層への出力の大きさを決める**非線形の関数**。</br>
入力値の値によって、次の層への信号のON/OFFや強弱を定める働きを持つ。</br>
線形な処理を非線形な活性化関数を通すことで、よりバラエティのある出力を作り出すことができる。</br>
例として、中間層用の活性化関数は、</br>
- ReLu関数
- シグモイド（ロジスティック）関数
- ステップ関数

### ステップ関数
<img width="223" alt="image" src="https://user-images.githubusercontent.com/57135683/147189005-1f1e0ecd-adcf-47f2-8c61-163930d8e203.png">  
閾値を超えたら1を出力する。パーセプトロンで利用された関数だが、線形分離可能なものしか学習できず、今は使われていない。</br>

### シグモイド関数
<img width="224" alt="image" src="https://user-images.githubusercontent.com/57135683/147189024-f5233382-2373-497c-98a6-f28856fda14e.png">  
0～1の間を緩やかに変化する関数で、信号の強弱を伝えられるようになり、予想ニューラルネットワークの普及のきっかけとなった。</br>
層が深くなると勾配消失が問題となる。</br>

### ReLu関数
<img width="209" alt="image" src="https://user-images.githubusercontent.com/57135683/147189057-c5b3b14a-e6df-423c-a820-a392b80aaf16.png">  
今の使われている関数。</br>
勾配消失問題とスパース化に貢献することで良い成果をもたらしている。</br>

## 確認テスト
> 線形と非線形の違いについて図を書いて簡易に説明せよ。
- 線形
<img width="284" alt="image" src="https://user-images.githubusercontent.com/57135683/147187450-399e6f0e-a9a7-4f31-a0ed-74ed2ca8ab04.png"></br>
線形は比例関係を満たす。
数学的な特徴では、
  - 加法性：f(x+y) = f(x) + f(y)
  - 斉次性：f(kx) = kf(x)
を満たす。

- 非線形
<img width="277" alt="image" src="https://user-images.githubusercontent.com/57135683/147187685-72e27ca0-fb63-4da3-91b9-514a01e99323.png"></br>
非線形は比例関係を満たさない。
数学的な特徴では、加法性、斉次性を満たさない。

> 配布されたソースコードより該当する箇所（z=f(u)）を抜き出せ。
```code
z1 = functions.sigmoid(u)
```

## 実装演習
上記で挙げた活性化関数の実装はそれぞれ以下のようになる。</br>
```code
# 中間層の活性化関数
# シグモイド関数（ロジスティック関数）
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# ReLU関数
def relu(x):
    return np.maximum(0, x)

# ステップ関数（閾値0）
def step_function(x):
    return np.where( x > 0, 1, 0)
```

# 出力層
## 誤差関数
### 要点まとめ
<img width="272" alt="image" src="https://user-images.githubusercontent.com/57135683/147194721-92e232d0-d6a1-4966-bfdd-db56c3e596ba.png">
ニューラルネットワークを学習させるには、</br>
まず、入力データとそれに対応した正解値を用意する。</br>

<img width="400" alt="image" src="https://user-images.githubusercontent.com/57135683/147195470-b2e92de9-6df3-40ee-a4b1-cbcae80d14ac.png">
実際に学習させるときは、</br>
入力データをニューラルネットワークに入力する。</br>
ニューラルネットワークの出力層からそれぞれのクラスの確率が出力させる。</br>

<img width="388" alt="image" src="https://user-images.githubusercontent.com/57135683/147195090-77a5dd24-ae96-4309-b6cf-e92b7d77c2d9.png">
その出力と正解値と比べ、</br>
どのくらい正解との誤差があるか、を誤差関数によって出力する。</br>

誤差関数の数値が小さければ、正しい答えに近いということを示し、
数値が大きければ、正しい答えからそれていることがわかる。

誤差関数には、二乗和誤差関数がある。

### 確認テスト
> 引き算ではなく二乗するかを述べよ</br>

誤差が正負でうち消すと誤差が正しく表せないため、2乗して正の値にして計算する。

> 下式の1/2はどういう意味か
<img src="https://latex.codecogs.com/svg.image?E_n(\boldsymbol{w})&space;=&space;&space;\frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2&space;=&space;\frac{1}{2}\left\|&space;\left(y&space;-&space;d\right)\right\|^2" title="E_n(\boldsymbol{w}) = \frac{1}{2}\sum_{j=1}^{I}\left(y_j-d_j\right)^2 = \frac{1}{2}\left\| \left(y - d\right)\right\|^2" />

ネットワークの学習を行うときに行う逆伝播法の計算で微分する際、式が簡単になるため。

### 実装演習
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

## 活性化関数
### 要点まとめ
中間層と出力層では用いる活性化関数が異なる。</br>
中間層では閾値前後で信号の強弱を調整するために活性化関数を用いたが、</br>
出力層では信号の大きさをそのままに我々が使いやすいように変換するのが目的。</br>
また分類問題では出力層の出力の総和は1になる必要がある。

例として出力層用の活性化関数は、</br>
- ソフトマックス関数
- 恒等写像
- シグモイド関数（ロジスティック関数）

使い分けは以下の通り</br>
<img width="395" alt="image" src="https://user-images.githubusercontent.com/57135683/147198517-b7de72b9-761a-4ead-b470-ab369e94dd3c.png">

### 確認テスト
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

### 実装演習
シグモイド関数の実装は、</br>
<img src="https://latex.codecogs.com/svg.image?f(u)=\frac{1}{1&plus;e^{-x}}" title="f(u)=\frac{1}{1+e^{-x}}" />
```code
def sigmoid(x):
    return 1/(1 + np.exp(-x))
```

ソフトマックス関数の実装は、</br>
<img src="https://latex.codecogs.com/svg.image?f(i,&space;u)=\frac{e^{u_i}}{\sum_{k=1}^{K}e^{u_k}}" title="f(i, u)=\frac{e^{u_i}}{\sum_{k=1}^{K}e^{u_k}}" />
```code
# ソフトマックス関数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
```

クロスエントロピー関数の実装は、</br>
<img width="398" alt="image" src="https://user-images.githubusercontent.com/57135683/147201488-4fcd6147-3aee-4c40-aeb6-178a14647c09.png">
```code
# クロスエントロピー
def cross_entropy_error(d, y):
    if y.ndim == 1:
        d = d.reshape(1, d.size)
        y = y.reshape(1, y.size)
        
    if d.size == y.size:
        d = d.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), d] + 1e-7)) / batch_size
```

# 勾配降下法
# 誤差逆伝播法
