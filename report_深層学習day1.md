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

出力層用の活性化関数は、</br>
- ソフトマックス関数
- 恒等写像
- シグモイド関数（ロジスティック関数）

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
# 勾配降下法
# 誤差逆伝播法
