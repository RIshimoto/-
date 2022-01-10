# 1.再帰型ニューラルネットワークの概念
## 1-1.要点まとめ
- RNNとは

  再帰型ニューラルネットワーク（以下、RNN）とは、時系列データに対応可能なニューラルネットワーク。</br>

- 時間列データとは？

  時系列的順序を追って一定間隔ごとに観察され、しかも相互に統計的な依存関係が認められるようなデータの系列。</br>
  音声データやテキストデータなど。</br>

- 時間的なつながりを学習させるには？

  <img width="300" alt="image" src="https://user-images.githubusercontent.com/57135683/148037994-a8c9476a-8f56-4c26-97a8-d9233733f8f5.png"></br>
  中間層の出力を出力層にだすと同時に次の中間層にもういっかい入れることで、時間的のつながりを学習させる。</br>
  
  <img width="300" alt="image" src="https://user-images.githubusercontent.com/57135683/148038147-c7f07b04-3024-4ea3-855a-eb1cafb458b9.png"></br>
  重みは３か所。入力層からの情報をよしなに加工してくれるW(in)、中間層から出力層への重みW(out)、前の中間層からの重みW。</br>
  この３つの重みを学習させる。</br>

- RNNの特徴は？

  時系列データを扱うには、初期の状態と過去の時間t-1の状態を保持し、そこから次の時間でのtを再帰的に求める再帰構造が必要になる。


  <img src="https://latex.codecogs.com/svg.image?\begin{align*}&\boldsymbol{u}^t=\boldsymbol{W}_{in}\boldsymbol{x}^t&plus;\boldsymbol{W}\boldsymbol{z}^{t-1}&plus;\boldsymbol{b}\\&\boldsymbol{z}^t=f\left(\boldsymbol{u}^t\right)\\&\boldsymbol{v}^{t}=\boldsymbol{W}_{(out)}\boldsymbol{z}^t&plus;\boldsymbol{c}\\&\boldsymbol{y}^{t}=g\left(\boldsymbol{v}^{t}\right)\end{align*}" title="\begin{align*}&\boldsymbol{u}^t=\boldsymbol{W}_{in}\boldsymbol{x}^t+\boldsymbol{W}\boldsymbol{z}^{t-1}+\boldsymbol{b}\\&\boldsymbol{z}^t=f\left(\boldsymbol{u}^t\right)\\&\boldsymbol{v}^{t}=\boldsymbol{W}_{(out)}\boldsymbol{z}^t+\boldsymbol{c}\\&\boldsymbol{y}^{t}=g\left(\boldsymbol{v}^{t}\right)\end{align*}" /></br>

  コードに起こすと、</br>
  ```code
  u[:,t+1] = np.dot(X, W_in)+np.dot(z[:,t].reshape(1, -1), W)
  z[:,t+1] = functions.sigmoid(u[:,t+1])
  y[:,t] = functions.sigmoid(np.dot(z[:,t+1].reshape(1,-1), W_out))
  ```

- BPTTとは？

  RNNにおいてのパラメータ調整法の一種。</br>

  * BPTTの数学的記述</br>
  
    <img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(in)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W_{(in)}}}\right]^T=\delta&space;^t\left[\boldsymbol{x}^t\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(in)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W_{(in)}}}\right]^T=\delta&space;^t\left[\boldsymbol{x}^t\right]^T\end{align*}" /></br>
    ```code
    np.dot(X.T, delta[:t].reshape(1, -1))
    ```

    <img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(out)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{W_{(out)}}}\right]^T=\delta^{out,&space;t}\left[\boldsymbol{z}^t\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(out)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{W_{(out)}}}\right]^T=\delta^{out, t}\left[\boldsymbol{z}^t\right]^T\end{align*}" /></br>
    ```code
    np.dot(z[:,t+1].reshape(-1, 1), delta_out[:,t].reshape(-1, 1))
    ```

    <img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W}}\right]^T=\delta^{t}\left[\boldsymbol{z}^{t-1}\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W}}\right]^T=\delta^{t}\left[\boldsymbol{z}^{t-1}\right]^T\end{align*}" /></br>
    ```code
    np.dot(z[:,t].reshape(-1, 1), delta[:,t].reshape(1, -1))
    ```
  
    <img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{b}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{b}}\right]^T=\delta^{t}\\\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{c}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{c}}\right]^T=\delta^{out,t}\\\end{align*}&space;" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{b}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{b}}\right]^T=\delta^{t}\\\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{c}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{c}}\right]^T=\delta^{out,t}\\\end{align*} " /></br>

  </br>

  * BPTTの全体像</br>
    <img width="411" alt="image" src="https://user-images.githubusercontent.com/57135683/148678639-d89898d4-c7bf-47bb-9480-286c2e0c4522.png"></br>
  
</br>

## 1-2.確認テスト
> RNNのネットワークには大きく分けて３つの重みがある。</br>
> 1つは入力から現在の中間層を定義する際にかけられる重み、１つは中間層から出力を定義する際にかけられる重みである。</br>
> 残り１つの重みについて説明せよ。</br>

　中間層からの中間層への重み。</br>

</br>

> 連鎖律の原理を使い、dz/dxを求めよ。</br>
> <img src="https://latex.codecogs.com/svg.image?\begin{align*}z=t^2\\t=x&plus;y\end{align*}" title="\begin{align*}z=t^2\\t=x+y\end{align*}" /></br>

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{dz}{dx}&=\frac{dz}{dt}\cdot\frac{dt}{dx}\\\frac{dz}{dt}&=2t\\\frac{dt}{dx}&=1\\\therefore\frac{dz}{dx}&=2t\cdot1\\&=2\left(x&plus;y\right)\end{align*}" title="\begin{align*}\frac{dz}{dx}&=\frac{dz}{dt}\cdot\frac{dt}{dx}\\\frac{dz}{dt}&=2t\\\frac{dt}{dx}&=1\\\therefore\frac{dz}{dx}&=2t\cdot1\\&=2\left(x+y\right)\end{align*}" /></br>

</br>

> 下図のy1をx・z0・z1・Win・W・Woutを用いて数式で表せ。</br>
> バイアスは任意の文字で定義せよ。</br>
> また中間層の出力にシグモイド関数g(x)を作用させよ。</br>
> <img width="320" alt="image" src="https://user-images.githubusercontent.com/57135683/148175014-84c74f98-5edc-4ad3-8a11-776e2a6a1486.png"></br>

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}s_1=W_{(in)}x_1&plus;Wz_0&plus;b\\y_1=g\left(W_{(out)}s_1&plus;c\right)\end{align*}&space;" title="\begin{align*}s_1=W_{(in)}x_1+Wz_0+b\\y_1=g\left(W_{(out)}s_1+c\right)\end{align*} " /></br>

</br>

> <img width="412" alt="image" src="https://user-images.githubusercontent.com/57135683/148679042-03fc3a7c-dfbf-4d13-896a-e2767b54fae8.png"></br>

　2

</br>

## 1-3.実装演習
題材としてバイナリ加算を扱う。</br>
```code
import numpy as np
from common import functions
import matplotlib.pyplot as plt

# def d_tanh(x):



# データを用意
# 2進数の桁数
binary_dim = 8
# 最大値 + 1
largest_number = pow(2, binary_dim)
# largest_numberまで2進数を用意
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)

input_layer_size = 2
hidden_layer_size = 16
output_layer_size = 1

weight_init_std = 1
learning_rate = 0.1

iters_num = 10000
plot_interval = 100

# ウェイト初期化 (バイアスは簡単のため省略)
W_in = weight_init_std * np.random.randn(input_layer_size, hidden_layer_size)
W_out = weight_init_std * np.random.randn(hidden_layer_size, output_layer_size)
W = weight_init_std * np.random.randn(hidden_layer_size, hidden_layer_size)

# Xavier


# He



# 勾配
W_in_grad = np.zeros_like(W_in)
W_out_grad = np.zeros_like(W_out)
W_grad = np.zeros_like(W)

u = np.zeros((hidden_layer_size, binary_dim + 1))
z = np.zeros((hidden_layer_size, binary_dim + 1))
y = np.zeros((output_layer_size, binary_dim))

delta_out = np.zeros((output_layer_size, binary_dim))
delta = np.zeros((hidden_layer_size, binary_dim + 1))

all_losses = []

for i in range(iters_num):
    
    # A, B初期化 (a + b = d)
    a_int = np.random.randint(largest_number/2)
    a_bin = binary[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2)
    b_bin = binary[b_int] # binary encoding
    
    # 正解データ
    d_int = a_int + b_int
    d_bin = binary[d_int]
    
    # 出力バイナリ
    out_bin = np.zeros_like(d_bin)
    
    # 時系列全体の誤差
    all_loss = 0    
    
    # 時系列ループ
    for t in range(binary_dim):
        # 入力値
        X = np.array([a_bin[ - t - 1], b_bin[ - t - 1]]).reshape(1, -1)
        # 時刻tにおける正解データ
        dd = np.array([d_bin[binary_dim - t - 1]])
        
        u[:,t+1] = np.dot(X, W_in) + np.dot(z[:,t].reshape(1, -1), W)
        z[:,t+1] = functions.sigmoid(u[:,t+1])

        y[:,t] = functions.sigmoid(np.dot(z[:,t+1].reshape(1, -1), W_out))


        #誤差
        loss = functions.mean_squared_error(dd, y[:,t])
        
        delta_out[:,t] = functions.d_mean_squared_error(dd, y[:,t]) * functions.d_sigmoid(y[:,t])        
        
        all_loss += loss

        out_bin[binary_dim - t - 1] = np.round(y[:,t])
    
    
    for t in range(binary_dim)[::-1]:
        X = np.array([a_bin[-t-1],b_bin[-t-1]]).reshape(1, -1)        

        delta[:,t] = (np.dot(delta[:,t+1].T, W.T) + np.dot(delta_out[:,t].T, W_out.T)) * functions.d_sigmoid(u[:,t+1])

        # 勾配更新
        W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(-1,1))
        W_grad += np.dot(z[:,t].reshape(-1,1), delta[:,t].reshape(1,-1))
        W_in_grad += np.dot(X.T, delta[:,t].reshape(1,-1))
    
    # 勾配適用
    W_in -= learning_rate * W_in_grad
    W_out -= learning_rate * W_out_grad
    W -= learning_rate * W_grad
    
    W_in_grad *= 0
    W_out_grad *= 0
    W_grad *= 0
    

    if(i % plot_interval == 0):
        all_losses.append(all_loss)        
        print("iters:" + str(i))
        print("Loss:" + str(all_loss))
        print("Pred:" + str(out_bin))
        print("True:" + str(d_bin))
        out_int = 0
        for index,x in enumerate(reversed(out_bin)):
            out_int += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out_int))
        print("------------")

lists = range(0, iters_num, plot_interval)
plt.plot(lists, all_losses, label="loss")
plt.show()
```

<img width="286" alt="image" src="https://user-images.githubusercontent.com/57135683/148674742-08b656ad-83e9-409d-be77-2c8078bf6ad8.png"></br>

中間層の活性化関数をrelu関数にしたとき、勾配爆発が起こりうまく学習ができていないことがわかる。</br>
<img width="280" alt="image" src="https://user-images.githubusercontent.com/57135683/148675081-cbcc83fd-0b27-4c49-9422-da9c31f382e6.png"></br>

</br>

# 2.LSTM
## 2-1.要点まとめ
　RNNの課題は、時系列をさかのぼれば、勾配が消失していく。</br>
　長い時系列の学習が困難。</br>

　構造自体を変えて解決したものがLSTM。</br>

　<img width="348" alt="image" src="https://user-images.githubusercontent.com/57135683/148670284-e5d89db4-2407-462c-8c42-053edd2a5329.png"></br>

  - CEC

    勾配消失および勾配爆発の解決策として、勾配が１であれば解決できる。</br>
    <img src="https://latex.codecogs.com/svg.image?\delta&space;^{t-z-1}=\delta^{t-z}{Wf'(u^{t-z-1})}=1&space;" title="\delta ^{t-z-1}=\delta^{t-z}{Wf'(u^{t-z-1})}=1 " /></br>
    <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E}{\partial&space;c^{t-1}}=\frac{\partial&space;E}{\partial&space;c^t}\cdot\frac{\partial&space;c^t}{\partial&space;c^{t-1}}=\frac{\partial&space;E}{\partial&space;c^t}\cdot\frac{\partial}{\partial&space;c^{t-1}}\cdot\left\{&space;a^t-c^{t-1}&space;\right\}=\frac{\partial&space;E}{\partial&space;c^t}" title="\frac{\partial E}{\partial c^{t-1}}=\frac{\partial E}{\partial c^t}\cdot\frac{\partial c^t}{\partial c^{t-1}}=\frac{\partial E}{\partial c^t}\cdot\frac{\partial}{\partial c^{t-1}}\cdot\left\{ a^t-c^{t-1} \right\}=\frac{\partial E}{\partial c^t}" /></br>

    ただし、CECには記憶機能のみで学習機能がないので、CECの周りに学習機能を配置する。</br>
  
  - 入力ゲート/出力ゲート

    それぞれのゲートへの入力値の重みを、重み行列W, Uで可変可能とする。
  
  - 忘却ゲート

    CECは過去の情報が保管され続けるため、忘却ゲートによって情報を忘却させる。

  - のぞき穴結合

    CECの保存されている過去の情報を、任意のタイミングで他のノードに伝播させたり、</br>
    あるいは任意のタイミングで忘却させたい。</br>
    そこでのぞき穴結合によって、CEC自身の値に重み行列を介して伝播可能にする。</br>

</br>

## 2-2.確認問題
> シグモイド関数を微分したとき、入力値が0の時に最大値をとる。その値として正しいもの。

　0.25</br>
 
 </br>
 
 > <img width="373" alt="image" src="https://user-images.githubusercontent.com/57135683/148669768-f48a7788-63cc-472f-a54e-14fb5ec905c9.png">

　1</br>
 
 </br>
 
> 以下の文章をLSTMに入力し、空欄に当てはまる単語を予測したいとする。</br>
> 文中の「とても」という言葉は空欄の予測においてなくなっても影響を及ぼさないと考えられる。</br>
> このような場合、どのゲートが作用すると考えられるか。</br>
> 「映画おもしろかったね。ところで、とてもお腹が空いたから何か___。」</br>

　忘却ゲート</br>

</br>

> <img width="386" alt="image" src="https://user-images.githubusercontent.com/57135683/148670769-cf1c3e21-cc24-4986-908f-a5215ad8688c.png"></br>

　3</br>
 
</br>

## 2-3.実装演習
　GRUのところでまとめて行う。</br>

</br>

# 3.GRU
## 3-1.要点まとめ

  LSTMでは、パラメータ数が多く、計算負荷が高くなる問題があった。</br>
  それを解消したのがGRUである。</br>
  
  GRUはパラメータを大幅に削減し、精度は同等またはそれ以上が望めるようになった構造。</br>
  計算負荷が低いのがメリット。</br>
  
  <img width="338" alt="image" src="https://user-images.githubusercontent.com/57135683/148670872-462ee843-a91a-4b51-99a1-362187e8ae6f.png"></br>

</br>

## 3-2.確認問題
> LSTMとCECが抱える課題についてそれぞれ簡潔に述べよ。</br>

  LSTMは、パラメータ数が多くなり計算量が多くなること。</br>
  CECは、学習機能がなこと。</br>

</br>

> <img width="389" alt="image" src="https://user-images.githubusercontent.com/57135683/148670891-85d49868-5784-4a85-ad23-4affa68c7f06.png"></br>

　4</br>
 
</br>

> LSTMとGRUの違いを簡潔に述べよ。</br>

  LSTMよりGRUのほうが計算量が少ない。</br>

</br>

## 3-3.実装演習

```code
import tensorflow as tf
import numpy as np
import re
import glob
import collections
import random
import pickle
import time
import datetime
import os

# logging levelを変更
tf.logging.set_verbosity(tf.logging.ERROR)

class Corpus:
    def __init__(self):
        self.unknown_word_symbol = "<???>" # 出現回数の少ない単語は未知語として定義しておく
        self.unknown_word_threshold = 3 # 未知語と定義する単語の出現回数の閾値
        self.corpus_file = "./corpus/**/*.txt"
        self.corpus_encoding = "utf-8"
        self.dictionary_filename = "./data_for_predict/word_dict.dic"
        self.chunk_size = 5
        self.load_dict()

        words = []
        for filename in glob.glob(self.corpus_file, recursive=True):
            with open(filename, "r", encoding=self.corpus_encoding) as f:

                # word breaking
                text = f.read()
                # 全ての文字を小文字に統一し、改行をスペースに変換
                text = text.lower().replace("\n", " ")
                # 特定の文字以外の文字を空文字に置換する
                text = re.sub(r"[^a-z '\-]", "", text)
                # 複数のスペースはスペース一文字に変換
                text = re.sub(r"[ ]+", " ", text)

                # 前処理： '-' で始まる単語は無視する
                words = [ word for word in text.split() if not word.startswith("-")]


        self.data_n = len(words) - self.chunk_size
        self.data = self.seq_to_matrix(words)

    def prepare_data(self):
        """
        訓練データとテストデータを準備する。
        data_n = ( text データの総単語数 ) - chunk_size
        input: (data_n, chunk_size, vocabulary_size)
        output: (data_n, vocabulary_size)
        """

        # 入力と出力の次元テンソルを準備
        all_input = np.zeros([self.chunk_size, self.vocabulary_size, self.data_n])
        all_output = np.zeros([self.vocabulary_size, self.data_n])

        # 準備したテンソルに、コーパスの one-hot 表現(self.data) のデータを埋めていく
        # i 番目から ( i + chunk_size - 1 ) 番目までの単語が１組の入力となる
        # このときの出力は ( i + chunk_size ) 番目の単語
        for i in range(self.data_n):
            all_output[:, i] = self.data[:, i + self.chunk_size] # (i + chunk_size) 番目の単語の one-hot ベクトル
            for j in range(self.chunk_size):
                all_input[j, :, i] = self.data[:, i + self.chunk_size - j - 1]

        # 後に使うデータ形式に合わせるために転置を取る
        all_input = all_input.transpose([2, 0, 1])
        all_output = all_output.transpose()

        # 訓練データ：テストデータを 4 : 1 に分割する
        training_num = ( self.data_n * 4 ) // 5
        return all_input[:training_num], all_output[:training_num], all_input[training_num:], all_output[training_num:]


    def build_dict(self):
        # コーパス全体を見て、単語の出現回数をカウントする
        counter = collections.Counter()
        for filename in glob.glob(self.corpus_file, recursive=True):
            with open(filename, "r", encoding=self.corpus_encoding) as f:

                # word breaking
                text = f.read()
                # 全ての文字を小文字に統一し、改行をスペースに変換
                text = text.lower().replace("\n", " ")
                # 特定の文字以外の文字を空文字に置換する
                text = re.sub(r"[^a-z '\-]", "", text)
                # 複数のスペースはスペース一文字に変換
                text = re.sub(r"[ ]+", " ", text)

                # 前処理： '-' で始まる単語は無視する
                words = [word for word in text.split() if not word.startswith("-")]

                counter.update(words)

        # 出現頻度の低い単語を一つの記号にまとめる
        word_id = 0
        dictionary = {}
        for word, count in counter.items():
            if count <= self.unknown_word_threshold:
                continue

            dictionary[word] = word_id
            word_id += 1
        dictionary[self.unknown_word_symbol] = word_id

        print("総単語数：", len(dictionary))

        # 辞書を pickle を使って保存しておく
        with open(self.dictionary_filename, "wb") as f:
            pickle.dump(dictionary, f)
            print("Dictionary is saved to", self.dictionary_filename)

        self.dictionary = dictionary

        print(self.dictionary)

    def load_dict(self):
        with open(self.dictionary_filename, "rb") as f:
            self.dictionary = pickle.load(f)
            self.vocabulary_size = len(self.dictionary)
            self.input_layer_size = len(self.dictionary)
            self.output_layer_size = len(self.dictionary)
            print("総単語数: ", self.input_layer_size)

    def get_word_id(self, word):
        # print(word)
        # print(self.dictionary)
        # print(self.unknown_word_symbol)
        # print(self.dictionary[self.unknown_word_symbol])
        # print(self.dictionary.get(word, self.dictionary[self.unknown_word_symbol]))
        return self.dictionary.get(word, self.dictionary[self.unknown_word_symbol])

    # 入力された単語を one-hot ベクトルにする
    def to_one_hot(self, word):
        index = self.get_word_id(word)
        data = np.zeros(self.vocabulary_size)
        data[index] = 1
        return data

    def seq_to_matrix(self, seq):
        print(seq)
        data = np.array([self.to_one_hot(word) for word in seq]) # (data_n, vocabulary_size)
        return data.transpose() # (vocabulary_size, data_n)

class Language:
    """
    input layer: self.vocabulary_size
    hidden layer: rnn_size = 30
    output layer: self.vocabulary_size
    """

    def __init__(self):
        self.corpus = Corpus()
        self.dictionary = self.corpus.dictionary
        self.vocabulary_size = len(self.dictionary) # 単語数
        self.input_layer_size = self.vocabulary_size # 入力層の数
        self.hidden_layer_size = 30 # 隠れ層の RNN ユニットの数
        self.output_layer_size = self.vocabulary_size # 出力層の数
        self.batch_size = 128 # バッチサイズ
        self.chunk_size = 5 # 展開するシーケンスの数。c_0, c_1, ..., c_(chunk_size - 1) を入力し、c_(chunk_size) 番目の単語の確率が出力される。
        self.learning_rate = 0.005 # 学習率
        self.epochs = 1000 # 学習するエポック数
        self.forget_bias = 1.0 # LSTM における忘却ゲートのバイアス
        self.model_filename = "./data_for_predict/predict_model.ckpt"
        self.unknown_word_symbol = self.corpus.unknown_word_symbol

    def inference(self, input_data, initial_state):
        """
        :param input_data: (batch_size, chunk_size, vocabulary_size) 次元のテンソル
        :param initial_state: (batch_size, hidden_layer_size) 次元の行列
        :return:
        """
        # 重みとバイアスの初期化
        hidden_w = tf.Variable(tf.truncated_normal([self.input_layer_size, self.hidden_layer_size], stddev=0.01))
        hidden_b = tf.Variable(tf.ones([self.hidden_layer_size]))
        output_w = tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.output_layer_size], stddev=0.01))
        output_b = tf.Variable(tf.ones([self.output_layer_size]))

        # BasicLSTMCell, BasicRNNCell は (batch_size, hidden_layer_size) が chunk_size 数ぶんつながったリストを入力とする。
        # 現時点での入力データは (batch_size, chunk_size, input_layer_size) という３次元のテンソルなので
        # tf.transpose や tf.reshape などを駆使してテンソルのサイズを調整する。

        input_data = tf.transpose(input_data, [1, 0, 2]) # 転置。(chunk_size, batch_size, vocabulary_size)
        input_data = tf.reshape(input_data, [-1, self.input_layer_size]) # 変形。(chunk_size * batch_size, input_layer_size)
        input_data = tf.matmul(input_data, hidden_w) + hidden_b # 重みWとバイアスBを適用。 (chunk_size, batch_size, hidden_layer_size)
        input_data = tf.split(input_data, self.chunk_size, 0) # リストに分割。chunk_size * (batch_size, hidden_layer_size)

        # RNN のセルを定義する。RNN Cell の他に LSTM のセルや GRU のセルなどが利用できる。
        cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_layer_size)
        outputs, states = tf.nn.static_rnn(cell, input_data, initial_state=initial_state)
        
        # 最後に隠れ層から出力層につながる重みとバイアスを処理する
        # 最終的に softmax 関数で処理し、確率として解釈される。
        # softmax 関数はこの関数の外で定義する。
        output = tf.matmul(outputs[-1], output_w) + output_b

        return output

    def loss(self, logits, labels):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return cost

    def training(self, cost):
        # 今回は最適化手法として Adam を選択する。
        # ここの AdamOptimizer の部分を変えることで、Adagrad, Adadelta などの他の最適化手法を選択することができる
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        return optimizer

    def train(self):
        # 変数などの用意
        input_data = tf.placeholder("float", [None, self.chunk_size, self.input_layer_size])
        actual_labels = tf.placeholder("float", [None, self.output_layer_size])
        initial_state = tf.placeholder("float", [None, self.hidden_layer_size])

        prediction = self.inference(input_data, initial_state)
        cost = self.loss(prediction, actual_labels)
        optimizer = self.training(cost)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # TensorBoard で可視化するため、クロスエントロピーをサマリーに追加
        tf.summary.scalar("Cross entropy: ", cost)
        summary = tf.summary.merge_all()

        # 訓練・テストデータの用意
        # corpus = Corpus()
        trX, trY, teX, teY = self.corpus.prepare_data()
        training_num = trX.shape[0]

        # ログを保存するためのディレクトリ
        timestamp = time.time()
        dirname = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")

        # ここから実際に学習を走らせる
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter("./log/" + dirname, sess.graph)

            # エポックを回す
            for epoch in range(self.epochs):
                step = 0
                epoch_loss = 0
                epoch_acc = 0

                # 訓練データをバッチサイズごとに分けて学習させる (= optimizer を走らせる)
                # エポックごとの損失関数の合計値や（訓練データに対する）精度も計算しておく
                while (step + 1) * self.batch_size < training_num:
                    start_idx = step * self.batch_size
                    end_idx = (step + 1) * self.batch_size

                    batch_xs = trX[start_idx:end_idx, :, :]
                    batch_ys = trY[start_idx:end_idx, :]

                    _, c, a = sess.run([optimizer, cost, accuracy],
                                       feed_dict={input_data: batch_xs,
                                                  actual_labels: batch_ys,
                                                  initial_state: np.zeros([self.batch_size, self.hidden_layer_size])
                                                  }
                                       )
                    epoch_loss += c
                    epoch_acc += a
                    step += 1

                # コンソールに損失関数の値や精度を出力しておく
                print("Epoch", epoch, "completed ouf of", self.epochs, "-- loss:", epoch_loss, " -- accuracy:",
                      epoch_acc / step)

                # Epochが終わるごとにTensorBoard用に値を保存
                summary_str = sess.run(summary, feed_dict={input_data: trX,
                                                           actual_labels: trY,
                                                           initial_state: np.zeros(
                                                               [trX.shape[0],
                                                                self.hidden_layer_size]
                                                           )
                                                           }
                                       )
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            # 学習したモデルも保存しておく
            saver = tf.train.Saver()
            saver.save(sess, self.model_filename)

            # 最後にテストデータでの精度を計算して表示する
            a = sess.run(accuracy, feed_dict={input_data: teX, actual_labels: teY,
                                              initial_state: np.zeros([teX.shape[0], self.hidden_layer_size])})
            print("Accuracy on test:", a)


    def predict(self, seq):
        """
        文章を入力したときに次に来る単語を予測する
        :param seq: 予測したい単語の直前の文字列。chunk_size 以上の単語数が必要。
        :return:
        """

        # 最初に復元したい変数をすべて定義してしまいます
        tf.reset_default_graph()
        input_data = tf.placeholder("float", [None, self.chunk_size, self.input_layer_size])
        initial_state = tf.placeholder("float", [None, self.hidden_layer_size])
        prediction = tf.nn.softmax(self.inference(input_data, initial_state))
        predicted_labels = tf.argmax(prediction, 1)

        # 入力データの作成
        # seq を one-hot 表現に変換する。
        words = [word for word in seq.split() if not word.startswith("-")]
        x = np.zeros([1, self.chunk_size, self.input_layer_size])
        for i in range(self.chunk_size):
            word = seq[len(words) - self.chunk_size + i]
            index = self.dictionary.get(word, self.dictionary[self.unknown_word_symbol])
            x[0][i][index] = 1
        feed_dict = {
            input_data: x, # (1, chunk_size, vocabulary_size)
            initial_state: np.zeros([1, self.hidden_layer_size])
        }

        # tf.Session()を用意
        with tf.Session() as sess:
            # 保存したモデルをロードする。ロード前にすべての変数を用意しておく必要がある。
            saver = tf.train.Saver()
            saver.restore(sess, self.model_filename)

            # ロードしたモデルを使って予測結果を計算
            u, v = sess.run([prediction, predicted_labels], feed_dict=feed_dict)

            keys = list(self.dictionary.keys())


            # コンソールに文字ごとの確率を表示
            for i in range(self.vocabulary_size):
                c = self.unknown_word_symbol if i == (self.vocabulary_size - 1) else keys[i]
                print(c, ":", u[0][i])

            print("Prediction:", seq + " " + ("<???>" if v[0] == (self.vocabulary_size - 1) else keys[v[0]]))

        return u[0]


def build_dict():
    cp = Corpus()
    cp.build_dict()

if __name__ == "__main__":
    #build_dict()

    ln = Language()

    # 学習するときに呼び出す
    #ln.train()

    # 保存したモデルを使って単語の予測をする
    ln.predict("some of them looks like")
```
<img width="359" alt="image" src="https://user-images.githubusercontent.com/57135683/148679725-627006b1-2eca-4f8b-8f10-c422f76b3e2c.png">


</br>

# 4.双方向RNN
## 4-1.要点まとめ
過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデル。</br>
文章の推敲や機械学習などで用いられる。</br>
<img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/148670934-1fcaa812-1701-43fc-a09e-8ecc5f19adbc.png"></br>

</br>

## 4-2.確認問題
> <img width="359" alt="image" src="https://user-images.githubusercontent.com/57135683/148670920-9bfc875e-9272-48cd-be01-b02f1403d123.png"></br>

　4</br>
 
</br>

# 5.Seq2Seq
## 5-1.要点まとめ
<img width="409" alt="image" src="https://user-images.githubusercontent.com/57135683/148673338-17f837fd-56d7-47af-924f-beeebe620419.png"></br>

Seq2Seqとは、Encoder-Decoderモデルの一種を指す。</br>
機械対話や、機械翻訳などに使用されている。</br>

- Encoder RNN

  ユーザーがインプットしたテキストデータを、単語などのトークンに区切って渡す構造。</br>
  <img width="232" alt="image" src="https://user-images.githubusercontent.com/57135683/148671001-1e144f91-28db-4f1f-8af8-69f74ff7543a.png"></br>
  Taking: 文章を単語などのトークン毎に分割し、トークンごとのIDに分割する。</br>
  Embedding: IDから、そのトークンを表す分散表現ベクトルに変換。</br>
  Encoder RNN: ベクトルを順番にRNNに入力していく。</br>
  
  処理手順としては、
  1.  vec1をRNNに入力し、hidden stateを出力。</br>
    このhidden stateと次の入力vec2をまたRNNに入力してきたhidden stateを出力という流れを繰り返す。</br>
  2. 最後のvecを入れたときのhidden stateをfinal stateとして取っておく。</br>
    このfinal stateがthought vectorと呼ばれ、入力した文の意味を表すベクトルとなる。</br>

- Decoder RNN
  
  システムがアウトプットデータを、単語などのトークンごとに生成する構造。
  <img width="239" alt="image" src="https://user-images.githubusercontent.com/57135683/148671115-3de4ec0c-94b6-4958-9569-01a3d54b4302.png">

  処理手順としては、
  1.Decoder RNN

    Encoder RNNのfinal state(thought vector)から、</br>
    各tokenの生成確率を出力して、final stateをDecoder RNNのinitial stateとして設定し、</br>
    Embeddingを入力。
    
  2.Sampling

    生成確率にもとづいてtokenをランダムに選ぶ。

  3.Embedding

    2で選ばれたtokenをEmbeddingしてDecoder RNNへの次の入力とします。
    
  4.Detokenize

    1-3を繰り返して、2で得られたtokenを文字列に直す。

- HRED

  過去のn-1個の発話から次の発話を生成する。</br>
  Seq2seqでは、会話の文脈無視で、応答がなされたが、</br>
  HREDでは、前の単語の流れに即して応答されるため、より人間らしい文章が生成される。</br>
  
  Seq2seqにContext RNN加えたものである。</br>
  Context RNNとは、Encoderのまとめた各文書の系列をまとめて、これまでの会話コンテキスト全体を表すベクトルに変換する構造。これによって過去の発話履歴を加味した返答ができる。</br>
  
  バリーエーションに富んだ会話ができないのが課題。</br>
  
- VHRED

  課題を解決するために、HREDにVAEの潜在変数の概念を追加したもの。</br>
  
- VAE
  * オートエンコーダー

    教師なし学習のひとつ。</br>
    入力データから潜在変数zに変換するニューラルネットワークをEncoder。</br>
    逆に潜在変数zをインプットとして元画像を復元するニューラルネットワークをDecoder。</br>
    
    次元削減が行えるのがメリット。</br>
    
  * VAE

    VAEは潜在変数zに確率変数z～N(0,1)を仮定したもの。</br>
    VAEは、データを潜在変数zの確率分布という構造に押し込めることを可能とする。</br>

## 5-2.確認テスト
> <img width="359" alt="image" src="https://user-images.githubusercontent.com/57135683/148671876-9f9ed7f0-90c1-4485-9ac4-5e516149d349.png">

　2</br>
 
</br>

> <img width="430" alt="image" src="https://user-images.githubusercontent.com/57135683/148671882-5f5406c0-cf04-4f4e-bc33-090a2119f40a.png">

　1</br>
 
</br>

> VAEに関する下記の説明文中の空欄に当てはまる言葉を答えよ。</br>
> 自己符号器の潜在変数に___を導入したもの。

　確率分布</br>

</br>

## 5-3.実装演習
EncoderDecoderモデル</br>
```code
class EncoderDecoder(nn.Module):
    """EncoderとDecoderの処理をまとめる"""
    def __init__(self, input_size, output_size, hidden_size):
        """
        :param input_size: int, 入力言語の語彙数
        :param output_size: int, 出力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, batch_X, lengths_X, max_length, batch_Y=None, use_teacher_forcing=False):
        """
        :param batch_X: tensor, 入力系列のバッチ, size=(max_length, batch_size)
        :param lengths_X: list, 入力系列のバッチ内の各サンプルの文長
        :param max_length: int, Decoderの最大文長
        :param batch_Y: tensor, Decoderで用いるターゲット系列
        :param use_teacher_forcing: Decoderでターゲット系列を入力とするフラグ
        :return decoder_outputs: tensor, Decoderの出力, 
            size=(max_length, batch_size, self.decoder.output_size)
        """
        # encoderに系列を入力（複数時刻をまとめて処理）
        _, encoder_hidden = self.encoder(batch_X, lengths_X)
        
        _batch_size = batch_X.size(1)

        # decoderの入力と隠れ層の初期状態を定義
        decoder_input = torch.tensor([BOS] * _batch_size, dtype=torch.long, device=device) # 最初の入力にはBOSを使用する
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)
        decoder_hidden = encoder_hidden  # Encoderの最終隠れ状態を取得

        # decoderの出力のホルダーを定義
        decoder_outputs = torch.zeros(max_length, _batch_size, self.decoder.output_size, device=device) # max_length分の固定長

        # 各時刻ごとに処理
        for t in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_output
            # 次の時刻のdecoderの入力を決定
            if use_teacher_forcing and batch_Y is not None:  # teacher forceの場合、ターゲット系列を用いる
                decoder_input = batch_Y[t].unsqueeze(0)
            else:  # teacher forceでない場合、自身の出力を用いる
                decoder_input = decoder_output.max(-1)[1]
                
        return decoder_outputs
 ```
 
訓練</br>
```code
# 訓練
best_valid_bleu = 0.

for epoch in range(1, num_epochs+1):
    train_loss = 0.
    train_refs = []
    train_hyps = []
    valid_loss = 0.
    valid_refs = []
    valid_hyps = []
    # train
    for batch in train_dataloader:
        batch_X, batch_Y, lengths_X = batch
        loss, gold, pred = compute_loss(
            batch_X, batch_Y, lengths_X, model, optimizer, 
            is_train=True
            )
        train_loss += loss
        train_refs += gold
        train_hyps += pred
    # valid
    for batch in valid_dataloader:
        batch_X, batch_Y, lengths_X = batch
        loss, gold, pred = compute_loss(
            batch_X, batch_Y, lengths_X, model, 
            is_train=False
            )
        valid_loss += loss
        valid_refs += gold
        valid_hyps += pred
    # 損失をサンプル数で割って正規化
    train_loss = np.sum(train_loss) / len(train_dataloader.data)
    valid_loss = np.sum(valid_loss) / len(valid_dataloader.data)
    # BLEUを計算
    train_bleu = calc_bleu(train_refs, train_hyps)
    valid_bleu = calc_bleu(valid_refs, valid_hyps)

    # validationデータでBLEUが改善した場合にはモデルを保存
    if valid_bleu > best_valid_bleu:
        ckpt = model.state_dict()
        torch.save(ckpt, ckpt_path)
        best_valid_bleu = valid_bleu

    print('Epoch {}: train_loss: {:5.2f}  train_bleu: {:2.2f}  valid_loss: {:5.2f}  valid_bleu: {:2.2f}'.format(
            epoch, train_loss, train_bleu, valid_loss, valid_bleu))
        
    print('-'*80)
```
<img width="475" alt="image" src="https://user-images.githubusercontent.com/57135683/148737204-60ae8f13-5a34-429f-a406-101b6bd5b6bd.png"></br>

評価</br>
```code
# BLEUの計算
test_dataloader = DataLoader(test_X, test_Y, batch_size=1, shuffle=False)
refs_list = []
hyp_list = []

for batch in test_dataloader:
    batch_X, batch_Y, lengths_X = batch
    pred_Y = model(batch_X, lengths_X, max_length=20)
    pred = pred_Y.max(dim=-1)[1].view(-1).data.cpu().tolist()
    refs = batch_Y.view(-1).data.cpu().tolist()
    refs_list.append(refs)
    hyp_list.append(pred)
bleu = calc_bleu(refs_list, hyp_list)
print(bleu)
```
<img width="99" alt="image" src="https://user-images.githubusercontent.com/57135683/148737292-da2a0b56-c3b3-4f0c-b304-f646db16a84e.png"></br>

</br>

# 6.Word2vec
## 6-1.要点まとめ
RNNでは、単語のような可変長の文字列をNNに伝えることができない。</br>
そこで固定長形式で単語を表したものがWord2vec。</br>
学習データからボキャブラリを作成。</br>

大規模データの分散表現の学習が、現実的な計算速度とメモリ量で実現可能にした。</br>

</br>

## 6-2.確認問題
> RNNとword2vec、seq2seq、Attentionの違いを簡潔に述べよ。

　RNNは、時系列データを処理するのに適したネットワーク。</br>
　word2は、vec単語の分散表現ベクトルを得る手法。</br>
　seq2seqは、一つの時系列データから別の時系列データを得るネットワーク。</br>
　Attentionは、時系列データの中身の関連性にそれぞれ重みを付ける。</br>

</br>

## 6-3.関連記事
Word2Vectの活用事例としては、
- レコメンドの分析
- レビュー分析
- 機械翻訳
- 質疑応答システム  
等がある。

# 7.Attention Mechanism
## 7-1.要点まとめ
Seq2seqは２単語でも１００単語でも、固定次元ベクトルの中に入力しなければならなず長い文章への対応が難しい。</br>

文章が長くなるほどそのシーケンスの内部表現の次元も大きくなっていく仕組みがAttentionMechanismである。</br>
これは入力と出力のどの単語が関連しているのかの関連度を学習する仕組みになっている。</br>

## 7-2.確認問題
> seq2seqとHRED、HREDとVHREDの違いを簡潔に述べよ。

　seq2seqは、一つの時系列データから別の時系列データを得るネットワーク。</br>
　HREDは、seq2seqの機構にそれまでの文脈の意味ベクトルを解釈に加えられるようにしたもの。</br>
　VHREDは、HREDが文脈に対して当たり障りのない返答しかできなくなった際の解決策。</br>

</br>

## 7-3.関連記事
