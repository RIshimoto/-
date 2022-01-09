# 1.再帰型ニューラルネットワークの概念
## 1-1.要点まとめ
- RNNとは
　再帰型ニューラルネットワーク（以下、RNN）とは、時系列データに対応可能なニューラルネットワーク。</br>

- 時間列データとは？  
  時系列的順序を追って一定間隔ごとに観察され、しかも相互に統計的な依存関係が認められるようなデータの系列。</br>
  音声データやテキストデータなど。</br>

 - 時間的なつながりを学習させるには？  
  <img width="300" alt="image" src="https://user-images.githubusercontent.com/57135683/148037994-a8c9476a-8f56-4c26-97a8-d9233733f8f5.png"></br>
  中間層の出力を出力層にだすと同時に次の中間層にもういっかい入れることで、時間的のつながりを学習させる。</br></br>
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

- BPTTとは？</br>
RNNにおいてのパラメータ調整法の一種。

BPTTの数学的記述１
　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(in)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W_{(in)}}}\right]^T=\delta&space;^t\left[\boldsymbol{x}^t\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(in)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W_{(in)}}}\right]^T=\delta&space;^t\left[\boldsymbol{x}^t\right]^T\end{align*}" /></br>
```code
np.dot(X.T, delta[:t].reshape(1, -1))
```

</br>

　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(out)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{W_{(out)}}}\right]^T=\delta^{out,&space;t}\left[\boldsymbol{z}^t\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W_{(out)}}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{W_{(out)}}}\right]^T=\delta^{out, t}\left[\boldsymbol{z}^t\right]^T\end{align*}" /></br>
```code
np.dot(z[:,t+1].reshape(-1, 1), delta_out[:,t].reshape(-1, 1))
```

</br>

　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W}}\right]^T=\delta^{t}\left[\boldsymbol{z}^{t-1}\right]^T\end{align*}" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{W}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{W}}\right]^T=\delta^{t}\left[\boldsymbol{z}^{t-1}\right]^T\end{align*}" /></br>
```code
np.dot(z[:,t].reshape(-1, 1), delta[:,t].reshape(1, -1))
```

</br>

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{b}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{b}}\right]^T=\delta^{t}\\\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{c}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{c}}\right]^T=\delta^{out,t}\\\end{align*}&space;" title="\begin{align*}\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{b}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{u}^t}\left[\frac{\partial\boldsymbol{u}^t}{\partial\boldsymbol{b}}\right]^T=\delta^{t}\\\frac{\partial&space;\boldsymbol{E}}{\partial\boldsymbol{c}}&=\frac{\partial\boldsymbol{E}}{\partial\boldsymbol{v}^t}\left[\frac{\partial\boldsymbol{v}^t}{\partial\boldsymbol{c}}\right]^T=\delta^{out,t}\\\end{align*} " /></br>
 
BPTTの数学的記述２

## 1-2.確認テスト
> RNNのネットワークには大きく分けて３つの重みがある。</br>
> 1つは入力から現在の中間層を定義する際にかけられる重み、１つは中間層から出力を定義する際にかけられる重みである。</br>
> 残り１つの重みについて説明せよ。

　中間層からの中間層への重み。</br>

</br>

> 連鎖律の原理を使い、dz/dxを求めよ。</br>
> <img src="https://latex.codecogs.com/svg.image?\begin{align*}z=t^2\\t=x&plus;y\end{align*}" title="\begin{align*}z=t^2\\t=x+y\end{align*}" /></br>

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{dz}{dx}&=\frac{dz}{dt}\cdot\frac{dt}{dx}\\\frac{dz}{dt}&=2t\\\frac{dt}{dx}&=1\\\therefore\frac{dz}{dx}&=2t\cdot1\\&=2\left(x&plus;y\right)\end{align*}" title="\begin{align*}\frac{dz}{dx}&=\frac{dz}{dt}\cdot\frac{dt}{dx}\\\frac{dz}{dt}&=2t\\\frac{dt}{dx}&=1\\\therefore\frac{dz}{dx}&=2t\cdot1\\&=2\left(x+y\right)\end{align*}" /></br>

</br>

> 下図のy1をx・z0・z1・Win・W・Woutを用いて数式で表せ。
> バイアスは任意の文字で定義せよ。
> また中間層の出力にシグモイド関数g(x)を作用させよ。
> <img width="320" alt="image" src="https://user-images.githubusercontent.com/57135683/148175014-84c74f98-5edc-4ad3-8a11-776e2a6a1486.png">

　<img src="https://latex.codecogs.com/svg.image?\begin{align*}s_1=W_{(in)}x_1&plus;Wz_0&plus;b\\y_1=g\left(W_{(out)}s_1&plus;c\right)\end{align*}&space;" title="\begin{align*}s_1=W_{(in)}x_1+Wz_0+b\\y_1=g\left(W_{(out)}s_1+c\right)\end{align*} " /></br>

</br>

## 1-3.実装演習
題材としてバイナリ加算を扱う。</br>
```code
```
BPTTの
```code
    
    for t in range(binary_dim)[::-1]:
        X = np.array([a_bin[-t-1],b_bin[-t-1]]).reshape(1, -1)        

        delta[:,t] = (np.dot(delta[:,t+1].T, W.T) + np.dot(delta_out[:,t].T, W_out.T)) * functions.d_sigmoid(u[:,t+1])

        # 勾配更新
        W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(-1,1))
        W_grad += np.dot(z[:,t].reshape(-1,1), delta[:,t].reshape(1,-1))
        W_in_grad += np.dot(X.T, delta[:,t].reshape(1,-1))

```

# 2.LSTM
## 2-1.要点まとめ
RNNの課題は、時系列をさかのぼれば、勾配が消失していく。</br>
長い時系列の学習が困難。

構造自体を変えて解決したものがLSTM。</br>

<img width="348" alt="image" src="https://user-images.githubusercontent.com/57135683/148670284-e5d89db4-2407-462c-8c42-053edd2a5329.png">

- CEC

  勾配消失および勾配爆発の解決策として、勾配が１であれば解決できる。</br>
  <img src="https://latex.codecogs.com/svg.image?\delta&space;^{t-z-1}=\delta^{t-z}{Wf'(u^{t-z-1})}=1&space;" title="\delta ^{t-z-1}=\delta^{t-z}{Wf'(u^{t-z-1})}=1 " /></br>
<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;E}{\partial&space;c^{t-1}}=\frac{\partial&space;E}{\partial&space;c^t}\cdot\frac{\partial&space;c^t}{\partial&space;c^{t-1}}=\frac{\partial&space;E}{\partial&space;c^t}\cdot\frac{\partial}{\partial&space;c^{t-1}}\cdot\left\{&space;a^t-c^{t-1}&space;\right\}=\frac{\partial&space;E}{\partial&space;c^t}" title="\frac{\partial E}{\partial c^{t-1}}=\frac{\partial E}{\partial c^t}\cdot\frac{\partial c^t}{\partial c^{t-1}}=\frac{\partial E}{\partial c^t}\cdot\frac{\partial}{\partial c^{t-1}}\cdot\left\{ a^t-c^{t-1} \right\}=\frac{\partial E}{\partial c^t}" /></br>

  ただし、CECには記憶機能のみで学習機能がないので、CECの周りに学習機能を配置する。
  
- 入力ゲート/出力ゲート

  それぞれのゲートへの入力値の重みを、重み行列W, Uで可変可能とする。
  
- 忘却ゲート

  CECは過去の情報が保管され続けるため、忘却ゲートによって情報を忘却させる。

- のぞき穴結合

  CECの保存されている過去の情報を、任意のタイミングで他のノードに伝播させたり、</br>
  あるいは任意のタイミングで忘却させたい。</br>
  そこでのぞき穴結合によって、CEC自身の値に重み行列を介して伝播可能にする。</br>

## 2-2.確認問題
> シグモイド関数を微分したとき、入力値が0の時に最大値をとる。その値として正しいもの。

　0.25
 
 </br>
 
 > <img width="373" alt="image" src="https://user-images.githubusercontent.com/57135683/148669768-f48a7788-63cc-472f-a54e-14fb5ec905c9.png">

　1
 
 </br>
 
> 以下の文章をLSTMに入力し、空欄に当てはまる単語を予測したいとする。</br>
> 文中の「とても」という言葉は空欄の予測においてなくなっても影響を及ぼさないと考えられる。</br>
> このような場合、どのゲートが作用すると考えられるか。</br>
> 「映画おもしろかったね。ところで、とてもお腹が空いたから何か___。」

  忘却ゲート

</br>

> <img width="386" alt="image" src="https://user-images.githubusercontent.com/57135683/148670769-cf1c3e21-cc24-4986-908f-a5215ad8688c.png">

　3
 
</br>

# 3.GRU
## 3-1.要点まとめ

  LSTMでは、パラメータ数が多く、計算負荷が高くなる問題があった。</br>
  それを解消したのがGRUである。</br>
  
  GRUはパラメータを大幅に削減し、精度は同等またはそれ以上が望めるようになった構造。</br>
  計算負荷が低いのがメリット。</br>
  
  <img width="338" alt="image" src="https://user-images.githubusercontent.com/57135683/148670872-462ee843-a91a-4b51-99a1-362187e8ae6f.png">

## 3-2.確認問題
> LSTMとCECが抱える課題についてそれぞれ簡潔に述べよ。

  LSTMは、パラメータ数が多くなり計算量が多くなること。
  CECは、学習機能がなこと。

</br>

> <img width="389" alt="image" src="https://user-images.githubusercontent.com/57135683/148670891-85d49868-5784-4a85-ad23-4affa68c7f06.png">

　4.
 
</br>

> LSTMとGRUの違いを簡潔に述べよ。

  LSTMよりGRUのほうが計算量が少ない。</br>

</br>

## 3-3.実装演習
```code
```

# 4.双方向RNN
## 4-1.要点まとめ
過去の情報だけでなく、未来の情報を加味することで、精度を向上させるためのモデル。</br>
文章の推敲や機械学習などで用いられる。</br>
<img width="206" alt="image" src="https://user-images.githubusercontent.com/57135683/148670934-1fcaa812-1701-43fc-a09e-8ecc5f19adbc.png"></br>

</br>

## 4-2.確認問題
<img width="359" alt="image" src="https://user-images.githubusercontent.com/57135683/148670920-9bfc875e-9272-48cd-be01-b02f1403d123.png">

　4
 
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
    
  3.Sampling
    生成確率にもとづいてtokenをランダムに選ぶ。

  4.Embedding
    2で選ばれたtokenをEmbeddingしてDecoder RNNへの次の入力とします。
    
  5.Detokenize
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

　2
 
</br>

> <img width="430" alt="image" src="https://user-images.githubusercontent.com/57135683/148671882-5f5406c0-cf04-4f4e-bc33-090a2119f40a.png">

　1.
 
</br>

> seq2seqとHRED、HREDとVHREDの違いを簡潔に述べよ。

seq2seqは、一つの時系列データから別の時系列データを得るネットワーク。</br>
HREDは、seq2seqの機構にそれまでの文脈の意味ベクトルを解釈に加えられるようにしたもの。</br>

VHREDは、HREDが文脈に対して当たり障りのない返答しかできなくなった際の解決策。</br>

</br>

> VAEに関する下記の説明文中の空欄に当てはまる言葉を答えよ。</br>
> 自己符号器の潜在変数に___を導入したもの。

　確率分布
 

</br>

# 6.Word2vec
RNNでは、単語のような可変長の文字列をNNに伝えることができない。</br>
そこで固定長形式で単語を表したものがWord2vec。</br>
学習データからボキャブラリを作成。</br>

大規模データの分散表現の学習が、現実的な計算速度とメモリ量で実現可能にした。</br>

</br>

# 7.Attention Mechanism
## 7-1.要点まとめ
Seq2seqは２単語でも１００単語でも、固定次元ベクトルの中に入力しなければならなず長い文章への対応が難しい。</br>

文章が長くなるほどそのシーケンスの内部表現の次元も大きくなっていく仕組みがAttentionMechanismである。</br>
これは入力と出力のどの単語が関連しているのかの関連度を学習する仕組みになっている。</br>

## 7-2.確認問題
> RNNとword2vec、seq2seq、Attentionの違いを簡潔に述べよ。

　RNNは、時系列データを処理するのに適したネットワーク。</br>
　word2は、vec単語の分散表現ベクトルを得る手法。</br>
　seq2seqは、一つの時系列データから別の時系列データを得るネットワーク。</br>
　Attentionは、時系列データの中身の関連性にそれぞれ重みを付ける。</br>
