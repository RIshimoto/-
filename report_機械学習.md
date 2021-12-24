# 1.線形回帰モデル
## 1-1.要点まとめ
　回帰問題（ある入力から出力を予測する問題）を**直線で**予測する、教師あり機械学習モデルの一つ。 </br>


　回帰で扱うデータは、  
　入力が、m次元のベクトル、</br>
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{x}=\left(x_1,x_2,...,x_m\right)^T\in&space;\boldsymbol{R}^m" title="\boldsymbol{x}=\left(x_1,x_2,...,x_m\right)^T\in \boldsymbol{R}^m" /></br>
　出力が、スカラー値である。</br>
　　<img src="https://latex.codecogs.com/svg.image?y\in&space;\boldsymbol{R}^1" title="y\in \boldsymbol{R}^1" /></br>
　この入力xが1次元の場合を**単回帰**、2次元以上を**重回帰**という。</br>
  
  
　以下のように入力xに対し、</br>
　m次元パラメータwと線形結合（入力とパラメータの内積）したものを予測値として出力する。</br>
　　<img src="https://latex.codecogs.com/svg.image?\widehat{y}=&space;\boldsymbol{w^{T}}\boldsymbol{x}&plus;x_{0}=\sum_{j}^{m}w_{j}x_{j}&plus;x_{0}" title="\widehat{y}= \boldsymbol{w^{T}}\boldsymbol{x}&plus;x_{0}=\sum_{j}^{m}w_{j}x_{j}&plus;x_{0}" /></br>
　この予測値と教師データから、</br>
　このモデルのパラメータを学習データを用いて、**最小二乗法**により推定していく。</br>


　ただし、データには回帰直線に誤差が加わり観測されていると仮定し、</br>
　誤差項εを含めた式になる。</br> 
　　<img src="https://latex.codecogs.com/svg.image?\widehat{y}&space;=&space;\boldsymbol{w^{T}}\boldsymbol{x}&space;&plus;&space;x_{0}&plus;&space;\varepsilon"> 

## 1-2.実装演習
　ボストンデータハウジングのデータを用いて住宅の価格を予測を行う。</br>

　まずは線形単回帰分析について。  
　部屋の数のみから価格を予測する。</br>

　まず、平均室数を説明変数とし、
```code
  # 説明変数
  data = df.loc[:, ['RM']].values
```
　住宅の価格がいくらになるかを目的変数とする。
  ```code
  # 目的変数
  target = df.loc[:, 'PRICE'].values
  ```
　それらのデータに対し、パラメータを調整した線形回帰モデルを構築する。

  ```code
  # オブジェクト生成
  model = LinearRegression()
  # fit関数でパラメータ推定
  model.fit(data, target)
  ```

  このモデルに未知データを与え予測を行う。</br>
  ```code
  #予測
  model.predict([[10]])
  ```

  実行結果は、部屋数が10個の場合、値段は56.35と出力される。</br>
  <img width="151" alt="image" src="https://user-images.githubusercontent.com/57135683/147052855-ccc234a2-cc89-4c66-bac4-b3526b46cf78.png">

次に重回帰分析について。</br>
犯罪率と部屋の数から価格を予測する。</br>

まず、犯罪率と平均室数を説明変数とし、
住宅の価格がいくらになるかを目的変数とする。
```code
# 説明変数
data2 = df.loc[:, ['CRIM', 'RM']].values
# 目的変数
target2 = df.loc[:, 'PRICE'].values
```

それらのデータに対し、パラメータを調整した線形回帰モデルを構築する。

```code
# オブジェクト生成
model2 = LinearRegression()
# fit関数でパラメータ推定
model2.fit(data2, target2)
```

このモデルに未知データを与え予測を行う。</br>
```code
#予測
model2.predict([[0.2, 7]])
```
実行結果は、犯罪率が0.2、部屋数が7個の場合、値段は29.43と出力される。</br>
<img width="150" alt="image" src="https://user-images.githubusercontent.com/57135683/147057241-155848c9-95c9-4dea-a1e8-81536272354b.png">


# 2.非線形回帰モデル
## 2-1.要点まとめ
　回帰問題（ある入力から出力を予測する問題）を**非直線で**予測する機械学習モデルのこと。</br>
</br>
　回帰関数として、基底関数（多項式やガウス型基底など）と呼ばれる既知の非線形関数とパラメータベクトルの線形結合を使用する。</br>
　多項式：</br>
　　<img src="https://latex.codecogs.com/svg.image?\phi_j&space;=&space;x^{j}"></br>
　ガウス型基底：</br>
　　<img src="https://latex.codecogs.com/svg.image?\phi_j(x)&space;=&space;exp\{\frac{\left&space;(&space;x&space;-&space;\mu&space;_{j}\right&space;)^{2}}{2h_{j}}\}"></br>
  </br>
　未知パラメータは、線形回帰モデルと同様に最小二乗法や最尤法により推定。</br>
　線形回帰モデルと変わりはなく、入力xを基底関数によって非線形化して線形結合する。</br>
　しかし、多くの基底関数を用意してしまうと過学習が起こる。</br>
 </br>
　過学習の対策としては、**学習データの数を増やす**、**不要な基底関数の削除**、**正規化法**があげられる。</br>
　また、適切なモデルは**交差検証法**で決定する。</br>
　そして、汎化性能を評価するため、ホールドアウト法、クロスバリデーションなどがある。</br>

## 2-2.実装演習
```code
```

# 3.主成分分析
## 3-1.要点まとめ
　教師なし学習の一種であり、次元圧縮の手法。</br>
</br>
　多変量データの持つ構造を情報損失をなるべく少なくして、より少数個の指標に圧縮する。</br>
　これにより、分析や可視化が可能になる。</br>
</br>
　方法としては、</br>
　圧縮した際の情報の量を分散ととらえ、線形変換後の分散が最大となるように射影軸を探索する。</br>

　学習データを</br>
　　<img src="https://latex.codecogs.com/svg.image?X_i&space;=&space;\left(&space;x_{i1},&space;x_{i2},&space;...,&space;x_{im}&space;\right)"></br>
　と表現すると、データ行列は</br>
　　<img src="https://latex.codecogs.com/svg.image?\bar{X}&space;=&space;\left(&space;x_{i1}-\bar{x},&space;x_{i2}-\bar{x}&space;,&space;...,&space;x_{im}-\bar{x}&space;\right)^T">  </br>
　となり、このn個のデータをaという主成分で変換してあげたものを考える。</br>
</br>
　変換後のベクトルは</br>
　　<img src="https://latex.codecogs.com/svg.image?s_j&space;=&space;\bar{X}a_j">  </br>
　となり、この分散が最大となるものを求める。</br>
　　<img src="https://latex.codecogs.com/svg.image?Var(s_j)&space;=&space;\frac{1}{n}s_{j}^Ts_j&space;=&space;\frac{1}{n}\left(&space;\bar{X}a_j&space;\right&space;)\left(&space;\bar{X}a_j&space;\right&space;)^T&space;=&space;a_j^TVar(\bar{X})a_j"></br>

　制約条件</br>
　　<img src="https://latex.codecogs.com/svg.image?a_j^Ta_j&space;=&space;1"></br>
　を考慮すると、ラグランジュ関数は</br>
　　<img src="https://latex.codecogs.com/svg.image?E(a_j)&space;=&space;a_j^TVar(\bar{X})a_j&space;-&space;\lambda&space;\left(&space;a_j^Ta_j&space;-&space;1\right&space;)"></br>
　となり、これを微分して最適解を求めると  </br>
　　<img src="https://latex.codecogs.com/svg.image?Var(\bar{X})a_j&space;=&space;\lambda&space;a_j"></br>
　となる。</br>
　これより、データ行列の共分散行列の固有ベクトルが分散を最大にする軸になるということになる。</br>
 </br>
　射影後の分散は、</br>
　　<img src="https://latex.codecogs.com/svg.image?Var(s_j)&space;=&space;a_1^TVar(\bar{X})a_1&space;=&space;\lambda_1&space;a_1^Ta_1&space;=&space;\lambda_1"> </br>
　より、固有値がそのまま射影先の分散になる。</br>

　分散共分散行列は正定値対称行列なので固有値は必ず0以上・固有値ベクトルは直交なので、</br>
　固有値の大きい順に固有ベクトルを射影先の軸にすればよい。</br>

　また、圧縮後の情報ロスは寄与率を求めることでわかる。</br>
　第1～元次元文の主成分の分散は元の分散に一致し、</br>
　第k主成分の分散は主成分に対応する固有値に一致しているため、</br>
　全分散の分散に対する第k主成分の分散の割合が寄与率となる。 </br>
　　<img src="https://latex.codecogs.com/svg.image?c_k&space;=&space;\frac{\lambda_k}{\sum_{i=1}^{m}&space;\lambda_i}">  </br>

　第1-k主成分まで圧縮した際の情報損失の割合を示す累積寄与率といい、</br>
　主成分の総分散に対する第1~k主成分の分散の割合となる。</br>
　　<img src="https://latex.codecogs.com/svg.image?c_k&space;=&space;\frac{\sum_{j=1}^{k}&space;\lambda_j}{\sum_{i=1}^{m}&space;\lambda_i}"></br>

## 3-2.実装演習
　乳がんデータの分析を行う。</br>
　ロジスティック回析を行うことで、30次元のデータを97％で分類できた。

　pcaで30次元を2次元や3次元に落とすことでどこまで情報が落ちてしまうか、定量的に測らなければいけない。</br>
　それを測る指標として、寄与率を調べる。</br>
```code
pca = PCA(n_components=30)
pca.fit(X_train_scaled)
plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
```
　<img width="307" alt="image" src="https://user-images.githubusercontent.com/57135683/146600082-360f7cce-ea7f-4d31-8dce-6da1ed242a88.png">
　実行結果によると、第2主成分まで情報を圧縮しても65％ほどは情報量を維持できることがわかる。

実際に２次元に落としてみると、</br>
```code
# PCA
# 次元数2まで圧縮
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
print('X_train_pca shape: {}'.format(X_train_pca.shape))
# X_train_pca shape: (426, 2)

# 寄与率
print('explained variance ratio: {}'.format(pca.explained_variance_ratio_))
# explained variance ratio: [ 0.43315126  0.19586506]

# 散布図にプロット
temp = pd.DataFrame(X_train_pca)
temp['Outcome'] = y_train.values
b = temp[temp['Outcome'] == 0]
m = temp[temp['Outcome'] == 1]
plt.scatter(x=b[0], y=b[1], marker='o') # 良性は○でマーク
plt.scatter(x=m[0], y=m[1], marker='^') # 悪性は△でマーク
plt.xlabel('PC 1') # 第1主成分をx軸
plt.ylabel('PC 2') # 第2主成分をy軸
```
<img width="335" alt="image" src="https://user-images.githubusercontent.com/57135683/146600580-94f21c69-131d-4f40-a3fe-61cb3cfc61c1.png"></br>
実行結果より、次元をおとしたことにより、元の97％の境界から良性と悪性の判定の堺があいまいになっていることがわかる。</br>
しかし、もともと30次元のデータが可視化できるようになったことがメリットである。</br>
