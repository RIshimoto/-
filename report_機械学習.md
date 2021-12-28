# 1.線形回帰モデル
<details><summary>クリックすると展開されます</summary>
   
## 1-1.要点まとめ
　回帰問題（ある入力から出力を予測する問題）を**直線で**予測する、教師あり機械学習モデルの一つ。 </br>


　回帰で扱うデータは、  
　入力が、m次元のベクトル、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{x}=\left(x_1,x_2,...,x_m\right)^T\in&space;\boldsymbol{R}^m" title="\boldsymbol{x}=\left(x_1,x_2,...,x_m\right)^T\in \boldsymbol{R}^m" /></br>
  
　出力が、スカラー値である。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?y\in&space;\boldsymbol{R}^1" title="y\in \boldsymbol{R}^1" /></br>
  
　この入力xが1次元の場合を**単回帰**、2次元以上を**重回帰**という。</br>
  
  </br>
  
　以下のように入力xに対し、</br>
　m次元パラメータwと線形結合（入力とパラメータの内積）したものを予測値として出力する。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?\widehat{y}=&space;\boldsymbol{w^{T}}\boldsymbol{x}&plus;x_{0}=\sum_{j}^{m}w_{j}x_{j}&plus;x_{0}" title="\widehat{y}= \boldsymbol{w^{T}}\boldsymbol{x}&plus;x_{0}=\sum_{j}^{m}w_{j}x_{j}&plus;x_{0}" /></br>
  
　ただし、データには回帰直線に誤差が加わり観測されていると仮定し、</br>
　誤差項εを含めた式になる。</br> 
 
　　<img src="https://latex.codecogs.com/svg.image?\widehat{y}&space;=&space;\boldsymbol{w^{T}}\boldsymbol{x}&space;&plus;&space;x_{0}&plus;&space;\varepsilon"></br>
  
  </br>
  
　この予測値と教師データから、</br>
　このモデルのパラメータを学習データを用いて、**最小二乗法**により推定していく。</br>
 
　最小二乗法とは学習データの平均二乗誤差を最小とするパラメータを探索する方法。</br>
　平均二乗法の勾配が0になる点を求めればよい。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?MSE_{train}=\frac{1}{n_train}\sum_{i=1}{n_train}\left(\hat{y_i}^{(train)}y_i^{(train)}\right)^2" title="MSE_{train}=\frac{1}{n_train}\sum_{i=1}{n_train}\left(\hat{y_i}^{(train)}y_i^{(train)}\right)^2" /></br>
 
 　　<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;}{\partial&space;\boldsymbol{w}}MSE_{train}=0" title="\frac{\partial }{\partial \boldsymbol{w}}MSE_{train}=0" /></br>

　　<img src="https://latex.codecogs.com/svg.image?\therefore&space;\boldsymbol{\hat{w}}=\left(\boldsymbol{X^{(train)}}^T\boldsymbol{X^{(train)}}\right)^{-1}\boldsymbol{X^{(train)}}^T\boldsymbol{y^{(train)}}" title="\therefore \boldsymbol{\hat{w}}=\left(\boldsymbol{X^{(train)}}^T\boldsymbol{X^{(train)}}\right)^{-1}\boldsymbol{X^{(train)}}^T\boldsymbol{y^{(train)}}" /></br>


</br>

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

</br>

</details>

# 2.非線形回帰モデル
<details><summary>クリックすると展開されます</summary>
  
## 2-1.要点まとめ
　回帰問題（ある入力から出力を予測する問題）を**非直線で**予測する機械学習モデルのこと。</br>
</br>
　回帰関数として、基底関数と呼ばれる既知の非線形関数とパラメータベクトルの線形結合を使用。</br>
　未知パラメータは線形回帰モデルと同様に最小二乗法や最尤法により推定する。</br>
 
　よく使われる基底関数</br>
 
  - 多項式：</br>
　　<img src="https://latex.codecogs.com/svg.image?\phi_j&space;=&space;x^{j}"></br>
  
  - ガウス型基底：</br>
　　<img src="https://latex.codecogs.com/svg.image?\phi_j(x)&space;=&space;exp\{\frac{\left&space;(&space;x&space;-&space;\mu&space;_{j}\right&space;)^{2}}{2h_{j}}\}"></br>
    
  </br>
  
　線形回帰モデルと変わりはなく、入力xを基底関数によって非線形化して線形結合する。</br>
　しかし、多くの基底関数を用意してしまうと過学習が起こる。</br>
 </br>
　過学習の対策としては、</br>
 
   - **学習データの数を増やす**
   - **不要な基底関数の削除**
   - **正規化法**

　があげられる。</br>
 
　また、適切なモデルは**交差検証法**で決定する。</br>
　そして、汎化性能を評価するため、ホールドアウト法、クロスバリデーションなどがある。</br>

</br>

## 2-2.実装演習
　非線形モデルでより表現力をあげることができる。</br>
```code
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
data = data.reshape(-1,1)
target = target.reshape(-1,1)
clf.fit(data, target)

p_lin = clf.predict(data)

plt.scatter(data, target, label='data')
plt.plot(data, p_lin, color='darkorange', marker='', linestyle='-', linewidth=1, markersize=6, label='linear regression')
plt.legend()
print(clf.score(data, target))
```
<img width="276" alt="image" src="https://user-images.githubusercontent.com/57135683/147532842-1f207017-c44c-40ee-a6e3-99e070bed873.png"></br>


```code
from sklearn.kernel_ridge import KernelRidge

clf = KernelRidge(alpha=0.0002, kernel='rbf')
clf.fit(data, target)

p_kridge = clf.predict(data)

plt.scatter(data, target, color='blue', label='data')

plt.plot(data, p_kridge, color='orange', linestyle='-', linewidth=3, markersize=6, label='kernel ridge')
plt.legend()
#plt.plot(data, p, color='orange', marker='o', linestyle='-', linewidth=1, markersize=6)
```
<img width="284" alt="image" src="https://user-images.githubusercontent.com/57135683/147532815-08b5f6df-77f6-4d08-b48b-031e5a350fa9.png"></br>

</br>

</details>

# 3.ロジスティック回帰
<details><summary>クリックすると展開されます</summary>
   
## 3-1.要点まとめ
　分類問題（ある入力からクラスに分類する）を解くための教師あり機械学習モデル。</br>
　m次元パラメータの線形結合をシグモイド関数に入力し、ラベルy=１になる確率を出力する。</br>
 
　シグモイド関数:</br>
　　入力は実数、出力は必ず0から1の値。クラス１に分類される確率を表現</br>

　　<img src="https://latex.codecogs.com/svg.image?\sigma&space;\left(x\right)=\frac{1}{1&plus;exp\left(&space;-ax\right)}" title="\sigma \left(x\right)=\frac{1}{1+exp\left( -ax\right)}" /></br>

</br>

　シグモイド関数の微分はシグモイド関数で表現できるため、尤度関数の微分の際に便利。</br>

　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\frac{\partial&space;\sigma\left(x\right)}{\partial&space;x}&=\frac{\partial&space;}{\partial&space;x}\left(\frac{1}{1&plus;exp\left(-ax\right)}\right)\\&=\left(-1\right)\cdot{1&plus;exp(-ax)}^{-2}\cdot&space;exp\left(-ax\right)\cdot\left(-a\right)\\&=\frac{a\cdot&space;exp\left(-ax\right)}{1&plus;exp(-ax)}^2\\&=\frac{a}{1&plus;exp(-ax)}\cdot\frac{1&plus;exp(-ax)-1}{1&plus;exp(-ax)}\\&=a\sigma(x)\cdot(1-\sigma(x))\end{align*}&space;" title="\begin{align*}\frac{\partial&space;\sigma\left(x\right)}{\partial&space;x}&=\frac{\partial&space;}{\partial&space;x}\left(\frac{1}{1&plus;exp\left(-ax\right)}\right)\\&=\left(-1\right)\cdot{1&plus;exp(-ax)}^{-2}\cdot exp\left(-ax\right)\cdot\left(-a\right)\\&=\frac{a\cdot exp\left(-ax\right)}{1&plus;exp(-ax)}^2\\&=\frac{a}{1&plus;exp(-ax)}\cdot\frac{1&plus;exp(-ax)-1}{1&plus;exp(-ax)}\\&=a\sigma(x)\cdot(1-\sigma(x))\end{align*}&space;" /></br>

</br>

　シグモイド関数の出力をY=1になる確率に対応させると、</br>

　　<img src="https://latex.codecogs.com/svg.image?P\left(Y=y|\boldsymbol{x}\right)=\sigma\left(w_0&space;&plus;&space;w_1x_1&space;&plus;&space;...&space;&plus;&space;w_mx_m\right)" title="P\left(Y=y|\boldsymbol{x}\right)=\sigma\left(w_0 + w_1x_1 + ... + w_mx_m\right)" /></br>
  
　となる。</br>

</br>

　特徴量Xとラベルベクトルyが与えられたときには、その行列Xからyが生じる確率を考えると、</br>

　　<img src="https://latex.codecogs.com/svg.image?P\left(\boldsymbol{y}|\boldsymbol{X}\right)=\prod_{k=1}^{n}\left[&space;\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)^{y_k}\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)^{1-y_k}\right]" title="P\left(\boldsymbol{y}|\boldsymbol{X}\right)=\prod_{k=1}^{n}\left[ \sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)^{y_k}\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)^{1-y_k}\right]" /></br>

　この確率を最大化することを考える。対数をとって、</br>

　　<img src="https://latex.codecogs.com/svg.image?E\left(\boldsymbol{w}\right)=-\log&space;P\left(\boldsymbol{y}|\boldsymbol{X}\right)=-\sum_{k=1}^{n}\left[&space;y_k\log\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)&plus;\left(1-y_k\right)\log\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)\right]" title="E\left(\boldsymbol{w}\right)=-\log P\left(\boldsymbol{y}|\boldsymbol{X}\right)=-\sum_{k=1}^{n}\left[ y_k\log\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)+\left(1-y_k\right)\log\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)\right]" /></br>

　この最適値を求める。</br>

　　<img src="https://latex.codecogs.com/svg.image?\begin{align*}\nabla&space;E\left(\boldsymbol{w}\right)&=-\sum_{k=1}^{n}\left[&space;y_k\nabla\log\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)&plus;\left(1-y_k\right)\nabla\log\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)\right]\\&=\sum_{k=1}^{n}\left(\sigma\left(\boldsymbol{w}^T\widetilde{\boldsymbol{x_k}}\right)-y_k\right)\widetilde{x_k}^T\end{align*}" title="\begin{align*}\nabla E\left(\boldsymbol{w}\right)&=-\sum_{k=1}^{n}\left[ y_k\nabla\log\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)+\left(1-y_k\right)\nabla\log\left(1-\sigma\left(\boldsymbol{w^T}\boldsymbol{\widetilde{x_k}}\right)\right)\right]\\&=\sum_{k=1}^{n}\left(\sigma\left(\boldsymbol{w}^T\widetilde{\boldsymbol{x_k}}\right)-y_k\right)\widetilde{x_k}^T\end{align*}" /></br>

</br>

　この式を計算機を用いて逐次的に解く。これを勾配降下法という。</br>

　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{k&plus;1}=\boldsymbol{w}^{k}&plus;\eta\nabla&space;E\left(\boldsymbol{x}\right)" title="\boldsymbol{w}^{k&plus;1}=\boldsymbol{w}^{k}&plus;\eta\nabla E\left(\boldsymbol{x}\right)" /></br>

</br>

　しかし、nが巨大となった時にデータをオン目盛りに載せる容量が足りない、計算量が膨大になるなどの問題から、</br>
　確率的勾配法降下法(SGD)を利用する。</br>

　　<img src="https://latex.codecogs.com/svg.image?\boldsymbol{w}^{k&plus;1}=\boldsymbol{w}^{k}&plus;\eta\nabla&space;E_i\left(\boldsymbol{x}\right)" title="\boldsymbol{w}^{k&plus;1}=\boldsymbol{w}^{k}&plus;\eta\nabla E_i\left(\boldsymbol{x}\right)" /></br>

　確率勾配降下法はデータをひとつずつランダムに（「確率的に」）選んでパラメータを更新する。</br>
　勾配降下法でパラメータを１回更新するのと同じ計算量でパラメータをn回更新できるので効率よく最適な解を探索可能。</br>

</br>

　このモデルの性能を測る指標には、混合行列で評価する。</br>
　　<img width="362" alt="image" src="https://user-images.githubusercontent.com/57135683/147525123-1cec45d9-daa3-450b-8440-680c920ec20f.png"></br>

  - **再現率**(Recall)

    「本当にPositiveなもの」の中からPositiveと予測できる割合。</br>
    「誤りが多くても多少多くても抜け漏れは少ない」予測したい際に利用。例、病院の検診など</br>
    
    　　<img src="https://latex.codecogs.com/svg.image?\frac{TP}{TP&plus;FN}" title="\frac{TP}{TP+FN}" /></br>
      
  　　<img width="184" alt="image" src="https://user-images.githubusercontent.com/57135683/147525380-9b3e03f8-a967-45df-9cb0-67fe540f9e3f.png"></br>

</br>

  - **適合率**(Precision)

   　モデルが「Positiveと予測」したものの中で本当にPositiveである割合。</br>
   　見逃しが多くてもより正解な予測をしたい際に利用。例、スパムメールなど</br>
    
   　　<img src="https://latex.codecogs.com/svg.image?\frac{TP}{TP&plus;FP}" title="\frac{TP}{TP+FP}" /></br>
    
   　<img width="200" alt="image" src="https://user-images.githubusercontent.com/57135683/147525354-40d06a73-7cbd-44ed-a7cd-4c22d34a13f1.png"></br>

</br>

## 3-2.実装演習　
　タイタニックの乗客データを利用し、ロジスティック回帰モデルを作成し、特定の乗客がどれくらい生き残れるかを予測する。</br>

</br>

　チケット価格から情報から生存情報を判別する。</br>
 ```code
   #運賃だけのリストを作成
   data1 = titanic_df.loc[:, ["Fare"]].values

   #生死フラグのみのリストを作成
   label1 =  titanic_df.loc[:,["Survived"]].values

   from sklearn.linear_model import LogisticRegression
   model=LogisticRegression()
   model.fit(data1, label1)
 ```
 
 　実際に推定を行うと$61のチケットを買った人はなくなったことがわかる。</br>
```code
   model.predict([[61]])
```
　<img width="77" alt="image" src="https://user-images.githubusercontent.com/57135683/147536598-6635a24f-e1b7-409d-9063-9d8bf193a8cd.png"></br>

　また死亡する確率は、</br>
```code
   model.predict_proba([[62]])
```
　<img width="214" alt="image" src="https://user-images.githubusercontent.com/57135683/147536693-eceee2a6-4764-4643-88ef-db1df48fe800.png"></br>
　とわかる。</br>
 
 </br>
 
　次に今あるデータから新しい特徴量を作って、それからロジスティック回帰を行ってみる。</br>
　高い階級の女性は死亡率が低いと仮定し、</br>
　「Pclass」と「Gender」のデータから新しい特徴量「Pclass_Gender」を作る。
```code
   titanic_df['Pclass_Gender'] = titanic_df['Pclass'] + titanic_df['Gender']
```
　<img width="598" alt="image" src="https://user-images.githubusercontent.com/57135683/147537138-f50bb5a4-2817-4339-95d2-03ea388c015c.png"></br>
 
　Pclass_GenderとAgeの関係から生存率を見てみると、</br>
　年齢が低い、階級が高い、女性の特徴があると生き残りやすいということがわかる。</br>
　<img width="391" alt="image" src="https://user-images.githubusercontent.com/57135683/147537335-18fe07f1-3bdb-4f8d-844b-3b19887e27fe.png"></br>

</br>

</details>

# 4.主成分分析
<details><summary>クリックすると展開されます</summary>

## 4-1.要点まとめ
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
 
　射影後の分散は、</br>
 
　　<img src="https://latex.codecogs.com/svg.image?Var(s_j)&space;=&space;a_1^TVar(\bar{X})a_1&space;=&space;\lambda_1&space;a_1^Ta_1&space;=&space;\lambda_1"> </br>
  
　より、固有値がそのまま射影先の分散になる。</br>

</br>

　分散共分散行列は正定値対称行列なので固有値は必ず0以上・固有値ベクトルは直交なので、</br>
　固有値の大きい順に固有ベクトルを射影先の軸にすればよい。</br>

　また、圧縮後の情報ロスは寄与率を求めることでわかる。</br>
 
　第1～元次元文の主成分の分散は元の分散に一致し、</br>
　第k主成分の分散は主成分に対応する固有値に一致しているため、</br>
　全分散の分散に対する第k主成分の分散の割合が寄与率となる。 </br>
 
　　<img src="https://latex.codecogs.com/svg.image?c_k&space;=&space;\frac{\lambda_k}{\sum_{i=1}^{m}&space;\lambda_i}">  </br>

</br>

　第1-k主成分まで圧縮した際の情報損失の割合を示す累積寄与率といい、</br>
　主成分の総分散に対する第1~k主成分の分散の割合となる。</br>
 
　　<img src="https://latex.codecogs.com/svg.image?c_k&space;=&space;\frac{\sum_{j=1}^{k}&space;\lambda_j}{\sum_{i=1}^{m}&space;\lambda_i}"></br>

</br>

## 4-2.実装演習
　乳がんデータの分析を行う。</br>
　ロジスティック回析を行うことで、30次元のデータを97％で分類できた。

　pcaで30次元を2次元や3次元に落とすことでどこまで情報が落ちてしまうか、定量的に測らなければいけない。</br>
　それを測る指標として、寄与率を調べる。</br>
```code
pca = PCA(n_components=30)
pca.fit(X_train_scaled)
plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
```
　<img width="307" alt="image" src="https://user-images.githubusercontent.com/57135683/146600082-360f7cce-ea7f-4d31-8dce-6da1ed242a88.png"></br>
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

</br>

</details>

# 5.サポートベクターマシーン
## 5-1.要点まとめ
　サポートベクターマシーンは２クラス分類問題の代表的な手法なひとつ。</br>
 
</br>

　マージンを最大化する直線（超平面）により分類する。</br>
　マージンの最大化とは、それぞれのクラスに属する点群のうち、</br>
　直線に一番近い点と直線の距離を考え、その距離を最大化するもの。</br>
 
</br>

　特徴ベクトルｘがどちらのクラスに属するか決定する関数を**決定関数**、</br>
　クラスを分類する境界を**分類境界**という。</br>
　分類境界に最も近いデータxのことを**サポートベクトル**という。</br>


</br>

　SVMには、完璧に分類できる決定関数が存在すると仮定する**ハードマージン**と多少の誤分類を許容する**ソフトマージン**がある。</br>

</br>

　線形分離ではうまく分類できないケースには**カーネルトリック**と呼ばれる手法で</br>
　データを**カーネル関数**によってデータを高次元へ拡張することで、非線形分離を行うことが可能になる。</br>

　代表的なカーネル関数は次の通り。</br>

  - 多項式カーネル

  　　<img src="https://latex.codecogs.com/svg.image?K\left(x_i,&space;x_j\right)=[\boldsymbol{x_i}^T\boldsymbol{x_j}&plus;c]^d" title="K\left(x_i, x_j\right)= [\boldsymbol{x_i}^T\boldsymbol{x_j}+c]^d" />

  - ガウスカーネル

  　　<img src="https://latex.codecogs.com/svg.image?K\left(x_i,&space;x_j\right)=exp\left(-\gamma&space;\left\|\boldsymbol{x_i}-\boldsymbol{x_j}\right\|^2\right)" title="K\left(x_i, x_j\right)=exp\left(-\gamma \left\|\boldsymbol{x_i}-\boldsymbol{x_j}\right\|^2\right)" />
  
  - シグモイドカーネル
  　　<img src="https://latex.codecogs.com/svg.image?K\left(x_i,&space;x_j\right)=\tanh\left(b\boldsymbol{x_i}^T\boldsymbol{x_j}&plus;c\right)" title="K\left(x_i, x_j\right)=\tanh\left(b\boldsymbol{x_i}^T\boldsymbol{x_j}+c\right)" />

</br>

## 5-2.実装演習
```code
```
