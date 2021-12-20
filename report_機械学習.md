# 主成分分析
## 1.要点まとめ
教師なし学習の一種であり、次元圧縮の手法。</br>
多変量データの持つ構造を情報損失をなるべく少なくして、より少数個の指標に圧縮する。</br>
これにより、分析や可視化が可能になる。</br>
</br>
方法としては、</br>
圧縮した際の情報の量を分散ととらえ、線形変換後の分散が最大となるように射影軸を探索する。</br>

学習データを</br>
　<img src="https://latex.codecogs.com/gif.latex?X_i&space;=&space;\left(&space;x_{i1},&space;x_{i2},&space;...,&space;x_{im}&space;\right)"></br>
と表現すると、</br>
データ行列は</br>
　<img src="https://latex.codecogs.com/gif.latex?\bar{X}&space;=&space;\left(&space;x_{i1}-\bar{x},&space;x_{i2}-\bar{x}&space;,&space;...,&space;x_{im}-\bar{x}&space;\right)^T">  </br>
となり、このn個のデータをaという主成分で変換してあげたものを考える。</br>
変換後のベクトルは</br>
　<img src="https://latex.codecogs.com/gif.latex?s_j&space;=&space;\bar{X}a_j">  </br>
となり、この分散が最大となるものを求める。</br>
<img src="https://latex.codecogs.com/gif.latex?Var(s_j)&space;=&space;\frac{1}{n}s_{j}^Ts_j&space;=&space;\frac{1}{n}\left(&space;\bar{X}a_j&space;\right&space;)\left(&space;\bar{X}a_j&space;\right&space;)^T&space;=&space;a_j^TVar(\bar{X})a_j"></br>

制約条件</br>
　<img src="https://latex.codecogs.com/gif.latex?a_j^Ta_j&space;=&space;1"></br>
を考慮すると、ラグランジュ関数は</br>
　<img src="https://latex.codecogs.com/gif.latex?E(a_j)&space;=&space;a_j^TVar(\bar{X})a_j&space;-&space;\lambda&space;\left(&space;a_j^Ta_j&space;-&space;1\right&space;)"></br>
となり、これを微分して最適解を求めると  </br>
　<img src="https://latex.codecogs.com/gif.latex?Var(\bar{X})a_j&space;=&space;\lambda&space;a_j"></br>
となる。</br>
これより、データ行列の共分散行列の固有ベクトルが分散を最大にする軸になるということになる。</br>
射影後の分散は、</br>
　<img src="https://latex.codecogs.com/gif.latex?Var(s_j)&space;=&space;a_1^TVar(\bar{X})a_1&space;=&space;\lambda_1&space;a_1^Ta_1&space;=&space;\lambda_1"> </br>
より、固有値がそのまま射影先の分散になる。</br>

分散共分散行列は正定値対称行列なので固有値は必ず0以上・固有値ベクトルは直交なので、</br>
固有値の大きい順に固有ベクトルを射影先の軸にすればよい。</br>

また、圧縮後の情報ロスは寄与率を求めることでわかる。</br>
第1～元次元文の主成分の分散は元の分散に一致し、</br>
第k主成分の分散は主成分に対応する固有値に一致しているため、</br>
全分散の分散に対する第k主成分の分散の割合が寄与率となる。 </br>
　<img src="https://latex.codecogs.com/gif.latex?c_k&space;=&space;\frac{\lambda_k}{\sum_{i=1}^{m}&space;\lambda_i}">  </br>

第1-k主成分まで圧縮した際の情報損失の割合を示す累積寄与率といい、</br>
主成分の総分散に対する第1~k主成分の分散の割合となる。</br>
　<img src="https://latex.codecogs.com/gif.latex?c_k&space;=&space;\frac{\sum_{j=1}^{k}&space;\lambda_j}{\sum_{i=1}^{m}&space;\lambda_i}"></br>

## 2.実装演習
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
