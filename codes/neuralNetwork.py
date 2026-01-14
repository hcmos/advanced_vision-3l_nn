# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pdb

# クラス
class neuralNetwork():
    #-------------------
    # 1. 学習データの初期化
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    # Y: 出力データ（データ数×次元数のnumpy.ndarray）
    # hDim: 中間層のノード数（整数スカラー）
    # activeType: 活性化関数の種類（1:シグモイド関数、2:ReLU関数、3:Leaky ReLU関数）
    def __init__(self,X,Y,hDim=10,activeType=1):
        # 学習データの設定
        self.xDim = X.shape[1]
        self.yDim = Y.shape[1]
        self.hDim = hDim

        self.activeType = activeType
        self.leaky_relu_alpha = 0.1

        # パラメータの初期値の設定
        self.w1 = np.random.normal(size=[self.xDim,self.hDim])
        self.w2 = np.random.normal(size=[self.hDim,self.yDim])
        self.b1 = np.random.normal(size=[1,self.hDim])
        self.b2 = np.random.normal(size=[1,self.yDim])

        # log(0)を回避するための微小値
        self.smallV = 10e-8
    #-------------------

    #-------------------
    # 2. 最急降下法を用いてモデルパラメータの更新
    # alpha: 学習率（実数スカラー）
    def update(self,X,Y,alpha=0.1):

        # 行列Xに「1」の要素を追加
        dNum = len(X)
        Z = np.append(X,np.ones([dNum,1]),axis=1)

        # 予測
        P,H,S = self.predict(X)

        # 予測の差の計算
        error = P - Y

        # 各階層のパラメータの準備
        V2 = np.concatenate([self.w2,self.b2],axis=0)
        V1 = np.concatenate([self.w1,self.b1],axis=0)

        # 入力層と中間層の間のパラメータの更新
        if self.activeType == 1:  # シグモイド関数
            term1 = np.matmul(error,self.w2.T)
            term2 = term1 * (1-H) * H
            grad1 = 1/dNum * np.matmul(Z.T,term2)

        elif self.activeType == 2: # ReLU関数
            Ms = np.ones_like(S)
            Ms[S<=0] = 0
            term1 = np.matmul(error,self.w2.T)
            grad1 = 1/dNum * np.matmul(Z.T,term1*Ms)

        elif self.activeType == 3:  # Leaky ReLU関数
            Ms = np.ones_like(S)
            Ms[S<=0] = self.leaky_relu_alpha
            term1 = np.matmul(error,self.w2.T)
            grad1 = 1/dNum * np.matmul(Z.T,term1*Ms)

        V1 -= alpha * grad1

        # 中間層と出力層の間のパラメータの更新
        # 行列Xに「1」の要素を追加
        H = np.append(H,np.ones([dNum,1]),axis=1)
        grad2 = 1/dNum * np.matmul(H.T,error)
        V2 -= alpha * grad2

        # パラメータw1,b1,w2,b2の決定
        self.w1 = V1[:-1]
        self.w2 = V2[:-1]
        self.b1 = V1[[-1]]
        self.b2 = V2[[-1]]
    #-------------------

    #-------------------
    # 3. 予測
    # X: 入力データ（データ数×次元数のnumpy.ndarray）
    def predict(self,x):
        s = np.matmul(x,self.w1) + self.b1
        H = self.activation(s)
        f_x = np.matmul(H,self.w2) + self.b2

        return 1/(1+np.exp(-f_x)),H,s
    #-------------------

    #-------------------
    # 4. 活性化関数
    # s: 中間データ（データ数×次元数のnumpy.ndarray）
    def activation(self,s):

        if self.activeType == 1:  # シグモイド関数
            h = 1/(1+np.exp(-s))

        elif self.activeType == 2:  # ReLU関数
            h = s
            h[h<=0] = 0

        elif self.activeType == 3:  # Leaky ReLU関数
            h = np.where(s>0,s,self.leaky_relu_alpha*s)

        return h
    #-------------------

    #-------------------
    # 5. 交差エントロピー損失
    # X: 入力データ（次元数×データ数のnumpy.ndarray）
    # Y: 出力データ（データ数×次元数のnumpy.ndarray）
    def CE(self,X,Y):
        P,_,_ = self.predict(X)

        if self.yDim == 1:
            loss = -np.mean(Y*np.log(P+self.smallV)+(1-Y)*np.log(1-P+self.smallV))
        else:
            loss = -np.mean(Y*np.log(P+self.smallV))

        return loss
    #-------------------

    #-------------------
    # 6. 正解率の計算
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # thre: 閾値（スカラー）
    def accuracy(self,X,Y,thre=0.5):
        P,_,_= self.predict(X)

        # 予測値Pをラベルに変換
        if self.yDim == 1:
            P[P>thre] = 1
            P[P<=thre] = 0
        else:
            P = np.argmax(P,axis=1)
            Y = np.argmax(Y,axis=1)

        # 正解率
        accuracy = np.mean(Y==P)
        return accuracy
    #-------------------

    #-------------------
    # 7. 真値と予測値のプロット（入力ベクトルが1次元の場合）
    # X:入力データ（次元数×データ数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # xLabel:x軸のラベル（文字列）
    # yLabel:y軸のラベル（文字列）
    # fName：画像の保存先（文字列）
    def plotModel1D(self,X=[],Y=[],xLabel="",yLabel="",fName=""):
        plt.rcParams['font.family'] = 'MS Gothic'
        fig = plt.figure(figsize=(6,4),dpi=100)

        # 予測値
        P,_ = self.predict(X)

        # 真値と予測値のプロット
        plt.plot(X,Y,'b.',label="真値")
        plt.plot(X,P,'r.',label="予測")

        # 各軸の範囲とラベルの設定
        plt.yticks([0,0.5,1])
        plt.ylim([-0.1,1.1])
        plt.xlim([np.min(X),np.max(X)])
        plt.xlabel(xLabel,fontsize=14)
        plt.ylabel(yLabel,fontsize=14)
        plt.grid()
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------

    #-------------------
    # 8. 真値と予測値のプロット（入力ベクトルが2次元の場合）
    # X:入力データ（データ数×次元数のnumpy.ndarray）
    # Y:出力データ（データ数×次元数のnumpy.ndarray）
    # xLabel:x軸のラベル（文字列）
    # yLabel:y軸のラベル（文字列）
    # title:タイトル（文字列）
    # fName：画像の保存先（文字列）
    def plotModel2D(self,X=[],Y=[],xLabel="",yLabel="",title="",fName=""):
        plt.rcParams['font.family'] = 'MS Gothic'
        #fig = plt.figure(figsize=(6,4),dpi=100)
        plt.close()

        # 真値のプロット（クラスごとにマーカーを変更）
        plt.plot(X[Y[:,0]==0,0],X[Y[:,0]==0,1],'cx',markersize=14,label="ラベル0")
        plt.plot(X[Y[:,0]==1,0],X[Y[:,0]==1,1],'m.',markersize=14,label="ラベル1")

        # 予測値のメッシュの計算
        X1,X2 = plt.meshgrid(plt.linspace(np.min(X[:,0]),np.max(X[:,0]),50),plt.linspace(np.min(X[:,1]),np.max(X[:,1]),50))
        Xmesh = np.hstack([np.reshape(X1,[-1,1]),np.reshape(X2,[-1,1])])
        Pmesh,_,_ = self.predict(Xmesh)
        Pmesh = np.reshape(Pmesh,X1.shape)

        # 予測値のプロット
        CS = plt.contourf(X1,X2,Pmesh,linewidths=2,cmap="bwr",alpha=0.3,vmin=0,vmax=1)

        # カラーバー
        CB = plt.colorbar(CS)
        CB.ax.tick_params(labelsize=14)

        # 各軸の範囲とラベルの設定
        plt.xlim([np.min(X[:,0]),np.max(X[:,0])])
        plt.ylim([np.min(X[:,1]),np.max(X[:,1])])
        plt.title(title,fontsize=14)
        plt.xlabel(xLabel,fontsize=14)
        plt.ylabel(yLabel,fontsize=14)
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------

    #-------------------
    # 8. 学習と評価損失のプロット
    # trEval:学習の損失
    # teEval:評価の損失
    # yLabel:y軸のラベル（文字列）
    # fName:画像の保存先（文字列）
    def plotEval(self,trEval,teEval,ylabel="損失",fName=""):
        fig = plt.figure(figsize=(6,4),dpi=100)

        # 損失のプロット
        plt.plot(trEval,'b',label="学習")
        plt.plot(teEval,'r',label="評価")

        # 各軸の範囲とラベルの設定
        plt.xlabel("反復",fontsize=14)
        plt.ylabel(ylabel,fontsize=14)
        plt.ylim([0,1.1])
        plt.legend()

        # グラフの表示またはファイルへの保存
        if len(fName):
            plt.savefig(fName)
        else:
            plt.show()
    #-------------------
