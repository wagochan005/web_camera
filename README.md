# web_camera
ウェブカメラを使って、物体検出＆物体の色(RGB値)を取得するプログラムです。

## ファイル詳細
・web_camera.py : ウェブカメラ動作検知後色取得プログラム(main)です。<br>
・

## 作成動機
ある作業に、プリンターから連続で画像を印刷するものがある。<br>
たまに機械のエラーで全く印刷されず、真っ白な紙が出てくる場合がある。<br>
基準の白紙の色(RGB)に対して、プリントされた紙が白でないことを確認するために、プリンターから出てきた紙を検出し、その紙の中心付近の色を取得して<br>
基準の白色(RGB)と取得した色(RGB)が異なれば正常、親しいRGB値であれば異常(印刷されていない)と判断するプログラムを作成しようと考えた。(←ここは作成途中)


## 動作内容
1.webカメラを起動します。<br>
2.物体の動きを検知します。<br>
3.物体の輪郭を取得します。<br>
4.物体の中心付近の色(RGB値)を取得します。<br>
5.基準の色に対して、近いRGB値でないかを確認する。<br>
6.異常なしであれば継続。<br>
7.撮影した画像は保存、取得したRGB値もログとして出力する。<br>

## 大まかな流れ
1.基準の紙の色を登録する
2.プリントする枚数分の紙をプリントされたら撮影していく
3.基準の色と撮影した紙の色が違うことを確認する
4.基準の色と差がない場合は、ログに警告を出す。
5.予定枚数分完了したら、処理を終了する。

## 作成メモ
初作成のプログラム。うまく動けばいいのだが。。。
OPENCVを使用した物体認識＆色取得プログラム。
詳細はQiita(https://qiita.com/Chima005) に記録予定
