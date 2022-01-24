# ライブラリのインポート
import cv2
import os
import matplotlib.pyplot as plt
%matplotlib inline
import itertools
import numpy as np
import copy
import json
import numpy as np
import time
import datetime

main_number = 1
end_number = int(input("処理枚数を入力してください"))
now = datetime.datetime.now()
current = now.strftime("%Y-%m-%d-%H-%M")
folder_name = "./capture_data/capture_data_" + current

base_R = 150
base_G = 150
base_B = 150

# フォルダ作成
os.makedirs(folder_name, exist_ok = True)
base_path = os.path.join(folder_name, "camera_capture")

# 基準のRGB値をtxtに出力しておく
with open("{}/RGB_log.txt".format(folder_name), mode = "a") as f:
    f.writelines("0" + " : " + "base_color" + " : " + str(base_R) + ", G : " + str(base_G) + ", B : " + str(base_B))
    f.write("\n")

def main():
    
    global main_number
    global end_number
    global folder_name
    
    os.makedirs(folder_name, exist_ok = True)
    base_path = os.path.join(folder_name, "camera_capture")
    # カメラキャプチャの開始
    cap = cv2.VideoCapture(0)
    
    #ここはカメラのフレームサイズ、FTP値によって変更する
    cap.set(cv2.CAP_PROP_FPS, 30)           # カメラFPSを設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840) # カメラ画像の横幅を設定3840
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # カメラ画像の縦幅を設定2160
    
    # フレームの初期化
    img1 = img2 = img3 = get_image(cap)
    th = 5000 # しきい値

    while True:
        # Enterキーが押されたら終了
        if cv2.waitKey(1000)  == 13: break #()内の秒数はミリ秒 1000ms= 1s
        # 差分を調べる
        diff = check_image(img1, img2, img3)
        cv2.imshow("test", img1)
        # 差分がthの値以上なら動きがあったと判定
        cnt = cv2.countNonZero(diff) # 二値化した画像の白色部分の面積を算出
        if cnt > th:
            time.sleep(1)

            # 写真を保存
            cv2.imwrite(base_path + str(main_number) + ".png", img3)
            break
            
        else:
            cv2.imshow('PUSH ENTER KEY', diff)
        # 比較用の画像を保存
        img1, img2, img3 = (img2, img3, get_image(cap))
    
    # 画像の読み込み
    image = cv2.imread("{}/camera_capture{}.png".format(folder_name, main_number))
    image = image[30 : -30, 30 : -30, :]
#     cv2.imshow("test2", image)   
    
    # 後始末
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    # 画像の高さ、幅、チャンネル数を取得
    height, width, channel = image.shape


    # 画像をグレースケール化
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 画像をCanny法により、エッジを検出する
    threshold1 = 100
    threshold2 = 200
    edge = cv2.Canny(img_gray, threshold1, threshold2)
#     cv2.imshow("edge", edge)
    
    # 膨張処理　境界線を太くする
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilates = cv2.dilate(edge, kernel)
    
    # 後始末
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    
    # 輪郭の検出
    contours, hierarchy = cv2.findContours(dilates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 輪郭の見つける
    
    # 小さい輪郭は削除
    contours = list(filter(lambda x: cv2.contourArea(x) > 10000, contours))
    
    # 座標を抽出
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)

    # 輪郭の描画
    cv2.drawContours(image, contours, -1, color=(255, 255, 255), thickness=2)

    # 画像のクローズ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

    # JSON用の辞書配列
    export_array = {
        "width" : width,
        "height" : height,
        "results" : []
    }

    # 4つの頂点をマーキング
    cv2.circle(img=image, center=(x, y), radius=10, color=(255, 255, 255), thickness=2)
    cv2.circle(img=image, center=(x + w, y), radius=10, color=(255, 255, 255), thickness=2)
    cv2.circle(img=image, center=(x, y + h), radius=10, color=(255, 255, 255), thickness=2)
    cv2.circle(img=image, center=(x + w, y + h), radius=10, color=(255, 255, 255), thickness=2)

    # 4つの頂点から中点を算出する
    center_x = int(x + (w / 2))
    center_y = int(y + (h / 2))

    # 4つの頂点の中点が中心の半径50の円を書く
    cv2.circle(img=image, center=(center_x, center_y), radius=50, color=(0, 255, 255), thickness=2)

    # 中点回りの範囲の色情報を取得するためのBOXを作る
    boxFromX = center_x - 50
    boxFromY = center_y - 50
    boxToX = center_x + 50
    boxToY = center_y + 50
    imgBox = image[boxFromY : boxToY, boxFromX : boxToX]

    cv2.rectangle(image, (boxFromX, boxFromY), (boxToX, boxToY), (255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0)

    # RGB値を出力する
    # flattenで一元化してmeanで平均を取得
    b = imgBox.T[0].flatten().mean()
    g = imgBox.T[1].flatten().mean()
    r = imgBox.T[2].flatten().mean()

    # RGB値を取得する
    print("R : " + str(int(r)))
    print("G : " + str(int(g)))
    print("B : " + str(int(b)))
    RGB = [str(int(r)), str(int(g)), str(int(b))]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap = "gray")  
    # 後始末
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    cap.release()

    # RGB値の出力結果をtxtに出力
    with open("{}/RGB_log.txt".format(folder_name), mode = "a") as f:
        now_print = datetime.datetime.now()
        current_time = now_print.strftime("%Y-%m-%d-%H-%M-%S")
        f.writelines(str(main_number) + " : " + current_time + " R : " + RGB[0] + ", G : " + RGB[1] + ", B : " + RGB[2])
        f.write("\n")
    
    # 実行回数に到達すると回帰から抜ける。
    print("main_number : " + str(main_number))
    if main_number != end_number:
        main_number += 1
        cap.release()
        main()
    else:
        cap.release()
    

# 画像に動きがあったか調べる関数
def check_image(img1, img2, img3):
    # グレイスケール画像に変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_RGB2GRAY)
    # 絶対差分を調べる
    diff1 = cv2.absdiff(gray1, gray2)
    diff2 = cv2.absdiff(gray2, gray3)
    # 論理積を調べる
    diff_and = cv2.bitwise_and(diff1, diff2)
    # 白黒二値化
    _, diff_wb = cv2.threshold(diff_and, 30, 255, cv2.THRESH_BINARY)
#     diff_wb = cv2.adaptiveThreshold(diff_and, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    # ノイズの除去
    diff = cv2.medianBlur(diff_wb, 5)
    return diff

# カメラから画像を取得する
def get_image(cap):
    img = cap.read()[1]
    img = cv2.resize(img, (600, 400)) # 扱いやすいフレームサイズに変更する
    return img

main()