from mmdet.apis import DetInferencer
import numpy as np
import glob
import cv2
import os
import pickle
import shutil

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()        
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            
            return
        
        







#入力はドラレコ動画を画像にし保存したパス(str)
#出力は物体検知の結果（人と信号機）
def video_analysis(path = "./temp/video/result"):
    #files = sorted(glob.glob("./temp/video/result/*"))
    files = sorted(glob.glob(path+"/*"))
    
    #推論結果を保存するファイルの設定
    folder_output_pic = "./temp/output_car_front"
    folder_output_bbox = "./temp/output_car_bbox"

    if not os.path.exists(folder_output_pic):
        os.mkdir(folder_output_pic)
    else:
        shutil.rmtree(folder_output_pic)
        os.mkdir(folder_output_pic)
    if not os.path.exists(folder_output_bbox):
        os.mkdir(folder_output_bbox)

    #コンフィグの設定
    model_name = 'rtmdet_tiny_8xb32-300e_coco'
    #チェックポイントの選択
    checkpoint = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    #デバイスの設定（CPUかGPU）
    device = 'cuda:0'

    # DetInferencer（推論機）の初期化
    inferencer = DetInferencer(model_name, checkpoint, device)
    
    person_all = []
    tlight_all = []
    
    #動画を読み込んだ後に何フレームあるかを入力する
    for i,ele in enumerate(files):
        #i = 3220
        print(i)
        # Use the detector to do inference
        #img = './car_front/'+f'{i:06}'+'_img.jpg'
        img = ele
        result = inferencer(img, out_dir=folder_output_pic)
        #print(result)
        #print(result['predictions'][0]['labels'])
        label_person = np.where(np.array(result['predictions'][0]['labels']) == 0)[0] #ラベル0は人
        label_tlight = np.where(np.array(result['predictions'][0]['labels']) == 9)[0] #ラベル9は信号機
        score30 = np.where(np.array(result['predictions'][0]['scores']) >= 0.30)[0]
        #print(label_person)
        #print(score_person)
        
        person_pred_label = list(set(label_person) & set(score30))
        tlight_pred_label = list(set(label_tlight) & set(score30))
        
        bboxes_person = [result['predictions'][0]['bboxes'][i] for i in person_pred_label]
        bboxes_tlight = [result['predictions'][0]['bboxes'][i] for i in tlight_pred_label]
        
        person_all.append(bboxes_person)
        tlight_all.append(bboxes_tlight)
    
    f = open(folder_output_bbox+"/bboxes_person.txt", 'wb')
    pickle.dump(person_all, f)
    f = open(folder_output_bbox+"/bboxes_tlight.txt", 'wb')
    pickle.dump(tlight_all, f)
    
    
    
        
        
        
        
    pass



#入力は画像、出力は赤が一定以上検出されたかどうか
def red_detect(img):
    # 赤色は２つの領域にまたがります！！
    # np.array([色彩, 彩度, 明度])
    # 各値は適宜設定する！！
    LOW_COLOR1 = np.array([0, 64, 150]) # 各最小値を指定
    HIGH_COLOR1 = np.array([30, 255, 255]) # 各最大値を指定
    LOW_COLOR2 = np.array([150, 64, 150])
    HIGH_COLOR2 = np.array([179, 255, 255])

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # RGB => YUV(YCbCr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # claheオブジェクトを生成
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0]) # 輝度にのみヒストグラム平坦化
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # YUV => RGB

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGRからHSVに変換

    bin_img1 = cv2.inRange(hsv, LOW_COLOR1, HIGH_COLOR1) # マスクを作成
    bin_img2 = cv2.inRange(hsv, LOW_COLOR2, HIGH_COLOR2)
    mask = bin_img1 + bin_img2 # 必要ならマスクを足し合わせる
    masked_img = cv2.bitwise_and(img, img, mask= mask) # 元画像から特定の色を抽出
    #cv2.imwrite("./out_img.jpg", masked_img) # 書き出す


    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    """
    fig_mask = plt.figure()
    ax_mask = fig_mask.add_subplot(1,1,1)
    ax_mask.imshow(mask)

    fig_crop = plt.figure()
    ax_crop = fig_crop.add_subplot(1,1,1)
    ax_crop.imshow(masked_img)
    """
    
    return True if sum(sum(mask)) >= 5000 else False


#入力は画像、出力は緑が一定以上検出されたかどうか
def green_detect(img):
    # 緑色は1つの領域にまたがります！！
    # np.array([色彩, 彩度, 明度])
    # 各値は適宜設定する！！
    LOW_COLOR1 = np.array([30, 64, 150]) # 各最小値を指定
    HIGH_COLOR1 = np.array([90, 255, 255]) # 各最大値を指定


    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) # RGB => YUV(YCbCr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # claheオブジェクトを生成
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0]) # 輝度にのみヒストグラム平坦化
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # YUV => RGB

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGRからHSVに変換

    bin_img1 = cv2.inRange(hsv, LOW_COLOR1, HIGH_COLOR1) # マスクを作成
    mask = bin_img1
    masked_img = cv2.bitwise_and(img, img, mask= mask) # 元画像から特定の色を抽出
    #cv2.imwrite("./out_img.jpg", masked_img) # 書き出す


    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

    """
    fig_mask = plt.figure()
    ax_mask = fig_mask.add_subplot(1,1,1)
    ax_mask.imshow(mask)

    fig_crop = plt.figure()
    ax_crop = fig_crop.add_subplot(1,1,1)
    ax_crop.imshow(masked_img)
    """
    
    return True if sum(sum(mask)) >= 5000 else False






def detection_person(path = "./temp/output_car_bbox/bboxes_person.txt",brake = [0]*300):
    
    
    

    f = open(path,"rb")
    bbox_person = pickle.load(f)
    brake = [0]*len(bbox_person)#ブレーキを踏んでいないダミーデータ
    
    area_max = []
    
    #print(bbox_person)
    
    
    
    
    #各フレームに対する検知した人の面積を計算する
    for i,ele in enumerate(bbox_person):
        area_all = []
        for j,elej in enumerate(ele):
            area = abs(elej[2] - elej[0])*abs(elej[3] - elej[1])
            #print(j,":",area)
            
            area_all.append(area)
        
        if len(area_all) > 0:
            area_max.append(sorted(area_all,reverse=True)[0])
        else:
            area_max.append(0)
            
    
            
            
    person_detect = np.where(np.array(area_max) >= 15000,1,0)
    
    #人が検知されたときに急ブレーキを踏んでいるフレームを算出
    result = np.array(brake)*np.array(person_detect)
    
    #print(person_detect)
    
    print(result)
    
    #[0]*len(bbox_person)
    
    
    
    
            

    #print(area_max)

    
    

    pass

    
    return ""


#後で引数に動画から保存した画像のpathを追加しろ
def detection_tlight(path = "./temp/output_car_bbox/bboxes_tlight.txt",brake = [0]*300):
    
    img_basepath = "./temp/video/result"###後で引数に追加
    img_files = sorted(glob.glob(img_basepath+"/*"))
    
    red_flag = False
    green_flag = False
    
    #####################################################
    #動画を分割した画像からheight,wigthを算出する
    width = 1920 
    height = 1080
    #####################################################

    f = open(path,"rb")
    bbox_tlight = pickle.load(f)
    
    #ブレーキを踏んでいないダミーデータ
    brake = [0]*len(bbox_tlight)
    #ブレーキを踏んでいるダミーデータ
    brake[250:] = [1]*(len(brake)-250)
    
    #赤信号が検知されたフレームを保持
    tlight_red = []
    
    #各フレームに対する検知した信号機が指定範囲内（車載から中央付近）にあるかで分類
    for i,per_frame in enumerate(bbox_tlight):
        #img_filesから動画を分解した画像を読み込む
        img = cv2.imread(img_files[i])
        
        #各フレームで信号機が検知されていたら
        if len(per_frame) > 0:            
            #指定範囲内で検出された信号機を検出(中央付近の方が良いか？)(tempの中身)
            temp = []
            for j,ele in enumerate(per_frame):
                if ele[0] >= 500 and ele[2] <= width-500:
                    temp.append(ele)
        #print(i,temp)
        if len(temp) > 0:
            #色を調べる
            #信号機の周りを見て赤が検出かつ青が検出されなかったら赤信号（の可能性が高い）
            #eleは検知された信号機の座標
            for j,ele in enumerate(temp):
                crop_img = img[int(ele[1]):int(ele[3]),int(ele[0]):int(ele[2])]
                """
                if i == 12:
                    print("img:",img)
                """
                red_flag = red_detect(crop_img)
                green_flag = green_detect(crop_img)
                
                if red_flag == True and green_flag == False:
                    tlight_red.append(1)
                    break
                if j == len(temp)-1:
                    #信号機は検知されたが赤でなかった場合
                    tlight_red.append(0)
        #信号機が検知されていない場合
        else:
            tlight_red.append(0)
            
            
        
        
        
        
        #temp[0]#検出された信号機の中で最もスコアが高く
        
    #print(len(tlight_red))
    #print(tlight_red)
    
    #赤信号が検知されたときに急ブレーキを踏んでいるフレームを算出
    result = np.array(brake)*np.array(tlight_red)
    
    #print(person_detect)
    
    print(result)
            



    return ""







def main(input):
    #save_all_frames(動画のパス, 画像の保存先, 保存画像のbasename)
    ###save_all_frames('./video_tlight.mp4', './temp/video/result', 'img')
    #video_analysis(ドラレコ動画を画像にし、保存したパス(str))
    ###video_analysis('./temp/video/result')
    #person_score = detection_person()
    tlight_score = detection_tlight()
    
    print("end")
    
    


if __name__ == "__main__":
    main("")


