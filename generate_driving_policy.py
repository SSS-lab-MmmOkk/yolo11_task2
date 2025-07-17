import pandas as pd
import numpy as np
import cv2
import os

# --- 定数定義 ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
EGO_VEHICLE_POSITION = (FRAME_WIDTH / 2, FRAME_HEIGHT)
TARGET_CLASSES = ['cyclist', 'person', 'car']
DISTANCE_THRESHOLDS = {
    'NEAR': FRAME_HEIGHT * 0.6,
    'MID': FRAME_HEIGHT * 0.3
}
# 日本語フォントのパス（環境に合わせて変更が必要な場合があります）
# 一般的なLinux環境で利用可能なIPAフォントを指定
FONT_PATH_LINUX = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"


def get_distance_to_ego(obj_bbox):
    """オブジェクトと自車の距離を簡易的に計算"""
    x1, y1, x2, y2 = obj_bbox
    bottom_center_y = y2
    return FRAME_HEIGHT - bottom_center_y

def analyze_frame_for_policy(frame_df):
    """1フレーム分のデータから運転方針を決定"""
    target_objects = frame_df[frame_df['class_name'].isin(TARGET_CLASSES)].copy()
    if target_objects.empty:
        return "周囲の状況を確認し、安全に走行します。"

    target_objects['bbox'] = target_objects.apply(lambda row: (row['x1'], row['y1'], row['x2'], row['y2']), axis=1)
    target_objects['distance'] = target_objects['bbox'].apply(get_distance_to_ego)
    most_critical_object = target_objects.loc[target_objects['distance'].idxmin()]

    obj_class = most_critical_object['class_name']
    obj_bbox = most_critical_object['bbox']
    obj_bottom_y = obj_bbox[3]

    obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
    if obj_center_x < FRAME_WIDTH / 3:
        horizontal_pos = "左前方"
    elif obj_center_x > FRAME_WIDTH * 2 / 3:
        horizontal_pos = "右前方"
    else:
        horizontal_pos = "前方"

    if obj_bottom_y > DISTANCE_THRESHOLDS['NEAR']:
        policy = f"{horizontal_pos}に{obj_class}が近接。減速し車間距離を確保。"
    elif obj_bottom_y > DISTANCE_THRESHOLDS['MID']:
        policy = f"{horizontal_pos}の{obj_class}に注意。追従します。"
    else:
        policy = f"遠方の{obj_class}を認識。引き続き注意。"

    vx = most_critical_object['velocity_x']
    if vx > 5:
        policy += " (右へ移動中)"
    elif vx < -5:
        policy += " (左へ移動中)"

    return policy

def draw_text(img, text, pos, font_size, color, thickness):
    """OpenCVで日本語を描画するラッパー関数"""
    # PIL(Pillow)を使って日本語を描画する
    try:
        from PIL import ImageFont, ImageDraw, Image
        font = ImageFont.truetype(FONT_PATH_LINUX, font_size)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError) as e:
        # PILがない、またはフォントファイルが見つからない場合は、英数字のみの描画にフォールバック
        if isinstance(e, ImportError):
            print("警告: Pillowがインストールされていません。'pip install Pillow' を実行してください。")
        else:
            print(f"警告: フォントファイルが見つかりません ({FONT_PATH_LINUX})。日本語が表示されません。")

        # 英数字のみで描画
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 40, color, thickness, cv2.LINE_AA)
        return img


def visualize_policy_on_video(video_path, tracking_csv_path, output_video_path):
    """動画に運転方針とバウンディングボックスを重畳表示する"""
    try:
        df = pd.read_csv(tracking_csv_path)
    except FileNotFoundError:
        print(f"エラー: {tracking_csv_path} が見つかりません。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル {video_path} を開けません。")
        return

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 現在のフレームに対応するデータを抽出
        frame_df = df[df['frame_id'] == frame_id]

        if not frame_df.empty:
            # 運転方針を生成
            policy_text = analyze_frame_for_policy(frame_df)

            # 検出オブジェクトのバウンディングボックスを描画
            for _, row in frame_df.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{row['class_name']} ID:{row['track_id']}"
                frame = draw_text(frame, label, (x1, y1 - 35), 30, (0, 255, 0), 2)


            # 運転方針テキストを描画
            frame = draw_text(frame, policy_text, (50, 50), 40, (255, 255, 255), 2)

        out.write(frame)
        frame_id += 1

        if frame_id % 30 == 0:
            print(f"処理中... フレーム {frame_id}")

    cap.release()
    out.release()
    print(f"可視化ビデオを {output_video_path} に保存しました。")


if __name__ == "__main__":
    VIDEO_PATH = "Solving the Long-Tail_cyclist.mp4"
    TRACKING_RESULTS_PATH = "task1_simple_output/tracking_results.csv"
    OUTPUT_POLICY_PATH = "task1_simple_output/driving_policy.txt"
    OUTPUT_VIDEO_PATH = "task1_simple_output/policy_visualization.mp4"

    # テキスト版の方針生成（これは残しておく）
    # generate_driving_policy(TRACKING_RESULTS_PATH, OUTPUT_POLICY_PATH)

    # 可視化版の生成
    visualize_policy_on_video(VIDEO_PATH, TRACKING_RESULTS_PATH, OUTPUT_VIDEO_PATH)
