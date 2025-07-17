"""
task1_detection_tracking_simple.py
Deep SORTを使わない簡易追跡版

YOLOv11による検出 + シンプルなIoUベース追跡
Deep SORTのインストール問題を回避
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy.spatial.distance import cdist

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTracker:
    """シンプルなIoUベース追跡クラス"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        初期化
        
        Args:
            max_age: トラック最大保持フレーム数
            min_hits: 確信トラックになるための最小ヒット数
            iou_threshold: IoU閾値
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_counter = 0
        
        logger.info("Simple Tracker初期化完了")
    
    def update(self, detections: Dict, frame_id: int) -> List[Dict]:
        """
        追跡更新
        
        Args:
            detections: 検出結果
            frame_id: フレーム番号
            
        Returns:
            追跡結果リスト
        """
        # 検出結果を処理
        det_boxes = np.array(detections['boxes']) if detections['boxes'] else np.empty((0, 4))
        det_scores = np.array(detections['scores']) if detections['scores'] else np.empty(0)
        det_classes = detections['names'] if detections['names'] else []
        
        # 既存トラックの予測
        for track in self.tracks:
            track['age'] += 1
            track['hits_streak'] = 0
        
        # マッチング実行
        if len(det_boxes) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
                det_boxes, [track['bbox'] for track in self.tracks]
            )
            
            # マッチしたトラック更新
            for m in matched:
                det_idx, trk_idx = m[0], m[1]
                self.tracks[trk_idx]['bbox'] = det_boxes[det_idx].tolist()
                self.tracks[trk_idx]['confidence'] = float(det_scores[det_idx])
                self.tracks[trk_idx]['class_name'] = det_classes[det_idx]
                self.tracks[trk_idx]['hits'] += 1
                self.tracks[trk_idx]['hits_streak'] += 1
                self.tracks[trk_idx]['age'] = 0
                self.tracks[trk_idx]['frame_id'] = frame_id
                
                # 速度計算
                self._update_velocity(self.tracks[trk_idx], frame_id)
            
            # 未マッチング検出から新しいトラック作成
            for i in unmatched_dets:
                new_track = self._create_track(
                    det_boxes[i], det_scores[i], det_classes[i], frame_id
                )
                self.tracks.append(new_track)
        
        elif len(det_boxes) > 0:
            # 初回検出
            for i in range(len(det_boxes)):
                new_track = self._create_track(
                    det_boxes[i], det_scores[i], det_classes[i], frame_id
                )
                self.tracks.append(new_track)
        
        # 古いトラック削除
        self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
        
        # 確信トラックのみ返却
        confirmed_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits or track['hits_streak'] >= 1:
                confirmed_tracks.append({
                    'track_id': track['track_id'],
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'class_name': track['class_name'],
                    'velocity': track['velocity'],
                    'frame_id': frame_id
                })
        
        return confirmed_tracks
    
    def _create_track(self, bbox: np.ndarray, score: float, class_name: str, frame_id: int) -> Dict:
        """新しいトラック作成"""
        self.track_id_counter += 1
        return {
            'track_id': self.track_id_counter,
            'bbox': bbox.tolist(),
            'confidence': float(score),
            'class_name': class_name,
            'hits': 1,
            'hits_streak': 1,
            'age': 0,
            'velocity': [0.0, 0.0],
            'history': [{'frame_id': frame_id, 'bbox': bbox.tolist()}],
            'frame_id': frame_id
        }
    
    def _update_velocity(self, track: Dict, frame_id: int) -> None:
        """速度更新"""
        track['history'].append({
            'frame_id': frame_id,
            'bbox': track['bbox']
        })
        
        # 履歴長制限
        if len(track['history']) > 10:
            track['history'] = track['history'][-10:]
        
        # 速度計算（過去2フレーム使用）
        if len(track['history']) >= 2:
            curr_bbox = track['history'][-1]['bbox']
            prev_bbox = track['history'][-2]['bbox']
            
            curr_center = [(curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2]
            prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2]
            
            frame_diff = track['history'][-1]['frame_id'] - track['history'][-2]['frame_id']
            
            if frame_diff > 0:
                vx = (curr_center[0] - prev_center[0]) / frame_diff
                vy = (curr_center[1] - prev_center[1]) / frame_diff
                track['velocity'] = [float(vx), float(vy)]
    
    def _associate_detections_to_trackers(self, detections: np.ndarray, trackers: List) -> Tuple:
        """検出結果とトラッカーのマッチング"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # IoU計算
        iou_matrix = self._compute_iou_matrix(detections, np.array(trackers))
        
        # ハンガリアンアルゴリズムの簡易版（貪欲法）
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))
        
        # IoU閾値以上のペアを見つける
        while len(unmatched_detections) > 0 and len(unmatched_trackers) > 0:
            # 最大IoUを見つける
            max_iou = 0
            max_det_idx = -1
            max_trk_idx = -1
            
            for det_idx in unmatched_detections:
                for trk_idx in unmatched_trackers:
                    iou = iou_matrix[det_idx, trk_idx]
                    if iou > max_iou and iou > self.iou_threshold:
                        max_iou = iou
                        max_det_idx = det_idx
                        max_trk_idx = trk_idx
            
            if max_det_idx >= 0:
                matched_indices.append([max_det_idx, max_trk_idx])
                unmatched_detections.remove(max_det_idx)
                unmatched_trackers.remove(max_trk_idx)
            else:
                break
        
        return (np.array(matched_indices), 
                np.array(unmatched_detections), 
                np.array(unmatched_trackers))
    
    def _compute_iou_matrix(self, det_boxes: np.ndarray, trk_boxes: np.ndarray) -> np.ndarray:
        """IoU行列計算"""
        iou_matrix = np.zeros((len(det_boxes), len(trk_boxes)))
        
        for d in range(len(det_boxes)):
            for t in range(len(trk_boxes)):
                iou_matrix[d, t] = self._compute_iou(det_boxes[d], trk_boxes[t])
        
        return iou_matrix
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """IoU計算"""
        # box: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# 他のクラスは task1_detection_tracking.py から流用
class YOLODetector:
    """YOLO物体検出クラス"""
    
    def __init__(self, model_path: str = "yolo11n.pt", confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str) -> None:
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLOモデル読み込み完了: {model_path}")
        except Exception as e:
            logger.error(f"YOLOモデル読み込み失敗: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        if self.model is None:
            raise RuntimeError("YOLOモデルが読み込まれていません")
        
        try:
            results = self.model(frame, verbose=False)
            return self._format_detections(results[0])
        except Exception as e:
            logger.error(f"物体検出エラー: {e}")
            return {'boxes': [], 'scores': [], 'classes': [], 'names': []}
    
    def _format_detections(self, result) -> Dict:
        detections = {
            'boxes': [],
            'scores': [],
            'classes': [],
            'names': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            valid_indices = scores >= self.confidence_threshold
            
            if np.any(valid_indices):
                detections['boxes'] = boxes[valid_indices].tolist()
                detections['scores'] = scores[valid_indices].tolist()
                detections['classes'] = classes[valid_indices].tolist()
                detections['names'] = [result.names[int(cls)] for cls in classes[valid_indices]]
        
        return detections

class VideoProcessor:
    """動画処理クラス"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0
        self._open_video()
    
    def _open_video(self) -> None:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"動画ファイルを開けません: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"動画読み込み完了: {self.video_path}")
        logger.info(f"総フレーム数: {self.total_frames}, FPS: {self.fps}")
    
    def get_next_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict]:
        if self.cap is None:
            return False, None, {}
        
        ret, frame = self.cap.read()
        if ret:
            metadata = {
                'frame_id': self.frame_count,
                'timestamp': self.frame_count / self.fps if self.fps > 0 else 0,
                'total_frames': self.total_frames
            }
            self.frame_count += 1
            return True, frame, metadata
        
        return False, None, {}
    
    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class OutputManager:
    """出力管理クラス"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.statistics = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'processing_times': []
        }
    
    def add_result(self, frame_metadata: Dict, tracking_results: List[Dict], processing_time: float) -> None:
        for track in tracking_results:
            result = {
                'frame_id': frame_metadata['frame_id'],
                'timestamp': frame_metadata['timestamp'],
                'track_id': track['track_id'],
                'x1': track['bbox'][0],
                'y1': track['bbox'][1],
                'x2': track['bbox'][2],
                'y2': track['bbox'][3],
                'confidence': track['confidence'],
                'class_name': track['class_name'],
                'velocity_x': track['velocity'][0],
                'velocity_y': track['velocity'][1]
            }
            self.results.append(result)
        
        self.statistics['total_frames'] += 1
        self.statistics['total_detections'] += len(tracking_results)
        self.statistics['processing_times'].append(processing_time)
        
        unique_tracks = set(track['track_id'] for track in tracking_results)
        self.statistics['total_tracks'] = max(self.statistics['total_tracks'], len(unique_tracks))
    
    def save_results(self, filename: str = "tracking_results.csv") -> None:
        if not self.results:
            logger.warning("保存する結果がありません")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        logger.info(f"追跡結果保存完了: {csv_path}")
        
        self._save_statistics()
    
    def _save_statistics(self) -> None:
        if self.statistics['processing_times']:
            avg_time = np.mean(self.statistics['processing_times'])
            max_time = np.max(self.statistics['processing_times'])
            min_time = np.min(self.statistics['processing_times'])
            
            stats = {
                'total_frames': self.statistics['total_frames'],
                'total_detections': self.statistics['total_detections'],
                'total_unique_tracks': self.statistics['total_tracks'],
                'average_processing_time': float(avg_time),
                'max_processing_time': float(max_time),
                'min_processing_time': float(min_time),
                'average_fps': float(1.0 / avg_time) if avg_time > 0 else 0
            }
            
            stats_path = self.output_dir / "processing_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"処理統計保存完了: {stats_path}")
            logger.info(f"平均処理時間: {avg_time:.3f}s, 平均FPS: {stats['average_fps']:.1f}")

class SimpleDetectionTrackingPipeline:
    """簡易版検出・追跡パイプライン"""
    
    def __init__(self, 
                 video_path: str,
                 model_path: str = "yolo11n.pt",
                 output_dir: str = "output",
                 confidence_threshold: float = 0.5,
                 max_age: int = 30,
                 min_hits: int = 3):
        
        self.detector = YOLODetector(model_path, confidence_threshold)
        self.tracker = SimpleTracker(max_age, min_hits)
        self.video_processor = VideoProcessor(video_path)
        self.output_manager = OutputManager(output_dir)
        
        logger.info("簡易版検出・追跡パイプライン初期化完了")
    
    def run(self, visualize: bool = False, save_video: bool = False) -> None:
        logger.info("パイプライン実行開始")
        
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 動画の実際の解像度を取得
            frame_width = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_processor.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_writer = cv2.VideoWriter(
                str(self.output_manager.output_dir / "tracking_output.mp4"),
                fourcc, self.video_processor.fps, (frame_width, frame_height)
            )
        
        try:
            while True:
                start_time = time.time()
                
                success, frame, metadata = self.video_processor.get_next_frame()
                if not success:
                    break
                
                detections = self.detector.detect_objects(frame)
                tracking_results = self.tracker.update(detections, metadata['frame_id'])
                
                processing_time = time.time() - start_time
                self.output_manager.add_result(metadata, tracking_results, processing_time)
                
                if visualize or save_video:
                    vis_frame = self._visualize_results(frame, tracking_results)
                    
                    if visualize:
                        cv2.imshow("Simple Detection & Tracking", vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if save_video and video_writer:
                        video_writer.write(vis_frame)
                
                if metadata['frame_id'] % 30 == 0:
                    progress = metadata['frame_id'] / metadata['total_frames'] * 100
                    logger.info(f"処理進捗: {progress:.1f}% ({metadata['frame_id']}/{metadata['total_frames']})")
        
        except KeyboardInterrupt:
            logger.info("ユーザーによる中断")
        except Exception as e:
            logger.error(f"パイプライン実行エラー: {e}")
            raise
        finally:
            self.video_processor.close()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            self.output_manager.save_results()
            logger.info("パイプライン実行完了")
    
    def _visualize_results(self, frame: np.ndarray, tracking_results: List[Dict]) -> np.ndarray:
        vis_frame = frame.copy()
        
        for track in tracking_results:
            x1, y1, x2, y2 = map(int, track['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"ID:{track['track_id']} {track['class_name']} {track['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            vx, vy = track['velocity']
            if abs(vx) > 1 or abs(vy) > 1:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                end_x = int(center_x + vx * 5)
                end_y = int(center_y + vy * 5)
                cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)
        
        return vis_frame

def main():
    """メイン関数"""
    VIDEO_PATH = "Solving the Long-Tail_cyclist.mp4"
    MODEL_PATH = "yolo11n.pt"
    OUTPUT_DIR = "task1_simple_output"
    
    pipeline = SimpleDetectionTrackingPipeline(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        confidence_threshold=0.3,
        max_age=50,
        min_hits=3
    )
    
    try:
        pipeline.run(visualize=True, save_video=True)
    except Exception as e:
        logger.error(f"実行エラー: {e}")

if __name__ == "__main__":
    main()
