# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import deque

import numpy as np
import torch

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """Extended STrack for YOLOv8 with object tracking features."""

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """Initialize BOTrack with temporal parameters."""
        # Convert to numpy array
        if isinstance(tlwh, (list, tuple)):
            tlwh = np.array(tlwh, dtype=np.float32)
        elif isinstance(tlwh, torch.Tensor):
            tlwh = tlwh.cpu().numpy().astype(np.float32)

        # Validate input
        if len(tlwh) < 4:
            raise ValueError(f"Invalid box: expected >=4 values, got {len(tlwh)}")

        # Take first 4 elements
        box_4 = tlwh[:4]

        # Detect format: xyxy vs tlwh
        is_xyxy = (box_4[2] > box_4[0] and box_4[3] > box_4[1] and
                   (box_4[2] - box_4[0]) > 5 and (box_4[3] - box_4[1]) > 5)

        if is_xyxy:
            # Convert xyxy to tlwh
            x1, y1, x2, y2 = box_4
            tlwh_converted = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        else:
            # Already tlwh
            tlwh_converted = box_4.astype(np.float32)

        # Prepare for parent class
        xywh_score = np.concatenate([tlwh_converted, [score]])

        # Call parent
        super().__init__(xywh_score, score, cls)

        # Initialize ReID features
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None and len(feat) > 0:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        """Update feature vector with exponential moving average."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        """Predict future state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate track with updated features."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id):
        """Update track with new information."""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        """Get current bounding box in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """Predict mean and covariance for multiple tracks."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        """Convert tlwh to xywh format."""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box from tlwh to xywh."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """Extended BYTETracker for YOLOv8 with ReID and GMC."""

    def __init__(self, args, frame_rate=30):
        """Initialize with ReID module and GMC algorithm."""
        super().__init__(args, frame_rate)

        # ReID parameters
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        # Initialize ReID encoder
        if args.with_reid:
            try:
                from .fast_reid_wrapper_fixed import FastReIDWrapper

                reid_model_path = getattr(args, 'reid_model_path', 'sheep_reid_128d.pth')
                self.encoder = FastReIDWrapper(
                    model_path=reid_model_path,
                    device='cuda',
                    fp16=True,
                    verbose=False
                )
                print(f"✅ ReID encoder initialized: {reid_model_path}")
            except Exception as e:
                print(f"⚠️  ReID encoder failed: {e}")
                print("   Using pure IoU tracking")
                self.encoder = None
        else:
            self.encoder = None

        # GMC and debug flags
        self.gmc = GMC(method=args.gmc_method)
        self._debug_printed = False
        self._frame_count = 0

    def get_kalmanfilter(self):
        """Return KalmanFilterXYWH instance."""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """Initialize tracks with detections."""
        self._frame_count += 1

        if len(dets) == 0:
            return []

        # Convert to numpy
        if isinstance(dets, torch.Tensor):
            dets = dets.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(cls, torch.Tensor):
            cls = cls.cpu().numpy()

        scores = np.asarray(scores).flatten()
        cls = np.asarray(cls).flatten()

        # Take first 4 columns
        if dets.ndim == 2 and dets.shape[1] > 4:
            dets_4col = dets[:, :4].copy()
        else:
            dets_4col = dets.copy()

        # Detect format and convert to xyxy for ReID
        if len(dets_4col) > 0:
            sample_box = dets_4col[0]
            is_xyxy = (sample_box[2] > sample_box[0] and
                      sample_box[3] > sample_box[1] and
                      (sample_box[2] - sample_box[0]) > 5)

            if is_xyxy:
                dets_for_reid = dets_4col
                format_name = "xyxy"
            else:
                # Convert tlwh to xyxy
                x, y, w, h = dets_4col[:, 0], dets_4col[:, 1], dets_4col[:, 2], dets_4col[:, 3]
                dets_for_reid = np.stack([x, y, x + w, y + h], axis=1)
                format_name = "tlwh"
        else:
            dets_for_reid = dets_4col
            format_name = "unknown"

        # Debug output (first 3 frames)
        if self._frame_count <= 3:
            print(f"\n🔍 [Frame {self._frame_count}] init_track")
            print(f"  Boxes: {len(dets_4col)} ({format_name})")
            print(f"  First 3 boxes: {dets_4col[:3]}")
            if format_name == "tlwh":
                print(f"  Converted: {dets_for_reid[:3]}")
            print(f"  Scores: [{scores.min():.3f}, {scores.max():.3f}]")
            if img is not None:
                print(f"  Image: {img.shape}")

        # Extract ReID features
        features = None
        if self.args.with_reid and self.encoder is not None and img is not None:
            try:
                features = self.encoder.inference(img, dets_for_reid)

                if self._frame_count <= 3:
                    print(f"  ✅ ReID features: {features.shape}")
                    non_zero = np.count_nonzero(features.sum(axis=1))
                    print(f"     Non-zero: {non_zero}/{len(features)}")
                    for i in range(min(3, len(features))):
                        norm = np.linalg.norm(features[i])
                        print(f"     Feature {i}: L2={norm:.4f}")

                if len(features) != len(dets_4col):
                    if not self._debug_printed:
                        print(f"⚠️  Feature count mismatch")
                        self._debug_printed = True
                    features = None

            except Exception as e:
                if not self._debug_printed:
                    print(f"⚠️  ReID failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self._debug_printed = True
                features = None

        # Create BOTrack objects
        tracks = []
        for i, (box, score, c) in enumerate(zip(dets_4col, scores, cls)):
            feat = None
            if features is not None:
                feat = features[i]
                if np.linalg.norm(feat) < 1e-6:
                    feat = None

            try:
                track = BOTrack(
                    tlwh=box,
                    score=float(score),
                    cls=int(c),
                    feat=feat
                )
                tracks.append(track)
            except Exception as e:
                if self._frame_count <= 3 and i < 3:
                    print(f"  ❌ Track {i} failed: {e}")
                continue

        if self._frame_count <= 3:
            tracks_with_feat = sum(1 for t in tracks if t.curr_feat is not None)
            print(f"  ✅ Created {len(tracks)}/{len(dets_4col)} tracks")
            print(f"  📊 With features: {tracks_with_feat}/{len(tracks)}\n")

        return tracks

    def get_dists(self, tracks, detections):
        """Calculate distance matrix with IoU and ReID."""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        # ReID matching
        if self.args.with_reid and self.encoder is not None:
            try:
                tracks_valid = sum(
                    1 for t in tracks
                    if hasattr(t, 'curr_feat') and t.curr_feat is not None
                    and np.linalg.norm(t.curr_feat) > 1e-6
                )

                dets_valid = sum(
                    1 for d in detections
                    if hasattr(d, 'curr_feat') and d.curr_feat is not None
                    and np.linalg.norm(d.curr_feat) > 1e-6
                )

                if tracks_valid > 0 and dets_valid > 0:
                    emb_dists = matching.embedding_distance(tracks, detections) / 2.0

                    # 🔍 添加调试：记录 ReID 改变了多少距离
                    iou_only = dists.copy()
                    emb_dists[emb_dists > self.appearance_thresh] = 1.0
                    emb_dists[dists_mask] = 1.0
                    dists = np.minimum(dists, emb_dists)

                    # 统计有多少匹配被 ReID 改善
                    changed = np.sum(dists < iou_only)
                    if changed > 0:
                        print(f"🔍 Frame: ReID 改善了 {changed} 个匹配")

            except Exception:
                pass

        return dists

    def multi_predict(self, tracks):
        """Predict mean and covariance for multiple tracks."""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """Reset tracker to initial state."""
        super().reset()
        self.gmc.reset_params()
        self._debug_printed = False
        self._frame_count = 0

        if self.encoder is not None and hasattr(self.encoder, 'print_stats'):
            print("\n" + "=" * 60)
            print("Tracking finished - ReID stats")
            print("=" * 60)
            self.encoder.print_stats()