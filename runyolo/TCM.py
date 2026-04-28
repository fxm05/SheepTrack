# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH


class BOTrack(STrack):
    """
    带观测校正的扩展版BOTrack类，用于处理长遮挡和加速度异常情况。

    新增特性：
        - 观测历史记录
        - 加速度异常检测
        - 观测校正机制
    """

    shared_kalman = KalmanFilterXYWH()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        """初始化BOTrack对象，增加观测校正相关参数。"""
        super().__init__(tlwh, score, cls)

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        # ========== 观测校正相关参数 ==========
        self.observation_history = deque([], maxlen=50)  # 观测历史
        self.velocity_history = deque([], maxlen=30)  # 速度历史
        self.acceleration_history = deque([], maxlen=20)  # 加速度历史

        # 阈值参数
        self.max_occlusion_frames = 30  # 长遮挡阈值
        self.acceleration_threshold = 15.0  # 加速度异常阈值
        self.observation_correction_window = 10  # 观测校正窗口大小

        # 状态标记
        self.occlusion_count = 0  # 遮挡帧计数
        self.need_correction = False  # 是否需要校正
        self.last_observation = None  # 上一次观测

    def update_features(self, feat):
        """更新特征向量并应用指数移动平均平滑。"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def calculate_velocity(self, current_pos, previous_pos):
        """计算速度向量。"""
        if previous_pos is None:
            return np.zeros(2)
        return current_pos[:2] - previous_pos[:2]

    def calculate_acceleration(self, current_velocity, previous_velocity):
        """计算加速度向量。"""
        if previous_velocity is None or len(previous_velocity) == 0:
            return np.zeros(2)
        return current_velocity - previous_velocity

    def detect_acceleration_anomaly(self):
        """检测加速度异常。"""
        if len(self.acceleration_history) < 3:
            return False

        # 计算最近加速度的范数
        recent_acc = np.array(list(self.acceleration_history)[-3:])
        acc_magnitude = np.linalg.norm(recent_acc, axis=1)

        # 如果加速度突然增大，判定为异常
        if np.max(acc_magnitude) > self.acceleration_threshold:
            return True

        # 检测加速度方向突变
        if len(self.acceleration_history) >= 2:
            acc_diff = np.linalg.norm(recent_acc[-1] - recent_acc[-2])
            if acc_diff > self.acceleration_threshold * 0.8:
                return True

        return False

    def observation_based_correction(self):
        """
        基于观测历史的校正方法。
        当检测到长遮挡或加速度异常时，使用历史观测数据重构轨迹。
        """
        if len(self.observation_history) < self.observation_correction_window:
            return

        # 获取最近的观测窗口
        recent_obs = list(self.observation_history)[-self.observation_correction_window:]

        # 使用线性回归拟合轨迹
        positions = np.array([obs[:2] for obs in recent_obs])  # 取中心点位置
        time_steps = np.arange(len(positions))

        # 对x和y分别进行线性拟合
        if len(positions) >= 3:
            # 使用多项式拟合（2次）来捕捉非线性运动
            poly_x = np.polyfit(time_steps, positions[:, 0], min(2, len(positions) - 1))
            poly_y = np.polyfit(time_steps, positions[:, 1], min(2, len(positions) - 1))

            # 预测当前位置
            predicted_x = np.polyval(poly_x, len(positions))
            predicted_y = np.polyval(poly_y, len(positions))

            # 更新mean状态（使用预测位置校正）
            self.mean[0] = predicted_x
            self.mean[1] = predicted_y

            # 重新计算速度
            if len(positions) >= 2:
                velocity_x = positions[-1, 0] - positions[-2, 0]
                velocity_y = positions[-1, 1] - positions[-2, 1]
                self.mean[4] = velocity_x
                self.mean[5] = velocity_y

            # 增加协方差，反映不确定性
            self.covariance[:2, :2] *= 1.5

    def predict(self):
        """预测目标未来状态，集成观测校正机制。"""
        mean_state = self.mean.copy()

        # 如果不是跟踪状态，重置速度
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
            self.occlusion_count += 1
        else:
            self.occlusion_count = 0

        # 检测是否需要观测校正
        is_long_occlusion = self.occlusion_count > self.max_occlusion_frames
        is_acceleration_anomaly = self.detect_acceleration_anomaly()

        if is_long_occlusion or is_acceleration_anomaly:
            self.need_correction = True

        # 执行卡尔曼滤波预测
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        # 如果需要校正，应用观测校正
        if self.need_correction:
            self.observation_based_correction()
            self.need_correction = False

    def update(self, new_track, frame_id):
        """更新轨迹，同时更新观测历史。"""
        # 更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # 记录观测
        current_pos = new_track.tlwh_to_xywh(new_track.tlwh)
        self.observation_history.append(current_pos.copy())

        # 计算速度
        if self.last_observation is not None:
            velocity = self.calculate_velocity(current_pos, self.last_observation)
            self.velocity_history.append(velocity)

            # 计算加速度
            if len(self.velocity_history) >= 2:
                acceleration = self.calculate_acceleration(
                    self.velocity_history[-1],
                    self.velocity_history[-2]
                )
                self.acceleration_history.append(acceleration)

        self.last_observation = current_pos

        # 调用父类update
        super().update(new_track, frame_id)

        # 重置遮挡计数
        self.occlusion_count = 0

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活轨迹，更新观测历史。"""
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # 记录观测
        current_pos = new_track.tlwh_to_xywh(new_track.tlwh)
        self.observation_history.append(current_pos.copy())
        self.last_observation = current_pos

        super().re_activate(new_track, frame_id, new_id)

        # 重置状态
        self.occlusion_count = 0
        self.need_correction = False

    @property
    def tlwh(self):
        """返回当前边界框位置（top left x, top left y, width, height）格式。"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        """使用共享卡尔曼滤波器预测多个目标轨迹的均值和协方差。"""
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
        """将tlwh边界框坐标转换为xywh格式。"""
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """将边界框从tlwh格式转换为xywh格式。"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    """
    带观测校正的BOTSORT跟踪器，增强对长遮挡和加速度异常的处理能力。
    """

    def __init__(self, args, frame_rate=30):
        """初始化BOTSORT跟踪器。"""
        super().__init__(args, frame_rate)

        # ReID模块
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = None

        self.gmc = GMC(method=args.gmc_method)

    def get_kalmanfilter(self):
        """返回KalmanFilterXYWH实例。"""
        return KalmanFilterXYWH()

    def init_track(self, dets, scores, cls, img=None):
        """使用检测框、分数、类别标签和可选的ReID特征初始化目标轨迹。"""
        if len(dets) == 0:
            return []

        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BOTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]
        else:
            return [BOTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]

    def get_dists(self, tracks, detections):
        """使用IoU和可选的ReID嵌入计算轨迹与检测之间的距离。"""
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > self.proximity_thresh

        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)

        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)

        return dists

    def multi_predict(self, tracks):
        """使用共享卡尔曼滤波器预测多个目标的均值和协方差。"""
        BOTrack.multi_predict(tracks)

    def reset(self):
        """重置BOTSORT跟踪器到初始状态。"""
        super().reset()
        self.gmc.reset_params()