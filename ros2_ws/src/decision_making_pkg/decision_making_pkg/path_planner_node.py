#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.interpolate import CubicSpline

#---------------Variable Setting---------------
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"   # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점
SUB_BEV_IMAGE_TOPIC = "roi_image"          # BEV/ROI 이미지 토픽 (mono8 예상)
PUB_TOPIC_NAME = "path_planning_result"    # 경로 계획 결과 퍼블리시
CAR_CENTER_POINT = (320, 179)              # BEV 이미지 좌표계에서 차량 앞 범퍼 중심 픽셀
SHOW_WINDOW = True                         # OpenCV 창 표시 여부
WINDOW_NAME = "Path Planning (overlay)"
#----------------------------------------------


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # 파라미터 선언
        self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.sub_bev_topic  = self.declare_parameter('sub_bev_topic', SUB_BEV_IMAGE_TOPIC).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point = tuple(self.declare_parameter('car_center_point', CAR_CENTER_POINT).value)
        self.show_window = bool(self.declare_parameter('show_window', SHOW_WINDOW).value)

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 브릿지/버퍼
        self.bridge = CvBridge()
        self.last_bev_img = None  # 최신 BEV/ROI 이미지 버퍼

        # 구독/퍼블리시
        self.lane_sub = self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)
        self.bev_sub  = self.create_subscription(Image, self.sub_bev_topic,  self.bev_image_callback, self.qos_profile)
        self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

        self.get_logger().info(
            f"SUB lane: {self.sub_lane_topic}, SUB bev: {self.sub_bev_topic} → PUB: {self.pub_topic} (show_window={self.show_window})"
        )

    # BEV/ROI 이미지 콜백
    def bev_image_callback(self, msg: Image):
        try:
            # roi_image는 mono8일 가능성이 높음. 실패 시 원본 encoding 사용
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                # 색상 오버레이 위해 BGR로 변환
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            except Exception:
                # bgr8이면 그대로
                img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_bev_img = img_bgr
        except Exception as e:
            self.get_logger().warn(f"cv_bridge bev: {e}")

    # LaneInfo 콜백
    def lane_callback(self, msg: LaneInfo):
        target_points = msg.target_points
        if len(target_points) < 3:
            return

        # 경로 계획 및 퍼블리시
        x_points, y_points, x_new, y_new = self.plan_path_arrays(target_points)

        # 퍼블리시
        path_msg = PathPlanningResult()
        path_msg.x_points = list(x_new)
        path_msg.y_points = list(y_new)
        self.publisher.publish(path_msg)

        # 시각화 (OpenCV)
        if self.show_window:
            self.draw_and_show(x_points, y_points, x_new, y_new)

    def plan_path_arrays(self, target_points):
        # TargetPoint -> (x,y)
        xs, ys = zip(*[(tp.target_x, tp.target_y) for tp in target_points])

        # 차량 중심점 추가
        xs = list(xs); ys = list(ys)
        xs.append(self.car_center_point[0]); ys.append(self.car_center_point[1])

        # y 기준 정렬 (영상 좌표계: y 증가 = 아래로)
        ys, xs = zip(*sorted(zip(ys, xs), key=lambda p: p[0]))

        # 스플라인
        if len(ys) < 3:
            # 안전장치 (이상 케이스)
            y_new = np.array(ys, dtype=float)
            x_new = np.array(xs, dtype=float)
        else:
            cs = CubicSpline(ys, xs, bc_type='natural')
            y_new = np.linspace(min(ys), max(ys), 100)
            x_new = cs(y_new)

        self.get_logger().info(f"Planning path with {len(ys)} pts → {len(y_new)} samples")
        return np.array(xs), np.array(ys), np.array(x_new), np.array(y_new)

    def draw_and_show(self, x_pts, y_pts, x_new, y_new):
        # 배경 이미지 준비
        if self.last_bev_img is not None:
            canvas = self.last_bev_img.copy()
            H, W = canvas.shape[:2]
        else:
            # 이미지가 아직 안 왔으면 적당한 캔버스 생성
            W = max(int(np.max([*x_pts, *x_new]) + 20), 640)
            H = max(int(np.max([*y_pts, *y_new]) + 20), 200)
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # 안전 클리핑
        def _clip_xy(x, y):
            x = int(np.clip(x, 0, W-1))
            y = int(np.clip(y, 0, H-1))
            return x, y

        # 타겟 포인트(빨강)
        for (x, y) in zip(x_pts, y_pts):
            u, v = _clip_xy(x, y)
            cv2.circle(canvas, (u, v), 3, (0, 0, 255), -1)

        # 스플라인 경로(초록 Polyline)
        pts = np.stack([x_new, y_new], axis=1).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H-1)
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)

        # 차량 중심(시안)
        u, v = _clip_xy(*self.car_center_point)
        cv2.circle(canvas, (u, v), 4, (255, 255, 0), -1)
        cv2.putText(canvas, "CAR", (u+6, v-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1, cv2.LINE_AA)

        # 창 표시
        cv2.imshow(WINDOW_NAME, canvas)
        # 'q' 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
