#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from cv_bridge import CvBridge
import numpy as np

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray
from .lib import camera_perception_func_lib as CPFL


#---------------Variable Setting---------------
SUB_TOPIC_NAME = "detections"          # Subscribe할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"    # Publish할 토픽 이름
ROI_IMAGE_TOPIC_NAME = "roi_image"     # ROI/BEV 이미지 퍼블리시 토픽
SHOW_IMAGE = True                      # 디버그 창 표시 여부
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        # 기본 파라미터
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        # BEV(버드아이뷰) 변환 파라미터 — 너가 올린 알고리즘 방식
        # 입력 이미지 해상도가 640x480이 아니어도 "비율"로 안전하게 적용됨
        self.bev_roi_h = int(self.declare_parameter('bev_roi_height', 200).value)     # ROI 높이(px)
        self.bev_roi_w = int(self.declare_parameter('bev_roi_width', 640).value)      # ROI 폭(px)
        self.bev_roi_y0 = int(self.declare_parameter('bev_roi_y0', 280).value)        # ROI 시작 y(px)
        self.bev_width_margin = int(self.declare_parameter('bev_width_margin', 215).value)  # 하단 벌림량(px)
        self.bev_clip_roi_minus = int(self.declare_parameter('bev_clip_roi_minus', 10).value) # ROI 높이 보정

        # 아래→위로 자르기(컷) 옵션 (비율) — GUI 없이 파라미터로 제어
        self.cut_enable = bool(self.declare_parameter('bev_cut_enable', True).value)
        self.cut_ratio  = float(self.declare_parameter('bev_cut_ratio', 0.00).value)  # 0.0~1.0

        # YOLO 클래스명 (필요 시 바꿀 수 있게)
        self.mask_class = self.declare_parameter('mask_class', 'track').value

        self.cv_bridge = CvBridge()

        # QoS
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # sub/pub
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher  = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

        self.get_logger().info(
            f"BEV params: roi(y0={self.bev_roi_y0}, h={self.bev_roi_h}, w={self.bev_roi_w}), "
            f"wm={self.bev_width_margin}, clip_minus={self.bev_clip_roi_minus}, "
            f"cut_enable={self.cut_enable}, cut_ratio={self.cut_ratio}"
        )

    # --------- BEV 변환(너가 준 방식) 함수 ----------
    def bev_warp_like_preproc(self, edge_img: np.ndarray) -> np.ndarray:
        """
        edge_img: mono8 (0/255)
        1) ROI(y0, h, w) 잘라오고
        2) src=직사각형, dst=하단 폭을 wm만큼 안쪽으로 넣은 사다리꼴
        3) 퍼스펙티브 워핑
        4) (옵션) 아래→위로 cut_ratio만큼 컷
        return: warped(cropped) mono8
        """
        h, w = edge_img.shape[:2]

        # ROI 사양 확정
        y0 = max(0, min(self.bev_roi_y0, h - 2))
        roi_h = max(2, min(self.bev_roi_h - self.bev_clip_roi_minus, h - y0))
        x0 = 0
        roi_w = max(2, min(self.bev_roi_w, w))

        roi = edge_img[y0:y0 + roi_h, x0:x0 + roi_w].copy()

        # 호모그래피: 직사각형 -> 하단이 안쪽으로 들어간 사다리꼴
        wm = int(min(self.bev_width_margin, (roi_w - 2) // 2))
        src = np.float32([[0, 0], [0, roi_h], [roi_w, 0], [roi_w, roi_h]])
        dst = np.float32([[0, 0], [wm, roi_h], [roi_w, 0], [roi_w - wm, roi_h]])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(roi, M, (roi_w, roi_h), flags=cv2.INTER_NEAREST)

        # 아래→위 컷
        if self.cut_enable and 0.0 < self.cut_ratio <= 1.0:
            cut_px = int(round(self.cut_ratio * roi_h))
            keep_h = max(1, roi_h - cut_px)
            warped = warped[:keep_h, :]

        return warped

    # ------------- 콜백 ---------------
    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        # 1) YOLO 마스크 -> 엣지 (mono8)
        lane_edge_image = CPFL.draw_edges(detection_msg, cls_name=self.mask_class, color=255)  # (H,W) mono

        # 2) (변경점) 버드아이뷰: 너가 준 알고리즘 방식으로 변환
        bev_image = self.bev_warp_like_preproc(lane_edge_image)

        # 3) 디버그 표시
        if self.show_image:
            cv2.imshow('edge_image', lane_edge_image)
            cv2.imshow('bev_image', bev_image)
            cv2.waitKey(1)

        # 4) 퍼블리시 (mono8)
        try:
            bev_u8 = cv2.convertScaleAbs(bev_image)  # 안전 변환
            bev_msg = self.cv_bridge.cv2_to_imgmsg(bev_u8, encoding="mono8")
            bev_msg.header.stamp = detection_msg.header.stamp
            self.roi_image_publisher.publish(bev_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert/publish BEV image: {e}")
            return

        # 5) 이하 로직(기울기/중앙선/타겟포인트)은 기존과 동일하게 BEV 결과를 사용
        grad = CPFL.dominant_gradient(bev_image, theta_limit=70)

        target_points = []
        for target_point_y in range(5, 155, 50):
            target_point_x = CPFL.get_lane_center(
                bev_image,
                detection_height=target_point_y,
                detection_thickness=10,
                road_gradient=grad,
                lane_width=300  # BEV 스케일 바뀌면 300→(필요 시) 350~420으로 튜닝
            )
            tp = TargetPoint()
            tp.target_x = int(round(target_point_x))
            tp.target_y = int(round(target_point_y))
            target_points.append(tp)

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points
        self.publisher.publish(lane)


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
