import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

#----------------------------------------------
TIMER = 0.05
IMAGE_CENTER_X = 320.0  # 이미지 중앙(640px 기준)
KP = 0.05           # PID 비례 게인
BASE_SPEED = 255        # 기본 속도

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_data = None
        self.traffic_light_data = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(
            DetectionArray, SUB_DETECTION_TOPIC_NAME, self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(
            PathPlanningResult, SUB_PATH_TOPIC_NAME, self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(
            String, SUB_TRAFFIC_LIGHT_TOPIC_NAME, self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(
            Bool, SUB_LIDAR_OBSTACLE_TOPIC_NAME, self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(MotionCommand, PUB_TOPIC_NAME, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(TIMER, self.timer_callback)

    # === 콜백 ===
    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    # === 메인 로직 ===
    def timer_callback(self):

        self.get_logger().info(f"[DEBUG] lidar={self.lidar_data}, traffic={self.traffic_light_data}, path={self.path_data is not None}")

        # 기본값 초기화
        lane_center_x, lane_center_y = None, None

        # 1️⃣ 장애물 감지 시 정지
        if self.lidar_data is not None and self.lidar_data.data is True:
            self.steering_command = 0
            self.left_speed_command = 0
            self.right_speed_command = 0

        # 2️⃣ 빨간불 감지 시 정지
        elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
            if self.detection_data:
                for detection in self.detection_data.detections:
                    if detection.class_name == 'traffic_light':
                        y_max = int(detection.bbox.center.position.y + detection.bbox.size.y / 2)
                        if y_max < 150:
                            self.steering_command = 0
                            self.left_speed_command = 0
                            self.right_speed_command = 0

        # 3️⃣ 경로 PID 조향
        else:
            if self.path_data is None:
                # 경로 없으면 정지
                self.steering_command = 0
                self.left_speed_command = 0
                self.right_speed_command = 0
            else:
                # path 중간점 선택
                mid_idx = len(self.path_data) // 2
                lane_center_x, lane_center_y = self.path_data[mid_idx]
                dx = lane_center_x - IMAGE_CENTER_X  # 이미지 중심 대비 오프셋

                self.get_logger().info(
                    f"[DEBUG] mid_idx={mid_idx}/{len(self.path_data)}, "
                    f"lane_center_x={lane_center_x:.2f}, lane_center_y={lane_center_y:.2f}, dx={dx:.2f}"
                )

                # PID 조향 (단순 비례)
                raw_steering = KP * dx
                self.steering_command = int(max(min(raw_steering, 7), -7))

                # 속도는 일정값 유지
                self.left_speed_command = BASE_SPEED
                self.right_speed_command = BASE_SPEED

        # 4️⃣ 최종 명령 퍼블리시
        motion_command_msg = MotionCommand()
        motion_command_msg.steering = self.steering_command
        motion_command_msg.left_speed = int(self.left_speed_command)
        motion_command_msg.right_speed = int(self.right_speed_command)
        self.publisher.publish(motion_command_msg)

        self.get_logger().info(
            f"steering: {self.steering_command}, left_speed: {self.left_speed_command}, right_speed: {self.right_speed_command}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
