'''Simple and lightweight module for working with RPLidar rangefinder scanners.

Usage example:

>>> from rplidar import RPLidar
>>> lidar = RPLidar('/dev/ttyUSB0')
>>>
>>> info = lidar.get_info()
>>> print(info)
>>>
>>> health = lidar.get_health()
>>> print(health)
>>>
>>> for i, scan in enumerate(lidar.iter_scans()):
...  print('%d: Got %d measures' % (i, len(scan)))
...  if i > 10:
...   break
...
>>> lidar.stop()
>>> lidar.stop_motor()
>>> lidar.disconnect()

For additional information please refer to the RPLidar class documentation.
'''
import logging
import sys
import time
import codecs
import serial
import struct

from collections import namedtuple


message = """
 ____    ___   ____    ____  
|  _ \  / _ \ / ___|  |___ \ 
| |_) || | | |\___ \    __) |
|  _ < | |_| | ___) |  / __/ 
|_| \_\ \___/ |____/  |_____|
                             
    _            _                 
   / \    _   _ | |_   ___         
  / _ \  | | | || __| / _ \  _____ 
 / ___ \ | |_| || |_ | (_) ||_____|
/_/   \_\ \__,_| \__| \___/        
                                   
__     __       _      _        _       
\ \   / /  ___ | |__  (_)  ___ | |  ___ 
 \ \ / /  / _ \| '_ \ | | / __|| | / _ \
  \ V /  |  __/| | | || || (__ | ||  __/
   \_/    \___||_| |_||_| \___||_| \___|  

"""
print(message)

print("ROS2 기반 자율주행 설계 및 구현")
print("Sungkyunkwan University Automation Lab.")

print("------------------Authors------------------")
print("Hyeong-Keun Hong <whaihong@g.skku.edu>")
print("Jinsun Lee <with23skku@g.skku.edu>")
print("Siwoo Lee <edenlee@g.skku.edu>")
print("Jae-Wook Jeon <jwjeon@skku.edu>")
print("------------------------------------------")




SYNC_BYTE = b'\xA5'
SYNC_BYTE2 = b'\x5A'

GET_INFO_BYTE = b'\x50'
GET_HEALTH_BYTE = b'\x52'

STOP_BYTE = b'\x25'
RESET_BYTE = b'\x40'

_SCAN_TYPE = {
    'normal': {'byte': b'\x20', 'response': 129, 'size': 5},
    'force': {'byte': b'\x21', 'response': 129, 'size': 5},
    'express': {'byte': b'\x82', 'response': 130, 'size': 84},
}

DESCRIPTOR_LEN = 7
INFO_LEN = 20
HEALTH_LEN = 3

INFO_TYPE = 4
HEALTH_TYPE = 6

# Constants & Command to start A2 motor
MAX_MOTOR_PWM = 1023
DEFAULT_MOTOR_PWM = 660
SET_PWM_BYTE = b'\xF0'

_HEALTH_STATUSES = {
    0: 'Good',
    1: 'Warning',
    2: 'Error',
}

def rotate_lidar_data (msg, offset = 0):
    
    offset = int(offset)
    if offset < 0 or offset >= 360:
        raise ValueError("offset must be between 0 and 359")
    
    msg.ranges = msg.ranges[offset:] + msg.ranges[:offset]
    msg.intensities = msg.intensities[offset:] + msg.intensities[:offset]

    return msg



def flip_lidar_data(msg, pivot_angle):
    
    pivot_angle = int(pivot_angle)
    if pivot_angle < 0 or pivot_angle >= 360:
        raise ValueError("pivot_angle must be between 0 and 359")
    
    length = len(msg.ranges)
    flipped_ranges = [0] * length
    flipped_intensities = [0]*length
    for i in range(length):
        new_angle = (2 * pivot_angle - i) % length
        flipped_ranges[new_angle] = msg.ranges[i]
        flipped_intensities[new_angle] = msg.intensities[i]
    
    msg.ranges = flipped_ranges
    msg.intensities = flipped_intensities

    return msg

def detect_object(ranges, start_angle, end_angle, range_min, range_max):
    
    # ranges는 라이다 센서값 입력                                                        

    # 각도 범위 지정
    # 예시 1) 
    # start_angle을 355도로, end_angle을 4도로 설정하면, 
    # 355도에서 4도까지의 모든 각도(355, 356, 357, 358, 359, 0, 1, 2, 3, 4도)가 포함.
    # 
    # 예시 2)
    # start_angle을 0도로, end_angle을 30도로 설정하면, 
    # 0도에서 30도까지의 모든 각도(0, 1, 2, ..., 30도)가 포함.
    # 
    # 예시 3)
    # start_angle을 180도로, end_angle을 190도로 설정하면, 
    # 180도에서 190도까지의 모든 각도(180, 181, 182, ..., 190도)가 포함. 

    # 거리범위 지정 
    # range_min보다 크거나 같고, range_max보다 작거나 같은 거리값을 포함.

    # 각도범위 and 거리범위에 라이다 센서값이 존재하면 True, 아니면 False 리턴. 
    
    num_readings = len(ranges)
    
    if start_angle > end_angle:
        end_angle += num_readings
    
    for i in range(start_angle, end_angle + 1):
        index = i % num_readings
        if range_min <= ranges[index] <= range_max:
            return True
    return False
"""
import numpy as np
from sklearn.cluster import DBSCAN

def detect_object(ranges, start_angle, end_angle, range_min, range_max, eps=0.15, min_samples=5) :
    
    ranges: 라이다 거리 배열
    start_angle, end_angle: 탐색할 각도 범위 (도 단위)
    range_min, range_max: 거리 필터링 범위 (m 단위)
    eps: DBSCAN 클러스터 간 거리 임계값 (m)
    min_samples: 클러스터로 인정될 최소 포인트 수
    

    num_readings = len(ranges)
    angle_increment = 360.0 / num_readings  # degree per step

    # 유효한 ROI 포인트 추출
    points = []
    if start_angle > end_angle:
        end_angle += num_readings

    for i in range(start_angle, end_angle + 1):
        idx = i % num_readings
        dist = ranges[idx]
        if range_min <= dist <= range_max:
            angle_deg = (idx * angle_increment) % 360
            angle_rad = np.deg2rad(angle_deg)
            x = dist * np.cos(angle_rad)
            y = dist * np.sin(angle_rad)
            points.append([x, y])

    if len(points) == 0:
        return False

    # DBSCAN 클러스터링
    points_np = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
    labels = db.labels_

    # 클러스터 라벨에서 이상치 제외하고 클러스터 개수 확인
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if num_clusters > 0 : return True
    else : return False
"""
class StabilityDetector:
    def __init__(self, consec_count):
        self.consec_count = consec_count
        self.detection_history = []
        self.current_state = False

    def check_consecutive_detections(self, detected):
        """ 감지가 연속으로 일어나는지 확인하는 함수

        Parameters:
        detected (bool): 현재 감지된 여부

        Returns:
        bool: 감지 상태가 변했는지 여부
        """
        self.detection_history.append(detected)
        if len(self.detection_history) > self.consec_count:
            self.detection_history.pop(0)

        if self.current_state and self.detection_history.count(False) >= self.consec_count:
            self.current_state = False
        elif not self.current_state and self.detection_history.count(True) >= self.consec_count:
            self.current_state = True

        return self.current_state

class RPLidarException(Exception):
    '''Basic exception class for RPLidar'''


def _b2i(byte):
    '''Converts byte to integer (for Python 2 compatability)'''
    return byte if int(sys.version[0]) == 3 else ord(byte)


def _showhex(signal):
    '''Converts string bytes to hex representation (useful for debugging)'''
    return [format(_b2i(b), '#02x') for b in signal]


def _process_scan(raw):
    '''Processes input raw data and returns measurement data'''
    new_scan = bool(_b2i(raw[0]) & 0b1)
    inversed_new_scan = bool((_b2i(raw[0]) >> 1) & 0b1)
    quality = _b2i(raw[0]) >> 2
    if new_scan == inversed_new_scan:
        raise RPLidarException('New scan flags mismatch')
    check_bit = _b2i(raw[1]) & 0b1
    if check_bit != 1:
        raise RPLidarException('Check bit not equal to 1')
    angle = ((_b2i(raw[1]) >> 1) + (_b2i(raw[2]) << 7)) / 64.
    distance = (_b2i(raw[3]) + (_b2i(raw[4]) << 8)) / 4.
    return new_scan, quality, angle, distance


def _process_express_scan(data, new_angle, trame):
    new_scan = (new_angle < data.start_angle) & (trame == 1)
    angle = (data.start_angle + (
            (new_angle - data.start_angle) % 360
            )/32*trame - data.angle[trame-1]) % 360
    distance = data.distance[trame-1]
    return new_scan, None, angle, distance


class RPLidar(object):
    '''Class for communicating with RPLidar rangefinder scanners'''

    def __init__(self, port, baudrate=115200, timeout=1, logger=None):
        '''Initilize RPLidar object for communicating with the sensor.

        Parameters
        ----------
        port : str
            Serial port name to which sensor is connected
        baudrate : int, optional
            Baudrate for serial connection (the default is 115200)
        timeout : float, optional
            Serial port connection timeout in seconds (the default is 1)
        logger : logging.Logger instance, optional
            Logger instance, if none is provided new instance is created
        '''
        self._serial = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._motor_speed = DEFAULT_MOTOR_PWM
        self.scanning = [False, 0, 'normal']
        self.express_trame = 32
        self.express_data = False
        self.motor_running = None
        if logger is None:
            logger = logging.getLogger('rplidar')
        self.logger = logger
        self.connect()

    def connect(self):
        '''Connects to the serial port with the name `self.port`. If it was
        connected to another serial port disconnects from it first.'''
        if self._serial is not None:
            self.disconnect()
        try:
            self._serial = serial.Serial(
                self.port, self.baudrate,
                parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout)
        except serial.SerialException as err:
            raise RPLidarException('Failed to connect to the sensor '
                                   'due to: %s' % err)

    def disconnect(self):
        '''Disconnects from the serial port'''
        if self._serial is None:
            return
        self._serial.close()

    def _set_pwm(self, pwm):
        payload = struct.pack("<H", pwm)
        self._send_payload_cmd(SET_PWM_BYTE, payload)

    @property
    def motor_speed(self):
        return self._motor_speed

    @motor_speed.setter
    def motor_speed(self, pwm):
        assert(0 <= pwm <= MAX_MOTOR_PWM)
        self._motor_speed = pwm
        if self.motor_running:
            self._set_pwm(self._motor_speed)

    def start_motor(self):
        '''Starts sensor motor'''
        self.logger.info('Starting motor')
        # For A1
        self._serial.setDTR(False)

        # For A2
        self._set_pwm(self._motor_speed)
        self.motor_running = True

    def stop_motor(self):
        '''Stops sensor motor'''
        self.logger.info('Stoping motor')
        # For A2
        self._set_pwm(0)
        time.sleep(.001)
        # For A1
        self._serial.setDTR(True)
        self.motor_running = False

    def _send_payload_cmd(self, cmd, payload):
        '''Sends `cmd` command with `payload` to the sensor'''
        size = struct.pack('B', len(payload))
        req = SYNC_BYTE + cmd + size + payload
        checksum = 0
        for v in struct.unpack('B'*len(req), req):
            checksum ^= v
        req += struct.pack('B', checksum)
        self._serial.write(req)
        self.logger.debug('Command sent: %s' % _showhex(req))

    def _send_cmd(self, cmd):
        '''Sends `cmd` command to the sensor'''
        req = SYNC_BYTE + cmd
        self._serial.write(req)
        self.logger.debug('Command sent: %s' % _showhex(req))

    def _read_descriptor(self):
        '''Reads descriptor packet'''
        descriptor = self._serial.read(DESCRIPTOR_LEN)
        self.logger.debug('Received descriptor: %s', _showhex(descriptor))
        if len(descriptor) != DESCRIPTOR_LEN:
            raise RPLidarException('Descriptor length mismatch')
        elif not descriptor.startswith(SYNC_BYTE + SYNC_BYTE2):
            raise RPLidarException('Incorrect descriptor starting bytes')
        is_single = _b2i(descriptor[-2]) == 0
        return _b2i(descriptor[2]), is_single, _b2i(descriptor[-1])

    def _read_response(self, dsize):
        '''Reads response packet with length of `dsize` bytes'''
        self.logger.debug('Trying to read response: %d bytes', dsize)
        while self._serial.inWaiting() < dsize:
            time.sleep(0.001)
        data = self._serial.read(dsize)
        self.logger.debug('Received data: %s', _showhex(data))
        return data

    def get_info(self):
        '''Get device information

        Returns
        -------
        dict
            Dictionary with the sensor information
        '''
        if self._serial.inWaiting() > 0:
            return ('Data in buffer, you can\'t have info ! '
                    'Run clean_input() to emptied the buffer.')
        self._send_cmd(GET_INFO_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != INFO_LEN:
            raise RPLidarException('Wrong get_info reply length')
        if not is_single:
            raise RPLidarException('Not a single response mode')
        if dtype != INFO_TYPE:
            raise RPLidarException('Wrong response data type')
        raw = self._read_response(dsize)
        serialnumber = codecs.encode(raw[4:], 'hex').upper()
        serialnumber = codecs.decode(serialnumber, 'ascii')
        data = {
            'model': _b2i(raw[0]),
            'firmware': (_b2i(raw[2]), _b2i(raw[1])),
            'hardware': _b2i(raw[3]),
            'serialnumber': serialnumber,
        }
        return data

    def get_health(self):
        '''Get device health state. When the core system detects some
        potential risk that may cause hardware failure in the future,
        the returned status value will be 'Warning'. But sensor can still work
        as normal. When sensor is in the Protection Stop state, the returned
        status value will be 'Error'. In case of warning or error statuses
        non-zero error code will be returned.

        Returns
        -------
        status : str
            'Good', 'Warning' or 'Error' statuses
        error_code : int
            The related error code that caused a warning/error.
        '''
        if self._serial.inWaiting() > 0:
            return ('Data in buffer, you can\'t have info ! '
                    'Run clean_input() to emptied the buffer.')
        self.logger.info('Asking for health')
        self._send_cmd(GET_HEALTH_BYTE)
        dsize, is_single, dtype = self._read_descriptor()
        if dsize != HEALTH_LEN:
            raise RPLidarException('Wrong get_info reply length')
        if not is_single:
            raise RPLidarException('Not a single response mode')
        if dtype != HEALTH_TYPE:
            raise RPLidarException('Wrong response data type')
        raw = self._read_response(dsize)
        status = _HEALTH_STATUSES[_b2i(raw[0])]
        error_code = (_b2i(raw[1]) << 8) + _b2i(raw[2])
        return status, error_code

    def clean_input(self):
        '''Clean input buffer by reading all available data'''
        if self.scanning[0]:
            return 'Cleanning not allowed during scanning process active !'
        self._serial.flushInput()
        self.express_trame = 32
        self.express_data = False

    def stop(self):
        '''Stops scanning process, disables laser diode and the measurement
        system, moves sensor to the idle state.'''
        self.logger.info('Stopping scanning')
        self._send_cmd(STOP_BYTE)
        time.sleep(.1)
        self.scanning[0] = False
        self.clean_input()

    def start(self, scan_type='normal'):
        '''Start the scanning process

        Parameters
        ----------
        scan : normal, force or express.
        '''
        if self.scanning[0]:
            return 'Scanning already running !'
        '''Start the scanning process, enable laser diode and the
        measurement system'''
        status, error_code = self.get_health()
        self.logger.debug('Health status: %s [%d]', status, error_code)
        if status == _HEALTH_STATUSES[2]:
            self.logger.warning('Trying to reset sensor due to the error. '
                                'Error code: %d', error_code)
            self.reset()
            status, error_code = self.get_health()
            if status == _HEALTH_STATUSES[2]:
                raise RPLidarException('RPLidar hardware failure. '
                                       'Error code: %d' % error_code)
        elif status == _HEALTH_STATUSES[1]:
            self.logger.warning('Warning sensor status detected! '
                                'Error code: %d', error_code)

        cmd = _SCAN_TYPE[scan_type]['byte']
        self.logger.info('starting scan process in %s mode' % scan_type)

        if scan_type == 'express':
            self._send_payload_cmd(cmd, b'\x00\x00\x00\x00\x00')
        else:
            self._send_cmd(cmd)

        dsize, is_single, dtype = self._read_descriptor()
        if dsize != _SCAN_TYPE[scan_type]['size']:
            raise RPLidarException('Wrong get_info reply length')
        if is_single:
            raise RPLidarException('Not a multiple response mode')
        if dtype != _SCAN_TYPE[scan_type]['response']:
            raise RPLidarException('Wrong response data type')
        self.scanning = [True, dsize, scan_type]

    def reset(self):
        '''Resets sensor core, reverting it to a similar state as it has
        just been powered up.'''
        self.logger.info('Reseting the sensor')
        self._send_cmd(RESET_BYTE)
        time.sleep(2)
        self.clean_input()

    def iter_measures(self, scan_type='normal', max_buf_meas=3000):
        '''Iterate over measures. Note that consumer must be fast enough,
        otherwise data will be accumulated inside buffer and consumer will get
        data with increasing lag.

        Parameters
        ----------
        max_buf_meas : int or False if you want unlimited buffer
            Maximum number of bytes to be stored inside the buffer. Once
            numbe exceeds this limit buffer will be emptied out.

        Yields
        ------
        new_scan : bool
            True if measures belongs to a new scan
        quality : int
            Reflected laser pulse strength
        angle : float
            The measure heading angle in degree unit [0, 360)
        distance : float
            Measured object distance related to the sensor's rotation center.
            In millimeter unit. Set to 0 when measure is invalid.
        '''
        self.start_motor()
        if not self.scanning[0]:
            self.start(scan_type)
        while True:
            dsize = self.scanning[1]
            if max_buf_meas:
                data_in_buf = self._serial.inWaiting()
                if data_in_buf > max_buf_meas:
                    self.logger.warning(
                        'Too many bytes in the input buffer: %d/%d. '
                        'Cleaning buffer...',
                        data_in_buf, max_buf_meas)
                    self.stop()
                    self.start(self.scanning[2])

            if self.scanning[2] == 'normal':
                raw = self._read_response(dsize)
                yield _process_scan(raw)
            if self.scanning[2] == 'express':
                if self.express_trame == 32:
                    self.express_trame = 0
                    if not self.express_data:
                        self.logger.debug('reading first time bytes')
                        self.express_data = ExpressPacket.from_string(
                                            self._read_response(dsize))

                    self.express_old_data = self.express_data
                    self.logger.debug('set old_data with start_angle %f',
                                      self.express_old_data.start_angle)
                    self.express_data = ExpressPacket.from_string(
                                        self._read_response(dsize))
                    self.logger.debug('set new_data with start_angle %f',
                                      self.express_data.start_angle)

                self.express_trame += 1
                self.logger.debug('process scan of frame %d with angle : '
                                  '%f and angle new : %f', self.express_trame,
                                  self.express_old_data.start_angle,
                                  self.express_data.start_angle)
                yield _process_express_scan(self.express_old_data,
                                            self.express_data.start_angle,
                                            self.express_trame)

    def iter_scans(self, scan_type='normal', max_buf_meas=3000, min_len=5):
        '''Iterate over scans. Note that consumer must be fast enough,
        otherwise data will be accumulated inside buffer and consumer will get
        data with increasing lag.

        Parameters
        ----------
        max_buf_meas : int
            Maximum number of measures to be stored inside the buffer. Once
            numbe exceeds this limit buffer will be emptied out.
        min_len : int
            Minimum number of measures in the scan for it to be yelded.

        Yields
        ------
        scan : list
            List of the measures. Each measurment is tuple with following
            format: (quality, angle, distance). For values description please
            refer to `iter_measures` method's documentation.
        '''
        scan_list = []
        iterator = self.iter_measures(scan_type, max_buf_meas)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan_list) > min_len:
                    yield scan_list
                scan_list = []
            if distance > 0:
                scan_list.append((quality, angle, distance))


class ExpressPacket(namedtuple('express_packet',
                               'distance angle new_scan start_angle')):
    sync1 = 0xa
    sync2 = 0x5
    sign = {0: 1, 1: -1}

    @classmethod
    def from_string(cls, data):
        packet = bytearray(data)

        if (packet[0] >> 4) != cls.sync1 or (packet[1] >> 4) != cls.sync2:
            raise ValueError('try to parse corrupted data ({})'.format(packet))

        checksum = 0
        for b in packet[2:]:
            checksum ^= b
        if checksum != (packet[0] & 0b00001111) + ((
                        packet[1] & 0b00001111) << 4):
            raise ValueError('Invalid checksum ({})'.format(packet))

        new_scan = packet[3] >> 7
        start_angle = (packet[2] + ((packet[3] & 0b01111111) << 8)) / 64

        d = a = ()
        for i in range(0,80,5):
            d += ((packet[i+4] >> 2) + (packet[i+5] << 6),)
            a += (((packet[i+8] & 0b00001111) + ((
                    packet[i+4] & 0b00000001) << 4))/8*cls.sign[(
                     packet[i+4] & 0b00000010) >> 1],)
            d += ((packet[i+6] >> 2) + (packet[i+7] << 6),)
            a += (((packet[i+8] >> 4) + (
                (packet[i+6] & 0b00000001) << 4))/8*cls.sign[(
                    packet[i+6] & 0b00000010) >> 1],)
        return cls(d, a, new_scan, start_angle)
