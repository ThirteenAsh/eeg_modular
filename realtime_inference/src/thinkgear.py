from __future__ import annotations

import json
import logging
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass
from threading import Thread, Lock
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ThinkGearConfig:
    connection_mode: str = "mock"
    com_port: str = "COM3"
    baud_rate: int = 57600
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 13854
    sample_rate: int = 512
    buffer_size: int = 1024
    use_mock: bool = True


@dataclass
class EEGData:
    raw: List[int]
    attention: int
    meditation: int
    delta: int
    theta: int
    alpha1: int
    alpha2: int
    beta1: int
    beta2: int
    gamma1: int
    gamma2: int
    timestamp: float


class ThinkGearCollector:
    """ThinkGear设备数据采集器 - 支持TCP、串口和模拟模式"""

    def __init__(self, cfg: ThinkGearConfig):
        self.cfg = cfg
        self._running = False
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        
        self._data_buffer: Deque[EEGData] = deque(maxlen=cfg.buffer_size)
        self._raw_buffer: Deque[int] = deque(maxlen=cfg.sample_rate * 2)
        
        self._last_attention = 50
        self._last_meditation = 50
        self._last_powers = [0] * 8
        
        is_mock = cfg.use_mock or cfg.connection_mode == "mock"
        
        if is_mock:
            logger.info("ThinkGearCollector initialized in MOCK mode")
        elif cfg.connection_mode == "tcp":
            logger.info(f"ThinkGearCollector initialized for TCP: {cfg.tcp_host}:{cfg.tcp_port}")
        elif cfg.connection_mode == "serial":
            logger.info(f"ThinkGearCollector initialized for {cfg.com_port} @ {cfg.baud_rate}bps")
        else:
            raise ValueError(f"Unknown connection_mode: {cfg.connection_mode}")

    def start(self):
        if self._running:
            logger.warning("Collector already running")
            return
        
        self._running = True
        self._thread = Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info("ThinkGearCollector started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("ThinkGearCollector stopped")

    def _collect_loop(self):
        is_mock = self.cfg.use_mock or self.cfg.connection_mode == "mock"
        if is_mock:
            self._mock_collect_loop()
        elif self.cfg.connection_mode == "tcp":
            self._tcp_collect_loop()
        elif self.cfg.connection_mode == "serial":
            self._serial_collect_loop()

    def _mock_collect_loop(self):
        """模拟数据采集（用于开发测试）"""
        t = 0.0
        
        while self._running:
            try:
                att = int(50 + 20 * np.sin(t / 30.0) + np.random.randn() * 3)
                med = int(50 + 25 * np.sin(t / 45.0 + 1.2) + np.random.randn() * 3)
                
                powers = [
                    int(50000 + 20000 * np.sin(t / 20.0) + np.random.randn() * 2000),
                    int(40000 + 15000 * np.sin(t / 22.0) + np.random.randn() * 1500),
                    int(30000 + 12000 * np.sin(t / 18.0) + np.random.randn() * 1200),
                    int(25000 + 10000 * np.sin(t / 19.0) + np.random.randn() * 1000),
                    int(20000 + 8000 * np.sin(t / 17.0) + np.random.randn() * 800),
                    int(18000 + 7000 * np.sin(t / 16.0) + np.random.randn() * 700),
                    int(15000 + 6000 * np.sin(t / 15.0) + np.random.randn() * 600),
                    int(12000 + 5000 * np.sin(t / 14.0) + np.random.randn() * 500),
                ]
                
                att = np.clip(att, 0, 100)
                med = np.clip(med, 0, 100)
                powers = [max(0, p) for p in powers]
                
                with self._lock:
                    self._last_attention = int(att)
                    self._last_meditation = int(med)
                    self._last_powers = powers
                    raw_val = int(1000 * np.sin(t / 5.0))
                    self._raw_buffer.append(raw_val)
                
                eeg_data = EEGData(
                    raw=list(self._raw_buffer)[-128:],
                    attention=self._last_attention,
                    meditation=self._last_meditation,
                    delta=powers[0],
                    theta=powers[1],
                    alpha1=powers[2],
                    alpha2=powers[3],
                    beta1=powers[4],
                    beta2=powers[5],
                    gamma1=powers[6],
                    gamma2=powers[7],
                    timestamp=time.time(),
                )
                
                with self._lock:
                    self._data_buffer.append(eeg_data)
                
                t += 1.0
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Mock collection error: {e}")
                time.sleep(0.1)

    def _tcp_collect_loop(self):
        """TCP模式采集 - 连接ThinkGear Connector"""
        sock = None
        buffer = ""
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            logger.info(f"Connecting to ThinkGear Connector at {self.cfg.tcp_host}:{self.cfg.tcp_port}...")
            sock.connect((self.cfg.tcp_host, self.cfg.tcp_port))
            logger.info("Connected to ThinkGear Connector!")
            
            config = {"enableRawOutput": False, "format": "Json"}
            sock.send(json.dumps(config).encode())
            logger.info("Sent configuration to ThinkGear Connector")
            
            sock.settimeout(0.1)
            
            while self._running:
                try:
                    data = sock.recv(4096)
                    if data:
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        while '\r' in buffer:
                            packet, buffer = buffer.split('\r', 1)
                            if packet.strip():
                                self._parse_json(packet.strip())
                
                except socket.timeout:
                    pass
                except Exception as e:
                    logger.warning(f"TCP receive error: {e}")
                    time.sleep(0.1)
            
        except ConnectionRefusedError:
            logger.error(f"Could not connect to ThinkGear Connector at {self.cfg.tcp_host}:{self.cfg.tcp_port}")
            logger.error("Please start ThinkGear Connector first!")
        except Exception as e:
            logger.error(f"TCP communication error: {e}")
        finally:
            if sock:
                sock.close()
                logger.info("TCP connection closed")

    def _serial_collect_loop(self):
        """串口模式采集 - 直接连接ThinkGear设备"""
        logger.warning("Serial mode not fully implemented yet")
        while self._running:
            time.sleep(1.0)

    def _parse_json(self, json_str: str):
        """解析ThinkGear Connector的JSON数据（和之前程序保持一致）"""
        try:
            data = json.loads(json_str)
            
            with self._lock:
                attention = data.get('attention', self._last_attention)
                meditation = data.get('meditation', self._last_meditation)
                
                if 'eSense' in data:
                    esense = data['eSense']
                    attention = esense.get('attention', attention)
                    meditation = esense.get('meditation', meditation)
                
                self._last_attention = attention
                self._last_meditation = meditation
                
                if 'eegPower' in data:
                    eeg_power = data['eegPower']
                    self._last_powers = [
                        eeg_power.get('delta', self._last_powers[0]),
                        eeg_power.get('theta', self._last_powers[1]),
                        eeg_power.get('lowAlpha', self._last_powers[2]),
                        eeg_power.get('highAlpha', self._last_powers[3]),
                        eeg_power.get('lowBeta', self._last_powers[4]),
                        eeg_power.get('highBeta', self._last_powers[5]),
                        eeg_power.get('lowGamma', self._last_powers[6]),
                        eeg_power.get('highGamma', self._last_powers[7]),
                    ]
                
                raw_value = int(sum(self._last_powers) / 800) if sum(self._last_powers) > 0 else 0
                self._raw_buffer.append(raw_value)
                
                if len(self._raw_buffer) >= 128:
                    eeg_data = EEGData(
                        raw=list(self._raw_buffer)[-128:],
                        attention=self._last_attention,
                        meditation=self._last_meditation,
                        delta=self._last_powers[0],
                        theta=self._last_powers[1],
                        alpha1=self._last_powers[2],
                        alpha2=self._last_powers[3],
                        beta1=self._last_powers[4],
                        beta2=self._last_powers[5],
                        gamma1=self._last_powers[6],
                        gamma2=self._last_powers[7],
                        timestamp=time.time(),
                    )
                    self._data_buffer.append(eeg_data)
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"JSON parse error: {e}")

    def get_latest_data(self) -> Optional[EEGData]:
        """获取最新的EEG数据"""
        with self._lock:
            if self._data_buffer:
                return self._data_buffer[-1]
        return None

    def get_multimodal_features(self, time_steps: int = 10, feat_dim: int = 4) -> Dict[str, np.ndarray]:
        """获取多模态特征数据（用于模型推理）"""
        with self._lock:
            result = {}
            
            for modality in ['filtered', 'powerspec', 'att', 'med']:
                arr = np.zeros((time_steps, feat_dim), dtype=np.float32)
                
                if modality == 'filtered':
                    for i in range(time_steps):
                        for j in range(feat_dim):
                            t_factor = (i + 1) / time_steps
                            arr[i, j] = 0.5 + 0.5 * np.sin(t_factor * 5.0 + j)
                
                elif modality == 'powerspec':
                    powers = np.array(self._last_powers, dtype=np.float32)
                    for i in range(time_steps):
                        t_factor = (i + 1) / time_steps
                        for j in range(feat_dim):
                            if j < len(powers):
                                arr[i, j] = powers[j % len(powers)] * t_factor * 0.001
                
                elif modality == 'att':
                    att = self._last_attention / 100.0
                    for i in range(time_steps):
                        t_factor = (i + 1) / time_steps
                        arr[i, 0] = att * t_factor
                        arr[i, 1] = att * t_factor * 0.8
                        arr[i, 2] = att * t_factor * 0.6
                        arr[i, 3] = att * t_factor * 0.4
                
                elif modality == 'med':
                    med = self._last_meditation / 100.0
                    for i in range(time_steps):
                        t_factor = (i + 1) / time_steps
                        arr[i, 0] = med * t_factor
                        arr[i, 1] = med * t_factor * 0.8
                        arr[i, 2] = med * t_factor * 0.6
                        arr[i, 3] = med * t_factor * 0.4
                
                result[modality] = arr
            
            return result
