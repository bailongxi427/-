import numpy as np
from scipy import signal
import pygame
import sys
import time
from collections import deque
import threading
from pylsl import StreamInlet, resolve_byprop
import random
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 1. 跨线程数据中枢 (Data Hub)
# ==========================================
class EEGDataHub:
    def __init__(self):
        # 实时信号值
        self.alpha_power = 0.0
        self.beta_power = 0.0

        # 状态标志
        self.is_clenching = False

        # 动作触发器
        self.is_single_blink_trigger = False
        self.is_double_blink_trigger = False

        # 统计计数器
        self.single_blink_count = 0
        self.double_blink_count = 0
        self.clench_count = 0


# ==========================================
# 2. LSL 实时数据分析后台线程 (稳定低延迟版)
# ==========================================
class LSLAnalysisThread(threading.Thread):
    def __init__(self, data_hub):
        super().__init__()
        self.data_hub = data_hub
        self.running = True

        # LSL 相关
        self.inlet = None
        self.sfreq = 128

        # 缓冲队列
        self.buffer_duration = 2  # 【修复】从1.5恢复到2秒
        self.data_buffer = deque(maxlen=256)
        self.time_buffer = deque(maxlen=256)

        # 算法阈值
        self.BLINK_THRESHOLD = 48.0  # 【修复】从45调回48，保持稳定
        self.CLENCH_THRESHOLD = 20.0
        self.idx_f3, self.idx_f4 = 3, 4
        self.idx_o1, self.idx_o2 = 0, 7

        # 逻辑冷却记录
        self.last_clench_time = 0
        self.MIN_CLENCH_INTERVAL = 3.0  # 【修复】恢复3.0

        # 双眨眼检测状态
        self.last_blink_time = 0
        self.pending_blink_time = 0
        self.pending_blink = False
        self.DOUBLE_BLINK_WINDOW = 0.6
        self.SINGLE_BLINK_COOLDOWN = 0.3  # 【修复】恢复0.3

        # 【修复】分析窗口改为0.8秒（平衡延迟和稳定性）
        self.analysis_window = 0.8  # 0.8秒窗口，延迟约400ms

        # 【修复】滤波器状态标志
        self.filter_initialized = False
        self.zi_alpha = None
        self.zi_beta = None
        self.zi_blink_f3 = None
        self.zi_blink_f4 = None
        self.zi_clench = None

    def connect(self, timeout=5):
        print("\n🔍 正在寻找 Cortex LSL 数据流...")
        streams = resolve_byprop('name', 'Cortex-lsl-data-sream', timeout=timeout)
        if not streams:
            print("❌ 未找到数据流")
            return False

        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        ch_count = info.channel_count()
        print(f"✅ 成功连接! 采样率: {self.sfreq} Hz, 通道数: {ch_count}")

        max_len = int(self.sfreq * self.buffer_duration)
        self.data_buffer = deque(maxlen=max_len)
        self.time_buffer = deque(maxlen=max_len)
        return True

    def _init_filters(self):
        """初始化滤波器状态"""
        nyquist = self.sfreq / 2

        # 设计滤波器系数
        self.b_alpha, self.a_alpha = signal.butter(4, [8 / nyquist, 14 / nyquist], btype='band')
        self.b_beta, self.a_beta = signal.butter(4, [14 / nyquist, 35 / nyquist], btype='band')
        self.b_blink, self.a_blink = signal.butter(4, [1 / nyquist, 10 / nyquist], btype='band')
        self.b_clench, self.a_clench = signal.butter(4, [20 / nyquist, 45 / nyquist], btype='band')

        # 初始化滤波器状态（用零初始化）
        self.zi_alpha = signal.lfilter_zi(self.b_alpha, self.a_alpha) * 0
        self.zi_beta = signal.lfilter_zi(self.b_beta, self.a_beta) * 0
        self.zi_blink_f3 = signal.lfilter_zi(self.b_blink, self.a_blink) * 0
        self.zi_blink_f4 = signal.lfilter_zi(self.b_blink, self.a_blink) * 0
        self.zi_clench = signal.lfilter_zi(self.b_clench, self.a_clench) * 0

        self.filter_initialized = True

    def run(self):
        if not self.inlet:
            if not self.connect():
                return

        # 初始化滤波器
        self._init_filters()

        print("🚀 LSL 实时分析线程已启动...")
        print(f"   分析窗口: {self.analysis_window}秒")
        print(f"   眨眼阈值: {self.BLINK_THRESHOLD}")

        window_size = int(self.sfreq * self.analysis_window)
        window_size = max(window_size, 50)  # 确保至少有50个样本

        # 【修复】用于峰值检测的缓冲区
        f3_history = deque(maxlen=window_size)
        f4_history = deque(maxlen=window_size)

        # 数据计数，用于定期处理
        samples_since_last = 0
        process_interval = int(self.sfreq * 0.1)  # 每0.1秒处理一次

        while self.running:
            try:
                # 【修复】使用更大的timeout，避免CPU占用过高
                samples, timestamps = self.inlet.pull_chunk(timeout=0.2, max_samples=30)

                if samples:
                    current_time = time.time()
                    for sample, ts in zip(samples, timestamps):
                        self.data_buffer.append(sample)
                        self.time_buffer.append(ts if ts else current_time)

                    samples_since_last += len(samples)

                    # 【修复】每积累一定数据或缓冲区足够时才处理
                    if samples_since_last >= process_interval and len(self.data_buffer) >= window_size:
                        samples_since_last = 0

                        # 获取完整的分析窗口数据
                        data_array = np.array(list(self.data_buffer))[-window_size:]

                        # ==========================================
                        # 1. Alpha / Beta 分析
                        # ==========================================
                        occipital_data = np.mean(data_array[:, [self.idx_o1, self.idx_o2]], axis=1)

                        # 使用滑动滤波
                        alpha_f, self.zi_alpha = signal.lfilter(
                            self.b_alpha, self.a_alpha, occipital_data, zi=self.zi_alpha)
                        beta_f, self.zi_beta = signal.lfilter(
                            self.b_beta, self.a_beta, occipital_data, zi=self.zi_beta)

                        # 取最近0.5秒计算功率
                        power_window = int(self.sfreq * 0.5)
                        self.data_hub.alpha_power = np.mean(alpha_f[-power_window:] ** 2)
                        self.data_hub.beta_power = np.mean(beta_f[-power_window:] ** 2)

                        # ==========================================
                        # 2. 咬牙检测
                        # ==========================================
                        clench_data = np.mean(data_array[:, [self.idx_o1, self.idx_o2]], axis=1)
                        clench_f, self.zi_clench = signal.lfilter(
                            self.b_clench, self.a_clench, clench_data, zi=self.zi_clench)

                        energy_window = int(0.2 * self.sfreq)
                        clench_energy = np.sqrt(np.mean(clench_f[-energy_window:] ** 2))

                        if clench_energy > self.CLENCH_THRESHOLD:
                            if current_time - self.last_clench_time > self.MIN_CLENCH_INTERVAL:
                                self.data_hub.is_clenching = True
                                self.data_hub.clench_count += 1
                                self.last_clench_time = current_time
                        else:
                            if current_time - self.last_clench_time > 1.0:
                                self.data_hub.is_clenching = False

                        # ==========================================
                        # 3. 眨眼检测
                        # ==========================================
                        if not self.data_hub.is_clenching:
                            f3_data = data_array[:, self.idx_f3]
                            f4_data = data_array[:, self.idx_f4]

                            # 滑动滤波
                            f3_filt, self.zi_blink_f3 = signal.lfilter(
                                self.b_blink, self.a_blink, f3_data, zi=self.zi_blink_f3)
                            f4_filt, self.zi_blink_f4 = signal.lfilter(
                                self.b_blink, self.a_blink, f4_data, zi=self.zi_blink_f4)

                            # 更新历史缓冲区
                            for val in f3_filt:
                                f3_history.append(val)
                            for val in f4_filt:
                                f4_history.append(val)

                            if len(f3_history) >= window_size:
                                f3_b = np.array(f3_history)
                                f4_b = np.array(f4_history)

                                min_dist = int(0.25 * self.sfreq)
                                peaks_f3, _ = signal.find_peaks(f3_b, height=self.BLINK_THRESHOLD, distance=min_dist)
                                peaks_f4, _ = signal.find_peaks(f4_b, height=self.BLINK_THRESHOLD, distance=min_dist)

                                if len(peaks_f3) > 0 and len(peaks_f4) > 0:
                                    t3_idx = peaks_f3[-1]
                                    t4_idx = peaks_f4[-1]
                                    time_diff = abs(t3_idx - t4_idx) / self.sfreq

                                    if time_diff < 0.15:
                                        current_blink_time = current_time

                                        # 防重复检测
                                        if current_blink_time - self.last_blink_time < self.SINGLE_BLINK_COOLDOWN:
                                            continue

                                        # 双眨眼检测
                                        if self.pending_blink:
                                            time_since_first = current_blink_time - self.pending_blink_time

                                            if time_since_first <= self.DOUBLE_BLINK_WINDOW:
                                                self.data_hub.double_blink_count += 1
                                                self.data_hub.is_double_blink_trigger = True
                                                print(f"👀👀 DOUBLE BLINK! Interval: {time_since_first:.2f}s")
                                                self.pending_blink = False
                                            else:
                                                self.data_hub.single_blink_count += 1
                                                self.data_hub.is_single_blink_trigger = True
                                                print(f"👀 SINGLE BLINK! Total: {self.data_hub.single_blink_count}")
                                                self.pending_blink_time = current_blink_time

                                            self.last_blink_time = current_blink_time

                                        else:
                                            self.pending_blink = True
                                            self.pending_blink_time = current_blink_time
                                            self.last_blink_time = current_blink_time
                                            print(f"⏳ First blink...")

                # 【修复】适当的休眠，避免CPU占用过高
                time.sleep(0.02)

            except Exception as e:
                print(f"[Warning] 分析循环异常: {e}")
                time.sleep(0.5)

    def stop(self):
        self.running = False


# ==========================================
# 3. Pygame 可视化主程序
# ==========================================
def main():
    pygame.init()
    try:
        pygame.mixer.init()
    except:
        pass

    WIDTH, HEIGHT = 1100, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cortex EEG Real-time Monitor")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)

    data_hub = EEGDataHub()

    # 启动后台数据处理线程
    lsl_thread = LSLAnalysisThread(data_hub)
    lsl_thread.start()

    # --- 视觉对象属性 ---
    obj_x = WIDTH // 2
    obj_y = HEIGHT // 2
    obj_base_radius = 60
    obj_radius = obj_base_radius
    obj_color = [100, 150, 255]
    obj_visible = True
    hide_time = 0

    # 平滑变量
    smooth_alpha = 0.0
    smooth_beta = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- 1. 获取最新数据 ---
        alpha = data_hub.alpha_power
        beta = data_hub.beta_power

        smooth_alpha = smooth_alpha * 0.9 + alpha * 0.1
        smooth_beta = smooth_beta * 0.9 + beta * 0.1

        # --- 2. 视觉动画逻辑 ---
        ratio = smooth_alpha / (smooth_beta + 0.0001)
        ratio = max(0.5, min(1.5, ratio))
        target_y = (HEIGHT // 2) + (ratio - 1.0) * 150
        obj_y += (target_y - obj_y) * 0.05
        obj_y = max(obj_radius + 50, min(HEIGHT - obj_radius - 50, obj_y))

        # 单次眨眼：改变颜色
        if data_hub.is_single_blink_trigger:
            obj_color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
            data_hub.is_single_blink_trigger = False

        # 双次眨眼：短暂消失
        current_time = pygame.time.get_ticks()
        if data_hub.is_double_blink_trigger:
            obj_visible = False
            hide_time = current_time
            data_hub.is_double_blink_trigger = False

        if not obj_visible and (current_time - hide_time > 500):
            obj_visible = True

        # 咬紧牙关：放大物体
        if data_hub.is_clenching:
            target_radius = obj_base_radius * 1.5
        else:
            target_radius = obj_base_radius
        obj_radius += (target_radius - obj_radius) * 0.1

        # --- 3. 渲染界面 ---
        screen.fill((20, 20, 30))

        if obj_visible:
            pygame.draw.circle(screen, obj_color, (int(obj_x), int(obj_y)), int(obj_radius))
            if data_hub.is_clenching:
                pygame.draw.circle(screen, (255, 100, 100), (int(obj_x), int(obj_y)), int(obj_radius) + 10, 3)
            else:
                pygame.draw.circle(screen, (255, 255, 255), (int(obj_x), int(obj_y)), int(obj_radius), 2)

        # --- 4. 绘制数据面板 ---
        panel_texts = [
            f"--- BCI Status ---",
            f"Alpha Power: {smooth_alpha:.1f}",
            f"Beta Power: {smooth_beta:.1f}",
            f"Alpha/Beta Ratio: {ratio:.2f}",
            f"",
            f"--- Event Stats ---",
            f"Single Blinks: {data_hub.single_blink_count}",
            f"Double Blinks: {data_hub.double_blink_count}",
            f"Clench Count: {data_hub.clench_count}",
            f"Clench Status: {'ACTIVE SHIELD' if data_hub.is_clenching else 'None'}",
            f"",
            f"--- Settings ---",
            f"Window: {lsl_thread.analysis_window}s",
            f"Blink Th: {lsl_thread.BLINK_THRESHOLD}"
        ]

        overlay = pygame.Surface((360, 420))
        overlay.set_alpha(150)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (20, 20))

        for i, text in enumerate(panel_texts):
            color = (255, 255, 100) if "Ratio" in text or "Window" in text else (200, 200, 200)
            if "ACTIVE" in text:
                color = (255, 100, 100)
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (30, 30 + i * 28))

        pygame.display.flip()
        clock.tick(60)

    print("\n🛑 正在安全关闭系统...")
    lsl_thread.stop()
    lsl_thread.join(timeout=2)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()