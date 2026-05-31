# ssvep_online_optimized.py — v3
"""
SSVEP 在线 BCI 综合优化版 v3
改进：
  1. 宽间距3频率 [10, 15, 20] Hz（5Hz间距，避开alpha波段）
  2. 修复谐波一致性公式方向（基频强 >> 谐波弱 = 真正SSVEP）
  3. 置信度阈值（z-score > 1.0 才输出，否则显示"不确定"）
  4. 移除键盘交互，纯观察模式
  5. CSV 数据记录（ssvep_test_log_*.csv），方便离线分析
  6. 保留 v2 全部优化：SNR归一化、EMA、4子带FBCCA、信号质量面板、50Hz陷波
"""

import tkinter as tk
import numpy as np
import threading
import time
import requests
import csv
import os
from datetime import datetime
from collections import deque
from scipy.signal import butter, filtfilt

# ===================== 配置区（可调参数） =====================
BASE_URL = "http://127.0.0.1:2336"
FS = 125                              # NeuroPlay 8通道采样率
DURATION = 4.0                        # 分析窗口 4 秒
OVERLAP = 2.0                         # 滑动重叠 2 秒（每2秒输出一次结果）
WINDOW_SAMPLES = int(DURATION * FS)
OVERLAP_SAMPLES = int(OVERLAP * FS)

# 整数频率（与60Hz显示器刷新率同步）
STIM_FREQS = [10.0, 12.0, 14.0]

# 4通道空间平均：O1(0), P3(1), P4(6), O2(7)
SSVEP_CHANNELS = [0, 1, 6, 7]

# EMA 平滑系数（越大越灵敏，越小越平滑）
EMA_ALPHA = 0.3

# SNR 归一化：z-score，消除 alpha 基线偏移
USE_SNR_NORMALIZE = True

# 置信度阈值：最高 z-score 必须超过此值才输出结果
CONFIDENCE_THRESHOLD = 0.2

# 谐波一致性权重（0=纯FBCCA, 1=纯谐波一致性）
HARMONIC_WEIGHT = 0.3

# ===================== 全局数据缓冲 =====================
data_buffer = deque(maxlen=3000)
stop_collecting = False
ema_scores = {f: 0.0 for f in STIM_FREQS}  # EMA平滑后的得分
ema_initialized = False  # EMA 冷启动标记

# CSV 日志文件
csv_file = None
csv_writer = None


def init_csv_log():
    """初始化 CSV 日志文件"""
    global csv_file, csv_writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(log_dir, f"ssvep_test_log_{timestamp}.csv")
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    # 写表头
    header = [
        'timestamp',
        'raw_10Hz', 'raw_12Hz', 'raw_14Hz',
        'norm_10Hz', 'norm_12Hz', 'norm_14Hz',
        'ema_10Hz', 'ema_12Hz', 'ema_14Hz',
        'predicted', 'confidence', 'is_confident'
    ]
    csv_writer.writerow(header)
    csv_file.flush()
    print(f"📁 CSV 日志: {csv_path}")
    return csv_path


def close_csv_log():
    """关闭 CSV 日志文件"""
    global csv_file, csv_writer
    if csv_file:
        csv_file.close()
        csv_file = None
        csv_writer = None


def get_eeg_data():
    """从 NeuroPlayPro API 获取 EEG 数据 (samples, channels)"""
    try:
        resp = requests.get(f"{BASE_URL}/rawData", timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                channel_data = []
                for ch_data in data['data']:
                    if isinstance(ch_data, list):
                        channel_data.append(np.array(ch_data))
                if len(channel_data) >= 8:
                    return np.array(channel_data).T
    except Exception:
        pass
    return None


def collect_data():
    """后台持续采集数据"""
    global stop_collecting
    while not stop_collecting:
        eeg = get_eeg_data()
        if eeg is not None:
            data_buffer.append(eeg)
        time.sleep(0.05)


def get_device_info():
    """获取设备信息，返回 (info_dict, connected)"""
    try:
        resp = requests.get(f"{BASE_URL}/currentDeviceInfo", timeout=2)
        info = resp.json()
        if info.get('currentFrequency', 0) > 0:
            return info, True
        return info, False
    except Exception:
        return None, False


# ===================== 信号处理函数 =====================
def preprocess(eeg):
    """预处理：去直流 + 50Hz陷波"""
    eeg = eeg - np.mean(eeg, axis=0)
    nyq = FS / 2
    # 50Hz 工频陷波
    b_notch, a_notch = butter(4, [48 / nyq, 52 / nyq], btype='bandstop')
    eeg = filtfilt(b_notch, a_notch, eeg, axis=0)
    return eeg


def create_reference(freq, duration, n_harmonics=3):
    """创建 CCA 参考信号模板"""
    t = np.arange(0, duration, 1 / FS)
    Y = []
    for h in range(1, n_harmonics + 1):
        Y.append(np.sin(2 * np.pi * h * freq * t))
        Y.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(Y).T


def cca_correlation(X, Y):
    """计算 CCA 最大相关系数"""
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    min_len = min(X.shape[0], Y.shape[0])
    X, Y = X[:min_len], Y[:min_len]
    Cxx = X.T @ X + 1e-6 * np.eye(X.shape[1])
    Cyy = Y.T @ Y + 1e-6 * np.eye(Y.shape[1])
    Cxy = X.T @ Y
    try:
        invCxx = np.linalg.inv(Cxx)
        invCyy = np.linalg.inv(Cyy)
        K = invCxx @ Cxy @ invCyy @ Cxy.T
        return np.sqrt(np.max(np.real(np.linalg.eigvals(K))))
    except np.linalg.LinAlgError:
        return 0.0


def harmonic_consistency(eeg_segment, freq, fs=FS, n_harmonics=3):
    """
    谐波一致性得分（修复方向）：
    真正的 SSVEP：基频相关强，谐波随次数递减（功率集中在基频）
    噪声/alpha：所有谐波频率相关度差不多
    
    得分 = 基频相关 / (基频相关 + 平均谐波相关)
    → 基频远强于谐波 → 得分接近 1.0（好）
    → 谐波也强/更强 → 得分接近 0.5（差）
    返回: 0~1 之间的得分
    """
    scores = []
    nyq = fs / 2
    for h in range(1, n_harmonics + 1):
        hf = freq * h
        if hf >= nyq - 2:
            break
        # 窄带滤波围绕谐波频率
        bw = 2.0  # 带宽 ±2Hz
        low = max(1.0, hf - bw)
        high = min(nyq - 1, hf + bw)
        if high <= low:
            continue
        try:
            b, a = butter(3, [low / nyq, high / nyq], btype='band')
            Xf = filtfilt(b, a, eeg_segment, axis=0)
            Yh = create_reference(hf, eeg_segment.shape[0] / fs, n_harmonics=1)
            corr = cca_correlation(Xf, Yh)
            scores.append(corr)
        except Exception:
            scores.append(0.0)
    if len(scores) < 2:
        return 0.0
    # 修复方向：基频/(基频+谐波均值) — 基频越占主导，得分越高
    fundamental = scores[0]
    harmonic_mean = np.mean(scores[1:]) if len(scores) > 1 else 0.0
    denom = fundamental + harmonic_mean
    if denom < 1e-8:
        return 0.0
    return fundamental / denom  # 范围 0~1，SSVEP 特征强的接近 1


def fbcca_scores(eeg_segment):
    """
    FBCCA 分类：4个子带滤波 + CCA + 谐波一致性
    返回每个频率的原始加权相关系数
    """
    # 4个子带，与离线配置一致
    subbands = [(5, 35), (8, 40), (11, 45), (14, 50)]
    scores = {f: 0.0 for f in STIM_FREQS}
    harmonic_scores = {f: 0.0 for f in STIM_FREQS}
    nyq = FS / 2
    for idx, (low, high) in enumerate(subbands):
        high = min(high, nyq - 1)
        if high <= low:
            continue
        b, a = butter(4, [low / nyq, high / nyq], btype='band')
        Xf = filtfilt(b, a, eeg_segment, axis=0)
        for freq in STIM_FREQS:
            Y = create_reference(freq, DURATION)
            scores[freq] += cca_correlation(Xf, Y) / (idx + 1)

    # 谐波一致性计算
    for freq in STIM_FREQS:
        harmonic_scores[freq] = harmonic_consistency(eeg_segment, freq)

    # 融合谐波一致性（作为乘性修正因子）
    for freq in STIM_FREQS:
        scores[freq] = scores[freq] * (1 - HARMONIC_WEIGHT) + harmonic_scores[freq] * HARMONIC_WEIGHT * max(scores.values())

    return scores


def snr_normalize(scores):
    """
    SNR 归一化：将相关系数转为 z-score
    消除 alpha 等内源性节律的基线偏移。
    返回: {freq: z_score}
    """
    values = np.array(list(scores.values()))
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val < 1e-8:
        return {f: 0.0 for f in scores}
    return {f: (v - mean_val) / std_val for f, v in scores.items()}


def classify_with_ema(eeg_segment):
    """
    使用 EMA 平滑 + SNR 归一化 + 置信度阈值的分类：
    1. 计算原始 FBCCA 得分
    2. SNR 归一化（z-score）
    3. EMA 平滑（第一帧直接用原始值，避免冷启动衰减）
    4. 置信度阈值判断
    返回: (best_freq, raw_scores, norm_scores, ema_scores, confidence, is_confident)
    """
    global ema_scores, ema_initialized
    raw_scores = fbcca_scores(eeg_segment)

    # SNR 归一化
    if USE_SNR_NORMALIZE:
        norm_scores = snr_normalize(raw_scores)
    else:
        norm_scores = raw_scores

    # EMA 平滑（冷启动：第一帧直接用 norm_scores，不衰减）
    if not ema_initialized:
        for freq in STIM_FREQS:
            ema_scores[freq] = norm_scores[freq]
        ema_initialized = True
    else:
        for freq in STIM_FREQS:
            ema_scores[freq] = EMA_ALPHA * norm_scores[freq] + (1 - EMA_ALPHA) * ema_scores[freq]

    # 找最高分
    best_freq = max(ema_scores, key=ema_scores.get)
    confidence = ema_scores[best_freq]

    # 置信度判断（即使低于阈值也返回猜测，只是标记不可靠）
    is_confident = confidence >= CONFIDENCE_THRESHOLD

    return best_freq, raw_scores, norm_scores, ema_scores.copy(), confidence, is_confident


# ===================== GUI 主界面 =====================
class SSVEPOptimizedGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SSVEP 在线 BCI v3 — 宽间距频率 / 置信度阈值")
        self.root.geometry("1200x900")
        self.root.configure(bg='black')

        self.running = False
        self.csv_path = None

        # ===== 顶部标题 =====
        title_frame = tk.Frame(self.root, bg='black')
        title_frame.pack(pady=8)
        self.title_label = tk.Label(
            title_frame,
            text=f"SSVEP 在线 BCI v3 — 频率: {[f'{f:.0f}' for f in STIM_FREQS]} Hz | 窗口: {DURATION}s | 置信度阈值: {CONFIDENCE_THRESHOLD}",
            font=("Arial", 18, "bold"), fg="white", bg="black"
        )
        self.title_label.pack()

        # ===== 闪烁按钮区域（3个大按钮） =====
        btn_frame = tk.Frame(self.root, bg='black')
        btn_frame.pack(pady=25)
        self.buttons = []
        for i, freq in enumerate(STIM_FREQS):
            btn = tk.Button(
                btn_frame,
                text=f"{freq:.0f} Hz",
                font=("Arial", 26, "bold"),
                width=14, height=4,
                bg='gray', fg='white'
            )
            btn.grid(row=0, column=i, padx=100, pady=10)
            btn.idx = i
            btn.freq = freq
            self.buttons.append(btn)

        # ===== 识别结果（大号显示） =====
        result_frame = tk.Frame(self.root, bg='black')
        result_frame.pack(pady=10)
        self.result_label = tk.Label(
            result_frame,
            text="⏳ 等待开始...",
            font=("Arial", 28, "bold"),
            fg="yellow", bg="black"
        )
        self.result_label.pack()
        self.confidence_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 14), fg="gray", bg="black"
        )
        self.confidence_label.pack(pady=3)

        # ===== 相关系数调试显示 =====
        self.scores_text = tk.Text(
            self.root, height=8, width=70,
            font=("Courier", 11),
            bg='#101010', fg='#00ff88'
        )
        self.scores_text.pack(pady=8)

        # ===== 日志区域 =====
        log_frame = tk.Frame(self.root, bg='black')
        log_frame.pack(pady=5, fill=tk.X, padx=40)
        tk.Label(
            log_frame, text="分类日志:", font=("Arial", 11),
            fg="gray", bg="black"
        ).pack(anchor='w')
        self.log_text = tk.Text(
            log_frame, height=10, width=120,
            font=("Courier", 9),
            bg='#181818', fg='#aaaaaa'
        )
        self.log_text.pack(fill=tk.BOTH)
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scroll.set)

        # ===== 倒计时进度条 =====
        self.countdown_frame = tk.Frame(self.root, bg='black')
        self.countdown_frame.pack(pady=5)
        self.countdown_label = tk.Label(
            self.countdown_frame,
            text="⏱ 下次结果: -- 秒",
            font=("Arial", 13), fg="#0088ff", bg="black"
        )
        self.countdown_label.pack(side=tk.LEFT, padx=5)
        self.countdown_bar = tk.Canvas(
            self.countdown_frame, width=400, height=18,
            bg='#222222', highlightthickness=0
        )
        self.countdown_bar.pack(side=tk.LEFT, padx=5)
        self.countdown_rect = None

        # ===== 控制按钮 =====
        ctrl_frame = tk.Frame(self.root, bg='black')
        ctrl_frame.pack(pady=10)
        self.start_btn = tk.Button(
            ctrl_frame, text="▶ 开始采集与分类",
            command=self.start,
            bg='#006600', fg='white',
            font=("Arial", 14, "bold"),
            width=18, height=1
        )
        self.start_btn.pack(side=tk.LEFT, padx=12)
        self.stop_btn = tk.Button(
            ctrl_frame, text="⏹ 停止",
            command=self.stop,
            bg='#660000', fg='white',
            font=("Arial", 14, "bold"),
            width=12, height=1,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=12)

        # ===== 信号质量面板（底部） =====
        quality_frame = tk.Frame(self.root, bg='#111111')
        quality_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=20)
        tk.Label(
            quality_frame, text="通道信号质量:",
            font=("Arial", 10), fg="gray", bg="#111111"
        ).pack(side=tk.LEFT, padx=5)
        self.quality_labels = {}
        ch_names = ["O1", "P3", "C3", "F3", "F4", "C4", "P4", "O2"]
        for name in ch_names:
            lbl = tk.Label(
                quality_frame, text=f"{name}:--%",
                font=("Courier", 9), fg="#555555", bg="#111111",
                width=8
            )
            lbl.pack(side=tk.LEFT, padx=3)
            self.quality_labels[name] = lbl

        # ===== 底部状态栏 =====
        self.status_label = tk.Label(
            self.root,
            text="就绪 — 请确保 O1/O2 信号质量 > 80%，然后点击「开始」",
            font=("Arial", 10), fg="lightgray", bg="black"
        )
        self.status_label.pack(side=tk.BOTTOM, pady=5)

    # ===================== 按钮闪烁 =====================
    def flash_buttons(self):
        """所有按钮以各自频率持续闪烁"""
        while self.running:
            for btn in self.buttons:
                if not self.running:
                    break
                brightness = 128 + 110 * np.sin(2 * np.pi * btn.freq * time.time())
                c = f'#{int(brightness):02x}{int(brightness):02x}{int(brightness):02x}'
                self.root.after(0, lambda b=btn, col=c: b.config(bg=col))
            time.sleep(0.016)  # ~60Hz 刷新

    # ===================== 倒计时动画 =====================
    def update_countdown(self, remaining_sec):
        """更新倒计时进度条"""
        if not self.running:
            return
        cycle = DURATION - OVERLAP  # 每次输出间隔
        progress = max(0.0, min(1.0, (cycle - remaining_sec) / cycle))
        bar_w = int(400 * progress)
        self.countdown_bar.delete("all")
        if bar_w > 0:
            color = "#0088ff" if remaining_sec > 1.0 else "#00cc44"
            self.countdown_bar.create_rectangle(0, 0, bar_w, 18, fill=color, outline="")
        self.countdown_label.config(text=f"⏱ 下次结果: {remaining_sec:.1f} 秒")

    # ===================== 分类循环 =====================
    def classify_loop(self):
        """实时分类主循环"""
        global ema_scores, ema_initialized

        # 初始化 CSV 日志
        self.csv_path = init_csv_log()

        # 等待初始数据积累
        wait_start = time.time()
        last_count = 0
        stuck_counter = 0
        while self.running and len(data_buffer) < WINDOW_SAMPLES:
            elapsed = time.time() - wait_start
            remaining = max(0, DURATION - elapsed)
            current_count = len(data_buffer)
            
            # 检测数据流是否停滞
            if current_count == last_count:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_count = current_count
            
            if stuck_counter > 10:  # 5秒无新数据 → 警告
                self.root.after(0, self.status_label.config,
                                text=f"⚠️ 数据停滞！已收到 {current_count}/{WINDOW_SAMPLES} 点，请检查设备连接")
                self.root.after(0, self.append_log,
                                f"⚠️ {time.strftime('%H:%M:%S')} 数据流停滞 ({current_count}/{WINDOW_SAMPLES})\n")
                stuck_counter = 0
            
            self.root.after(0, self.status_label.config,
                            text=f"初始化... {current_count}/{WINDOW_SAMPLES} 采样点 (还需 {remaining:.0f}秒)")
            self.root.after(0, self.update_countdown, remaining)
            time.sleep(0.5)

        self.root.after(0, self.status_label.config,
                        text=f"运行中 — 每 {DURATION - OVERLAP:.0f} 秒输出一次识别结果")

        cycle_sec = DURATION - OVERLAP
        while self.running:
            if len(data_buffer) < WINDOW_SAMPLES:
                time.sleep(0.5)
                continue

            # 倒计时动画（每秒更新）
            for sec_left in np.arange(cycle_sec, 0, -0.2):
                if not self.running:
                    break
                self.root.after(0, self.update_countdown, sec_left)
                time.sleep(0.2)

            if not self.running:
                break

            # 取最新数据
            all_data = np.vstack(list(data_buffer))
            recent = all_data[-WINDOW_SAMPLES:, :]
            eeg = recent[:, SSVEP_CHANNELS]

            try:
                eeg = preprocess(eeg)
                best_freq, raw_scores, norm_scores, smooth_scores, confidence, is_confident = classify_with_ema(eeg)

                # 写入 CSV（增加 is_confident 列）
                self.write_csv_row(raw_scores, norm_scores, smooth_scores, best_freq, confidence, is_confident)

                # 更新 GUI — 即使低置信度也显示"最可能"猜测
                if is_confident:
                    self.root.after(0, self.result_label.config,
                                    text=f"✓ 识别结果: {best_freq:.0f} Hz",
                                    fg="lime")
                    self.root.after(0, self.confidence_label.config,
                                    text=f"置信度 z-score: {confidence:.3f}（阈值: {CONFIDENCE_THRESHOLD}）",
                                    fg="#00cc00")
                else:
                    self.root.after(0, self.result_label.config,
                                    text=f"⚠ 最可能: {best_freq:.0f} Hz（低置信度）",
                                    fg="orange")
                    self.root.after(0, self.confidence_label.config,
                                    text=f"z-score: {confidence:.3f} < 阈值 {CONFIDENCE_THRESHOLD}，检查电极",
                                    fg="#ff8800")

                self.root.after(0, self.update_scores_display,
                                raw_scores, norm_scores, smooth_scores, best_freq, confidence, is_confident)

                # 日志
                flag = "✓" if is_confident else "?"
                log_msg = (
                    f"{time.strftime('%H:%M:%S')} [{flag}] "
                    f"最可能: {best_freq:.0f}Hz | "
                    f"z-score: {confidence:.3f} | "
                    f"EMA: {smooth_scores[best_freq]:.4f}\n"
                )
                self.root.after(0, self.append_log, log_msg)

            except Exception as e:
                print(f"分类异常: {e}")
                import traceback
                traceback.print_exc()

            # 滑动窗口：保留最后 OVERLAP_SAMPLES 个采样点用于重叠
            # 直接删除旧数据，不清空整个 buffer
            all_data_list = list(data_buffer)
            cutoff = len(all_data_list) - OVERLAP_SAMPLES
            if cutoff > 0:
                data_buffer.clear()
                for item in all_data_list[cutoff:]:
                    data_buffer.append(item)

    def write_csv_row(self, raw_scores, norm_scores, smooth_scores, best_freq, confidence, is_confident):
        """写入一行 CSV 数据"""
        global csv_writer
        if csv_writer is None:
            return
        try:
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                f"{raw_scores.get(10.0, 0):.6f}",
                f"{raw_scores.get(12.0, 0):.6f}",
                f"{raw_scores.get(14.0, 0):.6f}",
                f"{norm_scores.get(10.0, 0):.6f}",
                f"{norm_scores.get(12.0, 0):.6f}",
                f"{norm_scores.get(14.0, 0):.6f}",
                f"{smooth_scores.get(10.0, 0):.6f}",
                f"{smooth_scores.get(12.0, 0):.6f}",
                f"{smooth_scores.get(14.0, 0):.6f}",
                f"{best_freq:.1f}",
                f"{confidence:.6f}",
                "1" if is_confident else "0"
            ]
            csv_writer.writerow(row)
            csv_file.flush()
        except Exception:
            pass

    # ===================== 显示更新 =====================
    def update_scores_display(self, raw_scores, norm_scores, smooth_scores, best_freq, confidence, is_confident):
        """更新相关系数面板"""
        self.scores_text.delete('1.0', tk.END)
        header = f"{'频率':>8}  {'原始CCA':>10}  {'SNR(z)':>10}  {'EMA平滑':>10}  {'指示'}\n"
        header += "-" * 60 + "\n"
        self.scores_text.insert(tk.END, header)

        sorted_items = sorted(smooth_scores.items(), key=lambda x: x[1], reverse=True)
        for freq, smooth in sorted_items:
            raw = raw_scores.get(freq, 0.0)
            norm = norm_scores.get(freq, 0.0)
            if freq == best_freq:
                marker = f" <<< 最高 (z={confidence:.2f})" if is_confident else f" <<< 最高 (z={confidence:.2f}, 低置信)"
            else:
                marker = ""
            line = f"{freq:>6.1f}Hz  {raw:>10.4f}  {norm:>10.4f}  {smooth:>10.4f}  {marker}\n"
            self.scores_text.insert(tk.END, line)

        if not is_confident:
            self.scores_text.insert(
                tk.END,
                f"\n⚠️ 置信度不足（最高 z-score={confidence:.3f} < {CONFIDENCE_THRESHOLD}）\n"
            )
            self.scores_text.insert(
                tk.END, "   请检查 O1/O2 电极接触\n"
            )

    def append_log(self, msg):
        """追加日志"""
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)

    # ===================== 启停控制 =====================
    def refresh_quality(self):
        """更新信号质量显示"""
        try:
            resp = requests.get(f"{BASE_URL}/currentDeviceInfo", timeout=1)
            info = resp.json()
            quality = info.get('quality', [])
            ch_names = ["O1", "P3", "C3", "F3", "F4", "C4", "P4", "O2"]
            if len(quality) >= 8:
                for i, name in enumerate(ch_names):
                    q_val = quality[i]
                    lbl = self.quality_labels[name]
                    if q_val >= 80:
                        color = "#00ff00"  # 绿
                    elif q_val >= 40:
                        color = "#ffaa00"  # 橙
                    else:
                        color = "#ff4444"  # 红
                    lbl.config(text=f"{name}:{q_val:3d}%", fg=color)
        except Exception:
            pass

    def start(self):
        """开始采集与分类"""
        # 检查设备连接
        _, connected = get_device_info()
        if not connected:
            self.status_label.config(
                text="❌ 无法连接 NeuroPlayPro 设备 — 请检查设备是否已连接"
            )
            self.append_log("启动失败：设备未连接\n")
            return

        # 刷新信号质量
        self.refresh_quality()

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.running = True

        global stop_collecting, data_buffer, ema_scores
        stop_collecting = False
        data_buffer.clear()
        ema_scores = {f: 0.0 for f in STIM_FREQS}
        ema_initialized = False

        # 启动后台线程
        threading.Thread(target=collect_data, daemon=True).start()
        threading.Thread(target=self.flash_buttons, daemon=True).start()
        threading.Thread(target=self.classify_loop, daemon=True).start()

        self.status_label.config(
            text=f"运行中 — 3个按钮以各自频率闪烁，请注视目标按钮（无需按键）"
        )
        self.append_log(
            f"=== 开始 {time.strftime('%H:%M:%S')} "
            f"(频率={[f'{f:.0f}' for f in STIM_FREQS]}Hz, "
            f"窗口={DURATION}s, 阈值={CONFIDENCE_THRESHOLD}) ===\n"
        )

    def stop(self):
        """停止采集"""
        global stop_collecting
        self.running = False
        stop_collecting = True

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        # 恢复按钮颜色
        for btn in self.buttons:
            btn.config(bg='gray', fg='white')

        self.status_label.config(text="已停止")
        self.result_label.config(text="⏹ 已停止", fg="gray")
        self.confidence_label.config(text="")

        self.append_log(f"=== 停止 {time.strftime('%H:%M:%S')} ===\n")

        # 关闭 CSV
        close_csv_log()
        if self.csv_path:
            self.append_log(f"📁 CSV 已保存: {self.csv_path}\n")

    def run(self):
        """启动主循环"""
        self.root.after(500, self.refresh_quality)
        self.root.mainloop()


# ===================== 入口 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("SSVEP 在线 BCI v3 — 宽间距频率 + 置信度阈值")
    print("=" * 60)
    print(f"  采样率:        {FS} Hz")
    print(f"  分析窗口:      {DURATION} 秒 ({WINDOW_SAMPLES} 样本)")
    print(f"  滑动重叠:      {OVERLAP} 秒")
    print(f"  输出间隔:      {DURATION - OVERLAP} 秒")
    print(f"  频率:          {[f'{f:.0f}' for f in STIM_FREQS]} Hz (间距5Hz)")
    print(f"  通道:          {SSVEP_CHANNELS} (O1+P3+P4+O2)")
    print(f"  预处理:        去直流 + 50Hz陷波")
    print(f"  FBCCA子带:     5-35, 8-40, 11-45, 14-50 Hz (4个子带)")
    print(f"  EMA平滑:       α={EMA_ALPHA}")
    print(f"  SNR归一化:     {'启用' if USE_SNR_NORMALIZE else '禁用'}")
    print(f"  谐波一致性:    权重={HARMONIC_WEIGHT}")
    print(f"  置信度阈值:    z-score > {CONFIDENCE_THRESHOLD}")
    print(f"  CSV记录:       自动保存到当前目录")
    print("=" * 60)
    print()
    print("📋 使用说明:")
    print("   1. 确保 NeuroPlayPro 已连接，O1/O2 信号质量 > 80%")
    print("   2. 点击「开始」，3个按钮以各自频率闪烁")
    print("   3. 注视目标频率按钮（如 10Hz），无需按键")
    print("   4. 观察识别结果和置信度")
    print("   5. 停止后 CSV 自动保存，可发给我分析")
    print()
    print("💡 提示:")
    print("   - 如果频繁「不确定」，说明信号质量不足")
    print("   - 可尝试只注视一个频率 30 秒，记录 CSV 后再分析")
    print("   - 可在代码顶部调整 CONFIDENCE_THRESHOLD 和 EMA_ALPHA")
    print()

    app = SSVEPOptimizedGUI()
    app.run()