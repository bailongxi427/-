# minimal_gui_debug.py
import tkinter as tk
import numpy as np
import threading
import time
from scipy.signal import butter, filtfilt

print("程序启动...")

FS = 125
DURATION = 3.0
STIM_FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
              10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

print("配置加载完成")

class MockDataGenerator:
    def __init__(self):
        print("初始化数据生成器...")
        self.fs = FS
        self.data_buffer = []
        self.current_target = 0
        self.running = True
        threading.Thread(target=self._generate, daemon=True).start()
    
    def _generate(self):
        t = 0
        while self.running:
            sample = []
            target_freq = STIM_FREQS[self.current_target]
            for ch in range(2):
                noise = np.random.randn() * 5
                signal = 10 * np.sin(2 * np.pi * target_freq * t)
                sample.append(signal + noise)
            self.data_buffer.append(sample)
            if len(self.data_buffer) > self.fs * 10:
                self.data_buffer.pop(0)
            t += 1/self.fs
            time.sleep(1/self.fs)
    
    def get_data(self):
        n = int(DURATION * self.fs)
        if len(self.data_buffer) < n:
            return None
        return np.array(self.data_buffer[-n:])
    
    def set_target(self, idx):
        self.current_target = idx
    
    def stop(self):
        self.running = False

print("数据生成器类定义完成")

def classify(eeg):
    X = eeg - np.mean(eeg, axis=0)
    max_corr = -1
    best_freq = None
    for freq in STIM_FREQS:
        t = np.arange(0, DURATION, 1/FS)
        Y = np.column_stack([np.sin(2*np.pi*freq*t), np.cos(2*np.pi*freq*t)])
        Y = Y - np.mean(Y, axis=0)
        min_len = min(len(X), len(Y))
        Xt, Yt = X[:min_len], Y[:min_len]
        corr = np.corrcoef(Xt[:,0], Yt[:,0])[0,1]
        if abs(corr) > max_corr:
            max_corr = abs(corr)
            best_freq = freq
    return best_freq

print("分类函数定义完成")

class App:
    def __init__(self):
        print("创建主窗口...")
        self.root = tk.Tk()
        self.root.title("SSVEP BCI")
        self.root.geometry("800x600")
        self.root.configure(bg='black')
        
        print("创建按钮...")
        self.buttons = []
        for i, freq in enumerate(STIM_FREQS):
            btn = tk.Button(self.root, text=f"{freq} Hz", font=("Arial", 14),
                           width=10, height=2, bg='gray')
            btn.grid(row=i//4, column=i%4, padx=10, pady=10)
            self.buttons.append(btn)
        
        self.result = tk.Label(self.root, text="等待...", font=("Arial", 18),
                               fg="white", bg="black")
        self.result.grid(row=3, column=0, columnspan=4, pady=20)
        
        self.start_btn = tk.Button(self.root, text="开始", command=self.start,
                                   bg="green", fg="white", font=("Arial", 14))
        self.start_btn.grid(row=4, column=1, pady=10)
        
        self.stop_btn = tk.Button(self.root, text="停止", command=self.stop,
                                  bg="red", fg="white", font=("Arial", 14), state=tk.DISABLED)
        self.stop_btn.grid(row=4, column=2, pady=10)
        
        self.data_source = None
        self.running = False
        print("界面创建完成")
    
    def flash_buttons(self):
        while self.running:
            for i, btn in enumerate(self.buttons):
                brightness = 128 + 80 * np.sin(2 * np.pi * STIM_FREQS[i] * time.time())
                c = f'#{int(brightness):02x}{int(brightness):02x}{int(brightness):02x}'
                self.root.after(0, lambda b=btn, col=c: b.config(bg=col))
            time.sleep(0.02)
    
    def classify_loop(self):
        while self.running:
            eeg = self.data_source.get_data()
            if eeg is not None:
                pred = classify(eeg)
                idx = STIM_FREQS.index(pred)
                self.root.after(0, lambda: self.result.config(text=f"检测到: {pred} Hz"))
                for i, btn in enumerate(self.buttons):
                    if i == idx:
                        self.root.after(0, lambda b=btn: b.config(bg='green'))
                    else:
                        self.root.after(0, lambda b=btn: b.config(bg='gray'))
            time.sleep(0.5)
    
    def start(self):
        print("开始按钮被点击")
        self.data_source = MockDataGenerator()
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.flash_buttons, daemon=True).start()
        threading.Thread(target=self.classify_loop, daemon=True).start()
        print("分类线程已启动")
    
    def stop(self):
        print("停止按钮被点击")
        self.running = False
        if self.data_source:
            self.data_source.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def run(self):
        print("进入主循环...")
        self.root.mainloop()
        print("主循环结束")

if __name__ == "__main__":
    print("="*50)
    print("程序开始运行")
    print("="*50)
    app = App()
    app.run()
    print("程序结束")