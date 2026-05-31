# ssvep_fbcca.py 的完整正确版本
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

DATA_FOLDER = "./cca_ssvep"
FS = 256
ONSET_DELAY = 0.135
# 官方频率顺序
STIM_FREQS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
              10.25, 12.25, 14.25, 10.75, 12.75, 14.75]

def load_data(subject_num=1):
    file_path = f"{DATA_FOLDER}/s{subject_num}.mat"
    data = loadmat(file_path)
    eeg = data["eeg"]
    return eeg

def preprocess_eeg(eeg):
    n_targets, n_chans, n_samples, n_blocks = eeg.shape
    for target in range(n_targets):
        for ch in range(n_chans):
            for block in range(n_blocks):
                eeg[target, ch, :, block] -= np.mean(eeg[target, ch, :, block])
    return eeg

def create_reference(freq, duration, fs=FS, n_harmonics=3):
    t = np.arange(0, duration, 1/fs)
    Y = []
    for h in range(1, n_harmonics + 1):
        Y.append(np.sin(2 * np.pi * h * freq * t))
        Y.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(Y).T

def cca_correlation(X, Y):
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    min_len = min(X.shape[0], Y.shape[0])
    X = X[:min_len, :]
    Y = Y[:min_len, :]
    Cxx = X.T @ X + 1e-6 * np.eye(X.shape[1])
    Cyy = Y.T @ Y + 1e-6 * np.eye(Y.shape[1])
    Cxy = X.T @ Y
    try:
        inv_Cxx = np.linalg.inv(Cxx)
        inv_Cyy = np.linalg.inv(Cyy)
        K = inv_Cxx @ Cxy @ inv_Cyy @ Cxy.T
        eigenvalues = np.linalg.eigvals(K)
        max_corr = np.sqrt(np.max(np.real(eigenvalues)))
        return max_corr
    except:
        return 0

def fbcca_classify(eeg_segment, freqs, duration, fs=FS):
    subbands = [(5, 35), (8, 40), (11, 45), (14, 50)]
    correlations = {freq: 0 for freq in freqs}
    for subband_idx, (low, high) in enumerate(subbands):
        nyquist = fs / 2
        b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
        X_filtered = filtfilt(b, a, eeg_segment, axis=0)
        for freq in freqs:
            Y = create_reference(freq, duration, fs)
            corr = cca_correlation(X_filtered, Y)
            weight = 1.0 / (subband_idx + 1)
            correlations[freq] += weight * corr
    best_freq = max(correlations, key=correlations.get)
    return best_freq, correlations

def evaluate_subject(subject_num, duration):
    print(f"\n评估被试 {subject_num} (窗口: {duration}秒)")
    eeg = load_data(subject_num)
    eeg = preprocess_eeg(eeg)
    n_targets, n_chans, n_samples, n_blocks = eeg.shape
    onset_sample = int(ONSET_DELAY * FS)
    window_samples = int(duration * FS)
    if onset_sample + window_samples > n_samples:
        window_samples = n_samples - onset_sample
    trials_data = []
    trials_labels = []
    for block in range(n_blocks):
        for target in range(n_targets):
            trial_data = eeg[target, :, onset_sample:onset_sample+window_samples, block].T
            trials_data.append(trial_data)
            trials_labels.append(STIM_FREQS[target])
    n_trials = len(trials_data)
    correct = 0
    for i, trial in enumerate(trials_data):
        pred, _ = fbcca_classify(trial, STIM_FREQS, window_samples/FS)
        if pred == trials_labels[i]:
            correct += 1
    accuracy = correct / n_trials * 100
    print(f"结果: {correct}/{n_trials} = {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # 选择要运行的模式
    mode = input("选择模式: 1-测试窗口长度, 2-评估所有被试: ")
    
    if mode == "1":
        # 模式1: 测试不同窗口长度
        print("\n=== 测试不同窗口长度 ===")
        for duration in [1.0, 2.0, 3.0, 4.0]:
            acc = evaluate_subject(1, duration)
    else:
        # 模式2: 评估所有10个被试
        print("\n=== 评估所有被试 (窗口: 3.0秒) ===")
        DURATION = 3.0   #4.0准确性更高，但更慢
        accuracies = []
        for subj in range(1, 11):
            try:
                acc = evaluate_subject(subj, DURATION)
                accuracies.append(acc)
            except Exception as e:
                print(f"被试 {subj} 加载失败: {e}")
                accuracies.append(np.nan)
        
        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        valid_acc = [a for a in accuracies if not np.isnan(a)]
        if valid_acc:
            print(f"平均准确率: {np.mean(valid_acc):.2f}%")
            print(f"最高准确率: {np.max(valid_acc):.2f}%")
            print(f"最低准确率: {np.min(valid_acc):.2f}%")