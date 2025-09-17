import librosa
import numpy as np
import torch
import scipy.stats as st
from librosa.core import istft, stft
from scipy.signal.windows import hann

def wav_quantizing(wav_file, ori, downbeat_model, beat_tracker, downbeat_tracker, device, bpm=None):
    """

    Get beat timing of given wav_file. This module assumes wav has integer bpm.

    input : path of wav_file
    output : Beat Timing of given wav file in seconds.
    """
    y,sr = librosa.load(wav_file, sr=44100)
    mel_f = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000).T
    x = np.stack([np.dot(np.abs(np.mean(_stft(ori[key]), axis=-1))**2, mel_f) for key in ori])

    #Initialize Beat Transformer to estimate (down-)beat activation from demixed input
    model = downbeat_model
    model.eval()
    PARAM_PATH = {
        4: "ml_models/Beat-Transformer/checkpoint/fold_4_trf_param.pt", # 원래 다른 수도 있었는데, 용량 최적화를 위해 지움.
    }
    x = np.transpose(x, (0, 2, 1))
    x = np.stack([librosa.power_to_db(x[i], ref=np.max) for i in range(len(x))])
    x = np.transpose(x, (0, 2, 1))
    FOLD = 4
    model.load_state_dict(torch.load(PARAM_PATH[FOLD], map_location=torch.device('cuda'))['state_dict'])
    model.to(device)
    model.eval()

    model_input = torch.from_numpy(x).unsqueeze(0).float().to(device)
    activation, _ = model(model_input)

    beat_activation = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
    downbeat_activation = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()
    dbn_beat_pred = beat_tracker(beat_activation)

    combined_act = np.concatenate((np.maximum(beat_activation - downbeat_activation,
                                            np.zeros(beat_activation.shape)
                                            )[:, np.newaxis],
                                downbeat_activation[:, np.newaxis]
                                ), axis=-1)   #(T, 2)
    dbn_downbeat_pred = downbeat_tracker(combined_act)
    dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred[:, 1]==1][:, 0]

    beat_times_ori = dbn_beat_pred
    m_res = st.linregress(np.arange(len(beat_times_ori)),beat_times_ori)
    if bpm:
        bpms=[]
        if bpm>100:
            bpms = [bpm, bpm/2]
            bpm_ratios = [1,1/2]
        else:
            bpms = [bpm, bpm*2]
            bpm_ratios = [1,2]
    else:
        bpm = 60/m_res.slope

        # bpms=[]
        # if bpm>100:
        #     bpms = [round(bpm), round(bpm/2)]
        #     bpm_ratios = [1,1/2]
        # else:
        #     bpms = [round(bpm), round(bpm*2)]
        #     bpm_ratios = [1,2]
        bpms = [round(bpm)]
        bpm_ratios = [1]
    results=[]
    for i, int_bpm in enumerate(bpms):
        bpm_ratio = bpm_ratios[i]
        interpolated_beat_times = interpolate_beat_times(bpm_ratio, int_bpm, beat_times_ori)
        if i==0:
            time_shifted = beat_times_ori-interpolated_beat_times[0::bpm_ratio]
            mode_timing = st.mode(np.around(time_shifted,2)) # 이 매커니즘은 정 bpm에서 계산한걸 그대로 사용하는거로..
        beat_times = interpolated_beat_times +mode_timing.mode

        while beat_times[0]>60/int_bpm:
            beat_times=beat_times - 60/int_bpm
        if beat_times[0]<0:
            beat_times=beat_times + 60/int_bpm

        while len(y)/44100<beat_times[-1]: # if the beat_time has larger value than full song's length due to shift or something
            beat_times = beat_times[:-1]
        beat_times = beat_times[:-1] #

        time_gap = dbn_downbeat_pred[1:]-dbn_downbeat_pred[:-1]
        time_gap = np.round(time_gap/(beat_times[1]-beat_times[0]))
        if len(time_gap)==0:
            rhythm = 4
        else:
            rhythm = int(st.mode(time_gap).mode)
            if rhythm % 3 ==0:
                rhythm = 3
            else:
                rhythm = 4
        downbeat_time = np.remainder(dbn_downbeat_pred, (beat_times[1]-beat_times[0])*rhythm)
        start_downbeat_time = (downbeat_time - beat_times[0]) / (beat_times[1]-beat_times[0])
        start_downbeat_time = st.mode(np.round(start_downbeat_time)).mode
        start_downbeat_time = find_nearest(beat_times, beat_times[0] + start_downbeat_time * (beat_times[1]-beat_times[0]))
        
        results.append((beat_times.tolist(), start_downbeat_time , rhythm, int_bpm))
    return results

def interpolate_beat_times(bpm_ratio, int_bpm, beat_times):
    beat_steps_8th =  np.linspace(0, int(beat_times.size*bpm_ratio)-1, int(beat_times.size*bpm_ratio)) * (60 / int_bpm)
    return beat_steps_8th

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]




def _stft(data: np.ndarray, inverse: bool = False, length = None ):
        """
        Single entrypoint for both stft and istft. This computes stft and
        istft with librosa on stereo data. The two channels are processed
        separately and are concatenated together in the result. The
        expected input formats are: (n_samples, 2) for stft and (T, F, 2)
        for istft.

        Parameters:
            data (numpy.array):
                Array with either the waveform or the complex spectrogram
                depending on the parameter inverse
            inverse (bool):
                (Optional) Should a stft or an istft be computed.
            length (Optional[int]):

        Returns:
            numpy.ndarray:
                Stereo data as numpy array for the transform. The channels
                are stored in the last dimension.
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = 4096
        H = 1024
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {"win_length": None, "length": None} if inverse else {"n_fft": N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = (
                np.concatenate((np.zeros((N,)), data[:, c], np.zeros((N,))))
                if not inverse
                else data[:, :, c].T
            )
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            if inverse:
                s = s[N : N + length]
            s = np.expand_dims(s.T, 2 - inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2 - inverse)