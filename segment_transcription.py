import os
import numpy as np
import librosa
import soundfile
import demucs.separate
from wav_quantizer import wav_quantizing
from ml_models.AST.do_everything import vocal_trans
from music_info import Music_info
from ml_models.DilatedTransformer import Demixed_DilatedTransformerModel
from madmom.features.beats import DBNBeatTrackingProcessor
import shutil
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from utils import vocal_midi2note, quantize, chord_quantize, save_to_json
import time
import uuid

downbeat_model = Demixed_DilatedTransformerModel(attn_len=5, instr=5, ntoken=2,
                                            dmodel=256, nhead=8, d_hid=1024,
                                            nlayers=9,  norm_first=True)
beat_tracker = DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=44100/1024,
                                        transition_lambda=100, observation_lambda=6,
                                        num_tempi=None, threshold=0.2)
downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4],
                                                min_bpm=55.0, max_bpm=215.0, fps=44100/1024,
                                                transition_lambda=100, observation_lambda=6,
                                                num_tempi=None, threshold=0.2)

device = 'cuda'

def segment_transcription(audio_path):
    """
    개선된 segment_transcription 함수
    - 고유한 임시 폴더 사용으로 동시 처리 지원
    - try-finally로 안전한 파일 정리
    """
    wav_path = audio_path
    wav_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    # 고유한 폴더명 생성 (타임스탬프 + UUID)
    unique_id = f"{wav_name}_{int(time.time() * 1000)}_{str(uuid.uuid4())[:8]}"
    separated_base = f"separated_{unique_id}"
    
    print(f"Processing {wav_name} in temporary folder: {separated_base}")
    
    try:
        # 첫 번째 분리: piano vs no_piano
        print("Step 1: Separating piano...")
        demucs.separate.main([
            "--two-stems", "piano", 
            "-n", "htdemucs_6s", 
            "-o", separated_base, 
            wav_path
        ])   
        
        piano_wav_name = f"{separated_base}/htdemucs_6s/{wav_name}/piano.wav"
        others_name = f"{separated_base}/htdemucs_6s/{wav_name}/no_piano.wav"
        to_name = f"{separated_base}/htdemucs_6s/{wav_name}/{wav_name}.wav"
        
        # 파일명 변경
        if os.path.exists(others_name):
            os.rename(others_name, to_name)
        else:
            raise FileNotFoundError(f"Expected file not found: {others_name}")
        
        # 두 번째 분리: vocals, drums, bass, other
        print("Step 2: Separating vocals, drums, bass, other...")
        demucs.separate.main([
            "-n", "htdemucs", 
            "-o", separated_base, 
            to_name
        ])
        
        # 분리된 파일 경로들
        vocal_wav_name = f"{separated_base}/htdemucs/{wav_name}/vocals.wav"
        drum_wav_name = f"{separated_base}/htdemucs/{wav_name}/drums.wav"
        other_wav_name = f"{separated_base}/htdemucs/{wav_name}/other.wav"
        bass_wav_name = f"{separated_base}/htdemucs/{wav_name}/bass.wav"

        # 파일 존재 확인
        required_files = [vocal_wav_name, drum_wav_name, other_wav_name, bass_wav_name, piano_wav_name]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required separated file not found: {file_path}")

        vocal_wav_path = os.path.abspath(vocal_wav_name)
        drum_wav_path = os.path.abspath(drum_wav_name)
        other_wav_path = os.path.abspath(other_wav_name)
        bass_wav_path = os.path.abspath(bass_wav_name)
        abs_wav_path = os.path.abspath(wav_path)

        print("Step 3: Loading separated audio files...")
        vocals = librosa.load(vocal_wav_name, sr=44100, mono=False)[0]
        piano = librosa.load(piano_wav_name, sr=44100, mono=False)[0]
        drums = librosa.load(drum_wav_name, sr=44100, mono=False)[0]
        bass = librosa.load(bass_wav_name, sr=44100, mono=False)[0]
        other = librosa.load(other_wav_name, sr=44100, mono=False)[0]

        spleeter_dict = {
            'vocals': np.asarray(vocals).T, 
            'piano': np.asarray(piano).T, 
            'drums': np.asarray(drums).T, 
            'bass': np.asarray(bass).T, 
            'other': np.asarray(other).T
        }

        print("Step 4: Combining piano and other tracks...")
        real_others = librosa.load(piano_wav_name, sr=44100, mono=False)[0] + librosa.load(other_wav_name, sr=44100, mono=False)[0]
        soundfile.write(other_wav_name, real_others.T, 44100)

        print("Step 5: Quantizing audio...")
        quantize_result = wav_quantizing(wav_path, spleeter_dict, downbeat_model, beat_tracker, downbeat_tracker, device) 
        
        print("Step 6: Transcribing vocals...")
        vocal_notes = vocal_midi2note(vocal_trans(vocal_wav_path, device=device))
        
        # chord_info = transcript("chord", wav_path)[1]  # 주석 처리됨
        sav_path = wav_path[:-4] + ".json" 

        beat_times, downbeat_start, rhythm, bpm = quantize_result[0]
        chord_time_gap = (beat_times[1] - beat_times[0]) * rhythm
        vocal_infos = quantize(vocal_notes, beat_times, downbeat_start, chord_time_gap)
        # chord_infos = chord_quantize(chord_info, beat_times)  # 주석 처리됨
        
        print("Step 7: Creating music info object...")
        wav_music_info = Music_info(
                melody_info=None, 
                bass_info=None, 
                chord_info=None, 
                vocal_info=vocal_infos,
                chart_scale=None, 
                title=str(wav_name), 
                bpm=int(bpm), 
                rhythm=int(rhythm), 
                downbeat_start=float(downbeat_start),
                beat_times=beat_times, 
                boundaries=None, 
                segment_label=None, 
                link=None,
            )
            
        os.makedirs(os.path.dirname(sav_path), exist_ok=True)
        save_to_json(wav_music_info, sav_path)
        
        print(f"Successfully processed {wav_name} -> {sav_path}")
        return sav_path
        
    except Exception as e:
        print(f"Error processing {wav_name}: {str(e)}")
        raise e
        
    finally:
        # 해당 처리 세션의 임시 폴더만 정리
        if os.path.exists(separated_base):
            print(f"Cleaning up temporary folder: {separated_base}")
            try:
                shutil.rmtree(separated_base)
                print(f"Successfully cleaned up: {separated_base}")
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up {separated_base}: {cleanup_error}")
        else:
            print(f"Temporary folder {separated_base} not found (already cleaned up?)")
