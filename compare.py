import torch
import heapq
import jsonpickle
import os
import pandas as pd
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from compare_utils import remove_1, algorithmic_collate3, CompareHelper, quantize_image, infos_to_pianorolls, get_duration_in_interval, shift_image_optimized, piano_roll_to_chroma, calculate_correlation
import glob
from torch.utils.data import Dataset
import unicodedata

covers80_path = "covers80"
youtubecover_jsons = glob.glob(os.path.join(covers80_path, "*.json"))

def get_one_result(info_json):
    results = []
    device = torch.device('cpu')
    use_new_bpm = False
    inst = 'vocal'
    
    # info_json 처리
    test_dataset = TestDataset(info_json, use_new_bpm=use_new_bpm, inst=[inst])
    imgs, labels, points = test_dataset[0]
    test_images = [img for img in imgs]
    test_labels = [label for label in labels]
    test_points = [remove_1(point) for point in points]

    try:
        test_images = torch.cat(test_images).to(device)
    except:
        test_dataset = TestDataset(info_json, use_new_bpm=use_new_bpm, inst=['vocal'], condition=0)
        imgs, labels, points = test_dataset[0]
        test_images = [img for img in imgs]
        test_labels = [label for label in labels]
        test_points = [remove_1(point) for point in points]
        try:
            test_images = torch.cat(test_images).to(device)
        except Exception as e:
            test_dataset = TestDataset(info_json, use_new_bpm=use_new_bpm, inst=['vocal'], condition=0)
            imgs, labels, points = test_dataset[0]
            test_images = [img for img in imgs]
            test_labels = [label for label in labels]
            test_points = [remove_1(point) for point in points]
            try:
                test_images = torch.cat(test_images).to(device)
            except:
                print(e)
                return ["there is no note for this song"], []

    test_bpms = torch.tensor([label['bpm'] for label in labels])
    test_bpms_expanded = test_bpms[:, None]
    test_images_expanded = test_images[:, None, :, :].to(device)
    
    # youtubecover_jsons 처리
    additional_test_dataset = TestDataset2(youtubecover_jsons, inst=[inst], condition=0)
    additional_test_loader = DataLoader(additional_test_dataset, batch_size=40, collate_fn=algorithmic_collate3)
    
    compare_result = []
    max_heap_size = 1000
    
    for idx, (additional_library_images, additional_library_labels, additional_library_points) in tqdm(enumerate(additional_test_loader)):
        additional_library_images = torch.cat(additional_library_images).to(device)
        additional_library_images = additional_library_images.squeeze(1)
        additional_library_images_expanded = additional_library_images[None, :, :, :].to(device)
        additional_library_bpms = torch.tensor([label['bpm'] for label in additional_library_labels]).to(device)
        additional_library_bpms_expanded = additional_library_bpms[None, :]
        
        metrics = calculate_metric_optimized(
            test_images_expanded, 
            additional_library_images_expanded, 
            test_points, 
            additional_library_points, 
            test_bpms_expanded, 
            additional_library_bpms_expanded, 
            device
        )
        
        max_matching_score = torch.zeros_like(metrics)
        
        for i, test_label in enumerate(test_labels):
            for j, additional_library_label in enumerate(additional_library_labels):
                metric = metrics[i, j].item()
                # chord1 = test_labels[i]['chord']
                # chord2 = additional_library_labels[j]['chord']
                # matching_count = sum(c1 == c2 and c1 != 'Unknown' for c1, c2 in zip(chord1, chord2))
                # matching_score = [0, 0.02, 0.05, 0.09, 0.16]
                # max_matching_score[i, j] = matching_score[int(matching_count)]
                final_metric = (metric)
                if final_metric > 1:
                    final_metric = 1

                result_entry = CompareHelper([final_metric, test_label, additional_library_label, test_points[i], additional_library_points[j]])
                
                # heap 크기 제한 로직
                if len(compare_result) < max_heap_size:
                    heapq.heappush(compare_result, result_entry)
                else:
                    # heap이 가득 찬 경우, 최소값보다 큰 경우에만 교체
                    if result_entry.data[0] > compare_result[0].data[0]:
                        heapq.heappop(compare_result)  # 최소값 제거
                        heapq.heappush(compare_result, result_entry)  # 새로운 값 추가
    
    sorted_compare_results = sorted(compare_result, key=lambda x: x.data[0], reverse=True)
    
    return sorted_compare_results




class TestDataset(Dataset):
    def __init__(self, info_path, use_all=False, use_new_bpm=False, inst=['vocal','melody'],condition=4):
        if use_new_bpm:
            self.library_files = [info_path.replace(".json", "newbpm.json")]
        else:
            self.library_files = [info_path]
        self.info_path = info_path
        self.use_all = use_all
        self.inst = inst
        self.condition = condition
    def __len__(self):
        return 1#len(self.library_files) # use_new_bpm이어도 그냥 1임
    def get_chords(self, chord_info, time1, time2):
        if chord_info is None:
            return ['Unknown', 'Unknown', 'Unknown', 'Unknown']
        # time1과 time2 사이의 간격을 4등분
        intervals = [(time1 + i * (time2 - time1) / 4, time1 + (i + 1) * (time2 - time1) / 4) for i in range(4)]
        
        selected_chords = []

        for start_interval, end_interval in intervals:
            best_chord = None
            best_duration = 0
            
            for chord in chord_info:
                if chord['start'] <= end_interval and chord['end'] >= start_interval:
                    duration = get_duration_in_interval(chord, start_interval, end_interval)
                    if duration > best_duration:
                        best_duration = duration
                        best_chord = chord['chord']

            if best_chord:
                selected_chords.append(best_chord)
            else:
                selected_chords.append('Unknown')
        return selected_chords
    def get_structure(self, segment_label, time1, time2):
        max_overlap = 0
        target_label = None
        for segment in segment_label:
            # Calculate overlap between the segment and the time range
            overlap = min(segment['end'], time2) - max(segment['start'], time1)
            
            # If the overlap is negative, it means there is no overlap
            if overlap > 0:
                # Check if this is the maximum overlap found so far
                if overlap > max_overlap:
                    max_overlap = overlap
                    target_label = segment['label']

        return target_label
    def __getitem__(self, idx):
        images=[]
        labels=[]
        points=[]
        info_links = self.library_files 
        for info_link in info_links:
            with open(info_link, 'rb') as f:
                infos =jsonpickle.decode(f.read())
                test_piano, test_timing, test_point = infos_to_pianorolls(infos, self.use_all)
                one_bar_beat = (infos['beat_times'][1] - infos['beat_times'][0]) * infos['rhythm']
                for key in test_piano.keys():
                    if key in self.inst:
                        for time,image in test_piano[key].items():
                            second_values = [item[1] for item in test_point[key][time]]
                            unique_values = set(second_values)
                            condition = self.condition
                            if len(test_point[key][time]) > 4 and len(unique_values) >= 1:
                                image = torch.tensor(image).transpose(0, 1).unsqueeze(dim=0).float()  # 1, 128, 192(64)
                                time1 = infos['downbeat_start'] + one_bar_beat * int(test_timing[time])
                                time2 = time1 + 4 * one_bar_beat
                                chord = self.get_chords(infos['chord_info'], time1, time2)
                                title = unicodedata.normalize('NFC', infos['title'])
                                label = {
                                    "title": title,
                                    "bpm": infos['bpm'],
                                    "newbpm": infos['new_bpm'],
                                    "inst": key,
                                    "time": time1,
                                    "time2": time2,
                                    "link": infos['link'],
                                    "shift": 0,
                                    "platform": infos['platform'],
                                    "song_start": infos['downbeat_start'] + one_bar_beat * int(test_timing[0]),
                                    "song_end": infos['beat_times'][-1],
                                    "chord": chord,
                                    "used_time": None,
                                    "info_link": info_link
                                }
                                images.append(quantize_image(image))
                                labels.append(label)
                                points.append(test_point[key][time])
        return images, labels, points
    

def compare_titles(title1, title2):
    """특수문자와 공백을 모두 제거하고 소문자로 변환하여 비교"""
    def strip_to_basics(title):
        # 알파벳, 숫자만 남기고 전부 제거 후 소문자로 변환
        return ''.join(c.lower() for c in title if c.isalnum())
    
    return strip_to_basics(title1) == strip_to_basics(title2)


class TestDataset2(Dataset):
    def __init__(self, library_files, inst=['vocal','melody'],condition=4):
        self.library_files = library_files # 그냥 여기에 list를 다 박아야함
        self.use_all = True
        self.inst = inst
        self.condition = condition


    def __len__(self):
        return len(self.library_files) # use_new_bpm이어도 그냥 1임
    def get_chords(self, chord_info, time1, time2):
        if chord_info is None:
            return ['Unknown', 'Unknown', 'Unknown', 'Unknown']
        # time1과 time2 사이의 간격을 4등분
        intervals = [(time1 + i * (time2 - time1) / 4, time1 + (i + 1) * (time2 - time1) / 4) for i in range(4)]
        
        selected_chords = []

        for start_interval, end_interval in intervals:
            best_chord = None
            best_duration = 0
            
            for chord in chord_info:
                if chord['start'] <= end_interval and chord['end'] >= start_interval:
                    duration = get_duration_in_interval(chord, start_interval, end_interval)
                    if duration > best_duration:
                        best_duration = duration
                        best_chord = chord['chord']

            if best_chord:
                selected_chords.append(best_chord)
            else:
                selected_chords.append('Unknown')
        return selected_chords
    def get_structure(self, segment_label, time1, time2):
        max_overlap = 0
        target_label = None
        for segment in segment_label:
            # Calculate overlap between the segment and the time range
            overlap = min(segment['end'], time2) - max(segment['start'], time1)
            
            # If the overlap is negative, it means there is no overlap
            if overlap > 0:
                # Check if this is the maximum overlap found so far
                if overlap > max_overlap:
                    max_overlap = overlap
                    target_label = segment['label']

        return target_label
    def __getitem__(self, idx):
        images=[]
        labels=[]
        points=[]
        # 한 번에 하나의 파일만 처리하도록 수정
        info_link = self.library_files[idx]  # idx에 해당하는 파일만
        with open(info_link, 'rb') as f:
            infos =jsonpickle.decode(f.read())
            test_piano, test_timing, test_point = infos_to_pianorolls(infos, True)
            one_bar_beat = (infos['beat_times'][1] - infos['beat_times'][0]) * infos['rhythm']
            for key in test_piano.keys():
                if key in self.inst:
                    for time,image in test_piano[key].items():
                        second_values = [item[1] for item in test_point[key][time]]
                        unique_values = set(second_values)
                        title = unicodedata.normalize('NFC', infos['title'])
                        if len(test_point[key][time]) > 4 and len(unique_values) >= 1:
                            image = torch.tensor(image).transpose(0, 1).unsqueeze(dim=0).float()  # 1, 128, 192(64)
                            time1 = infos['downbeat_start'] + one_bar_beat * int(test_timing[time])
                            time2 = time1 + 4 * one_bar_beat
                            chord = self.get_chords(infos['chord_info'], time1, time2)
                            title = unicodedata.normalize('NFC', infos['title'])
                            label = {
                                "title": title,
                                "bpm": infos['bpm'],
                                "newbpm": infos['new_bpm'],
                                "inst": key,
                                "time": time1,
                                "time2": time2,
                                "shift": 0,
                                "platform": 'youtube',
                                "song_start": infos['downbeat_start'] + one_bar_beat * int(test_timing[0]),
                                "song_end": infos['beat_times'][-1],
                                "chord": chord,
                                "used_time": None,
                                "info_link": info_link
                            }
                            images.append(quantize_image(image))
                            labels.append(label)
                            points.append(test_point[key][time])
        return images, labels, points
    




def calculate_metric_optimized(images1, images2, points1, points2, bpms1, bpms2, device):
    images1 = piano_roll_to_chroma(images1)
    images2 = piano_roll_to_chroma(images2)
    min_length1 = min(images1.shape[0], len(points1))
    min_length2 = min(images2.shape[1], len(points2))
    images1 = images1[:min_length1]
    images2 = images2[:min_length2]
    points1 = points1[:min_length1]
    points2 = points2[:min_length2]
    bpms1 = bpms1[:,:min_length1] 
    bpms2 = bpms2[:,:min_length2] 

    rhythm_images2 = torch.zeros((images2.shape[1], 64)).to(device)
    if rhythm_images2.shape[0] < len(points2):
        rhythm_images2 = torch.zeros((len(points2), 64)).to(device)
    for j, points in enumerate(points2):
        if j < len(rhythm_images2):
            points_tensor = torch.tensor(points).to(device)
            indices = torch.round(points_tensor[:, 0] / 3.0).long()
            indices = torch.clamp(indices, max=63)
            rhythm_images2[j, indices] = 1

    # 모든 시프트 조합에 대한 이미지 계산 및 연결
    shifted_images1_list = []
    shifted_bpms1_list = []
    shift_count = 0
    for pitch_shifts in [0]: # 이 [0]을 pitch variation 등으로 구현해서 다른 변수를 넣을 수 있긴함
        for time_shifts in [-5,-4,-3,-2,-1 ,0,1,2,3,4,5]:
            shifted_images1_list.append(shift_image_optimized(images1, time_shifts, pitch_shifts))
            shifted_bpms1_list.append(bpms1)
            shift_count+=1
    shifted_images1_batch = torch.cat(shifted_images1_list, dim=0).to(device)
    shifted_bpms1_batch = torch.cat(shifted_bpms1_list, dim=0).to(device)
    # rhythm_images1 계산
    rhythm_images1_batch = torch.zeros((shifted_images1_batch.shape[0], 64)).to(device)
    dtw_images1_batch = torch.zeros_like(rhythm_images1_batch)

    for i, points in enumerate(points1):
        points_tensor = torch.tensor(points).to(device)
        start_times = torch.round(points_tensor[:, 0] / 3.0).long()
        pitches = points_tensor[:, 1].long()

        # 시간과 피치를 64와 128로 제한
        start_times = torch.clamp(start_times, max=63)
        pitches = torch.clamp(pitches, max=127)

        # 다음 노트의 시작 시간 계산
        end_times = torch.cat([start_times[1:], torch.tensor([64]).to(device)])
        # rhythm_images1_batch 채우기 (변경 없음)
        for k in range(len(shifted_images1_list)):
            rhythm_images1_batch[i + k * len(points1), start_times] = 1

                # dtw_images1_batch를 직접 채우기
            batch_index = i + k * len(points1)

            # 피치 값을 확장하여 각 구간에 설정
            for j in range(len(start_times)):
                dtw_images1_batch[batch_index, start_times[j]:end_times[j]] = pitches[j].float()

    
        # dtw_images2_batch 초기화
    dtw_images2_batch = torch.zeros_like(rhythm_images2).to(device)

    for j, points in enumerate(points2):
        if j < len(dtw_images2_batch):
            points_tensor = torch.tensor(points).to(device)
            start_times = torch.round(points_tensor[:, 0] / 3.0).long()
            pitches = points_tensor[:, 1].long()

            # 시간과 피치를 64와 128로 제한
            start_times = torch.clamp(start_times, max=63)
            pitches = torch.clamp(pitches, max=127)

            # 다음 노트의 시작 시간 계산
            end_times = torch.cat([start_times[1:], torch.tensor([64]).to(device)])

            # dtw_images2_batch 채우기
            batch_mask = torch.zeros(dtw_images2_batch.size(1)).to(device)

            # 피치 값을 확장하여 각 구간에 설정
            for i in range(len(start_times)):
                batch_mask[start_times[i]:end_times[i]] = pitches[i].float()

            dtw_images2_batch[j] = batch_mask

    min_bpm_optimized = torch.min(shifted_bpms1_batch, bpms2)
    max_bpm_optimized = torch.max(shifted_bpms1_batch, bpms2)
    bpm_ratio_optimized = (min_bpm_optimized / max_bpm_optimized)**0.65

    max_shift = 8
    correlation = calculate_correlation(rhythm_images1_batch, rhythm_images2, max_shift, device)

    #dtw = dtw_with_library(dtw_images1_batch, dtw_images2_batch)#batch_sequence_similarity(dtw_images1_batch, dtw_images2_batch) # 1에 가까울수록 유사도가 높음


    unique_pitches_intersection = ((shifted_images1_batch * images2).sum(dim=(3)) > 0).float().sum(dim=2)
    unique_pitches_image2 = (images2.sum(dim=(3)) > 0).float().sum(dim=2)
    unique_pitches_image1 = (shifted_images1_batch.sum(dim=(3)) > 0).float().sum(dim=2)

    difficulty = 1 / (1 + torch.exp(((unique_pitches_image2 + unique_pitches_image1) - 9) * -0.5))
    pitch_score = 2 * unique_pitches_intersection / (unique_pitches_image2 + unique_pitches_image1)
    final_pitch_score = pitch_score * difficulty

    total = (shifted_images1_batch + images2).clamp_(0, 1).sum(dim=(2, 3))
    intersection = (shifted_images1_batch * images2).sum(dim=(2, 3))
    ratio = intersection / total
    metrics =  (0.5 + 1 * final_pitch_score) * ((ratio) * (1.05) + 0.15 * torch.maximum(correlation, ratio)) * bpm_ratio_optimized # (0.6+1*mse_values) *
    metrics = metrics.clamp_(0, 1)
    metrics_reshaped = metrics.view(shift_count, -1, *metrics.shape[1:])
    max_metric, _ = torch.max(metrics_reshaped, dim=0)


    return max_metric