import torch
import numpy as np

def remove_1(points):
    filtered_points = [point for point in points if point[2] != 1]
    return filtered_points


class CompareHelper:
    def __init__(self, data):
        self.data = data

    def __lt__(self, other):
        return self.data[0] < other.data[0] 


def get_duration_in_interval(chord, start_interval, end_interval):
    """Interval 내에서 chord의 지속 시간을 반환합니다."""
    return min(chord['end'], end_interval) - max(chord['start'], start_interval)


def shift_image_optimized(image, x_shift, y_shift): # 이거 y랑 x랑 뒤집어야함.. time, pitch
    # 이미지를 x와 y 방향으로 동시에 시프트
    _, _, height, width = image.size()
    
    # torch.roll을 사용하여 이미지를 시프트
    shifted_image = torch.roll(image, shifts=(x_shift, y_shift), dims=(3, 2))
    
    # 시프트에 따라 이미지의 가장자리를 잘라냄
    if x_shift > 0:
        shifted_image[:, :, :, :x_shift] = 0
    elif x_shift < 0:
        shifted_image[:, :, :, x_shift:] = 0

    #if y_shift > 0:
    #    shifted_image[:, :, :y_shift, :] = 0
    #elif y_shift < 0:
    #    shifted_image[:, :, y_shift:, :] = 0
    return shifted_image


def algorithmic_collate3(batch):
    imgs, labels, points = zip(*batch)
    return_images = []
    return_labels = []
    return_points = []
    
    for img_list in imgs:
        return_images.extend(img_list)  # 한 단계 더 풀어줌
    for label in labels:
        return_labels.extend(label)
    for point in points:
        return_points.extend(point)
    
    return return_images, return_labels, return_points

def quantize_image(image):
    """
    Quantize the given image tensor.
    
    :param image: torch.Tensor, shape [1, 128, 192], binary values
    :return: torch.Tensor, shape [1, 128, 64], quantized values
    """

    quantized_image = torch.zeros(1, 128, 64)
    
    # Loop through each new pixel position
    for i in range(64):
        # Define the original image slice indexes
        
        # For the first slice, consider only first 2 columns
        if i == 0:
            start_idx = 0
            end_idx = start_idx + 2
        # For other slices, consider 3 columns
        else:
            start_idx = i * 3 - 1
            end_idx = start_idx + 3
        
        # Check if there's at least one '1' in the window
        quantized_image[:, :, i] = (image[:, :, start_idx:end_idx].sum(dim=2) > 0).float()
        
    return quantized_image

def piano_roll_to_chroma(piano_roll):
    """
    Convert a binary piano roll tensor to a binary chroma tensor.
    
    Parameters:
        piano_roll (torch.Tensor): The binary piano roll tensor with shape
            (batch_size, num_channels, num_pitches, num_frames).
            
    Returns:
        torch.Tensor: The binary chroma tensor with shape
            (batch_size, num_channels, 12, num_frames).
    """
    if piano_roll.shape[2] == 12:
        return piano_roll

    # Ensure the piano roll is binary
    binary_piano_roll = (piano_roll > 0).float()

    # Initialize chroma tensor
    chroma = torch.zeros(
        (binary_piano_roll.shape[0], binary_piano_roll.shape[1], 12, binary_piano_roll.shape[3]),
        device=binary_piano_roll.device,
    )
    
    # Sum along the pitch classes modulo 12 (pitches)
    for i in range(12):
        chroma[:, :, i, :] = binary_piano_roll[:, :, i::12, :].max(dim=2).values
    
    return chroma

def calculate_correlation(tensor1, tensor2, max_shift,device):
    #tensor1 = apply_gaussian_filter_1d_to_batch(tensor1,1.5)
    # 초기 최대 상관계수 행렬을 낮은 값으로 초기화
    max_correlation = torch.full((tensor1.size(0), tensor2.size(0)), float('-inf')).to(device)

    for shift in range(-max_shift, max_shift + 1):
        
        # tensor2를 시프트
        shifted_tensor2 = torch.roll(tensor2, shifts=shift, dims=1)
        #shifted_tensor2 = apply_gaussian_filter_1d_to_batch(torch.roll(tensor2, shifts=shift, dims=1),1.5)
        
        # 코사인 유사도 계산
        tensor1_norm = tensor1 / tensor1.norm(dim=1, keepdim=True)
        tensor2_norm = shifted_tensor2 / tensor2.norm(dim=1, keepdim=True)

        
        cosine_similarity = torch.mm(tensor1_norm, tensor2_norm.t())
        max_correlation = torch.max(max_correlation, cosine_similarity)
        """
        
         # L1 코사인 유사도라 해야하나..? 여튼 단순 노트 유사도 계산
        tensor1_expanded = tensor1.unsqueeze(1)
        tensor2_expanded = shifted_tensor2.unsqueeze(0)
        both_one = tensor1_expanded * tensor2_expanded

        # 두 벡터 모두에서 1인 요소의 개수 및 1인 요소의 총합 계산
        both_one_sum = both_one.sum(dim=2)
        total_one_sum = tensor1_expanded.sum(dim=2) + tensor2_expanded.sum(dim=2)
        metric_matrix = both_one_sum / total_one_sum
        max_correlation = torch.max(max_correlation, metric_matrix)
        """
        
    return max_correlation




def infos_to_pianorolls(info, use_all):
    pianorolls={}
    #chromas={} # chroma deprecated
    CONLON_points={}

    # melody_pianorolls={}
    # bass_pianorolls={}
    vocal_pianorolls={}
    # boundary_pianorolls={}

    #melody_chromas={}
    #bass_chromas={}
    #vocal_chromas={} 

    # melody_CONLON_points={}
    # bass_CONLON_points={}
    vocal_CONLON_points={}
    # boundary_CONLON_points={}

    start_points = infos_to_startpoint(info, use_all)

    #shift_val = np.argmax(chart_fit)
    shift_val = 0
    for idx, i in enumerate(start_points):
        #bass를 좀 깔끔하게 만듭니다. Heuristic함
        """
        cleansed_bass={}
        for key, bar in info.bass_info.items():
            if len(bar)>0:
                bar=np.array(bar)
                remain_notes=[]
                to_quantize = 16 # 16분 음표 하나당 최대 1개의 Note를 남깁니다.
                idx_quantize = 48/to_quantize
                for j in range(to_quantize):
                    bass_idx = np.where((bar[:,4]//idx_quantize == j))
                    notes = bar[bass_idx]
                    best_note = get_best_bass(chart_info, notes)
                    if best_note is not None:
                        remain_notes.append(best_note)
                cleansed_bass[key] = np.array(remain_notes)
        """
        # cleansed_bass = info['bass_info']
        # melody = [
        #     info['melody_info'].get(str(i), []) if info['melody_info'] is not None else [],
        #     info['melody_info'].get(str(i+1), []) if info['melody_info'] is not None else [],
        #     info['melody_info'].get(str(i+2), []) if info['melody_info'] is not None else [],
        #     info['melody_info'].get(str(i+3), []) if info['melody_info'] is not None else []
        # ]

        # bass = [
        #     info['bass_info'].get(str(i), []) if info['bass_info'] is not None else [],
        #     info['bass_info'].get(str(i+1), []) if info['bass_info'] is not None else [],
        #     info['bass_info'].get(str(i+2), []) if info['bass_info'] is not None else [],
        #     info['bass_info'].get(str(i+3), []) if info['bass_info'] is not None else []
        # ]

        vocal = [
            info['vocal_info'].get(str(i), []) if info['vocal_info'] is not None else [],
            info['vocal_info'].get(str(i+1), []) if info['vocal_info'] is not None else [],
            info['vocal_info'].get(str(i+2), []) if info['vocal_info'] is not None else [],
            info['vocal_info'].get(str(i+3), []) if info['vocal_info'] is not None else []
        ]

        # boundary = [
        #     info['boundaries'].get(str(i), []) if info['boundaries'] is not None else [],
        #     info['boundaries'].get(str(i+1), []) if info['boundaries'] is not None else [],
        #     info['boundaries'].get(str(i+2), []) if info['boundaries'] is not None else [],
        #     info['boundaries'].get(str(i+3), []) if info['boundaries'] is not None else []
        # ]
        #piano = [info.piano_info.get(str(i),[]),info.piano_info.get(str(i+1),[]),info.piano_info.get(str(i+2), []),info.piano_info.get(str(i+3),[])]

        # melody_pianoroll,  melody_CONLON_point = bar_notes_to_pianoroll(melody, shift_val)
        # bass_pianoroll, bass_CONLON_point = bar_notes_to_pianoroll(bass, shift_val)
        vocal_pianoroll,vocal_CONLON_point = bar_notes_to_pianoroll(vocal, shift_val)
        # boundary_pianoroll, boundary_CONLON_point = bar_notes_to_pianoroll(boundary, shift_val)
        #piano_pianoroll, piano_chroma, piano_CONLON_point = bar_notes_to_pianoroll(piano, shift_val)

        # melody_pianorolls[idx]=melody_pianoroll
        # bass_pianorolls[idx] = bass_pianoroll
        vocal_pianorolls[idx] = vocal_pianoroll
        # boundary_pianorolls[idx]= boundary_pianoroll
        #piano_pianorolls[idx] = piano_pianoroll

        #melody_chromas[idx]=melody_chroma
        #bass_chromas[idx] = bass_chroma
        #vocal_chromas[idx] = vocal_chroma
        #piano_chromas[idx] = piano_chroma

        # melody_CONLON_points[idx] = melody_CONLON_point
        # bass_CONLON_points[idx] = bass_CONLON_point
        vocal_CONLON_points[idx] = vocal_CONLON_point
        # boundary_CONLON_points[idx] = boundary_CONLON_point
        #piano_CONLON_points[idx] = piano_CONLON_point
        

    # pianorolls['melody'] = melody_pianorolls
    # pianorolls['bass'] = bass_pianorolls
    pianorolls['vocal'] = vocal_pianorolls
    # pianorolls['boundary'] = boundary_pianorolls
    #pianorolls['piano'] = piano_pianorolls

    #chromas['melody'] = melody_chromas
    #chromas['bass'] = bass_chromas
    #chromas['vocal'] = vocal_chromas 
    #chromas['piano'] = piano_chromas

    # CONLON_points['melody'] = melody_CONLON_points
    # CONLON_points['bass'] = bass_CONLON_points
    CONLON_points['vocal'] = vocal_CONLON_points
    # CONLON_points['boundary'] = boundary_CONLON_points
    #CONLON_points['piano'] = piano_CONLON_points


    return pianorolls, start_points, CONLON_points # chroma deprecated



def bar_notes_to_pianoroll(bars,shift_val):
    pianoroll = np.zeros((192,128)) #
    conlon_points = []
    for j, bar in enumerate(bars):
        j_offset = j * 48  # 반복되는 계산을 변수에 저장
        for note in bar:
            start, pitch, end = int(note[4]), int(note[2]), int(note[5])
            duration = (end - start + 1)
            start_idx = start + j_offset  # 인덱스 계산 최적화
            end_idx = end + j_offset + 1
            conlon_points.append([start_idx, pitch, duration])
            pianoroll[start_idx:end_idx, pitch] = 1  # 슬라이싱을 사용한 효율적인 할당
    return pianoroll, conlon_points

def infos_to_startpoint(info,use_all):
    downbeat_start = info['downbeat_start']
    

    boundary = round((info['beat_times'][-1] -downbeat_start)/(4*(info['beat_times'][1]-info['beat_times'][0])))-1

    song_structure_sp = [i for i in range(boundary+1)]
    song_structure_sp = refine_breakpoints_custom(song_structure_sp)
    if use_all:
        song_structure_sp = [i for i in range(song_structure_sp[-1])]
    return song_structure_sp

def refine_breakpoints_custom(breakpoints, interval=4):
    refined = []

    unique_breakpoints = []
    for point in breakpoints:
        if point not in unique_breakpoints and point>0: # 0빼고 시작이 애매하긴한데, 예를 들어 verse가 6에서 시작이면 0~4보냐 2~6을 보냐 차이.
            unique_breakpoints.append(point)

    # Determine the starting point
    if len(unique_breakpoints)==0:
        unique_breakpoints.append(0)
    starting_point = unique_breakpoints[0] % interval
    if starting_point != unique_breakpoints[0]:
        for point in range(starting_point, unique_breakpoints[0], interval):
            if point > -1:  # Ensure the point is positive
                refined.append(point)

    for i in range(len(unique_breakpoints)):
        # Add the current breakpoint
        refined.append(unique_breakpoints[i])

        # Check if there is a next breakpoint
        if i + 1 < len(unique_breakpoints):
            next_point = unique_breakpoints[i]
            while next_point + 2*interval <= unique_breakpoints[i + 1]:
                next_point += interval
                refined.append(next_point)
    if len(refined)==0:
        refined = [0]
    return refined
