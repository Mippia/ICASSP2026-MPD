import pretty_midi
import jsonpickle
def vocal_midi2note(midi):
    """
    """
    
    notes=[]
    for note in midi:
        pretty_note =pretty_midi.Note(velocity=100, start=note[0], end=note[1], pitch=note[2])
        notes.append(pretty_note)
    return notes


def quantize(notes, beat_times, downbeat_start, chord_time_gap):
    """
    어떤 Note가 몇번째 Bar의 몇번째 timing부터 몇번째 timing까지 나타나는지를 return해서 준다.

    Pianoroll의 Index를 넘겨준다? 라고 생각하면 적당히 맞다.

    ex) 1마디가 1초인 곡에서 연주 시간이 4.25~4.75인 음이 있고, 1마디를 48분 음표까지 고려한다면
    5번째 마디에 12~35까지 연주함.. 이라는 정보를 건네줌
    
    """
    first_beat = downbeat_start
    one_beat_time = beat_times[1]-beat_times[0] #그냥 1비트
    quantize_48th_time = one_beat_time/12 
    beat_num = chord_time_gap//one_beat_time * 12 # 4박자 곡이면 48, 3박자 곡이면 36 -> 이거 24나오면.. 시각화 망가지겠네? 
    max_idx=0
    for note in notes:
        start_idx = round((note.start-downbeat_start)/quantize_48th_time)
        end_idx = round((note.end-downbeat_start)/quantize_48th_time)
        if max_idx <int(start_idx // beat_num):
            max_idx = int(start_idx// beat_num)

    note_info={str(key) : [] for key in range(max_idx)}

    for note in notes:
        if note.start>downbeat_start: # 극초반의 일부 음표가 생략될 수도 있긴합니다.
            start_idx = round((note.start-downbeat_start)/quantize_48th_time)
            end_idx = round((note.end-downbeat_start)/quantize_48th_time)
            if end_idx == start_idx:
                end_idx+=1

            note_start = start_idx * quantize_48th_time + first_beat
            note_end  = end_idx * quantize_48th_time + first_beat
            note_pitch = note.pitch
            note_velocity = note.velocity

            bar_idx = int(start_idx // beat_num)
            bar_pos = start_idx % beat_num 
            bar_pos_end = end_idx % beat_num # 이거 때문에, 음 길이가 한 마디를 못넘어 감 *** 예를들어 beatnum이 48이고 35~67이라 하면 35 ~ 19 되었다가 if문 타면서 35~47됨. 
            if bar_pos_end<bar_pos and int(end_idx//beat_num) > bar_idx:
                bar_pos_end = (int(end_idx//beat_num) - bar_idx) * beat_num # 이제는 구현 함. 나중에 index에러 반드시 날거임

            if bar_pos_end<bar_pos:
                bar_pos_end = beat_num-1

            note = [float(note_start), float(note_end), int(note_pitch), int(note_velocity), int(bar_pos), int(bar_pos_end)]
            #note = {'start':note_start, 'end':note_end, 'pitch':note_pitch, 'velocity':note_velocity, 'start_idx':bar_pos, 'end_idx':bar_pos_end}
            if str(bar_idx) not in note_info:
                note_info[str(bar_idx)]=[note]
            else:
                note_info[str(bar_idx)].append(note)

    return note_info





def chord_quantize(chord_info, beat_times):
    """
    returns Quantized Chord info, First chord starting point and chord time(3박이냐 4박이냐에 따라 chord time이 달라집니다. 코드 변화가 한 마디 내에서 여러번 나올 수 있긴 하지만 전반적으로 마디 가장 처음 1번 이루어진다는 가정을 사용합니다.)
    first chord는 첫 Downbeat의 시작을 의미합니다. 다만 고쳐야할 것 같네요..
    """
    first_beat = beat_times[0]
    one_beat_time = beat_times[1]-beat_times[0]
    q_chord_info = []

    for chord in chord_info:
        chord_dict={}
        chord_dict['chord'] = chord['chord']
        chord_dict['start'] = float(round((chord['start']-first_beat)/one_beat_time) * one_beat_time + first_beat) # 0.2, 0.6, 1.0, 1.4 .... 가 있고 chord timing이 1.9라면 1.8을 return하는 코드
        end_time = round((chord['end']-first_beat)/one_beat_time) * one_beat_time + first_beat
        if end_time==chord_dict['start']:
            end_time += one_beat_time
        chord_dict['end'] = float(end_time)
        q_chord_info.append(chord_dict)
    
    return q_chord_info


def save_to_json(data, filename):
    """데이터를 JSON 파일로 저장합니다."""
    with open(filename, 'w', encoding='utf-8') as file:
        # JSON 형식으로 변환
        json_data = jsonpickle.encode(data, unpicklable=False)
        # 파일에 쓰기
        file.write(json_data)
