
class Music_info:
    def __init__(self,melody_info=None, bass_info=None, drum_info=None, chord_info=None, vocal_info=None, piano_info=None, chart_scale=None, 
                 title="default_title", bpm=None, rhythm = None, downbeat_start=None, beat_times=None, boundaries = None,
                   segment_label= None, link=None,platform=None, newbpm=None, key=None, structure_starting_point=None, structure_json=None, preview_music_path=None):

        self.melody_info = melody_info
        self.bass_info = bass_info
        self.drum_info = drum_info
        self.chord_info = chord_info
        self.vocal_info = vocal_info
        self.piano_info = piano_info # None for now
        self.chart_scale = chart_scale
        self.title = title
        self.bpm = bpm
        self.rhythm = rhythm
        self.downbeat_start = downbeat_start
        self.beat_times = beat_times
        self.boundaries = boundaries # toplines. idk why I used w
        self.segment_label = segment_label
        self.link = link
        self.preview_music_path = preview_music_path
        self.platform = platform
        self.new_bpm = newbpm
        self.key = key
        self.structure_starting_point = structure_starting_point
        self.structure_json = structure_json # 이게 진짜 어려움. lyric이나 chord, 곡 구조 등의 정보를 index키와 함께 저장해야함.



    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)