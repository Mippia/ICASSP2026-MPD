import os
import glob
from compare import get_one_result
from segment_transcription import segment_transcription
import sys

def inference(audio_path):
    segment_datas = segment_transcription(audio_path)
    result = get_one_result(segment_datas)
    final_result = result_formatting(result)
    return final_result

def result_formatting(result):
    """
    get_one_result에서 나온 결과를 포맷팅
    result: sorted list of CompareHelper objects
    """
    if not result or len(result) == 0:
        return {
            'matches': [],
            'message': 'No matches found'
        }
    
    # 에러 메시지 체크
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], str):
        return {
            'matches': [],
            'message': result[0]  # "there is no note for this song"
        }
    
    # 상위 3개 결과 추출
    top_3_results = []
    for i, compare_helper in enumerate(result[:3]):
        score = compare_helper.data[0]  # similarity score
        test_label = compare_helper.data[1]  # test song info
        library_label = compare_helper.data[2]  # matched song info
        
        # 라이브러리 레이블에서 정보 추출
        song_title = library_label.get('title', 'Unknown Song')
        library_time = library_label.get('time', 0)  # 매치된 구간의 시간
        library_time2 = library_label.get('time2', 0)
        
        # 테스트 레이블에서 정보 추출
        test_time = test_label.get('time', 0) if test_label else 0  # 입력 곡의 시간
        test_time2 = test_label.get('time2', 0) if test_label else 0
        
        match_info = {
            'rank': i + 1,
            'score': float(score),
            'song_title': song_title,
            'test_time': float(test_time),  # 입력 곡에서 매치된 시간
            'test_time2' : float(test_time2),
            'library_time': float(library_time),  # 라이브러리 곡에서 매치된 시간
            'library_time2': float(library_time2),
            'confidence': f"{score * 100:.1f}%",
            'time_match': f"Input: {test_time:.1f}s ↔ Library: {library_time:.1f}s"
        }
        
        top_3_results.append(match_info)
    
    return {
        'matches': top_3_results,
        'message': 'success'
    }


if __name__ == "__main__":
    audio_path = sys.argv[1]

    result = inference(audio_path)
    print(result)