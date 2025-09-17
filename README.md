<div align="center">

#  MUSIC PLAGIARISM DETECTION: PROBLEM FORMULATION AND A SEGMENT-BASED SOLUTION

<h3>Submitted @ ICCASP 2026</h3>

<p>
  <b>Seonghyeon Go*</b> · <b>Yumin Kim*</b> 
</p>

<p>MIPPIA Inc.</p>

[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://mippia.github.io/icassp-mpd/)
[![Demo Page](https://img.shields.io/badge/Demo-Page-red)](https://huggingface.co/spaces/mippia/MPD-demo)

</div>

## Transcription Demo Usage
```bash
python inference.py {wav_path}
```
This will analyze the audio file and provide simple transcription results with musical segment information, same as demo page.

Please use this demo for only understanding the concept of segment-level matching!

## SMP Dataset Overview

The SMP (Segment-based Music Plagiarism) dataset contains music plagiarism detection pairs with temporal segment annotations. Each row represents a pair of songs with identified similar segments.

## Dataset Structure

| Column | Description |
|--------|-------------|
| `ori_title` | Title of the original song |
| `comp_title` | Title of the comparison song |
| `ori_link` | YouTube link to the original song |
| `comp_link` | YouTube link to the comparison song |
| `relation` | Relationship type (`plag` for plagiarism) |
| `ori_times` | List of start times (in seconds) of similar segments in original song |
| `comp_times` | List of start times (in seconds) of similar segments in comparison song |
| `pair_number` | Unique identifier for song pairs |
| `acoustic_idx` | Unique identifier for segment pairs |

## Data Format
- **Time annotations**: JSON-formatted lists containing start times of similar segments
- **Temporal alignment**: `ori_times` and `comp_times` correspond to matching similar segments between songs
- **Segment duration**: Each segment represents a temporally coherent musical phrase or motif

## Statistics
- Total pairs: Multiple song pairs with plagiarism relationships
- Temporal annotations: Precise start times for similar musical segments
- Multi-language: Includes both English and Korean songs

## License
Our code and demo website are licensed under a 
  <a href="https://www.gnu.org/licenses/gpl-3.0.html" 
     class="text-blue-500 hover:underline">
    GPL License
  </a>.
