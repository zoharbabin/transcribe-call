import json

import numpy as np
import pytest

import transcribe_call as tc


def test_clock_under_an_hour():
    assert tc.clock(65) == '01:05'


def test_clock_over_an_hour():
    assert tc.clock(3725) == '1:02:05'


def test_match_name_fixes_ocr_noise():
    assert tc.match_name('Ann Leo', ['Ann Lee', 'bob.ray']) == 'Ann Lee'


def test_match_name_rejects_unrelated_text():
    assert tc.match_name('zzzzzz', ['Ann Lee', 'bob.ray']) is None


def _frame_with_box(x, y, w, h, thickness=4):
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[y:y + h, x:x + w] = (20, 224, 96)
    frame[y + thickness:y + h - thickness, x + thickness:x + w - thickness] = 0
    return frame


def test_active_tile_finds_green_outline():
    # the even-sized morphology kernels can shift the box by a pixel
    assert tc.active_tile(_frame_with_box(50, 40, 200, 120)) == pytest.approx((50, 40, 200, 120), abs=2)


def test_active_tile_ignores_small_green_content():
    frame = np.zeros((300, 400, 3), np.uint8)
    frame[100:120, 100:120] = (20, 224, 96)
    assert tc.active_tile(frame) is None


def _word(text, start, end):
    return {'word': text, 'start': start, 'end': end}


SEGS = [{'start': 0, 'end': 2, 'speaker': 'SPEAKER_00'}, {'start': 4, 'end': 6, 'speaker': 'SPEAKER_01'}]
NAMES = {'SPEAKER_00': 'Ann', 'SPEAKER_01': 'Bob'}


def test_build_turns_splits_on_speaker_change():
    words = [_word(' hello', 0.0, 0.5), _word(' there', 0.6, 1.0), _word(' hi', 5.0, 5.4)]
    assert [(t['speaker'], t['text']) for t in tc.build_turns(words, SEGS, NAMES)] == [
        ('Ann', ' hello there'), ('Bob', ' hi')]


def test_build_turns_splits_same_speaker_after_long_pause():
    segs = [{'start': 0, 'end': 10, 'speaker': 'SPEAKER_00'}]
    words = [_word(' one', 0.0, 0.5), _word(' two', 5.0, 5.5)]
    assert len(tc.build_turns(words, segs, NAMES)) == 2


def test_voice_names_uses_green_border_vote():
    segs = [{'start': 0, 'end': 10, 'speaker': 'SPEAKER_00'}]
    assert tc.voice_names(segs, ['Ann'] * 22, 2) == {'SPEAKER_00': 'Ann'}


def test_voice_names_falls_back_to_numbered_speakers():
    segs = [{'start': 0, 'end': 10, 'speaker': 'SPEAKER_00'}]
    assert tc.voice_names(segs, [None] * 22, 2) == {'SPEAKER_00': 'Speaker 1'}


def _write(tmp_path):
    turns = [{'speaker': 'Ann', 'start': 0, 'end': 1, 'text': ' hello', 'unsure': False}]
    recording = tmp_path / 'call.mov'
    return tc.write_outputs(recording, tmp_path, turns, ['Ann', 'Bob'])


def test_write_outputs_lists_participants_who_did_not_speak(tmp_path):
    assert 'Speakers: Ann. Did not speak: Bob' in _write(tmp_path).read_text()


def test_write_outputs_writes_json_next_to_markdown(tmp_path):
    _write(tmp_path)
    assert json.loads((tmp_path / 'call.transcript.json').read_text())[0]['text'] == ' hello'
