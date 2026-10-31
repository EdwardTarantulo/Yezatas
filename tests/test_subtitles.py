import re

import pytest

from yezatas.subtitles import srt_time, to_srt


def seg(start_ms: int, end_ms: int, text: str) -> dict:
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text}


def parse_cues(srt: str) -> list[tuple[int, int, int, str]]:
    """(number, start_ms, end_ms, text) of each cue."""
    cues = []
    for block in srt.strip("\n").split("\n\n"):
        number, times, *text = block.split("\n")
        m = re.fullmatch(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})", times)
        assert m, times
        a = [int(x) for x in m.groups()]
        cues.append((int(number), ((a[0] * 60 + a[1]) * 60 + a[2]) * 1000 + a[3],
                     ((a[4] * 60 + a[5]) * 60 + a[6]) * 1000 + a[7], "\n".join(text)))
    return cues


def test_srt_time_format() -> None:
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(61_005) == "00:01:01,005"
    assert srt_time(3_723_456) == "01:02:03,456"
    assert srt_time(-5) == "00:00:00,000"


def test_exact_output_for_small_input() -> None:
    srt = to_srt([seg(0, 2500, "Hello there."), seg(2500, 65_000, "Second  cue\nhere.")])
    assert srt == ("1\n00:00:00,000 --> 00:00:02,500\nHello there.\n"
                   "\n"
                   "2\n00:00:02,500 --> 00:01:05,000\nSecond cue here.\n")


def test_empty_input_and_empty_segments_make_no_cue() -> None:
    assert to_srt([]) == ""
    assert to_srt([seg(0, 1000, ""), seg(1000, 2000, "   \n ")]) == ""


def test_empty_segments_are_skipped_and_numbering_continues() -> None:
    srt = to_srt([seg(0, 1000, "One."), seg(1000, 2000, ""), seg(2000, 3000, "Two.")])
    assert [(n, t) for n, _, _, t in parse_cues(srt)] == [(1, "One."), (2, "Two.")]
    assert srt.endswith("Two.\n") and not srt.endswith("\n\n")


def test_long_text_wraps_into_two_lines_in_one_cue() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"  # 45 chars
    srt = to_srt([seg(1000, 5000, text)], max_line=30)
    cues = parse_cues(srt)
    assert len(cues) == 1
    assert cues[0][1:3] == (1000, 5000)
    lines = cues[0][3].split("\n")
    assert len(lines) == 2 and all(len(line) <= 30 for line in lines)
    assert " ".join(lines) == text


def test_very_long_text_splits_into_cues_inside_the_segment() -> None:
    words = [f"word{i:02d}" for i in range(40)]  # 40 words, 6 chars + space each
    text = " ".join(words)
    srt = to_srt([seg(10_000, 30_000, text)], max_line=20, max_lines=2)
    cues = parse_cues(srt)
    assert len(cues) > 2
    assert [c[0] for c in cues] == list(range(1, len(cues) + 1))
    assert cues[0][1] == 10_000 and cues[-1][2] == 30_000  # the segment's own start and end
    previous_end = 10_000
    for _, start, end, cue_text in cues:
        assert previous_end == start  # contiguous, increasing
        assert start < end <= 30_000
        assert all(len(line) <= 20 for line in cue_text.split("\n"))
        assert len(cue_text.split("\n")) <= 2
        previous_end = end
    assert " ".join(c[3].replace("\n", " ") for c in cues) == text  # no word lost or changed


def test_split_time_is_proportional_to_characters() -> None:
    text = "aaaaaaaaa bbbbbbbbb ccccccccc ddddddddd"  # four 9-char words
    cues = parse_cues(to_srt([seg(0, 4000, text)], max_line=9, max_lines=1))
    assert [(s, e) for _, s, e, _ in cues] == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]


def test_one_word_longer_than_a_line_is_broken() -> None:
    cues = parse_cues(to_srt([seg(0, 1000, "x" * 50)], max_line=20, max_lines=1))
    assert [len(c[3]) for c in cues] == [20, 20, 10]


def test_times_never_leave_their_segment_across_many_segments() -> None:
    segments = [seg(i * 7000, i * 7000 + 6500, "some words that are said here " * (i % 5 + 1))
                for i in range(30)]
    cues = parse_cues(to_srt(segments))
    bounds = [(s["start_ms"], s["end_ms"]) for s in segments]
    last_start = -1
    for _, start, end, _ in cues:
        assert start >= last_start and start <= end
        last_start = start
        assert any(a <= start and end <= b for a, b in bounds)


def test_end_before_start_is_clamped() -> None:
    cues = parse_cues(to_srt([seg(5000, 4000, "Odd.")]))
    assert cues[0][1:3] == (5000, 5000)


def test_bad_limits_are_refused() -> None:
    with pytest.raises(ValueError):
        to_srt([seg(0, 1, "x")], max_line=0)
