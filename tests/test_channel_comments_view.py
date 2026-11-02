from yezatas.comments import CATEGORIES as COMMENT_CATEGORIES
from yezatas.store import Comment, CommentReply, ReplyTemplate, Video
from yezatas.web.views import build_channel_comments_view

VID1 = "vidaaaaaaa1"
VID2 = "vidaaaaaaa2"


def _videos():
    return {
        VID1: Video(video_id=VID1, title="Cactus repotting guide"),
        VID2: Video(video_id=VID2, title="Second video"),
    }


def _comments():
    # newest first by published_at
    return [
        Comment(comment_id="c1", video_id=VID1, author="Alice", text="What soil?", published_at="2026-10-04T10:00:00Z", like_count=2, fetched_at="2026-10-05T10:00:00Z", category="question"),
        Comment(comment_id="c2", video_id=VID1, author="Bob", text="Nice!", published_at="2026-10-03T10:00:00Z", like_count=5, fetched_at="2026-10-05T10:00:00Z", category="praise"),
        Comment(comment_id="c3", video_id=VID2, author="Carol", text="Hello?", published_at="2026-10-02T10:00:00Z", like_count=0, fetched_at="2026-10-04T10:00:00Z", category="question"),
        Comment(comment_id="c4", video_id=VID2, author="Dave", text="random", published_at="2026-10-01T10:00:00Z", like_count=1, fetched_at="2026-10-03T10:00:00Z", category=""),
        Comment(comment_id="c5", video_id=VID1, author="Eve", text="Spam here", published_at="2026-09-30T10:00:00Z", like_count=0, fetched_at="2026-10-02T10:00:00Z", category="spam"),
    ]


def _replies():
    return [
        CommentReply(id=1, video_id=VID1, comment_id="c1", text="draft for c1", model="m1", created_at="2026-10-05T11:00:00Z", status="open", notes=()),
        CommentReply(id=2, video_id=VID2, comment_id="c4", text="draft for c4", model="m1", created_at="2026-10-05T11:00:00Z", status="open", notes=()),
        CommentReply(id=99, video_id=VID1, comment_id="", text="pinned draft", model="m1", created_at="2026-10-05T11:00:00Z", status="open", notes=()),
    ]


def _templates():
    return [ReplyTemplate(id=1, name="T1", body="Hi {name}, thanks!", created_at="2026-06-01T00:00:00Z", updated_at="2026-06-01T00:00:00Z")]


def test_rows_newest_first_and_meta():
    comments = _comments()
    replies = _replies()
    videos = _videos()
    cv = build_channel_comments_view(comments, replies, videos, _templates(), {})
    assert [r.comment.comment_id for r in cv.rows] == ["c1", "c2", "c3", "c4", "c5"]
    assert cv.rows[0].video_title == "Cactus repotting guide"
    assert cv.rows[0].video_href == f"/videos/{VID1}"
    assert cv.total == 5
    assert cv.video_count == 2
    assert cv.unsorted == 1  # c4 unsorted
    assert cv.fetched_text != ""


def test_category_filter_and_chips():
    comments = _comments()
    cv = build_channel_comments_view(comments, [], _videos(), [], {"c": "question"})
    assert all(r.comment.category == "question" for r in cv.rows)
    assert len(cv.rows) == 2
    # chip counts over scope (all videos)
    chip_map = {ch.key: ch for ch in cv.chips}
    assert chip_map["question"].count == 2
    assert chip_map["question"].active is True
    assert chip_map["question"].href == "/comments?c=question"
    # all chip href
    assert chip_map["all"].href == "/comments"


def test_video_filter_and_rail():
    comments = _comments()
    cv = build_channel_comments_view(comments, [], _videos(), [], {"v": VID2})
    assert cv.video_filter == VID2
    assert all(r.video_id == VID2 for r in cv.rows)
    # chip counts limited to that video
    chip_map = {ch.key: ch for ch in cv.chips}
    assert chip_map["all"].count == 2
    # rail still lists both, second active
    assert len(cv.videos) == 2
    # order by newest comment: VID1 first (c1 newest)
    assert cv.videos[0].video_id == VID1
    assert cv.videos[1].video_id == VID2
    assert cv.videos[1].active is True
    assert "v=vidaaaaaaa2" in chip_map["question"].href
    # unknown v ignored
    cv2 = build_channel_comments_view(comments, [], _videos(), [], {"v": "unknown123"})
    assert cv2.video_filter == ""


def test_draft_only():
    comments = _comments()
    replies = _replies()
    cv = build_channel_comments_view(comments, replies, _videos(), [], {"draft": "1"})
    assert cv.draft_only is True
    assert set(r.comment.comment_id for r in cv.rows) == {"c1", "c4"}
    # pinned draft never shows as row draft
    for r in cv.rows:
        assert r.draft is None or r.draft.comment_id != ""
    # draft_only_href and next_path
    cv_q = build_channel_comments_view(comments, replies, _videos(), [], {"c": "question"})
    assert cv_q.draft_only_href == "/comments?c=question&draft=1"
    assert cv_q.next_path == "/comments?c=question"
    cv_qd = build_channel_comments_view(comments, replies, _videos(), [], {"c": "question", "draft": "1"})
    assert cv_qd.draft_only_href == "/comments?c=question"
    assert cv_qd.next_path == "/comments?c=question&draft=1"


def test_rail_meta_and_unsorted():
    comments = _comments()
    cv = build_channel_comments_view(comments, [], _videos(), [], {})
    # VID1 has 1 question, 3 total
    v1 = next(v for v in cv.videos if v.video_id == VID1)
    assert v1.meta == "1 question · 3 comments"
    assert v1.unsorted == 0
    v2 = next(v for v in cv.videos if v.video_id == VID2)
    assert v2.meta == "1 question · 2 comments" or v2.meta == "1 question \u00b7 2 comments"
    # check contains question and count
    assert "1 comment" in v2.meta or "2 comments" in v2.meta
    # unsorted per video: VID2 has 1 unsorted
    assert v2.unsorted == 1


def test_empty():
    cv = build_channel_comments_view([], [], {}, [], {})
    assert cv.rows == ()
    assert cv.fetched_text == ""
    assert cv.total == 0
    assert cv.video_count == 0
