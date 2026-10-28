from consentcam.tracker import IoUTracker, RegionMemory, iou


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0


def test_track_survives_short_dropout():
    # the whole point: a face the detector loses for a few frames
    # must still be there (and still get blurred)
    tr = IoUTracker(hold=3)
    tr.update([(100, 100, 40, 40)])
    for _ in range(3):
        tracks = tr.update([])
        assert len(tracks) == 1 and tracks[0].misses > 0
    assert tr.update([]) == []


def test_same_face_keeps_id_while_moving():
    tr = IoUTracker()
    a = tr.update([(100, 100, 40, 40)])[0].tid
    b = tr.update([(106, 102, 40, 40)])[0].tid
    assert a == b


def test_new_tracks_start_untrusted():
    tr = IoUTracker()
    t = tr.update([(0, 0, 30, 30)])[0]
    assert not t.allowed


def test_region_memory_expires():
    m = RegionMemory(hold=2)
    m.update([(0, 0, 5, 5)])
    assert m.tick() and m.tick()
    assert m.tick() == []
