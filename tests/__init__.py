"""SafeVision test-suite.

Every test is written with the standard library ``unittest`` so it runs without
any extra dependency::

    python -m unittest discover -s tests -v
    python -m unittest discover -s tests -p "test_collision.py" -v

``pytest`` also works if you prefer it::

    pytest tests -q

The suite never needs YOLO weights or a network connection: the detector is
replaced by :class:`~ai_engine.detection.scripted.ScriptedDetector` and the
videos are generated on the fly.
"""
