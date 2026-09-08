from aq.common import cpu_peak_memory_bytes


def test_cpu_peak_memory_is_reported_in_bytes():
    peak = cpu_peak_memory_bytes()

    assert peak is None or isinstance(peak, int)
    assert peak is None or peak > 0
