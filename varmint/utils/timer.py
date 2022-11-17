import time
from collections import defaultdict


class TimerError(Exception):
    """A custom exception for the Timer class."""


class Timer():
    def __init__(self):
        self.aggregated_times = defaultdict(lambda: 0.0)
        self.aggregated_counts = defaultdict(lambda: 0)
        self._current_frames = []
        self._global_start = time.perf_counter()

    def start(self, clock_name):
        for frame_start, frame_name in self._current_frames:
            if frame_name == clock_name:
                raise TimerError(f'Already recording with clock name {clock_name}.')

        current_frame = (time.perf_counter(), clock_name)
        self._current_frames.append(current_frame)

    def stop(self, clock_name):
        if len(self._current_frames) == 0:
            raise TimerError(f'Exiting with no active timer.')

        last_frame_start, last_frame_name = self._current_frames[-1]
        if last_frame_name != clock_name:
            raise TimerError(f'Frame mismatch.')

        elapsed_time = time.perf_counter() - last_frame_start
        self.aggregated_times[clock_name] += elapsed_time
        self.aggregated_counts[clock_name] += 1
        self._current_frames.pop()

    def reset(self):
        if len(self._current_frames) > 0:
            print(f'WARNING: Active timer frames while resetting.')

        self.aggregated_times = defaultdict(lambda: 0.0)
        self.aggregated_counts = defaultdict(lambda: 0)
        self._current_frames = []
        self._global_start = time.perf_counter()

    def summarize(self, verbose=True):
        print(f'Summarizing Timer statistics:')
        total_time = time.perf_counter() - self._global_start
        max_width = max(len(s) for s in self.aggregated_times.keys())

        stats = {}
        for key in self.aggregated_times.keys():
            avg_time = self.aggregated_times[key] / self.aggregated_counts[key]
            percent_times = self.aggregated_times[key] / total_time
            stats[key] = {
                'count': self.aggregated_counts[key],
                'total_time': self.aggregated_times[key],
                'average_per_call': avg_time,
                'total_percentage': 100 * percent_times,
            }

        stats_list = [(key, stats[key]) for key in stats.keys()]
        stats_list = sorted(stats_list, key = lambda s: -s[1]['total_time'])
        if verbose:
            for key, vals in stats_list:
                print(f"\t{key:>{max_width}}: Count {vals['count']} | Total Time {vals['total_time']:0.6f} | Average Per Call {vals['average_per_call']:0.6f} | Total % {vals['total_percentage']:0.6}%.")

        return stats


class Time():
    def __init__(self, timer, clock_name):
        self.timer = timer
        self.clock_name = clock_name

    def __enter__(self):
        self.timer.start(self.clock_name)
        return self

    def __exit__(self, *exc_info):
        self.timer.stop(self.clock_name)


if __name__ == '__main__':
    t = Timer()
    with Time(t, 'first'):
        blah = 0
        for i in range(10000):
            blah += 1
    with Time(t, 'second'):
        blah = 0
        for i in range(100000):
            blah += 1
        with Time(t, 'third'):
            for i in range(10000):
                blah += 1
    with Time(t, 'first'):
        blah = 0
        for i in range(10000):
            blah += 1
    stats = t.summarize()
    print(stats)
