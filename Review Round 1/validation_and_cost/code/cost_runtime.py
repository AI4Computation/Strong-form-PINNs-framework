"""Process-local Windows resource measurements; no installed-environment edits."""
import ctypes
from ctypes import wintypes
import time


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize',
         'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
         'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage']]


class PowerStatus(ctypes.Structure):
    _fields_ = [('ACLineStatus', wintypes.BYTE), ('BatteryFlag', wintypes.BYTE),
                ('BatteryLifePercent', wintypes.BYTE), ('SystemStatusFlag', wintypes.BYTE),
                ('BatteryLifeTime', wintypes.DWORD), ('BatteryFullLifeTime', wintypes.DWORD)]


kernel = ctypes.WinDLL('kernel32', use_last_error=True)
psapi = ctypes.WinDLL('psapi', use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
kernel.GetSystemPowerStatus.argtypes = [ctypes.POINTER(PowerStatus)]
kernel.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME)] * 3
kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4


def memory():
    data = MemoryCounters()
    data.cb = ctypes.sizeof(data)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(rss_MiB=data.WorkingSetSize / 2**20,
                peak_rss_lifetime_MiB=data.PeakWorkingSetSize / 2**20,
                private_commit_MiB=data.PrivateUsage / 2**20,
                peak_private_commit_lifetime_MiB=data.PeakPagefileUsage / 2**20)


def ac_power():
    p = PowerStatus()
    if not kernel.GetSystemPowerStatus(ctypes.byref(p)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(ac_line_status=int(p.ACLineStatus), battery_percent=int(p.BatteryLifePercent))


def seconds(ft):
    return ((ft.dwHighDateTime << 32) + ft.dwLowDateTime) / 1e7


def system_times():
    idle, system, user = (wintypes.FILETIME() for _ in range(3))
    if not kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(system), ctypes.byref(user)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(idle=seconds(idle), total=seconds(system) + seconds(user), clock=time.perf_counter())


def process_cpu(pid):
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
            return None
        return seconds(system) + seconds(user)
    finally:
        kernel.CloseHandle(handle)


class PhasePeaks:
    """CUDA allocator maxima per measured phase; sampled RSS is explicitly weaker."""
    def __init__(self, torch):
        self.torch = torch
        self.peaks = {phase: dict(cuda_allocated_MiB=0., cuda_reserved_MiB=0.,
                                sampled_rss_MiB=0., sampled_private_commit_MiB=0.)
                      for phase in ['solver', 'evaluation']}

    def capture(self, phase):
        p = self.peaks[phase]
        m = memory()
        p['cuda_allocated_MiB'] = max(p['cuda_allocated_MiB'], self.torch.cuda.max_memory_allocated() / 2**20)
        p['cuda_reserved_MiB'] = max(p['cuda_reserved_MiB'], self.torch.cuda.max_memory_reserved() / 2**20)
        p['sampled_rss_MiB'] = max(p['sampled_rss_MiB'], m['rss_MiB'])
        p['sampled_private_commit_MiB'] = max(p['sampled_private_commit_MiB'], m['private_commit_MiB'])
        self.torch.cuda.reset_peak_memory_stats()
        return m


class GpuEngines:
    """Windows per-process engine counters, covering both integrated/discrete GPUs.

    Maximum external engine use is an interference screen, not total GPU use.
    No process titles, commands or private application paths are collected.
    """
    class Value(ctypes.Structure):
        _fields_ = [('status', wintypes.DWORD), ('value', ctypes.c_double)]

    def __init__(self):
        class Item(ctypes.Structure):
            _fields_ = [('name', wintypes.LPWSTR), ('data', GpuEngines.Value)]
        self.Item = Item
        self.dll = ctypes.WinDLL('pdh')
        self.query = ctypes.c_void_p(); self.counter = ctypes.c_void_p()
        self.dll.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        self.dll.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, wintypes.DWORD,
             ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        self.dll.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        status = self.dll.PdhOpenQueryW(None, 0, ctypes.byref(self.query))
        if status != 0:
            raise RuntimeError(f'PdhOpenQueryW: {status:#x}')
        status = self.dll.PdhAddEnglishCounterW(self.query, r'\GPU Engine(*)\Utilization Percentage', 0, ctypes.byref(self.counter))
        if status != 0:
            self.close()
            raise RuntimeError(f'PdhAddEnglishCounterW: {status:#x}')
        self.dll.PdhCollectQueryData(self.query)

    def sample(self, worker_pid=None):
        import re
        status = self.dll.PdhCollectQueryData(self.query)
        if status != 0:
            raise RuntimeError(f'PdhCollectQueryData: {status:#x}')
        size = wintypes.DWORD(); count = wintypes.DWORD()
        self.dll.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return dict(max_external_engine_pct=0., max_worker_engine_pct=0., active_external_pid_count=0)
        buf = ctypes.create_string_buffer(size.value)
        status = self.dll.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), buf)
        if status != 0:
            raise RuntimeError(f'PdhGetFormattedCounterArrayW: {status:#x}')
        items = ctypes.cast(buf, ctypes.POINTER(self.Item))
        external = []; own = []; pids = set()
        for i in range(count.value):
            item = items[i]
            if item.data.status not in [0, 1]:
                continue
            match = re.match(r'pid_(\d+)_', item.name or '')
            if not match:
                continue
            pid = int(match.group(1)); value = max(0., item.data.value)
            if worker_pid is not None and pid == worker_pid:
                own.append(value)
            else:
                external.append(value)
                if value > 1:
                    pids.add(pid)
        return dict(max_external_engine_pct=max(external, default=0.),
                    max_worker_engine_pct=max(own, default=0.), active_external_pid_count=len(pids))

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)
            self.query = ctypes.c_void_p()
