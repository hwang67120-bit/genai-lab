import os
if os.environ.get('RAM_BREAKDOWN_ACTIVE')=='1' and os.environ.get('RAM_ROOT_PID'):
    import ram_probe
    ram_probe.install()
