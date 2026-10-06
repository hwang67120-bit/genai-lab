"""CPU-only tail measurements; v3 thresholds are fixed, outputs are advisory only.

Source: outputs/tail-complexity-measure-20261006/measure_v3.py.
No prompt construction, model imports or settings changes happen here.
"""
import json
from pathlib import Path
import shutil

RULE_VERSION = "tail-complexity-v3-display-v1"

def color_families(rgb, mask):
    import numpy as np
    import cv2
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[mask]
    L = lab[:, 0] * 100 / 255; a = lab[:, 1] - 128; b = lab[:, 2] - 128
    C = np.hypot(a, b); H = (np.degrees(np.arctan2(b, a)) + 360) % 360; n = len(L)
    fam = []
    achro = (C < 18) & (L >= 30)
    if achro.sum() >= .10 * n: fam.append({'kind': 'achromatic', 'share': round(float(achro.mean()), 3)})
    chroma = C >= 18
    hist = np.array([((H >= i * 30) & (H < i * 30 + 30) & chroma).sum() for i in range(12)])
    occ = hist >= .01 * n  # 마스크 1% 이상 찬 구간만 "있음", 인접 구간끼리 합침(원형)
    groups, cur = [], []
    for i in range(12):
        if occ[i]: cur.append(i)
        elif cur: groups.append(cur); cur = []
    if cur:
        if groups and groups[0][0] == 0: groups[0] = cur + groups[0]
        else: groups.append(cur)
    for g in groups:
        s = hist[g].sum() / n
        if s >= .05: fam.append({'kind': 'hue', 'bins_deg': [i * 30 for i in g], 'share': round(float(s), 3)})
    return fam

def longest_path(skel):
    import numpy as np
    ys, xs = np.nonzero(skel); pts = set(zip(ys.tolist(), xs.tolist()))
    if not pts: return []
    nb = lambda p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and (p[0] + dy, p[1] + dx) in pts]
    def bfs(s):
        prev = {s: None}; q = [s]; last = s
        for p in q:
            for v in nb(p):
                if v not in prev: prev[v] = p; q.append(v)
            last = p
        return last, prev
    # 가장 큰 연결 성분 안에서 두 번 BFS
    a, _ = bfs(next(iter(pts))); b, prev = bfs(a); path = [b]
    while prev[path[-1]] is not None: path.append(prev[path[-1]])
    return path

def keep_parts(m):
    import numpy as np
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1: return m, m
    areas = st[1:, cv2.CC_STAT_AREA]; big = 1 + int(np.argmax(areas))
    keep = np.isin(lab, [1 + i for i, a in enumerate(areas) if a >= .05 * areas.max()])
    return keep, lab == big

def centerline_v2(main):
    import numpy as np
    import cv2
    sk = cv2.ximgproc.thinning(main.astype(np.uint8) * 255) > 0
    path = longest_path(sk)
    if len(path) < 2: return None
    dist = cv2.distanceTransform(main.astype(np.uint8), cv2.DIST_L2, 5)
    t = float(np.median([dist[y, x] for y, x in path]) * 2)
    P = np.array(path, np.float32)[:, ::-1]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; length = float(seg[-1])
    step = max(10.0, t / 2)
    s = np.arange(0, length, step); Q = np.c_[np.interp(s, seg, P[:, 0]), np.interp(s, seg, P[:, 1])]
    w = max(5, int(round(t / step))); w += (w % 2 == 0); h = w // 2
    Qs = np.array([Q[max(0, i - h):i + h + 1].mean(axis=0) for i in range(len(Q))])
    if length < 40 or len(Qs) < 3: turn = 0.0
    else:
        d = np.diff(Qs, axis=0); ang = np.arctan2(d[:, 1], d[:, 0]); da = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
        turn = float(np.degrees(np.abs(da).sum()))
    # 무늬용: 원 중심선을 짧은 간격(4px)으로
    s2 = np.arange(0, length, 4.0); C = np.c_[np.interp(s2, seg, P[:, 0]), np.interp(s2, seg, P[:, 1])]
    return dict(t=t, length=length, step=step, win=w, turn=turn, smooth=Qs, fine=C)

def profile(rgb, mask, C, t):
    import numpy as np
    import cv2
    from scipy.signal import find_peaks
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) * 100 / 255
    ys, xs = np.nonzero(mask); pts = np.c_[xs, ys].astype(np.float32)
    from scipy.spatial import cKDTree
    dd, idx = cKDTree(C).query(pts)
    ok = dd <= t; prof = np.full(len(C), np.nan)
    sums = np.bincount(idx[ok], L[ys[ok], xs[ok]], len(C)); cnt = np.bincount(idx[ok], None, len(C))
    prof[cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
    good = ~np.isnan(prof); prof = np.interp(np.arange(len(C)), np.nonzero(good)[0], prof[good])
    win = max(3, len(C) // 3); trend = np.convolve(np.pad(prof, win // 2, mode='edge'), np.ones(win) / win, mode='valid')[:len(C)]
    det = prof - trend
    valleys, pr = find_peaks(-det, prominence=8, distance=max(1, int(t / 2 / 4)))
    return prof, det, valleys, pr['prominences']


def measure_tail(rgb, segmentation, box):
    """Clip the mask, measure v3 values, then apply the display-only deferral."""
    import numpy as np
    from genai_lab.qwen_tail_edit import validate_box
    validate_box(box, (rgb.shape[1], rgb.shape[0]))
    mask = np.asarray(segmentation, dtype=bool).copy()
    if mask.shape != rgb.shape[:2]:
        raise ValueError("마스크 크기가 원본과 다릅니다.")
    x0, y0, x1, y1 = box
    mask[:y0] = False; mask[y1:] = False
    mask[:, :x0] = False; mask[:, x1:] = False
    mask, main = keep_parts(mask)
    if not mask.any():
        raise ValueError("선택 영역에 꼬리 마스크가 없습니다.")
    families = color_families(rgb, mask)
    center = centerline_v2(main)
    record = dict(status="completed", advisory_only=True, rule_version=RULE_VERSION,
                  box=list(box), mask_px=int(mask.sum()), main_px=int(main.sum()),
                  color_families=families, color_complex=len(families) >= 2)
    if center is None or center['t'] <= 0:
        record.update(thickness_px=None, centerline_px=None, turn_deg=None,
                      deep_valleys=None, pattern=None, curvy=None,
                      display_deferred=True, deferral_reason="중심선을 측정할 수 없음")
    else:
        _, _, valleys, prominence = profile(rgb, mask, center['fine'], center['t'])
        record.update(thickness_px=round(center['t'], 1), centerline_px=round(center['length'], 1),
                      resample_px=round(center['step'], 1), smooth_points=center['win'],
                      turn_deg=round(center['turn'], 1), valleys=len(valleys),
                      valley_prominence=[round(float(p), 1) for p in prominence],
                      deep_valleys=int((prominence >= 20).sum()),
                      pattern=int((prominence >= 20).sum()) >= 3,
                      curvy=center['length'] >= 40 and center['turn'] >= 120,
                      display_deferred=center['length'] / center['t'] < 3,
                      deferral_reason="짧은 덩어리형" if center['length'] / center['t'] < 3 else None)
    record['deferral_validated'] = False
    return record, mask


def advisory_text(record):
    """Present measured families and provisional geometry without inventing color names."""
    if record.get('status') != 'completed':
        return "복잡도 분석 불가 · 꼬리 고치기는 계속할 수 있습니다."
    lines = ["색: 여러 색" if record['color_complex'] else "색: 한 가지 색 계열"]
    deferred = record['display_deferred']
    if deferred:
        lines.append("무늬·곡선: 판단 보류 (" + record['deferral_reason'] + ")")
        lines.append("판단을 숨기는 보수 규칙이며, 아직 검증되지 않았습니다.")
    else:
        lines.append("추정: 반복 무늬 " + ("있음" if record['pattern'] else "없음") + " (확인 필요)")
        lines.append("추정: 곡선 " + ("많음" if record['curvy'] else "적음") + " (확인 필요)")
        if record['curvy']:
            lines.append("끝 모양 문장을 넣는 것을 고려해 주세요.")
    if record['color_complex'] or (not deferred and (record['pattern'] or record['curvy'])):
        lines.append("생성 단계에서 꼬리가 달라지기 쉬운 유형입니다. 꼬리 고치기를 권장합니다.")
    lines.append("겹친 그림이 실제 꼬리를 잡았는지 확인해 주세요. 분석은 안내용이며 편집 입력을 바꾸지 않습니다.")
    return "\n".join(lines)


def save_overlay(rgb, mask, box, directory):
    """Save diagnostic pixels only; never replace the actual reference crop."""
    import numpy as np
    from PIL import Image
    directory = Path(directory)
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / 'mask.png')
    overlay = rgb.copy()
    overlay[mask] = (rgb[mask].astype(np.float32) * .65 + np.array([0, 220, 130]) * .35).astype(np.uint8)
    Image.fromarray(overlay).crop(box).save(directory / 'overlay.png')


def persist_advisory(directory, spec, report):
    """Copy advisory artifacts outside TailEditSpec and the immutable Qwen request."""
    from genai_lab.qwen_record_io import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(report or {'status': 'disabled', 'advisory_only': True})
    if record.get('status') == 'completed':
        if record.get('source_sha256') != spec.source_sha256 or record.get('box') != list(spec.box):
            record = dict(status='unavailable', reason='선택 영역과 분석 결과가 다름', advisory_only=True)
        else:
            for name in ('mask', 'overlay'):
                source = record.pop(name + '_path', None)
                if source:
                    try:
                        shutil.copyfile(source, directory / (name + '.png'))
                        record[name + '_file'] = name + '.png'
                    except OSError as error:
                        record.setdefault('artifact_errors', {})[name] = str(error)
    write_json(directory / 'run.json', {'tail_complexity': record})
    return record


def annotate_finished_run(directory, report):
    """After the Qwen worker exits, add advisory metadata without touching its request."""
    from genai_lab.qwen_record_io import write_json
    path = Path(directory) / 'launcher.json'
    if not path.is_file():
        return  # Pre-launch failures still retain complexity/run.json.
    record = json.loads(path.read_text(encoding='utf-8'))
    record['tail_complexity'] = report
    write_json(path, record)
