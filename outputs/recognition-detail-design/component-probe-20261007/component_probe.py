"""Saved-mask diagnostic: retain components; suggest geometry, never infer a final tip."""
from collections import deque
import math
import cv2
import numpy as np

def neighbors(point, points):
    y, x = point
    return sorted((y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                  if (dy or dx) and (y+dy,x+dx) in points)

def outward_direction(endpoint, points, thickness):
    """Follow inward skeleton locally; return outward unit vector or unknown."""
    limit = max(6, math.ceil(2*thickness))
    queue = deque([(endpoint,0)])
    seen = {endpoint}
    last = endpoint
    while queue:
        node, depth = queue.popleft()
        last = node
        if depth >= limit:
            break
        for nxt in neighbors(node, points):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt,depth+1))
    vector = np.array([endpoint[1]-last[1], endpoint[0]-last[0]], float)
    norm = np.linalg.norm(vector)
    return None if norm == 0 else (vector/norm).tolist()

def collect_components(mask):
    """Return all visible components and skeleton endpoints in original coordinates."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    components = []
    for index in range(1,count):
        part = labels == index
        skeleton = cv2.ximgproc.thinning(part.astype(np.uint8)*255) > 0
        yy,xx = np.nonzero(skeleton)
        points = set(zip(yy.tolist(),xx.tolist()))
        distance = cv2.distanceTransform(part.astype(np.uint8),cv2.DIST_L2,5)
        thickness = float(np.median(distance[skeleton])*2) if points else 0.
        terminals = sorted(p for p in points if len(neighbors(p,points)) == 1)
        components.append(dict(id=index,area=int(stats[index,cv2.CC_STAT_AREA]),
            thickness=thickness,skeleton_xy=[[x,y] for y,x in sorted(points)],
            branch_pixels=sum(len(neighbors(p,points))>=3 for p in points),
            endpoints=[dict(xy=[p[1],p[0]],outward=outward_direction(p,points,thickness)) for p in terminals],
            status="candidates" if terminals else "no_endpoint"))
    return components

def connection_candidates(components):
    """Return unverified geometric links without changing pixels or selecting a tip."""
    links = []
    for i, left in enumerate(components):
        for right in components[i+1:]:
            gap_limit = 4*max(left["thickness"],right["thickness"])
            for a in left["endpoints"]:
                for b in right["endpoints"]:
                    vector = np.array(b["xy"],float)-a["xy"]
                    distance = float(np.linalg.norm(vector))
                    if not 0 < distance <= gap_limit or a["outward"] is None or b["outward"] is None:
                        continue
                    unit = vector/distance
                    angles = [math.degrees(math.acos(float(np.clip(np.dot(a["outward"],unit),-1,1)))),
                              math.degrees(math.acos(float(np.clip(np.dot(b["outward"],-unit),-1,1))))]
                    if max(angles)<=45:
                        links.append(dict(components=[left["id"],right["id"]],a=a["xy"],b=b["xy"],
                            gap_px=distance,angles=angles,status="geometry_only_occlusion_unverified"))
    return links

def inside_box(point, box):
    x,y = point
    x0,y0,x1,y1 = box
    return x0<=x<x1 and y0<=y<y1
