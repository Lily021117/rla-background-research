"""RGB-only paired hue augmentation. Never accepts official placeholder masks.

The sampling handle is produced by the official TrajectoryDataset. Occurrence
IDs are assigned by the sampler, not worker-local counters. No global RNG use.
"""
import hashlib
import json
import numpy as np
import cv2

CAMERAS = ['agentview_camera', 'wrist_camera']


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def hue_parameter(seed, split, record, occurrence):
    payload = json.dumps([int(seed), split, record, int(occurrence)],
                         sort_keys=True, separators=(',', ':')).encode()
    # Explicit PCG64 algorithm and digest-to-seed convention; no Python hash().
    digest = hashlib.sha256(payload).hexdigest()
    rng = np.random.Generator(np.random.PCG64(int(digest[:32], 16)))
    return float(rng.uniform(-0.25, 0.25)), digest


def hue_rgb(rgb, background, delta):
    """uint8 [...,H,W,3], bool [...,H,W]. Outside-mask bytes unchanged."""
    if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
        raise ValueError('Expected original decoded uint8 RGB')
    if background.dtype != np.bool_ or background.shape != rgb.shape[:-1]:
        raise ValueError('Expected explicit boolean background mask')
    if not np.isfinite(delta) or not -0.25 <= delta <= 0.25:
        raise ValueError('Hue offset outside frozen range')
    out = rgb.copy()
    if delta == 0:
        return out
    for src, dst, mask in zip(rgb.reshape(-1, *rgb.shape[-3:]),
                              out.reshape(-1, *rgb.shape[-3:]),
                              background.reshape(-1, *background.shape[-2:])):
        hsv = cv2.cvtColor(src.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = ((hsv[..., 0] / 360 + delta) % 1) * 360
        changed = np.rint(np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)*255).astype(np.uint8)
        dst[mask] = changed[mask]
    return out


class VerifiedMaskProvider:
    """Registry entries require original-RGB identity and reviewed alignment.

    Registry keys use suite/traj/frame/camera. Mask files are uint8 0/1 NPY.
    A signed-off record is necessary, not sufficient: it must refer to an
    independently reviewed alignment report. No registry is supplied as valid
    for the current converted dataset until that evidence actually exists.
    """
    def __init__(self, registry):
        self.registry = registry

    def get(self, suite, trajectory, frame, camera, rgb):
        key = f'{suite}/{trajectory}/{frame}/{camera}'
        if key not in self.registry:
            raise RuntimeError(f'BLOCKED: no verified background mask for {key}')
        entry = self.registry[key]
        if (entry.get('source_kind') not in ('sim_geom_whitelist', 'verified_segmentation')
            or entry.get('alignment_status') != 'reviewed_pass'
            or not entry.get('alignment_evidence_sha256')
            or entry.get('allowed_classes') != ['table', 'wall']
            or entry.get('original_rgb_sha256') != array_hash(rgb)):
            raise RuntimeError(f'BLOCKED: invalid mask provenance for {key}')
        with open(entry['path'], 'rb') as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Mask content changed')
        mask = np.load(entry['path'], allow_pickle=False)
        if mask.shape != rgb.shape[:2] or not np.isin(mask, [0, 1]).all():
            raise ValueError('Mask dimensions or labels invalid')
        return mask.astype(bool)


class PairedHueReader:
    """Replay official sampling decisions, then augment its original RGB.

    getitem_from_handle(..., apply_augmentation=False) is required: no hidden
    ColorJitter. Returned x and x_prime retain all metadata and camera geometry.
    DINO extraction belongs downstream and must process both RGB branches.
    """
    def __init__(self, official_dataset, mask_provider, seed=20261102, enabled=True):
        self.dataset = official_dataset
        self.masks = mask_provider
        self.seed = seed
        self.enabled = enabled

    def sample_training_record(self):
        """Uses the official suite/trajectory/start/horizon sampler unmodified.

        The caller persists the result with a global occurrence counter. This
        is intentionally separate from read(), so changing hue cannot consume
        sampler RNG or change the physical transition.
        """
        f = self.dataset.sample_frame_index()
        idx = f['dataset_idx']
        if not all(c in f['camera_keys'] for c in CAMERAS):
            raise ValueError('Missing official view; single-view fallback forbidden')
        return dict(f, version=1, camera_keys=list(CAMERAS), rgb_variant='base',
                    group_name=self.dataset.dataset_to_group[idx],
                    dataset_root=self.dataset.datasets[idx].root)

    def read(self, handle, occurrence, split):
        if split not in ('train', 'val') or (split == 'val' and occurrence != 0):
            raise ValueError('Validation uses frozen occurrence=0')
        if handle['camera_keys'] != CAMERAS:
            raise ValueError('Camera order must match the official two-view recipe')
        x = self.dataset.getitem_from_handle(handle, apply_augmentation=False)
        rgb = np.moveaxis(x['rgbs'], 2, -1)
        record = {k: handle[k] for k in ('group_name', 'traj_id', 'frame_id', 'sampled_horizon', 'camera_keys')}
        delta, key = hue_parameter(self.seed, split, record, occurrence)
        if self.enabled:
            frames = np.asarray(x['frame_id']).tolist()
            masks = np.stack([np.stack([self.masks.get(handle['group_name'], handle['traj_id'],
                int(frame), cam, rgb[t, c]) for c, cam in enumerate(CAMERAS)])
                for t, frame in enumerate(frames)])
            augmented = hue_rgb(rgb, masks, delta)
        else:
            masks = None
            augmented = rgb.copy()
            delta = 0.0
        xp = dict(x)
        xp['rgbs'] = np.moveaxis(augmented, -1, 2)
        return {'x': x, 'x_prime': xp, 'background_masks': masks,
                'provenance': {'handle': dict(handle), 'occurrence': occurrence,
                  'split': split, 'augmentation_key': key, 'delta_h': delta,
                  'original_rgb_hash': array_hash(rgb), 'augmented_rgb_hash': array_hash(augmented)}}
