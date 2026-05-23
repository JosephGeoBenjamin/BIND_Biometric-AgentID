""" Created with Claude and ChatGPT refactoring
"""

import cv2
import mediapipe as mp
import numpy as np
import glob
import os
import argparse
from tqdm import tqdm
import shutil

class BlazeFacePoseEstimator:
    """
    BlazeFace (MediaPipe Face Detection) -> 5 landmarks -> solvePnP -> pose + frontal check
    """

    _MODEL_POINTS = np.array([
        (0.0,   0.0,    0.0),     # nose
        (-30.0, -30.0, -30.0),    # left eye
        (30.0,  -30.0, -30.0),    # right eye
        (-25.0,  30.0, -30.0),    # left mouth
        (25.0,   30.0, -30.0),    # right mouth
    ], dtype=np.float64)

    def __init__(self,
                 yaw_thr=10,
                 pitch_thr=10,
                 roll_thr=5,
                 min_confidence=0.3):

        print("Yaw, Pitch, Roll threshs for fitering :", yaw_thr, pitch_thr, roll_thr,
              "Det Confidence: ", min_confidence)

        self.yaw_thr = yaw_thr
        self.pitch_thr = pitch_thr
        self.roll_thr = roll_thr

        self.mp_fd = mp.solutions.face_detection
        self.detector = self.mp_fd.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence
        )

        # IMPORTANT: avoid None dist-coeffs
        self.dist = np.zeros((4, 1), dtype=np.float64)

    def process(self, img: np.ndarray):
        h, w = img.shape[:2]

        keypoints = self._detect_5pts(img, w, h)
        if keypoints is None:
            return None, False

        cam = self._camera_matrix(w, h)

        rvec, tvec = self._solve_pnp(keypoints, cam, self.dist)
        if rvec is None:
            return None, False

        pose = self._rvec_to_euler(rvec)
        return pose, self._is_frontal(pose)

    def _detect_5pts(self, img, w, h):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        if not res.detections:
            return None

        det = res.detections[0]
        kps = det.location_data.relative_keypoints

        left_eye  = kps[0]
        right_eye = kps[1]
        nose      = kps[2]
        mouth     = kps[3]

        mx, my = mouth.x * w, mouth.y * h
        offset = 0.03 * w

        return np.array([
            (nose.x * w, nose.y * h),
            (left_eye.x * w, left_eye.y * h),
            (right_eye.x * w, right_eye.y * h),
            (mx - offset, my),
            (mx + offset, my),
        ], dtype=np.float64)

    def _solve_pnp(self, img_pts, cam, dist):
        # KEY FIX: use EPNP (stable for 4+ points, avoids DLT requirement)
        ok, rvec, tvec = cv2.solvePnP(
            self._MODEL_POINTS,
            img_pts,
            cam,
            dist,
            flags=cv2.SOLVEPNP_EPNP
        )
        return (rvec, tvec) if ok else (None, None)

    def _camera_matrix(self, w, h):
        f = float(w)
        return np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

    def _rvec_to_euler(self, rvec):
        rmat, _ = cv2.Rodrigues(rvec)

        pitch = -np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw   = -np.degrees(np.arcsin(np.clip(-rmat[2, 0], -1, 1)))
        roll  =  np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))

        return pitch, yaw, roll

    def _is_frontal(self, pose):
        if pose is None:
            return False

        pitch, yaw, roll = pose

        return (
            abs(yaw) <= self.yaw_thr and
            abs(pitch) <= self.pitch_thr and
            abs(roll) <= self.roll_thr
        )


def glob_load_images(glob_pattern):
    for path in glob.glob(glob_pattern):
        img = cv2.imread(path)
        if img is not None:
            yield path, img


def get_args():
    parser = argparse.ArgumentParser(description="Filter frontal faces using MediaPipe")

    parser.add_argument("--inp", type=str, required=True,
                        help="Input glob pattern (e.g. data/*.jpg)")

    parser.add_argument("--out", type=str, required=True,
                        help="Output folder for frontal images")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    inp_path = args.inp
    out_path = args.out

    os.makedirs(out_path, exist_ok=True)

    pose_estimator = BlazeFacePoseEstimator()

    kept = 0
    total = 0

    print("Globbing images ..... ")
    globbed_stuff = glob_load_images(inp_path)
    print("Glob..... Done!")


    for path, img in tqdm(globbed_stuff):
        total += 1

        pose, pose_flag = pose_estimator.process(img)

        if pose_flag:

            base_folder_path = os.path.dirname(path)
            out_folder_path  = out_path + os.path.basename(base_folder_path)

            shutil.copytree(base_folder_path, out_folder_path)

            kept += 1

    print("Total:", total)
    print("Kept:", kept)