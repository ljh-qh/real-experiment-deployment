import numpy as np
from omegaconf import DictConfig

from unirobot_infra.calibration.hand_eye import CameraIntrinsics
from unirobot_infra.camera.rgbd import RgbdFrame
from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)
ZED_WARMUP_FRAMES = 5


class ZedCapture:
    def __init__(self, name: str, cfg: DictConfig) -> None:
        self.name = name
        self.view = str(cfg.view)
        self.depth_mode = str(cfg.depth_mode)
        self.image_size = _image_size_tuple(cfg.image_size)
        self._sl = _import_pyzed()
        self._camera = self._sl.Camera()
        self._runtime = self._sl.RuntimeParameters()
        self._mat = self._sl.Mat()
        self._depth_mat = self._sl.Mat()
        self._open(cfg)
        self._warmup(ZED_WARMUP_FRAMES)

    def read(self) -> tuple[bool, np.ndarray | None]:
        status = self._camera.grab(self._runtime)
        if status != self._sl.ERROR_CODE.SUCCESS:
            return False, None
        self._camera.retrieve_image(self._mat, _zed_view(self._sl, self.view))
        image = self._mat.get_data()
        if image is None:
            return False, None
        frame = np.asarray(image).copy()
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = frame[..., ::-1].copy()
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            import cv2

            frame = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)
        return True, np.asarray(frame, dtype=np.uint8)

    def read_rgbd(self) -> tuple[bool, RgbdFrame | None]:
        if self.depth_mode == "NONE":
            raise RuntimeError("ZED depth_mode must not be NONE when reading RGB-D")
        status = self._camera.grab(self._runtime)
        if status != self._sl.ERROR_CODE.SUCCESS:
            return False, None
        self._camera.retrieve_image(self._mat, _zed_view(self._sl, self.view))
        self._camera.retrieve_measure(self._depth_mat, self._sl.MEASURE.DEPTH)
        image = self._mat.get_data()
        depth = self._depth_mat.get_data()
        if image is None or depth is None:
            return False, None
        frame = np.asarray(image).copy()
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        rgb = frame[..., ::-1].copy()
        depth_m = np.asarray(depth, dtype=np.float32).copy()
        if depth_m.ndim == 3:
            depth_m = depth_m[:, :, 0]
        native_size = (rgb.shape[1], rgb.shape[0])
        if native_size != self.image_size:
            import cv2

            rgb = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(depth_m, self.image_size, interpolation=cv2.INTER_NEAREST)
        return True, RgbdFrame(
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth_m=np.asarray(depth_m, dtype=np.float32),
            intrinsics=self.intrinsics(native_size=native_size),
        )

    def intrinsics(self, native_size: tuple[int, int] | None = None) -> CameraIntrinsics:
        camera_info = self._camera.get_camera_information()
        calibration = camera_info.camera_configuration.calibration_parameters
        camera_params = calibration.left_cam if self.view == "left" else calibration.right_cam
        if native_size is None:
            native_size = _camera_native_size(camera_info)
        scale_x = float(self.image_size[0]) / float(native_size[0])
        scale_y = float(self.image_size[1]) / float(native_size[1])
        camera_matrix = np.asarray(
            [
                [float(camera_params.fx) * scale_x, 0.0, float(camera_params.cx) * scale_x],
                [0.0, float(camera_params.fy) * scale_y, float(camera_params.cy) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return CameraIntrinsics(
            camera_matrix=camera_matrix,
            dist_coeffs=np.asarray(camera_params.disto, dtype=np.float64).reshape(-1),
            resolution=(self.image_size[0], self.image_size[1]),
        )

    def close(self) -> None:
        self._camera.close()

    def _open(self, cfg: DictConfig) -> None:
        sl = self._sl
        init_params = sl.InitParameters()
        init_params.camera_resolution = _zed_resolution(sl, str(cfg.resolution))
        init_params.camera_fps = int(cfg.fps)
        init_params.depth_mode = _zed_depth_mode(sl, self.depth_mode)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sensors_required = False
        init_params.sdk_verbose = int(cfg.sdk_verbose)
        serial_number = int(cfg.serial_number)
        if serial_number <= 0:
            raise ValueError("ZED SDK serial_number must be positive")
        init_params.set_from_serial_number(serial_number)

        logger.info(
            "启动 ZED SDK 相机 name=%s resolution=%s fps=%s view=%s",
            self.name,
            cfg.resolution,
            cfg.fps,
            self.view,
        )
        status = self._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED SDK camera {self.name} failed to open: {status}")
        logger.info("ZED SDK 相机已就绪 name=%s", self.name)

    def _warmup(self, warmup_frames: int) -> None:
        for _ in range(max(0, warmup_frames)):
            self._camera.grab(self._runtime)


def _import_pyzed():
    try:
        import pyzed.sl as sl
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ZED SDK Python binding pyzed is not installed in this environment. "
            "Install it with the Stereolabs ZED SDK Python wheel, for example "
            "`uv pip install /path/to/pyzed-*-cp312-cp312-linux_x86_64.whl`."
        ) from exc
    return sl


def _zed_resolution(sl, resolution: str):
    if not hasattr(sl.RESOLUTION, resolution):
        raise ValueError(f"unsupported ZED SDK resolution: {resolution}")
    return getattr(sl.RESOLUTION, resolution)


def _zed_depth_mode(sl, depth_mode: str):
    if not hasattr(sl.DEPTH_MODE, depth_mode):
        raise ValueError(f"unsupported ZED SDK depth_mode: {depth_mode}")
    return getattr(sl.DEPTH_MODE, depth_mode)


def _zed_view(sl, view: str):
    if view == "left":
        return sl.VIEW.LEFT
    if view == "right":
        return sl.VIEW.RIGHT
    raise ValueError("ZED SDK view must be left or right")


def _image_size_tuple(image_size) -> tuple[int, int]:
    values = tuple(int(value) for value in image_size)
    if len(values) != 2:
        raise ValueError("camera image_size must have length 2")
    return (values[0], values[1])


def _camera_native_size(camera_info) -> tuple[int, int]:
    resolution = camera_info.camera_configuration.resolution
    return (int(resolution.width), int(resolution.height))
