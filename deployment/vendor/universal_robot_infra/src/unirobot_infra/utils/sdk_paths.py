import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import sysconfig


@dataclass(frozen=True)
class SharpaManusSdkPaths:
    vendor_root: Path

    @property
    def python_package_layout(self) -> bool:
        """True for the flat install layout: vendor_root/sharpa/{__init__.py,sharpa.cpython-*.so}."""
        return (self.vendor_root / "sharpa" / "__init__.py").is_file()

    @property
    def client_dir(self) -> Path:
        return self.vendor_root / "client"

    @property
    def retarget_include_dir(self) -> Path:
        if self.python_package_layout:
            return self.vendor_root
        return self.vendor_root / "retargeting" / "include"

    @property
    def proto_dir(self) -> Path:
        if self.python_package_layout:
            return self.vendor_root
        return self.retarget_include_dir / "proto_hand"

    @property
    def sharpa_sdk_root(self) -> Path:
        if self.python_package_layout:
            return self.vendor_root
        return self.retarget_include_dir / "SharpaWaveSDK_4.6.6"

    @property
    def sharpa_python_dir(self) -> Path:
        if self.python_package_layout:
            return self.vendor_root
        return self.sharpa_sdk_root / "python"

    @property
    def sharpa_python_module(self) -> Path:
        if self.python_package_layout:
            pattern = f"sharpa.cpython-{sys.version_info.major}{sys.version_info.minor}-*.so"
            matches = sorted((self.vendor_root / "sharpa").glob(pattern))
            if matches:
                return matches[0]
            return self.vendor_root / "sharpa" / pattern.replace("*", "x86_64-linux-gnu")
        return self.sharpa_python_dir / f"python{sys.version_info.major}{sys.version_info.minor}" / "sharpa.so"

    @property
    def vendor_python_dir(self) -> Path:
        if self.python_package_layout:
            return self.vendor_root
        return self.vendor_root.parent / "python"


def prepend_sys_paths(paths: tuple[Path, ...]) -> None:
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def prepend_library_paths(paths: tuple[Path, ...]) -> None:
    current = os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else ""
    parts = [part for part in current.split(":") if part]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in parts:
            parts.insert(0, path_str)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def preload_current_libpython() -> Path:
    lib_dir = Path(sysconfig.get_config_var("LIBDIR"))
    version = sysconfig.get_config_var("VERSION")
    names = []
    for key in ("INSTSONAME", "LDLIBRARY"):
        name = sysconfig.get_config_var(key)
        if name and not name.endswith(".a"):
            names.append(name)
    names.extend((f"libpython{version}.so.1.0", f"libpython{version}.so"))

    candidates = []
    for name in names:
        lib_path = lib_dir / name
        if lib_path not in candidates:
            candidates.append(lib_path)

    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(", ".join(str(path) for path in candidates))
    lib_path = existing[0]
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    return lib_path


def sharpa_manus_sdk_paths(vendor_root: Path) -> SharpaManusSdkPaths:
    if not vendor_root.is_absolute():
        raise ValueError("robot.hardware.end_effector.sdk.vendor_root must be an absolute path")
    if not vendor_root.exists():
        raise FileNotFoundError(str(vendor_root))
    return SharpaManusSdkPaths(vendor_root=vendor_root)


def configure_sharpa_runtime(sdk_paths: SharpaManusSdkPaths) -> None:
    if sdk_paths.python_package_layout:
        for path in (sdk_paths.sharpa_python_dir, sdk_paths.sharpa_python_module):
            if not path.exists():
                raise FileNotFoundError(str(path))
        prepend_sys_paths((sdk_paths.sharpa_python_dir,))
        preload_current_libpython()
        return
    required_paths = (
        sdk_paths.retarget_include_dir,
        sdk_paths.proto_dir,
        sdk_paths.sharpa_sdk_root,
        sdk_paths.sharpa_sdk_root / "lib",
        sdk_paths.sharpa_python_dir,
        sdk_paths.sharpa_python_module,
        sdk_paths.vendor_python_dir,
        sdk_paths.vendor_python_dir / "casadi",
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
    prepend_sys_paths(
        (
            sdk_paths.retarget_include_dir,
            sdk_paths.proto_dir,
            sdk_paths.sharpa_python_dir,
            sdk_paths.vendor_python_dir,
        )
    )
    prepend_library_paths(
        (
            sdk_paths.sharpa_sdk_root / "lib",
            sdk_paths.vendor_python_dir / "casadi",
        )
    )
    preload_current_libpython()
