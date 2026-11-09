import math


NS_PER_S = 1_000_000_000


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def mm_to_m(mm: float) -> float:
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    return m * 1000.0


def s_to_ns(s: float) -> int:
    return int(round(s * NS_PER_S))


def ns_to_s(ns: int) -> float:
    return ns / NS_PER_S
