"""식별번호 생성·검증 (주민번호, 사업자등록번호 등)."""
import random

RRN_WEIGHTS = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
BIZ_WEIGHTS = [1, 3, 7, 1, 3, 7, 1, 3, 5]


def rrn_check_digit(first12: str) -> int:
    s = sum(int(d) * w for d, w in zip(first12, RRN_WEIGHTS))
    return (11 - s % 11) % 10


def rrn_is_valid(rrn: str) -> bool:
    digits = rrn.replace("-", "")
    return len(digits) == 13 and digits.isdigit() and rrn_check_digit(digits[:12]) == int(digits[12])


def biz_check_digit(first9: str) -> int:
    d = [int(c) for c in first9]
    s = sum(a * b for a, b in zip(d, BIZ_WEIGHTS)) + (d[8] * 5) // 10
    return (10 - s % 10) % 10


def biz_is_valid(bizno: str) -> bool:
    digits = bizno.replace("-", "")
    return len(digits) == 10 and biz_check_digit(digits[:9]) == int(digits[9])


def make_bizno(rng: random.Random, inst_type: str) -> str:
    """사업자등록번호 XXX-XX-XXXXX. 가운데 2자리: 개인 면세사업자(의원) 90번대, 비영리법인(병원) 82."""
    head = f"{rng.randint(101, 899):03d}"
    mid = f"{rng.choice([90, 91, 92, 96, 97, 98])}" if inst_type == "clinic" else rng.choice(["82", "81"])
    serial = f"{rng.randint(0, 9999):04d}"
    first9 = head + mid + serial
    return f"{head}-{mid}-{serial}{biz_check_digit(first9)}"


def make_license_no(rng: random.Random) -> str:
    return str(rng.randint(10000, 149999))


def make_phone(rng: random.Random, area: str) -> str:
    mid = rng.randint(1000, 9999) if rng.random() < 0.6 else rng.randint(200, 999)
    return f"{area}-{mid}-{rng.randint(0, 9999):04d}"
