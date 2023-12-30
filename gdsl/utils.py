from typing import Final
class COLORS:
    HEADER: Final[str] = '\033[95m'
    OKBLUE: Final[str] = '\033[94m'
    OKCYAN: Final[str] = '\033[96m'
    OKGREEN: Final[str] = '\033[92m'
    MAGENTA: Final[str] = '\033[95m'
    WARNING: Final[str] = '\033[93m'
    FAIL: Final[str] = '\033[91m'
    ENDC: Final[str] = '\033[0m'
    BOLD: Final[str] = '\033[1m'
    UNDERLINE: Final[str] = '\033[4m'


def green(v, fmt='{}'):
    return COLORS.OKGREEN + fmt.format(v) + COLORS.ENDC


def cyan(v, fmt='{}'):
    return COLORS.OKCYAN + fmt.format(v) + COLORS.ENDC


def blue(v, fmt='{}'):
    return COLORS.OKBLUE + fmt.format(v) + COLORS.ENDC


def yellow(v, fmt='{}'):
    return COLORS.WARNING + fmt.format(v) + COLORS.ENDC


def red(v, fmt='{}'):
    return COLORS.FAIL + fmt.format(v) + COLORS.ENDC
