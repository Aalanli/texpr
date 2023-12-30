class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    MAGENTA = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
