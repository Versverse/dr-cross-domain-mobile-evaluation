"""Binary-safe ADB input staging/result retrieval; runs only when explicitly invoked."""
from pathlib import Path
import argparse
import os
import subprocess

PACKAGE = 'com.example.aptosedgetest.fidelity20261016'


def long_path(path):
    value = os.path.abspath(os.fspath(path))
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\' + value
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adb', type=Path, default=None)
    parser.add_argument('--serial', required=True)
    commands = parser.add_subparsers(dest='command', required=True)
    push = commands.add_parser('push')
    push.add_argument('--archive', type=Path, required=True)
    pull = commands.add_parser('pull')
    pull.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adb = args.adb or Path(os.environ['ANDROID_SDK_ROOT']) / 'platform-tools/adb.exe'
    prefix = [str(adb), '-s', args.serial]
    if args.command == 'push':
        # Equivalent to: adb exec-in run-as PACKAGE tar -xf - -C files
        with long_path(args.archive).open('rb') as stream:
            subprocess.run(prefix + ['exec-in', 'run-as', PACKAGE, 'tar', '-xf', '-', '-C', 'files'],
                           stdin=stream, check=True)
    else:
        # Capture raw bytes, never PowerShell text redirection of a TAR stream.
        args.output = long_path(args.output)
        if args.output.exists():
            raise FileExistsError('Choose a new result TAR filename.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            subprocess.run(prefix + ['exec-out', 'run-as', PACKAGE, 'tar', '-cf', '-', '-C', 'files', 'fidelity_results'],
                           stdout=stream, check=True)
        print('Saved binary TAR: ' + str(args.output))


if __name__ == '__main__':
    main()
