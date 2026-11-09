"""Compatibility entry point for a profile-only build using the shared driver."""
import sys
from build_extensions import main

if __name__ == '__main__':
    main(['--modules', 'profile_geometry', *sys.argv[1:]])
