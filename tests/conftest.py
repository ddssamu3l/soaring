"""Shared test setup.

pygame runs HEADLESS for the whole test session: the SDL "dummy" video
driver renders into plain memory surfaces -- no window, no GPU, CI-safe.
The env var must be set before pygame initializes, hence at import time.
Tests draw onto their own pygame.Surface objects and assert on pixels.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402  (import must follow the env var)

pygame.init()
