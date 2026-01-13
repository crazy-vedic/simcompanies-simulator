# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification file for building the simtools executable."""

from pathlib import Path

# Get the data directory path
data_dir = Path("simtools/data")

# Collect all data files from the simtools/data directory
data_files = [
    (str(data_dir / "buildings.json"), "simtools/data"),
    (str(data_dir / "abundance_resources.json"), "simtools/data"),
    (str(data_dir / "seasonal_resources.json"), "simtools/data"),
    (str(data_dir / "__init__.py"), "simtools/data"),
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=data_files,
    hiddenimports=[
        "simtools",
        "simtools.api",
        "simtools.calculator",
        "simtools.cli",
        "simtools.genetic",
        "simtools.models",
        "simtools.models.building",
        "simtools.models.resource",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="simtools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
