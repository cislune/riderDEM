## Install
1. Install librealsense SDK https://github.com/realsenseai/librealsense/blob/master/doc/installation_windows.md

2. pip install numpy opencv-python matplotlib pyrealsense2

Note: make sure you are using USB 3 and Windows/Linux only

## Run
python riderDEM.py

## Usage
1. Select the area you want to capture after pressing 'm'
2. Select a reference area (area that will remain unchanged) after pressing 'r'
3. Take snapshots by pressing 's' after a pass (There must be no motion).
4. Undo snapshot by pressing 'y'

## Hotkeys
| Key | Action                                              |
| --- | --------------------------------------------------- |
| `s` | Save snapshot (first snapshot sets zero volume)     |
| `y` | Undo last snapshot (removes files and CSV row)      |
| `m` | Draw measurement ROI                                |
| `r` | Draw reference ROI                                  |
| `v` | Cycle visualization (hillshade / elevation / delta) |
| `f` | Flip volume sign                                    |
| `o` | Toggle rotation (0° ↔ 90°, relocks plane)           |
| `l` | Relock ground plane                                 |
| `q` | Quit                                                |

