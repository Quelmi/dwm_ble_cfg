# DWM1001 configuration through BLE API

This repository provides useful classes to both send and receive data through BLE based on [bleak](https://pypi.org/project/bleak/) in order to configure DWM1001 modules. Modules in this repository are based on [DWM1001 BLE API](https://www.decawave.com/dwm1001/api/). Python version used is python3.6.

## Installation

Copy this repository.
 
```bash
git clone https://github.com/esauortiz/dwm_ble_cfg.git
```

Install requirements .
```bash
pip install -r requirements.txt
```

## Usage
Basic configuration is done with ```dwm_cfg.py``` module. Nodes configuration, anchor operation mode and tag operation mode can be configured in ```.yaml``` files placed in ```params``` folder.
```bash
python dwm_cfg.py
```
### Custom configuration
This example uses already implemented ```PersistedPositionMsg``` class to set anchor position.
```python
from dwm1001_BleApi import BleConnectionHandler
from dwm1001_BleApi import PersistedPositionMsg

# BLE connection handler
ble_handler = BleConnectionHandler()
anchor_address = '00:11:22:33:FF:EE'

# msg
anchor_pose_msg = PersistedPositionMsg([10.0, 10.0, 0.0])

# send
ble_handler.send(anchor_address, anchor_pose_msg)
```
Other configurations could be set o read defining new ```BleMsg``` classes.