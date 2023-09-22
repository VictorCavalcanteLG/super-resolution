from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class MonitorExperimentsConfig:
    project_name: str
    project_configs: Dict[str, Union[str, float, int]]
