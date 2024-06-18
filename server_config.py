import yaml
from typing import Tuple, AnyStr


def get_server_info() -> Tuple[str, str]:

    with open('server_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    server_info = config.get('server', {})
    host = server_info.get('host', '127.0.0.1')
    port = server_info.get('port', 12345)
    return host, port
