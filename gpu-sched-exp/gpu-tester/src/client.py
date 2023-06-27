import json
import struct
from tcp_utils import TcpClient


def main():
    client = TcpClient("localhost", 12345)
    request = {
        "model_name": "fasterrcnn_resnet50_fpn",
        "model_weight": "FasterRCNN_ResNet50_FPN_Weights",
        "input_file_path": "../data-set/rene/0000000099.png",
        "output_file_path": "./logs",
        "output_file_name": "model_B",
        "priority": 1,
        "resize": False,
        "resize_size": [720, 1280],
        "batch_size": 1,
        "device_id": 1,
        "control": {
            "control": True,
            "controlsync": False,
            "controlEvent": False,
            "queue_limit": {
                "sync": 0,
                "event_group": 2
            }
        },
    }
    data = json.dumps(request).encode('utf-8')
    data_length_b = struct.pack('I', len(data))
    client.send(data_length_b)
    client.send(json.dumps(request).encode('utf-8'))
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
