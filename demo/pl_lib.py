import socket
import torch
import time
import pickle
import logging
from prettytable import PrettyTable, ALL
import termplotlib as tpl
import numpy as np

END_OF_MESSAGE = "\n\t".encode()
END_OF_GENERATE = "finish".encode()


def get_hist_str(data, bins=40, orientation="vertical"):
    counts, bin_edges = np.histogram(data, bins=bins)
    fig = tpl.figure()
    fig.hist(counts, bin_edges, force_ascii=False, orientation=orientation)
    return fig.get_string()


class CommProfiler:
    def __init__(self) -> None:
        self.amt = []
        self.encode_t = []
        self.send_t = []

    def add_comm(self, amt=0, encode_t=0, send_t=0):
        self.amt.append(amt)
        self.encode_t.append(encode_t)
        self.send_t.append(send_t)

    def get_report(self):
        amt = np.array(self.amt)
        encode_t = np.array(self.encode_t)
        send_t = np.array(self.send_t)

        table = PrettyTable()
        table.field_names = ["", "Histogram"]

        table.align[""] = "l"  # "l" 对应左对齐
        table.align["Histogram"] = "r"  # "l" 对应左对齐
        table.add_row(
            [
                f"""Bytes\n    Total:{np.sum(amt)}\n    AVG:{round(np.mean(amt),2)}""",
                get_hist_str(amt),
            ]
        )
        table.add_row(
            [
                f"""Encode Time\n    Total:{round(np.sum(encode_t), 6)} sec\n    AVG:{round(np.mean(encode_t), 6)} sec""",
                get_hist_str(amt),
            ]
        )
        table.add_row(
            [
                f"""Encode Throughput\n    AVG:{round(np.sum(amt)/np.sum(encode_t)/1e6,2)} MBps""",
                get_hist_str(amt / encode_t),
            ]
        )
        table.add_row(
            [
                f"""Send Time\n    Total:{round(np.sum(send_t), 6)} sec\n    AVG:{round(np.mean(send_t), 6)} sec""",
                get_hist_str(send_t),
            ]
        )
        table.add_row(
            [
                f"""Send bandwidth\n    AVG:{round(np.sum(amt)/np.sum(send_t)/1e6,2)} MBps""",
                get_hist_str(amt / send_t),
            ]
        )
        table.hrules = ALL
        print(table)


class AttnProfiler:
    def __init__(self) -> None:
        self.qkv = []
        self.lora = []

    def log_qkv(self, t):
        self.qkv.append(t)

    def log_lora(self, t):
        self.lora.append(t)

    def get_report(self):
        print(f"qkv mean time {np.mean(self.qkv)}")
        print(f"lora mean time {np.mean(self.lora)}")


def send_tensor(
    s: socket.socket,
    tensor: torch.Tensor,
    processing_method="numpy_pickle",
    trans_protocol="tcp",
    profiler: CommProfiler = None,
):
    # 记录开始时间
    if profiler is not None:
        start_time = time.time()

    data = ENCODING_MAP[processing_method](tensor)
    if profiler is not None:
        encode_time = time.time()
    logging.debug(f"Shape {list(tensor.shape)} tensor of size [{len(data)}] Bytes sent")
    # 发送数据
    SEND_METHOD_MAP[trans_protocol](s, data)

    # 记录结束时间
    if profiler is not None:
        end_time = time.time()

    # 计算耗时和速度
    if profiler is not None:
        profiler.add_comm(len(data), encode_time - start_time, end_time - encode_time)


def numpy_pickle_encoding(tensor):
    try:
        n = tensor.numpy()  # cpu
    except Exception:
        n = tensor.cpu().numpy()  # gpu
    return pickle.dumps(n)


def numpy_pickle_decoding(data):
    return torch.Tensor(pickle.loads(data))


def init_tcp_b(ip="127.0.0.1", port=12345):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    print("device connected")
    return s


def post_tcp_sends():
    pass


def tcp_sends(s: socket.socket, data):
    s.sendall(data + END_OF_MESSAGE)


def recv_tensor(
    s: socket.socket,
    buffer_size=1024,
    encoding_method="numpy_pickle",
    trans_protocol="tcp",
):
    """receive tensor

    Args:
        s (socket.socket): _description_
        buffer_size (int, optional): actually not used. Defaults to 1024.
        encoding_method (str, optional): convert `torch.Tensor` to `bytes`. Defaults to "numpy_pickle".
        trans_protocol (str, optional): Currently only implemented TCP. Defaults to "tcp".

    Returns:
        `torch.Tensor`
    """
    data = RECV_METHOD_MAP[trans_protocol](s, buffer_size)
    logging.debug(f"received data length {len(data)}")
    tensor = DECODING_MAP[encoding_method](data)

    return tensor


def init_tcp_cloud(ip="127.0.0.1", port=12345):
    """init tcp socket for central cloud

    Args:
        ip (str, optional): _description_. Defaults to "127.0.0.1".
        port (int, optional): _description_. Defaults to 12345.

    Returns:
        _type_: _description_
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((ip, port))
    print("cloud listen")
    s.listen(1)
    conn, addr = s.accept()
    print("cloud connected")
    return conn


def tcp_recv(s: socket.socket = None, buffer_size=1024):
    """recursively read tensor from buffer

    Args:
        s (socket.socket, optional): _description_. Defaults to None.
        buffer_size (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    data = b""
    while True:
        temp = s.recv(buffer_size)
        data += temp
        if temp.endswith(END_OF_MESSAGE):
            break
        if temp == END_OF_GENERATE:
            print("received END_OF_GENERATE")
            return

    return data.rstrip(END_OF_MESSAGE)


RECV_METHOD_MAP = {"tcp": tcp_recv}
SEND_METHOD_MAP = {"tcp": tcp_sends}
ENCODING_MAP = {
    "numpy_pickle": numpy_pickle_encoding,
}
DECODING_MAP = {
    "numpy_pickle": numpy_pickle_decoding,
}


class PLAB(torch.nn.Module):
    def __init__(self, in_features: int, rcd: int, rdc: int, out_features: int):
        """A,B matrices on Cloud, PrivateLorA of Q,K,V are stacked to execute in parallel

        Args:
            in_features (int):
            rcd (int):
            rdc (int):
            out_features (int):
        """
        super().__init__()
        self.lora_A = torch.nn.Parameter(
            torch.zeros((3, rcd, in_features)), requires_grad=False
        )  # 3 for q,k,v, stacked together for parallel execution
        self.lora_B = torch.nn.Parameter(
            torch.zeros((3, out_features, rdc)), requires_grad=False
        )

    def forward(self, x, s: socket.socket, buffer_size=int(2048e3)):
        x = x.to(self.lora_A.dtype).to(self.lora_A.device)
        # compress activations
        x = x @ self.lora_A.transpose(2, 1)
        # send compressed activations to edge device
        send_tensor(s, x)
        # receive comparessed and transformed activations from edge device
        x = recv_tensor(s, buffer_size).to(self.lora_B.device).to(self.lora_B.dtype)
        # de compress to hidden dimension
        x = x @ self.lora_B.transpose(2, 1)
        # chunk for q,k,v
        return x.chunk(3, 0)


class PLM(torch.nn.Module):
    def __init__(self, rcd: int, rdc: int, **kwargs) -> None:
        """PrivateLoRA M matrix
        1. receive compressed activations from cloud as input
        2. transform on activations
        3. send back activations

        Args:
            rcd (int): dimension of cloud 2 device
            rdc (int): dimension of device 2 cloud
        """
        super().__init__(**kwargs)
        # q,k,v lora stacked together
        self.lora_M = torch.nn.Parameter(
            torch.zeros((3, rdc, rcd)), requires_grad=False
        )

    def forward(
        self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        # receive compressed activations
        x = recv_tensor(s, buffer_size).to(self.lora_M.device).to(self.lora_M.dtype)
        # transform them
        x = x @ self.lora_M.transpose(2, 1)
        # send them back to cloud
        send_tensor(s, x, profiler=profiler)


class PLMStack(torch.nn.Module):
    def __init__(self, num_hidden_layers: int, rcd: int, rdc: int, **kwargs) -> None:
        """Stack of PrivateLoRA M

        Args:
            num_hidden_layers (int): number of m does not necessarily equal to number of decoder layers.
            rcd (int): dimension of M matrix, indicates transmission base for cloud 2 device connection
            rdc (int): dimension of M matrix, indicates transmission base for device 2 cloud connection
        """
        super().__init__(**kwargs)
        self.layers = torch.nn.ModuleList(
            [PLM(rcd, rdc) for _ in range(num_hidden_layers)]
        )

    def forward(
        self, s: socket.socket, buffer_size=int(2048e3), profiler: CommProfiler = None
    ):
        """
        Args:
            s (socket.socket):
            buffer_size (int, optional): useless but i'm lazy. Defaults to int(2048e3).
            profiler (CommProfiler, optional): if not None profiler will do performance profile. Defaults to None.
        """
        for i, layer in enumerate(self.layers):
            # print(f"{i} th layer")
            layer(s, buffer_size, profiler=profiler)
