"""Geriye uyumlu fused JSON/UDP yayimlayici."""

from .udp_json_publisher import UdpJsonPublisher


class UdpFusedPublisher(UdpJsonPublisher):
    def __init__(self, host, port, max_queue_size=1000, overflow_policy="drop_oldest"):
        super().__init__(host, port, max_queue_size, overflow_policy, name="fused")
