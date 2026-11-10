# Novasphere Dynamics - Product FAQ & Technical Reference

## General Architecture & Operating Principles

### What is Novasphere OS?
Novasphere OS is an edge-to-cloud distributed control plane designed for high-density fleets of autonomous mobile robots (AMRs), automated guided vehicles (AGVs), and industrial quadcopters. It provides sub-millisecond telemetry synchronization and deterministic task scheduling.

### What is the HyperSync Protocol?
HyperSync is Novasphere's proprietary UDP-based consensus and telemetry protocol. It replaces traditional WebSocket and MQTT connections for real-time telemetry, delivering guaranteed packet ordering and loss recovery across fluctuating wireless networks.

### What is ZeroGravity Failover?
ZeroGravity is an automated high-availability clustering system available in the Enterprise plan. If an edge gateway or cloud coordinator becomes unresponsive for more than 40 milliseconds, a peer node assumes state ownership immediately with zero packet loss.

### What is QuantumCache v4?
QuantumCache v4 is a distributed, vector-indexed in-memory cache used by Novasphere orchestrators to cache navigational point-clouds and routing policies locally on edge nodes. It reduces repetitive cloud trajectory computations by 85%.

---

## System Requirements & Integrations

### What operating systems and kernels are supported for Edge Nodes?
Novasphere Edge Agent currently supports:
- Ubuntu Linux 22.04 LTS and 24.04 LTS (Kernel 6.2 or higher)
- Debian 12 (Bookworm)
- Red Hat Enterprise Linux (RHEL) 9.2+
- Experimental support: Windows 11 Enterprise IoT (ARM64 and x86_64)

### What robotics middlewares are supported?
Native integration adapters are provided for:
1. ROS2 (Robot Operating System 2) Humble Hawksbill, Iron Irwini, and Jazzy Jalisco
2. PX4 Autopilot (firmware v1.14+)
3. Eclipse Zenoh (v1.0.0+)

---

## Reliability, SLAs & Troubleshooting

### What uptime SLA is guaranteed?
- Starter Tier: 99.5% service uptime SLA.
- Professional Tier: 99.9% service uptime SLA with financial service credits.
- Enterprise Tier: 99.995% service uptime SLA, backed by 24/7 mission-critical response.

### How do I troubleshoot error code `ERR_QUANTUM_TIMEOUT`?
`ERR_QUANTUM_TIMEOUT` occurs when an edge node does not receive a sync heartbeat within 350 milliseconds.
Resolution steps:
1. Verify network firewall rules permit inbound UDP on port 9872.
2. Ensure NTP or PTP clock synchronization drift is less than 5 milliseconds.
3. Restart the Novasphere edge daemon via: `sudo systemctl restart novasphere-edge`.

### How do I troubleshoot error code `ERR_SYNC_DESYNC_09`?
`ERR_SYNC_DESYNC_09` signifies a cryptographic state mismatch between the edge node cache and the regional cloud coordinator.
Resolution steps:
1. Run `novasphere-cli cache verify --deep`.
2. If discrepancies persist, flush local cache using `novasphere-cli cache flush --hard-reset`.
3. The coordinator will stream a fresh snapshot within 12 seconds.
