from enum import Enum

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AlertStatus(str, Enum):
    ACTIVE = "Active"
    ACKNOWLEDGED = "Acknowledged"
    ESCALATED = "Escalated"
    RESOLVED = "Resolved"

class BOPStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"

class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"

class AIMode(str, Enum):
    ACTIVE = "ACTIVE"
    EDGE_LOCAL = "EDGE LOCAL"
    STANDBY = "STANDBY"

class DetectionType(str, Enum):
    VIRTUAL_FENCE = "Virtual Fence Intrusion"
    VEHICLE = "Unrecognized Vehicle"
    LOITERING = "Suspicious Loitering"
    NIGHT_MOVEMENT = "Night Movement Detected"
    FACE = "Face Detection Match"
    ANPR = "ANPR Detection"
    PERSON = "Person Detected"
