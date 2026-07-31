from enum import Enum

class EntityType(str, Enum):
    COMPANY = "Company"
    SERVICE = "Service"
    TECHNOLOGY = "Technology"
    MAINTAINER = "Maintainer"
    INFRASTRUCTURE = "Infrastructure"
    VENDOR = "Vendor"
    REGULATION = "Regulation"

class RelationType(str, Enum):
    OWNS = "OWNS"
    DEPENDS_ON = "DEPENDS_ON"
    USES_TECH = "USES_TECH"
    MAINTAINED_BY = "MAINTAINED_BY"
    HOSTED_ON = "HOSTED_ON"
    PARTNERED_WITH = "PARTNERED_WITH"
    COMPLIES_WITH = "COMPLIES_WITH"
    LOCATED_IN = "LOCATED_IN"
    IMPACTS = "IMPACTS"
    REQUIRES_AUDIT = "REQUIRES_AUDIT"

class RouteChoice(str, Enum):
    VECTOR = "VECTOR"   # Direct fact lookup, definition, policy statement
    GRAPH = "GRAPH"     # Multi-hop relationships, dependency chains, cross-entity comparisons
    HYBRID = "HYBRID"   # Complex queries requiring both structural paths and rich passage context
