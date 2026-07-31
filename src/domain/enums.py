from enum import Enum

class EntityType(str, Enum):
    COMPANY = "Company"
    PERSON = "Person"
    PRODUCT = "Product"
    TECHNOLOGY = "Technology"
    REGULATION = "Regulation"
    LOCATION = "Location"

class RelationType(str, Enum):
    OWNS = "OWNS"
    USES_TECH = "USES_TECH"
    COMPLIES_WITH = "COMPLIES_WITH"
    DEPENDS_ON = "DEPENDS_ON"
    LOCATED_IN = "LOCATED_IN"
    PARTNERED_WITH = "PARTNERED_WITH"

class RouteChoice(str, Enum):
    VECTOR = "VECTOR"   # Direct fact lookup, definition, policy statement
    GRAPH = "GRAPH"     # Multi-hop relationships, dependency chains, cross-entity comparisons
    HYBRID = "HYBRID"   # Complex queries requiring both structural paths and rich passage context
