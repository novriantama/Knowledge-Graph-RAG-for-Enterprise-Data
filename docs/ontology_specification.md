# Enterprise IT & Software Architecture Ontology Specification

> **Rule of Graph Construction**: Constrain the ontology before writing extraction code. Define 5–10 Entity Types and 8–15 Relationship Types. An open-ended *"extract all entities"* prompt produces a fragmented, noisy graph that no Cypher template or model can reliably query.

---

## 1. Entity Types (7 Bounded Types)

| Entity Type | Description | Example Canonical Nodes |
| :--- | :--- | :--- |
| **`Company`** | Corporate entities, enterprise organizations, and direct legal subsidiaries. | `Acme Global Systems`, `Acme EU GmbH` |
| **`Service`** | Application microservices, API components, and core internal systems. | `Service-101 (API Gateway)`, `Service-102 (User Auth)` |
| **`Technology`** | Programming languages, frameworks, open-source libraries, DB engines, & message brokers. | `FastAPI`, `Pydantic`, `PostgreSQL`, `Redis`, `PyTorch` |
| **`Maintainer`** | Individual open-source maintainers, vendor collectives, or foundation maintainers. | `Supplier-X`, `Supplier-Y`, `Eclipse Foundation` |
| **`Infrastructure`** | Cloud regions, Kubernetes clusters, physical datacenters, and VPC networks. | `AWS us-east-1`, `EKS Cluster Alpha`, `AWS eu-central-1` |
| **`Vendor`** | External SaaS vendors, third-party API providers, and managed service partners. | `Auth0`, `Stripe`, `Cloudflare`, `Datadog` |
| **`Regulation`** | Security compliance frameworks, legal standards, and industry certifications. | `GDPR`, `EU CRA`, `SOC 2 Type II`, `ISO 27001`, `PCI-DSS v4.0` |

---

## 2. Relationship Types (10 Bounded Types)

Strict edge constraints defining allowed source and target entity pairings:

| Relationship Type | Source Entity Types | Target Entity Types | Semantics & Example |
| :--- | :--- | :--- | :--- |
| **`OWNS`** | `Company` | `Service`, `Infrastructure` | Corporate ownership or team stewardship. <br>*(e.g., `Acme EU GmbH` OWNS `AWS eu-central-1`)* |
| **`DEPENDS_ON`** | `Service`, `Technology` | `Service`, `Technology`, `Infrastructure` | Direct functional dependency. <br>*(e.g., `Service-101` DEPENDS_ON `Service-102`)* |
| **`USES_TECH`** | `Service` | `Technology` | Software stack composition. <br>*(e.g., `Service-102` USES_TECH `FastAPI`)* |
| **`MAINTAINED_BY`** | `Technology` | `Maintainer` | Upstream package maintainer. <br>*(e.g., `AnyIO` MAINTAINED_BY `Supplier-X`)* |
| **`HOSTED_ON`** | `Service`, `Technology` | `Infrastructure` | Execution runtime environment. <br>*(e.g., `Service-102` HOSTED_ON `EKS Cluster Alpha`)* |
| **`PARTNERED_WITH`** | `Company`, `Service` | `Vendor` | Third-party commercial integration. <br>*(e.g., `Service-104` PARTNERED_WITH `Stripe`)* |
| **`COMPLIES_WITH`** | `Company`, `Service` | `Regulation` | Legal & security audit compliance. <br>*(e.g., `Acme EU GmbH` COMPLIES_WITH `GDPR`)* |
| **`LOCATED_IN`** | `Infrastructure`, `Company` | `Infrastructure` | Geographic region placement. <br>*(e.g., `EKS Cluster Alpha` LOCATED_IN `AWS us-east-1`)* |
| **`IMPACTS`** | `Maintainer`, `Technology` | `Regulation`, `Service` | Transitive supply chain risk propagation. <br>*(e.g., `Supplier-X` IMPACTS `EU CRA`)* |
| **`REQUIRES_AUDIT`** | `Regulation` | `Infrastructure`, `Service` | Audit requirement constraint. <br>*(e.g., `GDPR` REQUIRES_AUDIT `PostgreSQL Database`)* |

---

## 3. Graph Schema Invariants & Cypher Rules

1. **Entity Node Isolation**: No node can exist without a valid `type` matching one of the 7 `EntityType` enums.
2. **Relationship Validation**: Any extracted relationship not belonging to the 10 allowed `RelationType` enums is rejected during extraction schema parsing.
3. **Citation Edge Tracking**: Every edge in Neo4j must persist a `source_chunk_ids` array property containing the original document chunk ID that justified the relationship link.
