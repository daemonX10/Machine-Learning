# 30 Availability & Reliability interview questions

> Source: [https://devinterview.io/questions/software-architecture-and-system-design/availability-and-reliability-interview-questions/](https://devinterview.io/questions/software-architecture-and-system-design/availability-and-reliability-interview-questions/)
> Scraped: 2026-02-20 00:40
> Total Questions: 30

---

## Table of Contents

1. [Availability & Reliability Fundamentals](#availability-reliability-fundamentals) (8 questions)
2. [Designing for Availability](#designing-for-availability) (5 questions)
3. [Monitoring & Incident Response](#monitoring-incident-response) (5 questions)
4. [Scaling & Performance](#scaling-performance) (4 questions)
5. [Reliability in Distributed Systems](#reliability-in-distributed-systems) (4 questions)
6. [Recovery Strategies](#recovery-strategies) (4 questions)

---

## Availability & Reliability Fundamentals

### 1. What is the difference between availability and reliability in the context of a software system?

**Type:** 📝 Question

**Availability** measures the proportion of time a system is **operational and accessible**, while **reliability** measures the probability a system performs its intended function **without failure** over a given period. A system can be highly available but unreliable (frequent short outages with fast recovery) or highly reliable but less available (rare failures but long recovery times).

- **Availability** = Uptime / (Uptime + Downtime), expressed as a percentage (e.g., 99.9%)
- **Reliability** = e^(-t/MTBF), probability of failure-free operation over time t
- **MTBF** (Mean Time Between Failures) is the key reliability metric
- **MTTR** (Mean Time to Repair) bridges both: Availability = MTBF / (MTBF + MTTR)

```
+--------------------------------------------------+
|         SYSTEM HEALTH METRICS                     |
+--------------------------------------------------+
|                                                    |
|  AVAILABILITY (% uptime)                          |
|  +--------+    +--------+    +--------+           |
|  | UP 99h |    | DOWN 1h|    | UP 99h |           |
|  +--------+    +--------+    +--------+           |
|  = 99% available                                  |
|                                                    |
|  RELIABILITY (failure-free probability)           |
|  +---+  +---+  +---+  +---+  +---+  +---+        |
|  | F |  | F |  | F |  | F |  | F |  | F |        |
|  +---+  +---+  +---+  +---+  +---+  +---+        |
|  6 failures in 200h --> MTBF = 33.3h              |
|                                                    |
|  RELATIONSHIP:                                    |
|  Availability = MTBF / (MTBF + MTTR)             |
|  High MTBF + Low MTTR = High Availability        |
+--------------------------------------------------+
```

| Aspect | Availability | Reliability |
|---|---|---|
| **Measures** | Uptime percentage | Failure-free probability |
| **Formula** | Uptime/(Uptime+Downtime) | e^(-t/MTBF) |
| **Key Metric** | Percentage (99.99%) | MTBF (hours) |
| **Focus** | Minimizing downtime | Preventing failures |
| **Improvement** | Redundancy, fast failover | Better components, testing |
| **Example** | 99.9% = 8.76h downtime/yr | MTBF = 10,000 hours |

```python
import math
from dataclasses import dataclass

@dataclass
class SystemMetrics:
    uptime_hours: float
    downtime_hours: float
    total_failures: int
    observation_period_hours: float

    @property
    def availability(self) -> float:
        """Calculate availability as percentage."""
        total = self.uptime_hours + self.downtime_hours
        return (self.uptime_hours / total) * 100 if total > 0 else 0

    @property
    def mtbf(self) -> float:
        """Mean Time Between Failures in hours."""
        return self.observation_period_hours / self.total_failures if self.total_failures > 0 else float('inf')

    @property
    def mttr(self) -> float:
        """Mean Time to Repair in hours."""
        return self.downtime_hours / self.total_failures if self.total_failures > 0 else 0

    def reliability(self, mission_time: float) -> float:
        """Probability of failure-free operation over mission_time hours."""
        if self.mtbf == float('inf'):
            return 1.0
        return math.exp(-mission_time / self.mtbf)

    def availability_from_mtbf_mttr(self) -> float:
        """Availability derived from MTBF and MTTR."""
        return self.mtbf / (self.mtbf + self.mttr) * 100

# Example: Web service over 1 year
metrics = SystemMetrics(
    uptime_hours=8750,
    downtime_hours=10,
    total_failures=12,
    observation_period_hours=8760
)

print(f"Availability: {metrics.availability:.4f}%")
print(f"MTBF: {metrics.mtbf:.1f} hours")
print(f"MTTR: {metrics.mttr:.2f} hours")
print(f"Reliability (24h mission): {metrics.reliability(24):.6f}")
print(f"Availability (from MTBF/MTTR): {metrics.availability_from_mtbf_mttr():.4f}%")
```

**AI/ML Application:** ML model serving platforms track both availability (API uptime percentage) and reliability (prediction consistency). **Model reliability** degrades through data drift even when the serving infrastructure stays available, requiring separate monitoring for **model staleness** vs **infrastructure health**.

**Real-World Example:** Amazon targets **99.99% availability** for DynamoDB but measures reliability separately through error rates. During the 2017 S3 outage, the system was unavailable for ~4 hours (availability hit) but remained reliable in that no data was lost or corrupted once restored.

> **Interview Tip:** Emphasize that availability and reliability are complementary but distinct. A system that crashes every 5 minutes but recovers in 1 second has ~99.7% availability but terrible reliability. Interviewers value candidates who understand this nuance and can explain the MTBF/MTTR relationship.

---

### 2. How do you define system availability and what are the key components to measure it?

**Type:** 📝 Question

**System availability** is defined as the percentage of time a system is **operational and capable of performing its required function** under stated conditions. It is measured through **Service Level Indicators (SLIs)**, governed by **Service Level Objectives (SLOs)**, and contractually bound by **Service Level Agreements (SLAs)**.

- **SLI** (Service Level Indicator): Quantitative measure (e.g., request success rate, latency P99)
- **SLO** (Service Level Objective): Target value for an SLI (e.g., 99.95% success rate)
- **SLA** (Service Level Agreement): Contract with consequences if SLO is breached
- **Error Budget** = 100% - SLO (e.g., 0.05% allowed failures for 99.95% SLO)

```
+-----------------------------------------------------------+
|              AVAILABILITY MEASUREMENT STACK                |
+-----------------------------------------------------------+
|                                                             |
|   SLA (Contract)     "99.9% uptime or credit issued"       |
|   +-----------------------------------------------+        |
|   |  SLO (Target)    "99.95% success rate"        |        |
|   |  +-------------------------------------------+|        |
|   |  |  SLI (Metric)  "successful_req / total"   ||        |
|   |  +-------------------------------------------+|        |
|   +-----------------------------------------------+        |
|                                                             |
|   ERROR BUDGET TRACKING:                                   |
|   Month: 30 days = 43,200 minutes                          |
|   SLO: 99.95%                                              |
|   Budget: 43,200 * 0.0005 = 21.6 minutes                  |
|                                                             |
|   [==========>        ] 12 min used / 21.6 min total       |
|   Remaining: 9.6 minutes of allowed downtime               |
+-----------------------------------------------------------+
```

| Component | Purpose | Example | Owner |
|---|---|---|---|
| **SLI** | Raw measurement | 99.97% success rate | Engineering |
| **SLO** | Internal target | 99.95% success rate | Engineering + Product |
| **SLA** | External contract | 99.9% with 10% credit | Business + Legal |
| **Error Budget** | Innovation allowance | 0.05% = 21.6 min/month | SRE Team |
| **Request-based** | Per-request success | Errors / Total requests | Monitoring |
| **Time-based** | Uptime windows | Good minutes / Total | Monitoring |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class MeasurementType(Enum):
    REQUEST_BASED = "request_based"
    TIME_BASED = "time_based"

@dataclass
class AvailabilityTracker:
    slo_target: float  # e.g., 99.95
    measurement_type: MeasurementType
    window_days: int = 30
    total_requests: int = 0
    failed_requests: int = 0
    total_minutes: int = 0
    bad_minutes: int = 0

    @property
    def current_availability(self) -> float:
        if self.measurement_type == MeasurementType.REQUEST_BASED:
            if self.total_requests == 0:
                return 100.0
            return ((self.total_requests - self.failed_requests) / self.total_requests) * 100
        else:
            if self.total_minutes == 0:
                return 100.0
            return ((self.total_minutes - self.bad_minutes) / self.total_minutes) * 100

    @property
    def error_budget_total(self) -> float:
        return 100.0 - self.slo_target

    @property
    def error_budget_remaining(self) -> float:
        error_rate = 100.0 - self.current_availability
        return max(0, self.error_budget_total - error_rate)

    @property
    def error_budget_consumed_pct(self) -> float:
        if self.error_budget_total == 0:
            return 100.0
        return ((self.error_budget_total - self.error_budget_remaining) / self.error_budget_total) * 100

    def record_requests(self, total: int, failed: int):
        self.total_requests += total
        self.failed_requests += failed

    def record_minutes(self, total: int, bad: int):
        self.total_minutes += total
        self.bad_minutes += bad

# Example usage
tracker = AvailabilityTracker(slo_target=99.95, measurement_type=MeasurementType.REQUEST_BASED)
tracker.record_requests(total=1_000_000, failed=400)

print(f"Current Availability: {tracker.current_availability:.4f}%")
print(f"SLO Target: {tracker.slo_target}%")
print(f"Error Budget Total: {tracker.error_budget_total:.4f}%")
print(f"Error Budget Remaining: {tracker.error_budget_remaining:.4f}%")
print(f"Error Budget Consumed: {tracker.error_budget_consumed_pct:.1f}%")
```

**AI/ML Application:** ML platforms define availability SLIs specific to inference: **prediction latency P99**, **model load time**, and **prediction success rate**. Error budgets let ML teams deploy risky model updates early in the month when budget is fresh, and freeze deployments when budget is low.

**Real-World Example:** Google SRE defines availability through **request-based SLIs** for most services. Gmail's SLO might be 99.99% of requests succeed within 300ms. Their error budget approach lets teams balance feature velocity against reliability.

> **Interview Tip:** Always mention the SLI/SLO/SLA hierarchy and error budgets. Explain that availability is not just "is the server up?" but a nuanced measurement that can be request-based or time-based, and that error budgets bridge the gap between development speed and operational stability.

---

### 3. Can you explain the concept of “ Five Nines ” and how it relates to system availability ?

**Type:** 📝 Question

**Five Nines** (99.999%) refers to a system availability target allowing only **5.26 minutes of downtime per year**. Each additional "nine" exponentially reduces allowed downtime and exponentially increases the **cost and complexity** of achieving it. The concept provides a universal shorthand for availability targets across the industry.

- **One Nine** (90%): 36.5 days downtime/year — acceptable for development environments
- **Two Nines** (99%): 3.65 days downtime/year — basic production systems
- **Three Nines** (99.9%): 8.76 hours downtime/year — standard web applications
- **Four Nines** (99.99%): 52.6 minutes downtime/year — enterprise-grade systems
- **Five Nines** (99.999%): 5.26 minutes downtime/year — critical infrastructure

```
+-----------------------------------------------------------+
|              AVAILABILITY NINES SCALE                      |
+-----------------------------------------------------------+
|                                                             |
|  Nines   Avail%     Downtime/Year    Cost Curve            |
|  +-----+---------+----------------+------------------+     |
|  |  1  |  90%    | 36.5 days      | $               |     |
|  |  2  |  99%    |  3.65 days     | $$              |     |
|  |  3  |  99.9%  |  8.76 hours    | $$$             |     |
|  |  4  |  99.99% | 52.6 minutes   | $$$$$$          |     |
|  |  5  |  99.999%|  5.26 minutes  | $$$$$$$$$$      |     |
|  |  6  |  99.9999| 31.5 seconds   | $$$$$$$$$$$$$   |     |
|  +-----+---------+----------------+------------------+     |
|                                                             |
|  COST vs AVAILABILITY:                                     |
|  Cost                                                      |
|   ^        *                                               |
|   |       *                                                |
|   |      *                                                 |
|   |    *                                                   |
|   |  *                                                     |
|   |*                                                       |
|   +---------> Nines (1  2  3  4  5  6)                    |
|                                                             |
|  ARCHITECTURE REQUIREMENTS BY LEVEL:                       |
|  3 nines --> single region + failover                      |
|  4 nines --> multi-AZ + auto-failover + monitoring         |
|  5 nines --> multi-region + active-active + chaos eng      |
+-----------------------------------------------------------+
```

| Nines Level | Downtime/Year | Downtime/Month | Architecture Required | Example Systems |
|---|---|---|---|---|
| **2 (99%)** | 3.65 days | 7.31 hours | Single server + backups | Internal tools |
| **3 (99.9%)** | 8.76 hours | 43.8 minutes | Redundant + failover | E-commerce sites |
| **4 (99.99%)** | 52.6 min | 4.38 minutes | Multi-AZ + auto-failover | Banking portals |
| **5 (99.999%)** | 5.26 min | 26.3 seconds | Multi-region active-active | Payment systems |
| **6 (99.9999%)** | 31.5 sec | 2.63 seconds | Custom hardware + software | Air traffic control |

```python
from dataclasses import dataclass

@dataclass
class AvailabilityLevel:
    nines: int
    name: str

    @property
    def percentage(self) -> float:
        return 100.0 - (100.0 / (10 ** self.nines))

    @property
    def downtime_per_year_minutes(self) -> float:
        minutes_per_year = 365.25 * 24 * 60
        return minutes_per_year * (1 - self.percentage / 100)

    @property
    def downtime_per_month_minutes(self) -> float:
        return self.downtime_per_year_minutes / 12

    @property
    def downtime_per_year_human(self) -> str:
        mins = self.downtime_per_year_minutes
        if mins >= 1440:
            return f"{mins / 1440:.2f} days"
        elif mins >= 60:
            return f"{mins / 60:.2f} hours"
        elif mins >= 1:
            return f"{mins:.2f} minutes"
        else:
            return f"{mins * 60:.2f} seconds"

    @property
    def estimated_relative_cost(self) -> int:
        return 10 ** (self.nines - 1)

def calculate_composite_availability(components: list[float]) -> float:
    """Serial components: multiply availabilities."""
    result = 1.0
    for avail in components:
        result *= avail / 100
    return result * 100

def calculate_redundant_availability(single_avail: float, n_replicas: int) -> float:
    """Parallel redundancy: 1 - (1 - A)^N."""
    return (1 - (1 - single_avail / 100) ** n_replicas) * 100

levels = [
    AvailabilityLevel(1, "One Nine"),
    AvailabilityLevel(2, "Two Nines"),
    AvailabilityLevel(3, "Three Nines"),
    AvailabilityLevel(4, "Four Nines"),
    AvailabilityLevel(5, "Five Nines"),
    AvailabilityLevel(6, "Six Nines"),
]

for level in levels:
    print(f"{level.name} ({level.percentage:.4f}%): {level.downtime_per_year_human} downtime/year")

# Composite: web + app + db in series
serial = calculate_composite_availability([99.99, 99.99, 99.99])
print(f"\nSerial (3 components at 99.99%): {serial:.6f}%")

# Redundant: 2 replicas of 99.9% component
redundant = calculate_redundant_availability(99.9, 2)
print(f"Redundant (2x 99.9%): {redundant:.6f}%")
```

**AI/ML Application:** ML inference services rarely achieve five nines for **end-to-end prediction pipelines** because model loading, feature stores, and preprocessing add serial dependencies. Teams often implement **model caching** and **fallback models** (simpler but faster) to maintain availability when the primary model is slow or unavailable.

**Real-World Example:** AWS targets **99.99% (four nines)** for EC2 and **99.999% (five nines)** for S3 durability (not availability). Achieving five nines of availability for S3 would mean only 5 minutes of downtime per year across all regions — they actually target 99.99% availability with 99.999999999% (eleven nines) durability.

> **Interview Tip:** Mention the exponential cost curve — each additional nine roughly 10x the cost. Also highlight the difference between **serial** (multiply) and **parallel** (1-(1-A)^N) availability calculations, as this is a favorite follow-up question.

---

### 4. How does redundancy contribute to the reliability of a system?

**Type:** 📝 Question

**Redundancy** is the duplication of critical components or functions to increase system reliability by ensuring that **no single failure causes a complete outage**. Types include **active-active** (all replicas serve traffic simultaneously), **active-passive** (standby takes over on failure), and **N+1** (one extra beyond minimum required).

- **Hardware Redundancy**: Duplicate servers, power supplies, network links, storage arrays
- **Software Redundancy**: Multiple application instances, database replicas, service meshes
- **Data Redundancy**: Replication (sync/async), RAID configurations, multi-region backups
- **Geographic Redundancy**: Multi-AZ, multi-region, multi-cloud deployments
- **Temporal Redundancy**: Retry mechanisms, checkpoint/restart capabilities

```
+-----------------------------------------------------------+
|              REDUNDANCY PATTERNS                           |
+-----------------------------------------------------------+
|                                                             |
|  ACTIVE-ACTIVE (Load Shared):                              |
|  Client --> LB --+--> Server A (active, 50% traffic)       |
|                  +--> Server B (active, 50% traffic)       |
|  Failover: instant (already serving)                       |
|                                                             |
|  ACTIVE-PASSIVE (Hot Standby):                             |
|  Client --> LB ----> Server A (active, 100% traffic)       |
|                      Server B (passive, standby)           |
|  Failover: seconds (health check + switchover)             |
|                                                             |
|  N+1 REDUNDANCY:                                           |
|  Need 3 servers for load --> Deploy 4                      |
|  +----+  +----+  +----+  +----+                            |
|  | S1 |  | S2 |  | S3 |  | S4 |  <-- spare                |
|  +----+  +----+  +----+  +----+                            |
|  Any 1 fails --> remaining 3 handle full load              |
|                                                             |
|  AVAILABILITY MATH:                                        |
|  Single:   A = 0.999 (99.9%)                               |
|  Parallel: A = 1-(1-0.999)^2 = 0.999999 (99.9999%)        |
|  Serial:   A = 0.999 * 0.999 = 0.998001 (99.8%)           |
+-----------------------------------------------------------+
```

| Redundancy Type | Failover Time | Resource Cost | Use Case | Complexity |
|---|---|---|---|---|
| **Active-Active** | ~0 (instant) | 2x+ resources | High-traffic web apps | High (state sync) |
| **Active-Passive** | Seconds-minutes | 2x resources (idle standby) | Databases, stateful services | Medium |
| **N+1** | Seconds | (N+1)/N overhead | Server farms, k8s pods | Low |
| **N+2** | Seconds | (N+2)/N overhead | Critical infrastructure | Low-Medium |
| **Multi-Region** | Minutes | 2x+ (cross-region) | Global services, DR | Very High |

```python
from dataclasses import dataclass

@dataclass
class RedundancyCalculator:
    """Calculate system availability with different redundancy strategies."""

    @staticmethod
    def single_component(availability: float) -> float:
        return availability

    @staticmethod
    def parallel_redundancy(availability: float, n: int) -> float:
        """Active-active or N+1: 1 - (1-A)^N"""
        return 1 - (1 - availability) ** n

    @staticmethod
    def serial_chain(availabilities: list[float]) -> float:
        """Components in series: multiply all."""
        result = 1.0
        for a in availabilities:
            result *= a
        return result

    @staticmethod
    def k_of_n(availability: float, k: int, n: int) -> float:
        """System works if at least k of n components work."""
        from math import comb
        prob = 0.0
        for i in range(k, n + 1):
            prob += comb(n, i) * (availability ** i) * ((1 - availability) ** (n - i))
        return prob

    @staticmethod
    def full_stack_availability(tiers: dict[str, dict]) -> float:
        """Calculate availability for a multi-tier architecture."""
        tier_avails = []
        for name, config in tiers.items():
            component_avail = config["availability"]
            replicas = config.get("replicas", 1)
            tier_avail = RedundancyCalculator.parallel_redundancy(component_avail, replicas)
            tier_avails.append(tier_avail)
            print(f"  {name}: {component_avail:.4f} x{replicas} = {tier_avail:.8f}")
        return RedundancyCalculator.serial_chain(tier_avails)

calc = RedundancyCalculator()

# Single vs redundant
single = 0.999
print(f"Single (99.9%):     {single:.6f}")
print(f"2x Parallel:        {calc.parallel_redundancy(single, 2):.9f}")
print(f"3x Parallel:        {calc.parallel_redundancy(single, 3):.12f}")

# Full stack example
print("\nFull-Stack Architecture:")
stack = {
    "Load Balancer": {"availability": 0.9999, "replicas": 2},
    "Web Servers":   {"availability": 0.999,  "replicas": 3},
    "App Servers":   {"availability": 0.999,  "replicas": 3},
    "Database":      {"availability": 0.999,  "replicas": 2},
    "Cache":         {"availability": 0.999,  "replicas": 2},
}
total = calc.full_stack_availability(stack)
nines = -1 * __import__('math').log10(1 - total)
print(f"\nTotal System Availability: {total:.9f} ({nines:.2f} nines)")
```

**AI/ML Application:** ML serving uses **model redundancy** at multiple levels: multiple model replicas behind a load balancer, **shadow models** (canary deployments), and **fallback models** (simpler model that serves when the primary GPU model is unavailable). Feature stores employ **read replicas** to ensure feature retrieval doesn't become a bottleneck.

**Real-World Example:** Netflix uses **active-active multi-region** redundancy across three AWS regions. If an entire region fails, traffic automatically shifts to the remaining two. Their **Chaos Monkey** randomly terminates instances to verify redundancy works in practice, not just in theory.

> **Interview Tip:** Always discuss the **cost-benefit tradeoff** of redundancy — it's not free. Mention the parallel availability formula `1-(1-A)^N` and explain that serial dependencies (load balancer → app → DB) multiply, reducing overall availability. A full-stack calculation impresses interviewers.

---

### 5. What is a single point of failure (SPOF) , and how can it be mitigated?

**Type:** 📝 Question

A **Single Point of Failure (SPOF)** is any component whose failure causes the **entire system to become unavailable**. SPOFs exist at every layer — hardware, software, network, data, and even people. Mitigation requires **identifying all SPOFs** through dependency mapping and then applying **redundancy, failover, and decoupling** strategies at each layer.

- **Hardware SPOFs**: Single server, single power supply, single disk, single network switch
- **Software SPOFs**: Single application instance, single database, monolithic deployment
- **Network SPOFs**: Single ISP, single DNS provider, single load balancer
- **Data SPOFs**: Single data center, single region, no backup strategy
- **Human SPOFs**: Single expert ("bus factor" of 1), no runbook documentation

```
+-----------------------------------------------------------+
|              SPOF IDENTIFICATION & MITIGATION              |
+-----------------------------------------------------------+
|                                                             |
|  BEFORE (SPOFs everywhere):                                |
|                                                             |
|  User --> [DNS] --> [LB] --> [App] --> [DB]                |
|            SPOF      SPOF    SPOF     SPOF                 |
|                                                             |
|  AFTER (redundancy at every layer):                        |
|                                                             |
|  User --> [DNS-1] --> [LB-1] --> [App-1] --> [DB-Primary]  |
|       --> [DNS-2]     [LB-2]     [App-2]     [DB-Replica]  |
|                                  [App-3]     [DB-Replica]  |
|                                                             |
|  SPOF ANALYSIS CHECKLIST:                                  |
|  +--------------------------------------------------+      |
|  | Layer         | Component  | Redundancy Strategy |      |
|  |---------------|------------|---------------------|      |
|  | DNS           | Provider   | Multi-provider      |      |
|  | Network       | ISP        | Dual ISP            |      |
|  | Load Balancer | LB node    | Active-passive pair |      |
|  | Compute       | App server | N+1 instances       |      |
|  | Database      | DB server  | Primary + replicas  |      |
|  | Storage       | Disk       | RAID + backups      |      |
|  | Power         | PSU        | Dual PSU + UPS      |      |
|  | People        | On-call    | Rotation + runbooks |      |
|  +--------------------------------------------------+      |
+-----------------------------------------------------------+
```

| SPOF Layer | Example | Mitigation Strategy | Detection Method |
|---|---|---|---|
| **DNS** | Single DNS provider | Multi-provider (Route53 + Cloudflare) | DNS probing |
| **Network** | Single ISP | Dual ISP with BGP failover | Network monitoring |
| **Load Balancer** | Single LB | Active-passive LB pair | Health checks |
| **Application** | Single instance | Auto-scaling group (min 2) | Instance health |
| **Database** | Single DB server | Primary-replica with failover | Replication lag |
| **Storage** | Single disk | RAID 10 + offsite backups | SMART monitoring |
| **Human** | Single expert | Cross-training + documentation | Bus factor audit |

```python
from dataclasses import dataclass, field
from enum import Enum

class Severity(Enum):
    CRITICAL = "critical"  # System down if this fails
    HIGH = "high"          # Major degradation
    MEDIUM = "medium"      # Partial degradation
    LOW = "low"            # Minor impact

@dataclass
class Component:
    name: str
    layer: str
    availability: float
    is_redundant: bool
    replicas: int = 1
    severity_if_failed: Severity = Severity.CRITICAL

    @property
    def effective_availability(self) -> float:
        if self.is_redundant and self.replicas > 1:
            return 1 - (1 - self.availability) ** self.replicas
        return self.availability

    @property
    def is_spof(self) -> bool:
        return not self.is_redundant or self.replicas <= 1

@dataclass
class SPOFAnalyzer:
    components: list[Component] = field(default_factory=list)

    def add(self, component: Component):
        self.components.append(component)

    def find_spofs(self) -> list[Component]:
        return [c for c in self.components if c.is_spof]

    def system_availability(self) -> float:
        result = 1.0
        for c in self.components:
            result *= c.effective_availability
        return result

    def report(self):
        spofs = self.find_spofs()
        print(f"Total Components: {len(self.components)}")
        print(f"SPOFs Found: {len(spofs)}")
        print(f"System Availability: {self.system_availability():.8f}")
        for s in spofs:
            print(f"  SPOF: {s.name} ({s.layer}) - Severity: {s.severity_if_failed.value}")

analyzer = SPOFAnalyzer()
analyzer.add(Component("DNS", "network", 0.9999, True, 2))
analyzer.add(Component("Load Balancer", "network", 0.9999, True, 2))
analyzer.add(Component("App Server", "compute", 0.999, True, 3))
analyzer.add(Component("Database", "data", 0.999, False, 1))  # SPOF!
analyzer.add(Component("Cache", "data", 0.999, True, 2))

analyzer.report()
```

**AI/ML Application:** In ML pipelines, common SPOFs include the **feature store** (if singular), the **model registry**, and the **GPU cluster scheduler**. Teams mitigate by deploying **offline feature caches**, **model artifact mirrors**, and **multi-queue GPU scheduling** with preemption to avoid single-dependency bottlenecks.

**Real-World Example:** GitHub experienced a major outage in 2018 when a **network partition** between their US East Coast data center and a secondary site caused a MySQL SPOF to trigger a cascading failure. They subsequently redesigned their architecture to eliminate database-layer SPOFs through better replication and partitioning.

> **Interview Tip:** When discussing SPOFs, walk through the entire request path from DNS to database and identify potential SPOFs at each layer. Mention the "bus factor" concept for human SPOFs — it shows you think about organizational reliability, not just technical.

---

### 6. Discuss the significance of Mean Time Between Failures (MTBF) in reliability engineering .

**Type:** 📝 Question

**Mean Time Between Failures (MTBF)** measures the average time a **repairable system** operates before experiencing a failure. It is the primary metric for **reliability engineering**, used for capacity planning, maintenance scheduling, spare parts inventory, and predicting system lifecycle costs. Higher MTBF indicates greater reliability.

- **MTBF** = Total Operating Time / Number of Failures
- **Failure Rate (λ)** = 1 / MTBF (failures per unit time)
- **Reliability Function**: R(t) = e^(-t/MTBF) — probability of surviving to time t
- **MTTF** (Mean Time to Failure) — used for **non-repairable** components (e.g., light bulbs)
- **MTBF** applies to **repairable** systems (e.g., servers, networks)

```
+-----------------------------------------------------------+
|              MTBF IN SYSTEM LIFECYCLE                      |
+-----------------------------------------------------------+
|                                                             |
|  BATHTUB CURVE (Failure Rate over Time):                   |
|                                                             |
|  Rate                                                      |
|   ^                                                        |
|   |*                                       *               |
|   | *                                    *                  |
|   |  *                                 *                    |
|   |   *                              *                     |
|   |    *****************************                       |
|   |    Early    Useful Life    Wear-out                    |
|   +----------------------------------------> Time          |
|                                                             |
|  MTBF TIMELINE:                                            |
|  |<-- Op -->|<R>|<-- Op -->|<R>|<-- Op -->|                |
|  +---------+---+---------+---+---------+                   |
|  | Running | F | Running | F | Running |                   |
|  +---------+---+---------+---+---------+                   |
|  |  500 hr |   |  700 hr |   |  600 hr |                   |
|                                                             |
|  MTBF = (500 + 700 + 600) / 3 = 600 hours                |
|  MTTR = avg repair time                                    |
|  Availability = MTBF / (MTBF + MTTR)                      |
+-----------------------------------------------------------+
```

| Metric | Full Name | Formula | Applies To | Example |
|---|---|---|---|---|
| **MTBF** | Mean Time Between Failures | Total Op Time / Failures | Repairable systems | Server: 50,000 hrs |
| **MTTF** | Mean Time to Failure | Total Time / Units Failed | Non-repairable items | SSD: 1.5M hours |
| **MTTR** | Mean Time to Repair | Total Repair Time / Repairs | Repairable systems | Server: 2 hours |
| **MTTD** | Mean Time to Detect | Detection Time / Incidents | Monitoring systems | Alert: 5 minutes |
| **Failure Rate** | Lambda (λ) | 1 / MTBF | All | 0.00002 per hour |

```python
import math
from dataclasses import dataclass

@dataclass
class ReliabilityMetrics:
    operating_hours: list[float]  # Hours between each failure
    repair_hours: list[float]     # Hours to repair each failure

    @property
    def num_failures(self) -> int:
        return len(self.operating_hours)

    @property
    def mtbf(self) -> float:
        return sum(self.operating_hours) / self.num_failures

    @property
    def mttr(self) -> float:
        return sum(self.repair_hours) / len(self.repair_hours)

    @property
    def failure_rate(self) -> float:
        return 1 / self.mtbf

    @property
    def availability(self) -> float:
        return self.mtbf / (self.mtbf + self.mttr)

    def reliability_at_time(self, t: float) -> float:
        """Probability of failure-free operation for t hours."""
        return math.exp(-t / self.mtbf)

    def predict_failures(self, period_hours: float) -> float:
        """Expected number of failures in a given period."""
        return period_hours * self.failure_rate

    def spare_parts_needed(self, period_hours: float, confidence: float = 0.95) -> int:
        """Number of spares needed for a confidence level."""
        from scipy.stats import poisson
        expected = self.predict_failures(period_hours)
        return int(poisson.ppf(confidence, expected)) + 1

# Example: server farm reliability analysis
metrics = ReliabilityMetrics(
    operating_hours=[720, 1100, 850, 950, 1200, 780],
    repair_hours=[2.5, 1.0, 3.0, 1.5, 0.5, 2.0]
)

print(f"MTBF: {metrics.mtbf:.1f} hours ({metrics.mtbf/24:.1f} days)")
print(f"MTTR: {metrics.mttr:.2f} hours")
print(f"Failure Rate: {metrics.failure_rate:.6f} per hour")
print(f"Availability: {metrics.availability:.6f} ({metrics.availability*100:.4f}%)")
print(f"Reliability (30 days): {metrics.reliability_at_time(720):.6f}")
print(f"Expected failures/year: {metrics.predict_failures(8760):.1f}")
```

**AI/ML Application:** MTBF for ML systems includes both **infrastructure failures** (GPU crashes, OOM) and **model failures** (accuracy drops, data pipeline breaks). Teams track **model MTBF** separately — how long a deployed model meets accuracy thresholds before requiring retraining or rollback.

**Real-World Example:** Hard drive manufacturers publish MTBF ratings (e.g., 1.2 million hours for enterprise SSDs). Google's study of 100,000+ drives showed actual failure rates were **2-10x higher** than manufacturer MTBF claims, leading to their multi-layered redundancy approach (GFS/Colossus replicates data 3x across different failure domains).

> **Interview Tip:** Distinguish MTBF (repairable) from MTTF (non-repairable) — this shows precision. Mention the **bathtub curve** with its three phases: early failures (burn-in), constant rate (useful life), and wear-out. Note that MTBF alone is insufficient; you need MTTR context to derive availability.

---

### 7. What is the role of Mean Time to Repair (MTTR) in maintaining system availability ?

**Type:** 📝 Question

**Mean Time to Repair (MTTR)** measures the average time to **restore a failed system to operational status**. It directly impacts availability through the formula `Availability = MTBF / (MTBF + MTTR)`. Reducing MTTR is often **more cost-effective** than increasing MTBF because it focuses on faster detection, diagnosis, and recovery rather than preventing all failures.

- **MTTD** (Mean Time to Detect): Time from failure occurrence to detection (monitoring latency)
- **MTTI** (Mean Time to Identify): Time from detection to root cause identification
- **MTTF** (Mean Time to Fix): Time from identification to fix implementation
- **MTTV** (Mean Time to Verify): Time from fix to verification that service is restored
- **MTTR** = MTTD + MTTI + MTTF + MTTV (total recovery time)

```
+-----------------------------------------------------------+
|              MTTR BREAKDOWN                                |
+-----------------------------------------------------------+
|                                                             |
|  INCIDENT TIMELINE:                                        |
|                                                             |
|  Failure    Detected   Identified   Fixed    Verified      |
|  Occurs     (Alert)    (Root Cause) (Deploy) (Confirmed)   |
|    |           |            |          |          |         |
|    v           v            v          v          v         |
|    +---MTTD---+----MTTI----+---MTTF---+---MTTV--+         |
|    |           |            |          |          |         |
|    |<-------------- MTTR (total) --------------->|         |
|                                                             |
|  EXAMPLE:                                                  |
|  MTTD = 5 min  (alert fires)                              |
|  MTTI = 15 min (find root cause)                           |
|  MTTF = 25 min (deploy fix)                               |
|  MTTV = 5 min  (verify restored)                          |
|  MTTR = 50 min total                                       |
|                                                             |
|  IMPACT ON AVAILABILITY:                                   |
|  MTBF=720h, MTTR=1h  --> A = 99.86%                       |
|  MTBF=720h, MTTR=0.5h --> A = 99.93% (halve MTTR)        |
|  MTBF=1440h, MTTR=1h --> A = 99.93% (double MTBF)        |
|  Reducing MTTR is more efficient!                          |
+-----------------------------------------------------------+
```

| MTTR Phase | Activity | Reduction Strategy | Tools |
|---|---|---|---|
| **MTTD** (Detect) | Alert fires | Better monitoring, lower thresholds | Prometheus, Datadog |
| **MTTI** (Identify) | Find root cause | Distributed tracing, log aggregation | Jaeger, ELK Stack |
| **MTTF** (Fix) | Deploy solution | Runbooks, automated rollback, feature flags | PagerDuty, Argo |
| **MTTV** (Verify) | Confirm recovery | Automated health checks, synthetic tests | Grafana, Pingdom |

```python
from dataclasses import dataclass, field
from statistics import mean, stdev

@dataclass
class IncidentRecord:
    failure_time_min: float
    detect_time_min: float
    identify_time_min: float
    fix_time_min: float
    verify_time_min: float

    @property
    def mttd(self) -> float:
        return self.detect_time_min - self.failure_time_min

    @property
    def mtti(self) -> float:
        return self.identify_time_min - self.detect_time_min

    @property
    def mttf(self) -> float:
        return self.fix_time_min - self.identify_time_min

    @property
    def mttv(self) -> float:
        return self.verify_time_min - self.fix_time_min

    @property
    def mttr(self) -> float:
        return self.verify_time_min - self.failure_time_min

@dataclass
class MTTRAnalyzer:
    incidents: list[IncidentRecord] = field(default_factory=list)

    def add_incident(self, failure: float, detect: float, identify: float, fix: float, verify: float):
        self.incidents.append(IncidentRecord(failure, detect, identify, fix, verify))

    def analyze(self):
        mttds = [i.mttd for i in self.incidents]
        mttis = [i.mtti for i in self.incidents]
        mttfs = [i.mttf for i in self.incidents]
        mttvs = [i.mttv for i in self.incidents]
        mttrs = [i.mttr for i in self.incidents]

        print(f"Incidents Analyzed: {len(self.incidents)}")
        print(f"Avg MTTD (Detect):   {mean(mttds):.1f} min (stdev: {stdev(mttds):.1f})")
        print(f"Avg MTTI (Identify): {mean(mttis):.1f} min (stdev: {stdev(mttis):.1f})")
        print(f"Avg MTTF (Fix):      {mean(mttfs):.1f} min (stdev: {stdev(mttfs):.1f})")
        print(f"Avg MTTV (Verify):   {mean(mttvs):.1f} min (stdev: {stdev(mttvs):.1f})")
        print(f"Avg MTTR (Total):    {mean(mttrs):.1f} min (stdev: {stdev(mttrs):.1f})")

        bottleneck = max(
            ("Detect", mean(mttds)), ("Identify", mean(mttis)),
            ("Fix", mean(mttfs)), ("Verify", mean(mttvs)),
            key=lambda x: x[1]
        )
        print(f"\nBottleneck Phase: {bottleneck[0]} ({bottleneck[1]:.1f} min avg)")

analyzer = MTTRAnalyzer()
analyzer.add_incident(0, 3, 20, 45, 50)   # Incident 1
analyzer.add_incident(0, 5, 35, 55, 60)   # Incident 2
analyzer.add_incident(0, 2, 10, 30, 35)   # Incident 3
analyzer.add_incident(0, 8, 40, 70, 78)   # Incident 4
analyzer.analyze()
```

**AI/ML Application:** ML systems have unique MTTR challenges: **model rollback** (switching to previous model version) can be done in minutes with proper versioning, but **retraining** a model after data pipeline corruption can take hours or days. Teams pre-compute **shadow model predictions** with the previous version to enable instant fallback.

**Real-World Example:** Amazon reduced MTTR for AWS services by implementing **automated remediation** — if CloudWatch detects a failing instance, Lambda functions automatically replace it without human intervention. This reduced MTTD to seconds and MTTF to minutes for many common failure types.

> **Interview Tip:** Break MTTR into its four sub-phases (Detect, Identify, Fix, Verify) and explain that the biggest wins often come from reducing MTTD through better monitoring. Mention that reducing MTTR by 50% has the same availability effect as doubling MTBF — but is usually much cheaper.

---

### 8. Can you differentiate between high availability (HA) and fault tolerance (FT) ?

**Type:** 📝 Question

**High Availability (HA)** aims to **minimize downtime** by ensuring rapid failover when components fail, accepting brief service interruptions. **Fault Tolerance (FT)** aims for **zero downtime** by operating through failures without any user-visible impact. FT is a stricter requirement than HA and typically requires **specialized hardware** or **synchronous replication** with significantly higher cost.

- **HA**: System recovers quickly from failures (seconds to minutes of downtime acceptable)
- **FT**: System continues operating without any interruption during failures
- HA uses **detection + failover**; FT uses **masking** (failure is invisible to users)
- HA targets 99.9%-99.99%; FT targets 99.999%+ with zero-perceptible downtime
- FT requires **duplicate everything** running in lockstep; HA can use cheaper active-passive

```
+-----------------------------------------------------------+
|         HIGH AVAILABILITY vs FAULT TOLERANCE               |
+-----------------------------------------------------------+
|                                                             |
|  HIGH AVAILABILITY:                                        |
|  Normal:    Client --> [Server A] (active)                 |
|                        [Server B] (standby)                |
|  Failure:   Client --> [Server A] X FAILS                  |
|             Client -/-> brief pause (2-30 sec)             |
|             Client --> [Server B] (promoted)               |
|  Result: Brief downtime during failover                    |
|                                                             |
|  FAULT TOLERANCE:                                          |
|  Normal:    Client --> [Server A] <=sync=> [Server B]      |
|             Both process every request simultaneously      |
|  Failure:   Client --> [Server A] X FAILS                  |
|             Client --> [Server B] (already has state)       |
|  Result: Zero downtime, zero data loss                     |
|                                                             |
|  SPECTRUM:                                                 |
|  Basic HA    Warm Standby   Hot Standby    Full FT         |
|  |-----------|-------------|-------------|                  |
|  Min failover   Fast failover  Near-zero   Zero impact     |
|  99.9%          99.99%         99.999%     99.9999%+       |
|  $              $$             $$$         $$$$$$           |
+-----------------------------------------------------------+
```

| Aspect | High Availability (HA) | Fault Tolerance (FT) |
|---|---|---|
| **Goal** | Minimize downtime | Zero downtime |
| **Downtime** | Seconds to minutes | None (invisible) |
| **Approach** | Detect + failover | Mask failures entirely |
| **State Sync** | Async replication OK | Synchronous replication required |
| **Cost** | 1.5-2x base | 3-10x base |
| **Complexity** | Medium | Very high |
| **Data Loss** | Possible (RPO > 0) | None (RPO = 0) |
| **Examples** | AWS RDS Multi-AZ | Boeing flight controls, RAID 1 |
| **Target** | 99.9%-99.99% | 99.999%+ |

```python
from dataclasses import dataclass
from enum import Enum
import time
import random

class FailoverType(Enum):
    HA_COLD = "cold_standby"       # Minutes to failover
    HA_WARM = "warm_standby"       # Seconds to failover
    HA_HOT = "hot_standby"         # Sub-second failover
    FT_ACTIVE = "fault_tolerant"   # Zero failover time

@dataclass
class SystemConfig:
    name: str
    failover_type: FailoverType
    num_replicas: int
    sync_replication: bool
    failover_time_seconds: float
    component_availability: float
    cost_multiplier: float

    @property
    def effective_availability(self) -> float:
        base = 1 - (1 - self.component_availability) ** self.num_replicas
        downtime_factor = self.failover_time_seconds / (365.25 * 24 * 3600)
        return base - downtime_factor

    @property
    def rpo_seconds(self) -> float:
        """Recovery Point Objective - how much data can be lost."""
        return 0.0 if self.sync_replication else 30.0

    @property
    def rto_seconds(self) -> float:
        """Recovery Time Objective - how long until recovery."""
        return self.failover_time_seconds

    def annual_downtime_minutes(self, expected_failures: int = 12) -> float:
        return (self.failover_time_seconds * expected_failures) / 60

configs = [
    SystemConfig("Basic HA", FailoverType.HA_COLD, 2, False, 300, 0.999, 1.5),
    SystemConfig("Warm Standby", FailoverType.HA_WARM, 2, False, 30, 0.999, 1.8),
    SystemConfig("Hot Standby", FailoverType.HA_HOT, 2, True, 2, 0.999, 2.5),
    SystemConfig("Full FT", FailoverType.FT_ACTIVE, 3, True, 0, 0.999, 5.0),
]

print(f"{'Config':<16} {'Failover':>10} {'RPO':>8} {'RTO':>8} {'Cost':>6} {'Downtime/yr':>14}")
print("-" * 70)
for c in configs:
    print(f"{c.name:<16} {c.failover_type.value:>10} {c.rpo_seconds:>6.0f}s {c.rto_seconds:>6.0f}s {c.cost_multiplier:>5.1f}x {c.annual_downtime_minutes():>10.1f} min")
```

**AI/ML Application:** ML serving systems typically use **HA** (not FT) because inference is often **idempotent** — a retry is acceptable. However, **training pipelines** with days-long GPU jobs may use **FT checkpointing** (every N minutes, save state to persistent storage) so that failures don't require restarting from scratch.

**Real-World Example:** Boeing 787 flight control systems use **triple modular redundancy (TMR)** — three independent computers running identical software. If one disagrees with the other two, its output is ignored. This is true fault tolerance: passengers never experience any interruption. In contrast, AWS RDS Multi-AZ provides HA with ~30-second failover during primary failure.

> **Interview Tip:** Frame HA vs FT as a **cost-benefit decision** — not every system needs FT. Mention specific numbers: HA typically costs 1.5-2x, while FT can cost 5-10x. Use RPO (data loss tolerance) and RTO (downtime tolerance) to determine which approach is appropriate for a given system.

---

## Designing for Availability

### 9. How would you architect a system for high availability ?

**Type:** 📝 Question

A **highly available architecture** eliminates single points of failure across every layer through **redundancy**, **automatic failover**, **geographic distribution**, and **graceful degradation**. The design follows the principle that every component will eventually fail, so the system must continue operating despite individual component failures.

- **Multi-AZ Deployment**: Distribute across availability zones within a region
- **Auto-Scaling**: Dynamically adjust capacity based on demand and health
- **Database Replication**: Primary-replica with automatic failover (RDS Multi-AZ)
- **Stateless Services**: No server-side session state — enables any-instance routing
- **Load Balancing**: Distribute traffic and route around unhealthy instances
- **Circuit Breakers**: Prevent cascading failures across service boundaries
- **Health Checks**: Continuous monitoring with automatic instance replacement

```
+-----------------------------------------------------------+
|         HIGH AVAILABILITY ARCHITECTURE                     |
+-----------------------------------------------------------+
|                                                             |
|  Region: us-east-1                                         |
|  +-----------------------------------------------------+   |
|  |  Route 53 (DNS) -- health check failover              |  |
|  +-----------------------------------------------------+   |
|           |                          |                      |
|     AZ-1 (us-east-1a)         AZ-2 (us-east-1b)           |
|  +-------------------+     +-------------------+           |
|  |    ALB Node 1     |     |    ALB Node 2     |          |
|  +-------------------+     +-------------------+           |
|  | +---+ +---+ +---+ |     | +---+ +---+ +---+ |          |
|  | |App| |App| |App| |     | |App| |App| |App| |          |
|  | +---+ +---+ +---+ |     | +---+ +---+ +---+ |          |
|  | Auto-Scaling Group |     | Auto-Scaling Group |          |
|  +-------------------+     +-------------------+           |
|           |                          |                      |
|  +-------------------+     +-------------------+           |
|  |   DB Primary      |<--->|   DB Standby      |          |
|  |   (sync replic)   |     |   (auto failover) |          |
|  +-------------------+     +-------------------+           |
|           |                          |                      |
|  +-------------------+     +-------------------+           |
|  |   ElastiCache     |     |   ElastiCache     |          |
|  |   Primary         |     |   Replica          |          |
|  +-------------------+     +-------------------+           |
|  +-----------------------------------------------------+   |
|  |  S3 (cross-AZ replication built-in, 11 nines)        |  |
|  +-----------------------------------------------------+   |
+-----------------------------------------------------------+
```

| HA Component | Strategy | Failover Time | Data Loss Risk |
|---|---|---|---|
| **DNS** | Route 53 health checks | 60-300 seconds | None |
| **Load Balancer** | Multi-AZ ALB/NLB | Automatic | None |
| **Web/App Tier** | Auto-scaling group (min 2 AZs) | Seconds | None (stateless) |
| **Database** | Multi-AZ primary-standby | 60-120 seconds | Minimal (sync) |
| **Cache** | Redis cluster with replicas | Seconds | Possible (async) |
| **Storage** | S3 cross-AZ redundancy | None (built-in) | None |
| **Queue** | SQS (distributed by design) | None (built-in) | None |

```python
from dataclasses import dataclass, field
from enum import Enum

class DeploymentTier(Enum):
    DNS = "dns"
    LOAD_BALANCER = "load_balancer"
    APPLICATION = "application"
    DATABASE = "database"
    CACHE = "cache"
    STORAGE = "storage"
    QUEUE = "queue"

@dataclass
class HAComponent:
    tier: DeploymentTier
    name: str
    availability_per_instance: float
    instances_per_az: int
    num_azs: int
    sync_replication: bool = False

    @property
    def total_instances(self) -> int:
        return self.instances_per_az * self.num_azs

    @property
    def tier_availability(self) -> float:
        """Availability with redundancy across AZs."""
        single_az = 1 - (1 - self.availability_per_instance) ** self.instances_per_az
        return 1 - (1 - single_az) ** self.num_azs

@dataclass
class HAArchitecture:
    name: str
    components: list[HAComponent] = field(default_factory=list)

    def add(self, component: HAComponent):
        self.components.append(component)

    @property
    def system_availability(self) -> float:
        result = 1.0
        for c in self.components:
            result *= c.tier_availability
        return result

    def report(self):
        print(f"\n{'='*60}")
        print(f"Architecture: {self.name}")
        print(f"{'='*60}")
        for c in self.components:
            print(f"  {c.tier.value:<15} {c.name:<20} "
                  f"Instances: {c.total_instances} "
                  f"Avail: {c.tier_availability:.10f}")
        avail = self.system_availability
        import math
        nines = -math.log10(1 - avail) if avail < 1 else float('inf')
        print(f"\n  System Availability: {avail:.10f} ({nines:.2f} nines)")
        downtime = (1 - avail) * 365.25 * 24 * 60
        print(f"  Annual Downtime: {downtime:.2f} minutes")

arch = HAArchitecture("Production E-Commerce")
arch.add(HAComponent(DeploymentTier.DNS, "Route 53", 0.9999, 1, 2))
arch.add(HAComponent(DeploymentTier.LOAD_BALANCER, "ALB", 0.9999, 1, 2))
arch.add(HAComponent(DeploymentTier.APPLICATION, "App Servers", 0.999, 3, 2))
arch.add(HAComponent(DeploymentTier.DATABASE, "RDS Multi-AZ", 0.999, 1, 2, sync_replication=True))
arch.add(HAComponent(DeploymentTier.CACHE, "ElastiCache", 0.999, 1, 2))
arch.report()
```

**AI/ML Application:** HA for ML systems requires **model artifact replication** across AZs (store in S3), **stateless inference containers** behind a load balancer, and **feature store replicas** in each AZ. GPU instances are expensive, so ML teams often use **spot instances** with automatic failover to on-demand when interrupted.

**Real-World Example:** Netflix's architecture spans three AWS regions with **active-active** deployment. Each region can handle 100% of traffic if needed. Their **Zuul gateway** routes traffic, and **Eureka** provides service discovery with health-based routing. During the 2012 Christmas Eve AWS outage, Netflix remained operational by shifting traffic to healthy regions.

> **Interview Tip:** Draw the architecture diagram with specific AWS/cloud services at each tier. Show that you understand the full stack from DNS through database. Mention that stateless application tiers are essential for HA because they allow any-instance routing and easy horizontal scaling.

---

### 10. What design patterns are commonly used to improve system availability ?

**Type:** 📝 Question

Several **design patterns** directly improve availability by addressing failure modes at different architectural layers. Key patterns include **circuit breaker** (prevent cascading failures), **bulkhead** (isolate failure domains), **retry with backoff** (handle transient faults), **graceful degradation** (serve reduced functionality), and **leader election** (coordinate redundant nodes).

- **Circuit Breaker**: Stops requests to failing services, fails fast, allows recovery time
- **Bulkhead**: Isolates components so failure in one doesn't exhaust shared resources
- **Retry with Exponential Backoff**: Retries transient failures with increasing delays
- **Graceful Degradation**: Serves reduced functionality when dependencies are unavailable
- **Leader Election**: Coordinates which replica is primary in distributed systems
- **Saga Pattern**: Manages distributed transactions with compensating actions
- **Sidecar/Ambassador**: Offloads reliability concerns (retries, circuit breaking) from app code

```
+-----------------------------------------------------------+
|         AVAILABILITY DESIGN PATTERNS                       |
+-----------------------------------------------------------+
|                                                             |
|  CIRCUIT BREAKER:                                          |
|  Closed --> [failures exceed threshold] --> Open            |
|    ^                                         |              |
|    |                                    [timeout]           |
|    |                                         |              |
|    +--- [success] <--- Half-Open <-----------+              |
|                                                             |
|  BULKHEAD ISOLATION:                                       |
|  +-----------+  +-----------+  +-----------+               |
|  | Service A |  | Service B |  | Service C |               |
|  | Pool: 20  |  | Pool: 15  |  | Pool: 10  |               |
|  | threads   |  | threads   |  | threads   |               |
|  +-----------+  +-----------+  +-----------+               |
|  B fails --> only B's pool exhausted, A and C unaffected   |
|                                                             |
|  RETRY WITH BACKOFF:                                       |
|  Attempt 1 --> fail --> wait 1s                            |
|  Attempt 2 --> fail --> wait 2s                            |
|  Attempt 3 --> fail --> wait 4s                            |
|  Attempt 4 --> fail --> wait 8s + jitter                   |
|  Attempt 5 --> circuit breaker opens                       |
|                                                             |
|  GRACEFUL DEGRADATION:                                     |
|  Normal:  Full features + recommendations + analytics      |
|  Level 1: Core features + cached recommendations           |
|  Level 2: Core features only (static fallback)             |
|  Level 3: Read-only mode (maintenance page for writes)     |
+-----------------------------------------------------------+
```

| Pattern | Problem Solved | How It Works | Example |
|---|---|---|---|
| **Circuit Breaker** | Cascading failures | Open/closed/half-open states | Hystrix, Resilience4j |
| **Bulkhead** | Resource exhaustion | Isolate thread/connection pools | Separate pools per service |
| **Retry + Backoff** | Transient faults | Exponential delays with jitter | AWS SDK retries |
| **Graceful Degradation** | Dependency failures | Reduce features progressively | Netflix without recs |
| **Leader Election** | Coordination | Consensus for primary selection | ZooKeeper, etcd |
| **Saga** | Distributed transactions | Compensating actions on failure | Order + Payment rollback |
| **Sidecar** | Cross-cutting concerns | Proxy handles retries, TLS | Envoy, Istio |

```python
import random
import time
from dataclasses import dataclass, field
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit is OPEN - failing fast")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

def retry_with_backoff(func, max_retries=5, base_delay=1.0, max_delay=60.0):
    """Retry with exponential backoff and jitter."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = delay * random.uniform(0, 0.3)
            print(f"  Attempt {attempt+1} failed, retrying in {delay+jitter:.2f}s")

# Demo
cb = CircuitBreaker(failure_threshold=3)
for i in range(7):
    try:
        def flaky():
            if random.random() < 0.6:
                raise Exception("Service unavailable")
            return "OK"
        result = cb.call(flaky)
        print(f"Call {i+1}: {result} (state={cb.state.value})")
    except Exception as e:
        print(f"Call {i+1}: {e} (state={cb.state.value})")
```

**AI/ML Application:** ML inference services use **circuit breakers** to fall back to simpler models when the primary GPU model is slow, **bulkheads** to isolate different model endpoints (recommendation vs search ranking), and **graceful degradation** that returns cached predictions or popular-items defaults when the model service is down.

**Real-World Example:** Netflix pioneered the circuit breaker pattern with **Hystrix** (now maintenance-mode, succeeded by Resilience4j). When their recommendation service fails, the UI degrades gracefully to show **trending content** instead of personalized recommendations — users still get a functional experience.

> **Interview Tip:** When discussing patterns, always mention the **failure mode** each pattern addresses. Circuit breaker prevents cascading failures, bulkhead prevents resource exhaustion, retry handles transient faults. Show you understand *when* to apply each pattern, not just *how* they work.

---

### 11. How can load balancing improve system availability, and what are some of its potential pitfalls?

**Type:** 📝 Question

**Load balancing** improves availability by distributing traffic across multiple healthy instances, **routing around failures**, and preventing any single instance from becoming overwhelmed. However, pitfalls include the load balancer itself becoming a **SPOF**, **uneven distribution** with certain algorithms, **session affinity** complications, and **thundering herd** problems during instance recovery.

- **Health-Check Routing**: Automatically removes unhealthy instances from the pool
- **Traffic Distribution**: Spreads load to prevent capacity exhaustion on any single node
- **Horizontal Scaling**: Enables adding instances without changing client configuration
- **Pitfall: LB as SPOF**: If the LB fails, everything behind it is unreachable
- **Pitfall: Sticky Sessions**: Session affinity reduces effective redundancy
- **Pitfall: Thundering Herd**: All connections rush to a recovered instance

```
+-----------------------------------------------------------+
|         LOAD BALANCING FOR AVAILABILITY                    |
+-----------------------------------------------------------+
|                                                             |
|  HEALTHY STATE:                                            |
|  Client --> LB --+--> Instance A [healthy] (33%)           |
|                  +--> Instance B [healthy] (33%)           |
|                  +--> Instance C [healthy] (33%)           |
|                                                             |
|  INSTANCE FAILURE:                                         |
|  Client --> LB --+--> Instance A [healthy] (50%)           |
|                  +--> Instance B [FAILED]  (removed)       |
|                  +--> Instance C [healthy] (50%)           |
|  Health check detects B down --> traffic redistributed     |
|                                                             |
|  PITFALL - LB AS SPOF:                                     |
|  Client --> [LB] X FAILS --> all traffic lost!             |
|  Fix: Deploy active-passive LB pair with VRRP/keepalived  |
|                                                             |
|  PITFALL - THUNDERING HERD:                                |
|  Instance B recovers --> LB sends ALL queued connections   |
|  +--------+                                                |
|  |Instance|  <-- 0 conn --> 500 conn suddenly!             |
|  |   B    |  Fix: slow-start (gradual ramp-up)            |
|  +--------+                                                |
|                                                             |
|  PITFALL - STICKY SESSIONS:                                |
|  User1 --> always Instance A (pinned)                      |
|  If A fails, User1 loses session state                     |
|  Fix: External session store (Redis)                       |
+-----------------------------------------------------------+
```

| LB Algorithm | Strength | Weakness | Best For |
|---|---|---|---|
| **Round Robin** | Simple, even distribution | Ignores server capacity | Homogeneous instances |
| **Least Connections** | Capacity-aware | Slow to react | Varying request duration |
| **Weighted** | Handles heterogeneous | Manual weight management | Mixed instance sizes |
| **IP Hash** | Session persistence | Uneven with few clients | Stateful applications |
| **Random** | Low overhead | Statistically uneven | Large server pools |
| **Least Response Time** | Performance-optimized | Costly health probes | Latency-sensitive apps |

```python
from dataclasses import dataclass, field
import random
from collections import defaultdict

@dataclass
class BackendInstance:
    name: str
    healthy: bool = True
    connections: int = 0
    weight: int = 1
    slow_start_remaining: int = 0

@dataclass
class LoadBalancer:
    instances: list[BackendInstance] = field(default_factory=list)
    health_check_interval: int = 10
    slow_start_window: int = 30
    algorithm: str = "round_robin"
    _rr_index: int = 0

    def healthy_instances(self) -> list[BackendInstance]:
        return [i for i in self.instances if i.healthy]

    def route(self) -> BackendInstance:
        healthy = self.healthy_instances()
        if not healthy:
            raise Exception("No healthy instances!")

        if self.algorithm == "round_robin":
            instance = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
        elif self.algorithm == "least_connections":
            instance = min(healthy, key=lambda i: i.connections)
        elif self.algorithm == "weighted_random":
            weights = [i.weight for i in healthy]
            instance = random.choices(healthy, weights=weights, k=1)[0]
        else:
            instance = random.choice(healthy)

        instance.connections += 1
        return instance

    def release(self, instance: BackendInstance):
        instance.connections = max(0, instance.connections - 1)

    def simulate_traffic(self, num_requests: int) -> dict:
        distribution = defaultdict(int)
        for _ in range(num_requests):
            try:
                inst = self.route()
                distribution[inst.name] += 1
                if random.random() < 0.3:
                    self.release(inst)
            except Exception:
                distribution["FAILED"] += 1
        return dict(distribution)

# Simulate normal vs failure scenario
lb = LoadBalancer(algorithm="least_connections")
lb.instances = [
    BackendInstance("server-1"), BackendInstance("server-2"),
    BackendInstance("server-3"), BackendInstance("server-4"),
]

print("Normal operation (4 healthy):")
dist = lb.simulate_traffic(1000)
for name, count in sorted(dist.items()):
    print(f"  {name}: {count} requests ({count/10:.1f}%)")

# Simulate failure
lb.instances[1].healthy = False
print("\nAfter server-2 failure (3 healthy):")
dist = lb.simulate_traffic(1000)
for name, count in sorted(dist.items()):
    print(f"  {name}: {count} requests ({count/10:.1f}%)")
```

**AI/ML Application:** ML model serving uses **model-aware load balancing** that routes based on model version, GPU memory availability, and batch queue depth. Specialized LB algorithms check **GPU utilization** rather than just connections — a GPU at 90% utilization with 2 connections is busier than a CPU instance with 10 connections.

**Real-World Example:** GitHub uses **GLB** (GitHub Load Balancer), a custom L4 load balancer built on DPDK. During their 2018 DDoS attack (1.35 Tbps), their LB tier distributed traffic to Akamai Prolexic for scrubbing while maintaining service for legitimate users — demonstrating LB's role in both availability and resilience.

> **Interview Tip:** Always mention the LB-as-SPOF pitfall — it shows you think critically. Discuss the **thundering herd** problem when instances recover and the **slow-start** mitigation. Mention health check types: TCP (fast, basic), HTTP (application-level), and custom (deep health including dependencies).

---

### 12. Explain the role of health checks in maintaining an available system.

**Type:** 📝 Question

**Health checks** are automated probes that continuously verify whether system components are **operational and capable of serving traffic**. They enable load balancers, orchestrators, and service meshes to **detect failures quickly** and route traffic only to healthy instances. Health checks operate at multiple depths: **liveness** (is it running?), **readiness** (can it accept traffic?), and **deep health** (are all dependencies OK?).

- **Liveness Check**: Confirms the process is running (TCP port open, HTTP 200 on /health)
- **Readiness Check**: Confirms the service can handle requests (DB connected, cache warm)
- **Deep Health Check**: Validates all dependencies (DB, cache, external APIs all reachable)
- **Startup Probe**: One-time check for slow-starting containers (prevents premature kill)
- **Shallow vs Deep**: Shallow checks (liveness) run frequently; deep checks run less often

```
+-----------------------------------------------------------+
|         HEALTH CHECK LAYERS                                |
+-----------------------------------------------------------+
|                                                             |
|  LIVENESS (is it alive?):                                  |
|  LB --> GET /healthz --> 200 OK                            |
|  Frequency: every 5s | Timeout: 2s | Threshold: 3 fails   |
|  Checks: process running, port open                        |
|                                                             |
|  READINESS (can it serve?):                                |
|  LB --> GET /ready --> 200 OK / 503 Not Ready              |
|  Frequency: every 10s | Timeout: 5s                        |
|  Checks: DB connection, cache connected, config loaded     |
|                                                             |
|  DEEP HEALTH (are dependencies OK?):                       |
|  Monitor --> GET /health/deep --> JSON status               |
|  Frequency: every 30s | Timeout: 10s                       |
|  Checks: DB query works, cache read/write, external APIs   |
|                                                             |
|  HEALTH CHECK FLOW:                                        |
|        Start                                               |
|          |                                                  |
|     [Startup Probe]                                         |
|          |                                                  |
|     Pass? --No--> Keep Waiting (don't kill yet)            |
|          |Yes                                               |
|     [Liveness Probe] --Fail x3--> Restart Container        |
|          |Pass                                              |
|     [Readiness Probe] --Fail--> Remove from LB Pool        |
|          |Pass                                              |
|     [Serving Traffic]                                       |
+-----------------------------------------------------------+
```

| Health Check Type | Frequency | Timeout | Failure Action | Checks |
|---|---|---|---|---|
| **Liveness** | 5-10s | 1-3s | Restart container/process | Port, process, basic HTTP |
| **Readiness** | 5-15s | 3-5s | Remove from LB pool | Dependencies connected |
| **Startup** | 5s (only at start) | 30-300s | Kill and restart | One-time initialization |
| **Deep Health** | 30-60s | 10-30s | Alert on-call team | Full dependency chain |
| **Synthetic** | 1-5 min | 30s | Page on-call | End-to-end user flow |

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"

@dataclass
class DependencyCheck:
    name: str
    check_fn: callable = None
    healthy: bool = True
    latency_ms: float = 0.0

@dataclass
class HealthChecker:
    service_name: str
    dependencies: list[DependencyCheck] = field(default_factory=list)
    consecutive_failures: int = 0
    failure_threshold: int = 3
    is_ready: bool = False

    def liveness_check(self) -> dict:
        """Basic check: is the process running?"""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    def readiness_check(self) -> dict:
        """Can this instance serve traffic?"""
        results = {}
        all_healthy = True
        for dep in self.dependencies:
            healthy = dep.healthy
            results[dep.name] = {
                "healthy": healthy,
                "latency_ms": dep.latency_ms
            }
            if not healthy:
                all_healthy = False

        self.is_ready = all_healthy
        return {
            "status": "ready" if all_healthy else "not_ready",
            "dependencies": results
        }

    def deep_health_check(self) -> dict:
        """Full dependency chain verification."""
        readiness = self.readiness_check()
        unhealthy = [
            name for name, info in readiness["dependencies"].items()
            if not info["healthy"]
        ]
        slow = [
            name for name, info in readiness["dependencies"].items()
            if info["latency_ms"] > 100
        ]

        if unhealthy:
            status = HealthStatus.UNHEALTHY
        elif slow:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return {
            "status": status.value,
            "service": self.service_name,
            "unhealthy_deps": unhealthy,
            "slow_deps": slow,
            "details": readiness["dependencies"]
        }

# Example
checker = HealthChecker("order-service")
checker.dependencies = [
    DependencyCheck("PostgreSQL", healthy=True, latency_ms=12),
    DependencyCheck("Redis Cache", healthy=True, latency_ms=2),
    DependencyCheck("Payment API", healthy=False, latency_ms=0),
    DependencyCheck("Inventory Service", healthy=True, latency_ms=150),
]

print("Liveness:", checker.liveness_check())
print("\nReadiness:", checker.readiness_check())
print("\nDeep Health:", checker.deep_health_check())
```

**AI/ML Application:** ML health checks include unique probes: **model loaded** (is the model in GPU memory?), **warm-up complete** (has the model processed test inputs?), and **prediction latency** (is inference within SLA?). Kubernetes readiness probes for ML pods verify that the model is loaded and warmed before routing traffic.

**Real-World Example:** Kubernetes uses **three probe types** (liveness, readiness, startup) that directly map to availability patterns. AWS ELB health checks run every 30 seconds by default with a 2-out-of-10 unhealthy threshold. Google Cloud Health Checks support gRPC probes — important for microservice architectures using Protocol Buffers.

> **Interview Tip:** Differentiate liveness from readiness — this is a common interview question on its own. Explain that a liveness failure restarts the pod (it's broken), while a readiness failure removes it from the pool (it's busy/initializing). Mention the cascading failure risk of overly aggressive deep health checks.

---

### 13. What is the purpose of a circuit breaker pattern in a distributed system?

**Type:** 📝 Question

The **circuit breaker pattern** prevents **cascading failures** in distributed systems by stopping requests to a failing downstream service, allowing it time to recover, and **failing fast** instead of making callers wait for timeouts. It operates in three states: **Closed** (normal — requests pass through), **Open** (tripped — requests fail immediately), and **Half-Open** (testing — limited requests test recovery).

- **Closed State**: All requests pass through; failures are counted against threshold
- **Open State**: Requests fail immediately with fallback response; no load on failing service
- **Half-Open State**: Limited probe requests test if service has recovered
- **Prevents**: Thread pool exhaustion, timeout cascades, resource starvation
- **Enables**: Fast failure response, graceful degradation, automatic recovery

```
+-----------------------------------------------------------+
|         CIRCUIT BREAKER STATE MACHINE                      |
+-----------------------------------------------------------+
|                                                             |
|                    success_threshold met                    |
|                  +------------------------+                 |
|                  |                        |                 |
|                  v                        |                 |
|  +----------+  requests  +------------+  |                 |
|  |  CLOSED  |----------->| HALF-OPEN  |--+                 |
|  |          |  fail >    |            |                     |
|  |  Normal  |  threshold | Test with  |                     |
|  |  traffic |            | limited    |                     |
|  +----------+            | requests   |                     |
|       ^                  +------------+                     |
|       |                       |                             |
|       | success_threshold     | failure during              |
|       | met                   | half-open                   |
|       |                       v                             |
|       |                  +----------+                       |
|       |                  |   OPEN   |                       |
|       +--[timeout]-------| Fail     |                       |
|                          | fast     |                       |
|                          +----------+                       |
|                                                             |
|  CASCADING FAILURE WITHOUT CIRCUIT BREAKER:                |
|  Service A --> Service B --> Service C (down)               |
|       |            |             X                          |
|       |            +-- timeout 30s (threads blocked)        |
|       +-- timeout 30s (threads blocked)                    |
|  All services fail due to thread exhaustion!               |
|                                                             |
|  WITH CIRCUIT BREAKER:                                     |
|  Service A --> Service B -X-> Service C (down)             |
|       |            |    CB OPEN: fail in 1ms               |
|       |            +-- returns fallback instantly           |
|       +-- gets degraded response (fast)                    |
|  Only Service C is down; A and B continue working          |
+-----------------------------------------------------------+
```

| State | Request Behavior | When Enters | When Exits | Duration |
|---|---|---|---|---|
| **Closed** | Pass through normally | Startup or recovery confirmed | Failure threshold exceeded | Default state |
| **Open** | Fail immediately (fallback) | Failures exceed threshold | Timeout period expires | 10-60 seconds typical |
| **Half-Open** | Limited test requests | Open timeout expires | Success or failure of tests | Until threshold met |

```python
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any, Optional

class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 5

@dataclass
class CircuitBreakerMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list = field(default_factory=list)

@dataclass
class CircuitBreaker:
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: State = State.CLOSED
    metrics: CircuitBreakerMetrics = field(default_factory=CircuitBreakerMetrics)
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0
    _half_open_calls: int = 0
    _fallback: Optional[Callable] = None

    def with_fallback(self, fallback_fn: Callable):
        self._fallback = fallback_fn
        return self

    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        self.metrics.total_calls += 1

        if self.state == State.OPEN:
            if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                self._transition_to(State.HALF_OPEN)
            else:
                self.metrics.rejected_calls += 1
                return self._execute_fallback(*args, **kwargs)

        if self.state == State.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self.metrics.rejected_calls += 1
                return self._execute_fallback(*args, **kwargs)
            self._half_open_calls += 1

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            return self._execute_fallback(*args, **kwargs)

    def _on_success(self):
        self.metrics.successful_calls += 1
        if self.state == State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(State.CLOSED)
        self._failure_count = 0

    def _on_failure(self):
        self.metrics.failed_calls += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.config.failure_threshold:
            self._transition_to(State.OPEN)

    def _transition_to(self, new_state: State):
        old = self.state
        self.state = new_state
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self.metrics.state_transitions.append(f"{old.value} -> {new_state.value}")

    def _execute_fallback(self, *args, **kwargs):
        if self._fallback:
            return self._fallback(*args, **kwargs)
        raise Exception(f"Circuit {self.name} is {self.state.value} - no fallback")

# Simulate degrading service
cb = CircuitBreaker("payment-service", CircuitBreakerConfig(failure_threshold=3, timeout_seconds=2))
cb.with_fallback(lambda: {"status": "fallback", "message": "Using cached response"})

def unreliable_service():
    if random.random() < 0.7:
        raise Exception("Service timeout")
    return {"status": "success"}

for i in range(15):
    result = cb.execute(unreliable_service)
    print(f"Call {i+1:2d}: state={cb.state.value:<10} result={result}")
    time.sleep(0.3)

print(f"\nMetrics: {cb.metrics.total_calls} total, {cb.metrics.successful_calls} success, "
      f"{cb.metrics.failed_calls} failed, {cb.metrics.rejected_calls} rejected")
print(f"Transitions: {cb.metrics.state_transitions}")
```

**AI/ML Application:** Circuit breakers protect ML pipelines from **slow model inference** — if a complex deep learning model exceeds latency SLAs, the circuit opens and routes to a **lightweight fallback model** (e.g., logistic regression or cached predictions). This prevents GPU saturation from cascading to upstream services.

**Real-World Example:** Netflix's Hystrix (now Resilience4j) implements circuit breakers across all microservice calls. When their recommendation engine is slow, the circuit opens and the UI falls back to **trending content**. Hystrix dashboard provides real-time circuit state visualization across hundreds of services.

> **Interview Tip:** Draw the three-state diagram (Closed → Open → Half-Open → Closed). Explain *why* the half-open state exists — it prevents the circuit from staying open forever and allows automatic recovery testing. Mention that circuit breakers require **fallback strategies** to be truly useful.

---

## Monitoring & Incident Response

### 14. What are some key indicators you would monitor to ensure system reliability ?

**Type:** 📝 Question

Key reliability indicators follow the **Four Golden Signals** framework (from Google SRE): **Latency**, **Traffic**, **Errors**, and **Saturation**. These are complemented by **USE metrics** (Utilization, Saturation, Errors) for infrastructure and **RED metrics** (Rate, Errors, Duration) for services. Together they provide comprehensive visibility into system health.

- **Latency**: Response time distribution (P50, P95, P99) — distinguish between successful and failed request latency
- **Traffic**: Request rate (RPS), throughput (bytes/sec), active connections
- **Errors**: Error rate (4xx, 5xx), error budget consumption, failed health checks
- **Saturation**: Resource usage approaching limits (CPU >80%, memory >90%, disk I/O, connection pool)
- **Availability**: Uptime percentage, successful requests / total requests
- **Durability**: Data loss events, replication lag, backup success rate

```
+-----------------------------------------------------------+
|         MONITORING INDICATOR FRAMEWORKS                    |
+-----------------------------------------------------------+
|                                                             |
|  FOUR GOLDEN SIGNALS (Google SRE):                         |
|  +-------------+  +-------------+                          |
|  | Latency     |  | Traffic     |                          |
|  | P50: 45ms   |  | 12,500 RPS  |                          |
|  | P99: 230ms  |  | 850 MB/s    |                          |
|  +-------------+  +-------------+                          |
|  +-------------+  +-------------+                          |
|  | Errors      |  | Saturation  |                          |
|  | 0.02% 5xx   |  | CPU: 65%    |                          |
|  | 0.5% 4xx    |  | Mem: 78%    |                          |
|  +-------------+  +-------------+                          |
|                                                             |
|  USE METHOD (Infrastructure):                              |
|  +--------+  +--------+  +--------+                        |
|  |Utiliz. |  |Satur.  |  |Errors  |                        |
|  |CPU 65% |  |Queue 12|  |Disk 0  |                        |
|  |Mem 78% |  |RunQ 3.2|  |Net 2   |                        |
|  +--------+  +--------+  +--------+                        |
|                                                             |
|  RED METHOD (Services):                                    |
|  +--------+  +--------+  +--------+                        |
|  | Rate   |  | Errors |  |Duration|                        |
|  |12.5K/s |  | 0.02%  |  |P99 23ms|                        |
|  +--------+  +--------+  +--------+                        |
|                                                             |
|  ALERT SEVERITY LEVELS:                                    |
|  P1 (Critical) --> Page immediately, all hands             |
|  P2 (High)     --> Page on-call engineer                   |
|  P3 (Medium)   --> Slack alert, fix in business hours      |
|  P4 (Low)      --> Ticket, fix in sprint                   |
+-----------------------------------------------------------+
```

| Indicator | What It Measures | Warning Threshold | Critical Threshold | Tool |
|---|---|---|---|---|
| **Latency P99** | Tail response time | >500ms | >2000ms | Prometheus/Grafana |
| **Error Rate** | Failed requests % | >0.1% | >1% | Datadog, New Relic |
| **CPU Utilization** | Compute saturation | >70% | >90% | CloudWatch, node_exporter |
| **Memory Usage** | Memory pressure | >80% | >95% | Prometheus |
| **Disk I/O** | Storage throughput | >70% capacity | >90% capacity | iostat, CloudWatch |
| **Replication Lag** | Data consistency | >1 second | >10 seconds | DB-specific metrics |
| **Error Budget** | SLO burn rate | >50% consumed | >80% consumed | SLO dashboards |

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class AlertSeverity(Enum):
    P1_CRITICAL = "P1"
    P2_HIGH = "P2"
    P3_MEDIUM = "P3"
    P4_LOW = "P4"

@dataclass
class MetricThreshold:
    name: str
    warning: float
    critical: float
    current: float
    unit: str = ""
    higher_is_worse: bool = True

    @property
    def severity(self) -> AlertSeverity:
        value = self.current
        if self.higher_is_worse:
            if value >= self.critical:
                return AlertSeverity.P1_CRITICAL
            elif value >= self.warning:
                return AlertSeverity.P2_HIGH
        else:
            if value <= self.critical:
                return AlertSeverity.P1_CRITICAL
            elif value <= self.warning:
                return AlertSeverity.P2_HIGH
        return AlertSeverity.P4_LOW

    @property
    def status(self) -> str:
        sev = self.severity
        if sev == AlertSeverity.P1_CRITICAL:
            return "CRITICAL"
        elif sev == AlertSeverity.P2_HIGH:
            return "WARNING"
        return "OK"

@dataclass
class ReliabilityDashboard:
    service_name: str
    metrics: list[MetricThreshold] = field(default_factory=list)

    def add_metric(self, name: str, warning: float, critical: float,
                   current: float, unit: str = "", higher_is_worse: bool = True):
        self.metrics.append(MetricThreshold(name, warning, critical, current, unit, higher_is_worse))

    def report(self):
        print(f"\n{'='*60}")
        print(f"Reliability Dashboard: {self.service_name}")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"{'='*60}")
        for m in self.metrics:
            status = m.status
            print(f"  [{status:>8}] {m.name:<25} {m.current:>8.2f}{m.unit} "
                  f"(warn: {m.warning}{m.unit}, crit: {m.critical}{m.unit})")

        alerts = [m for m in self.metrics if m.status != "OK"]
        print(f"\nActive Alerts: {len(alerts)}")
        for a in alerts:
            print(f"  {a.severity.value}: {a.name} = {a.current}{a.unit}")

dashboard = ReliabilityDashboard("order-service")
dashboard.add_metric("Latency P99", 500, 2000, 1250, "ms")
dashboard.add_metric("Error Rate", 0.1, 1.0, 0.05, "%")
dashboard.add_metric("CPU Usage", 70, 90, 82, "%")
dashboard.add_metric("Memory Usage", 80, 95, 76, "%")
dashboard.add_metric("Replication Lag", 1, 10, 0.3, "s")
dashboard.add_metric("Error Budget Remaining", 50, 20, 35, "%", higher_is_worse=False)
dashboard.report()
```

**AI/ML Application:** ML systems require additional indicators: **model prediction latency**, **feature freshness** (staleness of features in the store), **model accuracy drift** (comparing predictions to ground truth), and **GPU utilization/memory**. Prometheus with custom ML exporters tracks these alongside standard infrastructure metrics.

**Real-World Example:** Google SRE monitors the **Four Golden Signals** for all production services. Their internal tool **Borgmon** (now open-sourced as Prometheus) scrapes thousands of metrics endpoints every 15 seconds. They found that 80% of reliability issues are catchable by monitoring just latency P99 and error rate.

> **Interview Tip:** Name specific frameworks (Four Golden Signals, USE, RED) and explain when to use each. Mention that **percentiles** (P99, P95) are more useful than averages for latency because averages hide tail latency. Discuss **error budget burn rate** as a leading indicator of SLO violations.

---

### 15. How do you implement a monitoring system that accurately reflects system availability ?

**Type:** 📝 Question

An effective monitoring system combines **multi-layer data collection** (metrics, logs, traces), **user-centric SLIs**, **intelligent alerting** with severity tiers, and **real-time dashboards**. The key principle is measuring availability from the **user's perspective**, not just infrastructure health — a server at 100% CPU with all requests succeeding is available; a server at 10% CPU returning errors is not.

- **Metrics Pipeline**: Prometheus/Datadog collects time-series data at 15-60s intervals
- **Log Aggregation**: ELK/Splunk centralizes structured logs for analysis
- **Distributed Tracing**: Jaeger/Zipkin tracks requests across service boundaries
- **Synthetic Monitoring**: Automated scripts simulate user flows from external locations
- **Real User Monitoring (RUM)**: JavaScript agents measure actual user experience
- **SLO-Based Alerting**: Alert on error budget burn rate, not individual metric thresholds

```
+-----------------------------------------------------------+
|         MONITORING ARCHITECTURE                            |
+-----------------------------------------------------------+
|                                                             |
|  DATA SOURCES:                                             |
|  +--------+ +--------+ +--------+ +--------+              |
|  |Apps    | |Infra   | |Network | |External|              |
|  |metrics | |metrics | |metrics | |probes  |              |
|  +---+----+ +---+----+ +---+----+ +---+----+              |
|      |          |          |          |                     |
|      v          v          v          v                     |
|  +---------------------------------------------+          |
|  |  Collection Layer                            |          |
|  |  Prometheus | StatsD | FluentD | OTEL        |          |
|  +---------------------------------------------+          |
|               |                                            |
|               v                                            |
|  +---------------------------------------------+          |
|  |  Storage Layer                               |          |
|  |  +--------+ +--------+ +--------+           |          |
|  |  |Metrics | |Logs    | |Traces  |           |          |
|  |  |TSDB    | |Elastic | |Jaeger  |           |          |
|  |  +--------+ +--------+ +--------+           |          |
|  +---------------------------------------------+          |
|               |                                            |
|               v                                            |
|  +---------------------------------------------+          |
|  |  Analysis & Alerting                         |          |
|  |  +----------+ +----------+ +----------+     |          |
|  |  |SLO       | |Anomaly   | |Alert     |     |          |
|  |  |Dashboard | |Detection | |Manager   |     |          |
|  |  +----------+ +----------+ +----------+     |          |
|  +---------------------------------------------+          |
|               |                                            |
|               v                                            |
|  +---------------------------------------------+          |
|  |  Response                                    |          |
|  |  PagerDuty | Slack | Runbooks | Auto-Remed  |          |
|  +---------------------------------------------+          |
+-----------------------------------------------------------+
```

| Monitoring Layer | Tool Examples | Data Type | Retention | Resolution |
|---|---|---|---|---|
| **Metrics** | Prometheus, Datadog | Time-series numbers | 15-90 days | 15s-1min |
| **Logs** | ELK, Splunk, Loki | Structured text | 30-90 days | Per event |
| **Traces** | Jaeger, Zipkin, X-Ray | Request spans | 7-30 days | Per request |
| **Synthetic** | Pingdom, Catchpoint | Availability probes | 1+ year | 1-5 min |
| **RUM** | New Relic, Datadog RUM | User experience | 30-90 days | Per page load |
| **Profiling** | Pyroscope, Parca | CPU/memory profiles | 7-14 days | Continuous |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

@dataclass
class SLIWindow:
    """Sliding window SLI measurement."""
    window_minutes: int = 60
    _good_events: deque = field(default_factory=deque)
    _total_events: deque = field(default_factory=deque)

    def record(self, timestamp: datetime, is_good: bool):
        self._total_events.append(timestamp)
        if is_good:
            self._good_events.append(timestamp)
        self._prune(timestamp)

    def _prune(self, now: datetime):
        cutoff = now - timedelta(minutes=self.window_minutes)
        while self._total_events and self._total_events[0] < cutoff:
            self._total_events.popleft()
        while self._good_events and self._good_events[0] < cutoff:
            self._good_events.popleft()

    @property
    def availability(self) -> float:
        total = len(self._total_events)
        if total == 0:
            return 1.0
        return len(self._good_events) / total

@dataclass
class SLOMonitor:
    service_name: str
    slo_target: float  # e.g., 0.999
    window_days: int = 30
    sli_windows: dict = field(default_factory=dict)

    def add_sli(self, name: str, window_minutes: int = 60):
        self.sli_windows[name] = SLIWindow(window_minutes)

    def record_event(self, sli_name: str, is_good: bool, timestamp: datetime = None):
        ts = timestamp or datetime.now()
        if sli_name in self.sli_windows:
            self.sli_windows[sli_name].record(ts, is_good)

    @property
    def error_budget_total(self) -> float:
        return 1.0 - self.slo_target

    def error_budget_remaining(self, sli_name: str) -> float:
        if sli_name not in self.sli_windows:
            return self.error_budget_total
        current_error_rate = 1.0 - self.sli_windows[sli_name].availability
        return max(0, self.error_budget_total - current_error_rate)

    def burn_rate(self, sli_name: str) -> float:
        if sli_name not in self.sli_windows:
            return 0
        remaining = self.error_budget_remaining(sli_name)
        consumed = self.error_budget_total - remaining
        if self.error_budget_total == 0:
            return float('inf')
        return consumed / self.error_budget_total

    def report(self):
        print(f"\nSLO Monitor: {self.service_name} (target: {self.slo_target*100:.2f}%)")
        for name, window in self.sli_windows.items():
            avail = window.availability
            burn = self.burn_rate(name)
            remaining = self.error_budget_remaining(name)
            alert = " ALERT!" if burn > 0.5 else ""
            print(f"  {name}: {avail*100:.4f}% | burn rate: {burn:.2f} | "
                  f"budget left: {remaining*100:.4f}%{alert}")

# Simulate
monitor = SLOMonitor("api-gateway", slo_target=0.999)
monitor.add_sli("request_success", window_minutes=60)
monitor.add_sli("latency_p99_under_500ms", window_minutes=60)

import random
now = datetime.now()
for i in range(10000):
    ts = now - timedelta(seconds=random.randint(0, 3600))
    monitor.record_event("request_success", random.random() < 0.9985, ts)
    monitor.record_event("latency_p99_under_500ms", random.random() < 0.998, ts)

monitor.report()
```

**AI/ML Application:** ML monitoring adds **model observability** layers: prediction distribution monitoring (detect data drift), feature importance tracking, A/B test metric collection, and **model performance SLIs** (accuracy, recall, precision measured against delayed ground truth). Tools like **Evidently AI** and **WhyLabs** specialize in ML-specific monitoring.

**Real-World Example:** LinkedIn's monitoring system processes **2 trillion events/day**. They use a custom metrics pipeline (InGraphs) with SLO-based alerting. Instead of alerting on "CPU > 80%", they alert on "error budget burned 50% in the last hour" — this dramatically reduced alert fatigue while catching real issues faster.

> **Interview Tip:** Emphasize **SLO-based alerting** over threshold-based alerting — it's the modern approach. Mention the **three pillars of observability** (metrics, logs, traces) and explain that metrics tell you *what* is wrong, logs tell you *why*, and traces tell you *where* in the request path the problem occurs.

---

### 16. Discuss the importance of alerting and on-call rotations in maintaining system reliability . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Alerting** and **on-call rotations** form the human response layer of reliability engineering. Effective alerting ensures the **right person** gets the **right information** at the **right time** to resolve incidents before they impact users. Poor alerting leads to **alert fatigue** (too many false positives) or **missed incidents** (too few alerts), both of which degrade reliability.

- **Alert Tiers**: P1 (page immediately), P2 (page during hours), P3 (Slack/ticket), P4 (dashboard only)
- **On-Call Rotation**: Rotate primary/secondary every 1-2 weeks to prevent burnout
- **Escalation Policy**: Primary → Secondary → Team Lead → Director (with time-based escalation)
- **Alert Quality**: Signal-to-noise ratio, false positive rate, mean time to acknowledge
- **Runbooks**: Pre-written response procedures for each alert type
- **SLO-Based Alerts**: Alert on error budget burn rate, not raw metrics

```
+-----------------------------------------------------------+
|         ALERTING & ON-CALL ARCHITECTURE                    |
+-----------------------------------------------------------+
|                                                             |
|  ALERT FLOW:                                               |
|  Metrics --> Alert Rules --> Alert Manager --> Routing      |
|                                                    |        |
|                              +---------------------+        |
|                              |                              |
|                    +---------+---------+                    |
|                    |         |         |                    |
|                  P1/P2     P3       P4                     |
|                  PagerDuty  Slack    Dashboard              |
|                    |                                        |
|              +-----+------+                                |
|              |            |                                 |
|           Primary     Secondary                            |
|           On-Call     On-Call                               |
|              |            |                                 |
|           [5 min]     [15 min]                             |
|           no ack?     no ack?                              |
|              |            |                                 |
|           Escalate    Escalate                             |
|           to Team     to Director                          |
|           Lead                                             |
|                                                             |
|  ON-CALL ROTATION (2-week cycle):                          |
|  Week 1-2: Alice (primary), Bob (secondary)               |
|  Week 3-4: Bob (primary), Carol (secondary)               |
|  Week 5-6: Carol (primary), Alice (secondary)             |
|                                                             |
|  ALERT QUALITY METRICS:                                    |
|  Signal-to-Noise:  >80% actionable alerts                 |
|  MTTA (Acknowledge): <5 minutes for P1                    |
|  False Positive Rate: <10%                                 |
|  Pages per shift: <2 (sustainable on-call)                 |
+-----------------------------------------------------------+
```

| Alert Component | Purpose | Best Practice | Anti-Pattern |
|---|---|---|---|
| **Severity Levels** | Prioritize response | 4 tiers (P1-P4) | Everything is P1 |
| **Routing Rules** | Right person, right time | Service-based routing | Single email group |
| **Escalation** | Prevent missed incidents | Auto-escalate after timeout | No escalation path |
| **Runbooks** | Speed up resolution | Link in alert, keep updated | No documentation |
| **On-Call Schedule** | Fair distribution | 1-2 week rotations | Same person always |
| **Alert Suppression** | Reduce noise | Aggregate correlated alerts | No deduplication |
| **Burn Rate Alerts** | SLO-based detection | Multi-window burn rate | Raw metric thresholds |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class Severity(Enum):
    P1 = "critical"
    P2 = "high"
    P3 = "medium"
    P4 = "low"

@dataclass
class OnCallEngineer:
    name: str
    email: str
    phone: str
    role: str = "primary"

@dataclass
class OnCallSchedule:
    rotation_weeks: int = 2
    engineers: list[OnCallEngineer] = field(default_factory=list)
    _current_index: int = 0

    def current_primary(self) -> OnCallEngineer:
        return self.engineers[self._current_index % len(self.engineers)]

    def current_secondary(self) -> OnCallEngineer:
        return self.engineers[(self._current_index + 1) % len(self.engineers)]

    def rotate(self):
        self._current_index = (self._current_index + 1) % len(self.engineers)

@dataclass
class AlertRule:
    name: str
    severity: Severity
    condition: str
    runbook_url: str
    escalation_minutes: int = 5

@dataclass
class Alert:
    rule: AlertRule
    triggered_at: datetime
    acknowledged_at: datetime = None
    resolved_at: datetime = None
    acknowledged_by: str = ""
    is_false_positive: bool = False

    @property
    def time_to_ack_minutes(self) -> float:
        if self.acknowledged_at:
            return (self.acknowledged_at - self.triggered_at).total_seconds() / 60
        return float('inf')

    @property
    def time_to_resolve_minutes(self) -> float:
        if self.resolved_at:
            return (self.resolved_at - self.triggered_at).total_seconds() / 60
        return float('inf')

@dataclass
class AlertAnalytics:
    alerts: list[Alert] = field(default_factory=list)

    def add(self, alert: Alert):
        self.alerts.append(alert)

    def report(self):
        total = len(self.alerts)
        acked = [a for a in self.alerts if a.acknowledged_at]
        resolved = [a for a in self.alerts if a.resolved_at]
        false_pos = [a for a in self.alerts if a.is_false_positive]

        avg_ack = sum(a.time_to_ack_minutes for a in acked) / len(acked) if acked else 0
        avg_resolve = sum(a.time_to_resolve_minutes for a in resolved) / len(resolved) if resolved else 0

        print(f"Alert Analytics Report")
        print(f"  Total Alerts: {total}")
        print(f"  Acknowledged: {len(acked)} ({len(acked)/total*100:.0f}%)")
        print(f"  Resolved: {len(resolved)} ({len(resolved)/total*100:.0f}%)")
        print(f"  False Positives: {len(false_pos)} ({len(false_pos)/total*100:.0f}%)")
        print(f"  Avg MTTA: {avg_ack:.1f} minutes")
        print(f"  Avg MTTR: {avg_resolve:.1f} minutes")

        by_severity = {}
        for a in self.alerts:
            sev = a.rule.severity.value
            by_severity.setdefault(sev, 0)
            by_severity[sev] += 1
        print(f"  By Severity: {by_severity}")

# Demo
analytics = AlertAnalytics()
import random
now = datetime.now()
rules = [
    AlertRule("High Error Rate", Severity.P1, "error_rate > 1%", "https://runbook/errors", 5),
    AlertRule("Latency Spike", Severity.P2, "p99 > 2s", "https://runbook/latency", 10),
    AlertRule("Disk Usage", Severity.P3, "disk > 80%", "https://runbook/disk", 30),
]

for i in range(20):
    rule = random.choice(rules)
    triggered = now - timedelta(hours=random.randint(0, 168))
    alert = Alert(rule=rule, triggered_at=triggered)
    alert.acknowledged_at = triggered + timedelta(minutes=random.randint(1, 15))
    if random.random() < 0.8:
        alert.resolved_at = alert.acknowledged_at + timedelta(minutes=random.randint(5, 120))
    alert.is_false_positive = random.random() < 0.15
    analytics.add(alert)

analytics.report()
```

**AI/ML Application:** ML systems generate unique alerts: **model accuracy degradation** (drift detection triggers retraining), **feature pipeline delays** (stale features), and **GPU memory exhaustion** (OOM during inference batch). On-call ML engineers need specialized runbooks for model rollback vs infrastructure issues.

**Real-World Example:** PagerDuty reports that top-performing teams maintain a **signal-to-noise ratio >80%** and average **<2 pages per on-call shift**. Google SRE caps on-call at **25% of an engineer's time** and requires **50% of on-call time to be spent on project work**, not reactive firefighting.

> **Interview Tip:** Discuss alert fatigue as the primary failure mode of alerting systems. Mention that **SLO-based multi-window burn rate alerts** (Google SRE approach) dramatically reduce false positives compared to static thresholds. Show you understand the human side: rotation fairness, burnout prevention, and blameless culture.

---

### 17. What steps would you take to respond to an incident that reduces system availability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Incident response** follows a structured lifecycle: **Detection** → **Triage** → **Mitigation** → **Resolution** → **Post-Mortem**. The primary goal during an incident is to **restore service** (mitigate), not to find the root cause. Root cause analysis happens after service is restored. Effective incident management requires clear **roles**, **communication channels**, and **pre-defined runbooks**.

- **Incident Commander (IC)**: Coordinates response, makes decisions, communicates status
- **Technical Lead**: Diagnoses and implements fixes
- **Communications Lead**: Updates stakeholders, status page, customers
- **Scribe**: Documents timeline, actions taken, decisions made
- **Priority**: Mitigate first (restore service), investigate second (find root cause)

```
+-----------------------------------------------------------+
|         INCIDENT RESPONSE LIFECYCLE                        |
+-----------------------------------------------------------+
|                                                             |
|  Phase 1: DETECT (minimize MTTD)                          |
|  Alert fires --> On-call acknowledges --> Triage severity  |
|  [0-5 minutes]                                             |
|                                                             |
|  Phase 2: TRIAGE (classify & mobilize)                    |
|  Severity? --> P1: War room, all hands                     |
|            --> P2: IC + Tech Lead                           |
|            --> P3: On-call handles solo                     |
|  [5-15 minutes]                                            |
|                                                             |
|  Phase 3: MITIGATE (restore service ASAP)                 |
|  +---> Rollback deployment? (most common fix)              |
|  +---> Scale up resources?                                 |
|  +---> Enable feature flag (disable broken feature)?       |
|  +---> Failover to secondary region?                       |
|  +---> Apply known workaround from runbook?                |
|  [15-60 minutes]                                           |
|                                                             |
|  Phase 4: RESOLVE (fix root cause)                        |
|  Root cause identified --> Permanent fix deployed           |
|  --> Monitoring updated --> Runbook updated                 |
|  [Hours to days]                                           |
|                                                             |
|  Phase 5: POST-MORTEM (learn & improve)                   |
|  Timeline --> Root causes --> Action items --> Share        |
|  [Within 48 hours of incident]                             |
|                                                             |
|  COMMUNICATION TEMPLATE:                                   |
|  [Time] [Status] [Impact] [Next Update]                    |
|  "14:30 UTC - Investigating elevated error rates in        |
|   payment service. ~5% of transactions affected.           |
|   Next update in 15 minutes."                              |
+-----------------------------------------------------------+
```

| Phase | Goal | Key Activities | Typical Duration |
|---|---|---|---|
| **Detect** | Find the problem fast | Alert acknowledgment, initial assessment | 0-5 minutes |
| **Triage** | Classify severity | Determine impact, assign roles, open channel | 5-15 minutes |
| **Mitigate** | Restore service | Rollback, failover, scale, feature flag | 15-60 minutes |
| **Resolve** | Fix root cause | Debug, patch, test, deploy permanent fix | Hours-days |
| **Post-Mortem** | Prevent recurrence | Blameless review, action items, share learnings | Within 48 hours |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class IncidentSeverity(Enum):
    SEV1 = "Critical - service down for all users"
    SEV2 = "Major - significant degradation"
    SEV3 = "Minor - limited impact"
    SEV4 = "Low - cosmetic/minor issue"

class IncidentPhase(Enum):
    DETECTED = "detected"
    TRIAGING = "triaging"
    MITIGATING = "mitigating"
    RESOLVED = "resolved"
    POST_MORTEM = "post_mortem"

@dataclass
class TimelineEntry:
    timestamp: datetime
    phase: IncidentPhase
    action: str
    actor: str

@dataclass
class Incident:
    id: str
    title: str
    severity: IncidentSeverity
    detected_at: datetime
    phase: IncidentPhase = IncidentPhase.DETECTED
    timeline: list[TimelineEntry] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    root_cause: str = ""
    impact_summary: str = ""
    mitigated_at: datetime = None
    resolved_at: datetime = None

    def log_action(self, action: str, actor: str, phase: IncidentPhase = None):
        if phase:
            self.phase = phase
        self.timeline.append(TimelineEntry(datetime.now(), self.phase, action, actor))

    @property
    def mttm(self) -> timedelta:
        """Mean time to mitigate."""
        if self.mitigated_at:
            return self.mitigated_at - self.detected_at
        return timedelta(0)

    @property
    def mttr(self) -> timedelta:
        if self.resolved_at:
            return self.resolved_at - self.detected_at
        return timedelta(0)

    def generate_report(self) -> str:
        lines = [
            f"Incident Report: {self.id}",
            f"Title: {self.title}",
            f"Severity: {self.severity.value}",
            f"Impact: {self.impact_summary}",
            f"Root Cause: {self.root_cause}",
            f"Time to Mitigate: {self.mttm}",
            f"Time to Resolve: {self.mttr}",
            "",
            "Timeline:",
        ]
        for entry in self.timeline:
            lines.append(f"  [{entry.timestamp.strftime('%H:%M')}] [{entry.phase.value}] "
                        f"{entry.action} ({entry.actor})")
        lines.append("")
        lines.append("Action Items:")
        for i, item in enumerate(self.action_items, 1):
            lines.append(f"  {i}. {item}")
        return "\n".join(lines)

# Simulate an incident
inc = Incident(
    id="INC-2024-0142",
    title="Payment service returning 500 errors",
    severity=IncidentSeverity.SEV1,
    detected_at=datetime.now() - timedelta(hours=2)
)

inc.log_action("Alert triggered: payment error rate > 5%", "PagerDuty", IncidentPhase.DETECTED)
inc.log_action("On-call acknowledged, opened incident channel", "Alice", IncidentPhase.TRIAGING)
inc.log_action("Severity classified as SEV1, paging team lead", "Alice", IncidentPhase.TRIAGING)
inc.log_action("Identified recent deployment as potential cause", "Bob", IncidentPhase.MITIGATING)
inc.log_action("Rolled back deployment v2.4.1 -> v2.4.0", "Bob", IncidentPhase.MITIGATING)
inc.mitigated_at = inc.detected_at + timedelta(minutes=23)
inc.log_action("Error rate returned to normal, service restored", "Alice", IncidentPhase.RESOLVED)
inc.resolved_at = inc.detected_at + timedelta(minutes=35)

inc.root_cause = "Database connection pool exhaustion from missing query timeout in new endpoint"
inc.impact_summary = "~8% of payment transactions failed for 23 minutes"
inc.action_items = [
    "Add query timeout to all new database queries (mandatory code review check)",
    "Add connection pool utilization alert at 70% threshold",
    "Add integration test for connection pool under load",
    "Update deployment runbook with rollback criteria",
]

print(inc.generate_report())
```

**AI/ML Application:** ML incidents have unique response patterns: **model drift** requires retraining or rollback to a previous model version (not code rollback). **Feature pipeline failures** require fallback to cached features or default values. ML incident response runbooks must include **model-specific** rollback procedures separate from infrastructure runbooks.

**Real-World Example:** Atlassian's 2022 incident (14 days to restore some customers) demonstrated the importance of incident management at scale. Their post-mortem revealed gaps in automation for their deletion recovery process. In contrast, GitHub's 2018 incident (24 hours) was well-managed: they prioritized data integrity over speed, communicated transparently, and published a detailed post-mortem.

> **Interview Tip:** Emphasize "mitigate first, investigate later" — this shows operational maturity. Mention specific mitigation strategies: rollback (most common), feature flags, traffic shifting, and scaling. Discuss the **Incident Commander role** as the decision-maker who prevents chaos during major incidents.

---

### 18. How can post-mortem analysis improve future system reliability and availability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

A **blameless post-mortem** is a structured analysis conducted after an incident to determine **what happened**, **why it happened**, and **how to prevent recurrence**. The "blameless" aspect is critical — it focuses on **systemic failures** rather than individual blame, which encourages honest reporting and deeper root cause analysis. Post-mortems produce **concrete action items** that systematically improve reliability over time.

- **Blameless Culture**: Focus on systems, not individuals — people make mistakes; systems should prevent them
- **Root Cause Analysis**: Use **"5 Whys"** or **Ishikawa diagrams** to find underlying causes
- **Contributing Factors**: Multiple causes often combine — avoid single-cause narratives
- **Action Items**: Specific, assigned, trackable improvements with deadlines
- **Sharing**: Publish internally to spread learnings across teams
- **Follow-Up**: Track action item completion; unfinished items predict future incidents

```
+-----------------------------------------------------------+
|         POST-MORTEM PROCESS                                |
+-----------------------------------------------------------+
|                                                             |
|  TIMELINE:                                                 |
|  [Incident occurs] --> [Resolved] --> [Post-mortem]        |
|                                     (within 48 hours)      |
|                                                             |
|  POST-MORTEM STRUCTURE:                                    |
|  +---------------------------------------------------+     |
|  | 1. SUMMARY: What happened, impact, duration        |     |
|  | 2. TIMELINE: Minute-by-minute events               |     |
|  | 3. ROOT CAUSE: 5 Whys analysis                     |     |
|  | 4. IMPACT: Users affected, revenue lost             |     |
|  | 5. WHAT WENT WELL: Things that helped               |     |
|  | 6. WHAT WENT WRONG: Things that made it worse      |     |
|  | 7. ACTION ITEMS: Assigned, tracked, deadlined      |     |
|  +---------------------------------------------------+     |
|                                                             |
|  5 WHYS EXAMPLE:                                           |
|  Why 1: Service returned 500 errors                        |
|    --> Database connection pool exhausted                   |
|  Why 2: Why were connections exhausted?                    |
|    --> New query had no timeout, held connections           |
|  Why 3: Why no timeout?                                    |
|    --> Not in code review checklist                         |
|  Why 4: Why not in checklist?                              |
|    --> Previous incident action item not completed          |
|  Why 5: Why not completed?                                 |
|    --> No tracking system for post-mortem actions           |
|  ROOT CAUSE: Lack of action item tracking system           |
|                                                             |
|  ACTION ITEM TRACKING:                                     |
|  [Action] --> [Assigned] --> [Deadline] --> [Completed]    |
|  Items not completed predict FUTURE incidents              |
+-----------------------------------------------------------+
```

| Post-Mortem Element | Purpose | Key Questions | Output |
|---|---|---|---|
| **Summary** | Quick context | What happened? How long? | 2-3 sentence overview |
| **Timeline** | Sequence of events | When was each step taken? | Minute-by-minute log |
| **Root Cause** | Underlying failures | 5 Whys / Ishikawa | Systemic cause(s) |
| **Impact** | Quantify damage | Users affected? Revenue lost? | Metrics and numbers |
| **What Went Well** | Reinforce good practices | What helped resolve faster? | Practices to keep |
| **What Went Wrong** | Find improvement areas | What slowed recovery? | Systemic weaknesses |
| **Action Items** | Prevent recurrence | What concrete steps prevent this? | Assigned, deadlined tasks |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class ActionStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WONT_FIX = "wont_fix"

class ActionPriority(Enum):
    P1 = "must_fix_this_sprint"
    P2 = "fix_this_quarter"
    P3 = "nice_to_have"

@dataclass
class ActionItem:
    description: str
    assignee: str
    priority: ActionPriority
    deadline: datetime
    status: ActionStatus = ActionStatus.OPEN
    completed_at: datetime = None

@dataclass
class PostMortem:
    incident_id: str
    title: str
    date: datetime
    duration_minutes: int
    summary: str
    root_causes: list[str] = field(default_factory=list)
    five_whys: list[str] = field(default_factory=list)
    impact: dict = field(default_factory=dict)
    what_went_well: list[str] = field(default_factory=list)
    what_went_wrong: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)

    def completion_rate(self) -> float:
        if not self.action_items:
            return 1.0
        completed = sum(1 for a in self.action_items if a.status == ActionStatus.COMPLETED)
        return completed / len(self.action_items)

    def overdue_items(self) -> list[ActionItem]:
        now = datetime.now()
        return [a for a in self.action_items
                if a.status in (ActionStatus.OPEN, ActionStatus.IN_PROGRESS)
                and a.deadline < now]

    def report(self):
        print(f"Post-Mortem: {self.incident_id} - {self.title}")
        print(f"Date: {self.date.strftime('%Y-%m-%d')} | Duration: {self.duration_minutes} min")
        print(f"\nSummary: {self.summary}")
        print(f"\n5 Whys:")
        for i, why in enumerate(self.five_whys, 1):
            print(f"  Why {i}: {why}")
        print(f"\nRoot Causes: {', '.join(self.root_causes)}")
        print(f"\nImpact: {self.impact}")
        print(f"\nAction Items ({self.completion_rate()*100:.0f}% complete):")
        for a in self.action_items:
            overdue = " [OVERDUE]" if a in self.overdue_items() else ""
            print(f"  [{a.status.value}] {a.description} ({a.assignee}){overdue}")

pm = PostMortem(
    incident_id="INC-2024-0142",
    title="Payment Service 500 Errors",
    date=datetime.now() - timedelta(days=3),
    duration_minutes=23,
    summary="Database connection pool exhaustion caused 8% payment failures for 23 minutes.",
    root_causes=["Missing query timeout", "No connection pool monitoring"],
    five_whys=[
        "Payment service returned 500 errors",
        "Database connection pool was exhausted",
        "New query held connections without timeout",
        "Query timeout not in code review checklist",
        "Previous post-mortem action item not tracked to completion",
    ],
    impact={"users_affected": 12500, "transactions_failed": 1080, "revenue_impact": "$54,000"},
    what_went_well=["Alert fired within 2 minutes", "Rollback was fast (5 minutes)"],
    what_went_wrong=["Took 15 minutes to identify root cause", "No runbook for connection pool issues"],
)

pm.action_items = [
    ActionItem("Add query timeout to all DB queries", "Bob", ActionPriority.P1,
               datetime.now() - timedelta(days=1), ActionStatus.COMPLETED, datetime.now() - timedelta(days=2)),
    ActionItem("Add connection pool utilization alert", "Alice", ActionPriority.P1,
               datetime.now() + timedelta(days=4), ActionStatus.IN_PROGRESS),
    ActionItem("Create post-mortem action item tracker", "Carol", ActionPriority.P2,
               datetime.now() + timedelta(days=14), ActionStatus.OPEN),
]
pm.report()
```

**AI/ML Application:** ML post-mortems additionally analyze **model-specific failures**: Was the issue data drift, training data contamination, feature pipeline breakage, or GPU resource contention? Teams track **model incident patterns** separately from infrastructure incidents to identify systemic ML-specific failure modes.

**Real-World Example:** Google publishes a curated collection of post-mortems that have driven massive reliability improvements. Their post-mortem for the 2020 global authentication outage revealed that a **quota system change** cascaded to affect all Google services. The resulting action items led to redesigned quota management and cross-service dependency mapping.

> **Interview Tip:** Emphasize the word "blameless" and explain why it matters — engineers who fear blame will hide information, making root cause analysis impossible. Mention that **tracking action item completion rate** is critical — incomplete action items from previous post-mortems are a leading indicator of future incidents.

---

## Scaling & Performance

### 19. How does system scalability impact availability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Scalability** directly impacts availability because a system that cannot scale to meet demand will experience **resource exhaustion**, leading to degraded performance or complete outages. Conversely, **over-scaling** wastes resources and introduces unnecessary complexity. The relationship is bidirectional: poor scalability reduces availability under load, and scaling operations themselves can introduce availability risks if not handled gracefully.

- **Under-scaling**: Resource exhaustion → request queuing → timeouts → cascading failures
- **Over-scaling**: Wasted cost, increased complexity, more failure surface area
- **Scaling Speed**: Auto-scaling lag can cause temporary unavailability during traffic spikes
- **Stateful Scaling**: Databases and caches are harder to scale than stateless services
- **Scaling Failures**: Failed scaling operations (e.g., no available capacity) reduce availability
- **Data Consistency**: Scaling distributed databases risks temporary consistency issues

```
+-----------------------------------------------------------+
|         SCALABILITY-AVAILABILITY RELATIONSHIP              |
+-----------------------------------------------------------+
|                                                             |
|  DEMAND vs CAPACITY:                                       |
|                                                             |
|  Load                                                      |
|   ^                                                        |
|   |         /\    traffic spike                            |
|   |        /  \                                            |
|   |   ----/    \----  demand                               |
|   |  ===================  capacity (auto-scaled)           |
|   |  ============  capacity (fixed)                        |
|   +----------------------------------------> Time          |
|        ^         ^                                         |
|        OK      OUTAGE (fixed can't handle spike)           |
|                                                             |
|  AUTO-SCALING TIMELINE:                                    |
|  [Spike starts] --> [Metric crosses threshold]             |
|  --> [Cooldown period] --> [Scale-out initiated]           |
|  --> [Instance launching] --> [Health checks pass]         |
|  --> [Added to LB pool]                                    |
|  Total lag: 3-10 minutes (can miss short spikes!)          |
|                                                             |
|  SCALING STRATEGY BY TIER:                                 |
|  +-------------+------------------+-------------------+    |
|  | Tier        | Scaling Type     | Availability Risk |    |
|  |-------------|------------------|-------------------|    |
|  | Web/App     | Horizontal (easy)| Low (stateless)   |    |
|  | Cache       | Horizontal (mod) | Medium (resharding)|   |
|  | Database    | Vertical + read  | High (migrations) |    |
|  |             | replicas         |                   |    |
|  | Storage     | Horizontal (hard)| High (rebalancing)|    |
|  +-------------+------------------+-------------------+    |
+-----------------------------------------------------------+
```

| Scaling Approach | Availability Impact | Risk Level | Mitigation |
|---|---|---|---|
| **Horizontal (stateless)** | Positive — more capacity | Low | Health checks before serving |
| **Horizontal (stateful)** | Risk during resharding | Medium | Consistent hashing, migration |
| **Vertical** | Downtime during resize | High | Blue-green, live migration |
| **Auto-scaling out** | Lag during spike | Medium | Pre-warming, predictive scaling |
| **Auto-scaling in** | Capacity reduction | Low-Medium | Gradual drain, cooldown period |
| **Database scaling** | Migration risk | High | Online schema change, read replicas |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class ScalingAction(Enum):
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"

@dataclass
class AutoScalingPolicy:
    min_instances: int
    max_instances: int
    target_cpu: float = 70.0
    scale_out_cooldown_sec: int = 300
    scale_in_cooldown_sec: int = 600
    instance_warmup_sec: int = 120

@dataclass
class ScalingEvent:
    timestamp: datetime
    action: ScalingAction
    instances_before: int
    instances_after: int
    trigger_metric: str
    trigger_value: float
    success: bool = True

@dataclass
class AutoScaler:
    name: str
    policy: AutoScalingPolicy
    current_instances: int = 2
    events: list[ScalingEvent] = field(default_factory=list)
    _last_scale_out: datetime = None
    _last_scale_in: datetime = None

    def evaluate(self, cpu_avg: float, timestamp: datetime = None):
        ts = timestamp or datetime.now()

        if cpu_avg > self.policy.target_cpu and self.current_instances < self.policy.max_instances:
            if self._can_scale_out(ts):
                new_count = min(self.current_instances + 1, self.policy.max_instances)
                self._record_event(ts, ScalingAction.SCALE_OUT, new_count, "cpu_avg", cpu_avg)
                self.current_instances = new_count
                self._last_scale_out = ts

        elif cpu_avg < self.policy.target_cpu * 0.5 and self.current_instances > self.policy.min_instances:
            if self._can_scale_in(ts):
                new_count = max(self.current_instances - 1, self.policy.min_instances)
                self._record_event(ts, ScalingAction.SCALE_IN, new_count, "cpu_avg", cpu_avg)
                self.current_instances = new_count
                self._last_scale_in = ts

    def _can_scale_out(self, ts: datetime) -> bool:
        if not self._last_scale_out:
            return True
        return (ts - self._last_scale_out).total_seconds() > self.policy.scale_out_cooldown_sec

    def _can_scale_in(self, ts: datetime) -> bool:
        if not self._last_scale_in:
            return True
        return (ts - self._last_scale_in).total_seconds() > self.policy.scale_in_cooldown_sec

    def _record_event(self, ts, action, new_count, metric, value):
        self.events.append(ScalingEvent(ts, action, self.current_instances, new_count, metric, value))

    def report(self):
        print(f"AutoScaler: {self.name}")
        print(f"Current instances: {self.current_instances} (min={self.policy.min_instances}, max={self.policy.max_instances})")
        print(f"Scaling events: {len(self.events)}")
        for e in self.events[-5:]:
            print(f"  [{e.timestamp.strftime('%H:%M')}] {e.action.value}: "
                  f"{e.instances_before} -> {e.instances_after} (cpu={e.trigger_value:.1f}%)")

# Simulate traffic spike
import random
scaler = AutoScaler("web-tier", AutoScalingPolicy(min_instances=2, max_instances=10))
base_time = datetime.now()

for minute in range(60):
    ts = base_time + timedelta(minutes=minute)
    if 15 <= minute <= 30:
        cpu = random.uniform(75, 95)
    else:
        cpu = random.uniform(25, 50)
    scaler.evaluate(cpu, ts)

scaler.report()
```

**AI/ML Application:** ML inference services face unique scaling challenges: **GPU auto-scaling** is slow (minutes to provision GPU instances vs seconds for CPU). Teams use **predictive scaling** based on historical traffic patterns and pre-warm GPU instances before expected spikes. **Multi-model serving** (Triton) shares GPU memory across models to improve utilization.

**Real-World Example:** Twitter (X) scaled from handling 5,000 tweets/sec to 143,000+ during the 2014 World Cup semifinal. Their approach combined **pre-provisioned capacity** (predictive scaling days before), **cache warming**, and **graceful degradation** (timeline delivery prioritized over less-critical features like trending topics).

> **Interview Tip:** Discuss the **scaling lag problem** — auto-scaling takes minutes, but traffic spikes can be instantaneous. Mention mitigation strategies: predictive scaling, over-provisioning for known events, and graceful degradation during the gap. Distinguish horizontal (add instances) from vertical (bigger instance) scaling.

---

### 20. What strategies can be employed to scale a system while maintaining or improving reliability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

Scaling while maintaining reliability requires **stateless design**, **incremental rollouts**, **data partitioning**, and **capacity planning**. The key principle is that **scaling operations themselves should not reduce availability** — every scale event must be transparent to users. This means zero-downtime deployments, connection draining, and pre-tested scaling playbooks.

- **Stateless Services**: No server-side state → any instance handles any request → easy horizontal scaling
- **Connection Draining**: Complete in-flight requests before removing instances during scale-in
- **Blue-Green Scaling**: Run new capacity alongside old, shift traffic gradually
- **Database Read Replicas**: Scale reads horizontally without touching the write primary
- **Sharding**: Partition data across multiple nodes for write scalability
- **CDN/Edge Caching**: Offload static content to reduce origin load
- **Queue-Based Load Leveling**: Buffer spikes with message queues for async processing

```
+-----------------------------------------------------------+
|         RELIABLE SCALING STRATEGIES                        |
+-----------------------------------------------------------+
|                                                             |
|  STATELESS HORIZONTAL SCALING:                             |
|  +------+     +------+     +------+     +------+          |
|  |App-1 |     |App-2 |     |App-3 |     |App-4 |          |
|  +------+     +------+     +------+     +------+          |
|      \            |            |            /               |
|       \           |            |           /                |
|        +----------+----+-------+----------+                |
|                        |                                    |
|                  +----------+                               |
|                  | Shared   |                               |
|                  | State    |                               |
|                  | (Redis/  |                               |
|                  |  DB)     |                               |
|                  +----------+                               |
|                                                             |
|  QUEUE-BASED LOAD LEVELING:                                |
|  +-------+     +-------+     +----------+                  |
|  |Spike  | --> |Message| --> |Workers   |                  |
|  |Traffic|     |Queue  |     |(scalable)|                  |
|  |1000/s |     |buffer |     |process   |                  |
|  +-------+     +-------+     |at 100/s  |                  |
|                              +----------+                  |
|  Queue absorbs spikes; workers scale independently         |
|                                                             |
|  DATABASE SCALING PATTERN:                                 |
|  Writes --> [Primary DB]                                   |
|                |  sync                                      |
|                v                                            |
|  Reads  --> [Replica 1] [Replica 2] [Replica 3]           |
|  Scale reads by adding replicas (no primary impact)        |
+-----------------------------------------------------------+
```

| Strategy | Scales | Availability Impact | Complexity | Cost |
|---|---|---|---|---|
| **Horizontal + Stateless** | Compute | Positive (more capacity) | Low | Linear |
| **Read Replicas** | Read throughput | Positive (offload primary) | Medium | Linear |
| **Sharding** | Write throughput | Risk during resharding | High | Sub-linear |
| **CDN/Edge** | Static content | Positive (reduce origin) | Low | Per-request |
| **Queue Leveling** | Async workload | Positive (absorb spikes) | Medium | Per-message |
| **Connection Pooling** | DB connections | Reduces connection churn | Low | Minimal |
| **Caching** | Read throughput | Risk of stale data | Medium | Memory cost |

```python
from dataclasses import dataclass, field
from enum import Enum

class ScalingDimension(Enum):
    COMPUTE = "compute"
    READ_THROUGHPUT = "read_throughput"
    WRITE_THROUGHPUT = "write_throughput"
    STORAGE = "storage"
    NETWORK = "network"

@dataclass
class ScalingStrategy:
    name: str
    dimension: ScalingDimension
    max_scale_factor: float
    availability_impact: str
    zero_downtime: bool
    complexity: str

@dataclass
class CapacityPlanner:
    strategies: list[ScalingStrategy] = field(default_factory=list)
    current_load: dict = field(default_factory=dict)
    max_capacity: dict = field(default_factory=dict)

    def add_strategy(self, strategy: ScalingStrategy):
        self.strategies.append(strategy)

    def set_current(self, dimension: str, load: float, capacity: float):
        self.current_load[dimension] = load
        self.max_capacity[dimension] = capacity

    def headroom(self, dimension: str) -> float:
        if dimension not in self.current_load:
            return 1.0
        return 1 - (self.current_load[dimension] / self.max_capacity[dimension])

    def plan(self):
        print("Capacity Planning Report")
        print("=" * 60)
        for dim in self.current_load:
            load = self.current_load[dim]
            cap = self.max_capacity[dim]
            hr = self.headroom(dim)
            status = "OK" if hr > 0.3 else "WARNING" if hr > 0.1 else "CRITICAL"
            print(f"  [{status:>8}] {dim:<20} {load:.0f}/{cap:.0f} ({hr*100:.0f}% headroom)")

        print("\nRecommended Strategies:")
        for dim in self.current_load:
            if self.headroom(dim) < 0.3:
                applicable = [s for s in self.strategies if s.dimension.value == dim]
                for s in applicable:
                    print(f"  {dim}: {s.name} (zero-downtime: {s.zero_downtime}, complexity: {s.complexity})")

planner = CapacityPlanner()
planner.add_strategy(ScalingStrategy("Horizontal Auto-Scaling", ScalingDimension.COMPUTE, 10, "positive", True, "low"))
planner.add_strategy(ScalingStrategy("Read Replicas", ScalingDimension.READ_THROUGHPUT, 5, "positive", True, "medium"))
planner.add_strategy(ScalingStrategy("Sharding", ScalingDimension.WRITE_THROUGHPUT, 100, "risk during migration", True, "high"))
planner.add_strategy(ScalingStrategy("CDN Offload", ScalingDimension.NETWORK, 50, "positive", True, "low"))

planner.set_current("compute", 750, 1000)
planner.set_current("read_throughput", 8500, 10000)
planner.set_current("write_throughput", 450, 2000)
planner.set_current("network", 850, 1000)
planner.set_current("storage", 700, 1000)

planner.plan()
```

**AI/ML Application:** ML scaling strategies include **model distillation** (smaller model for high-traffic endpoints), **batched inference** (accumulate requests for efficient GPU utilization), and **multi-tier serving** (simple model for 90% of requests, complex model only for edge cases). Feature stores scale through **read replicas** and **pre-computed feature snapshots**.

**Real-World Example:** Slack scales to handle **3.5 billion messages per day** using a combination of MySQL sharding (by workspace), Redis caching (hot channel data), and edge CDN (static assets). Their reliability comes from **connection draining** during deployments and **queue-based load leveling** for search indexing.

> **Interview Tip:** Focus on the principle that scaling operations must be **transparent to users**. Mention connection draining, health check warm-up periods, and gradual traffic shifting. Discuss **capacity planning** as a proactive approach — don't wait for autoscaling to react.

---

### 21. Describe how caching can affect system reliability and what are some trade-offs. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

Caching **improves reliability** by reducing load on backend services and providing **fallback data** when origins are down. However, caching introduces **data staleness**, **cache stampede** risks, **memory pressure**, and **cold start vulnerabilities**. The trade-off is fundamentally between **speed/availability** and **data freshness/consistency**.

- **Positive**: Reduces database load, provides stale-while-revalidate fallback, absorbs traffic spikes
- **Negative**: Stale data, cache stampede, cold start vulnerability, memory exhaustion
- **Cache Stampede**: Cache expires → all requests hit backend simultaneously → overload
- **Cache Poisoning**: Bad data cached → all users receive incorrect responses
- **Cache Warming**: Pre-populate cache to avoid cold start after deployment/restart
- **Consistency**: Write-through for consistency vs write-back for performance

```
+-----------------------------------------------------------+
|         CACHING RELIABILITY TRADE-OFFS                     |
+-----------------------------------------------------------+
|                                                             |
|  POSITIVE IMPACT (availability):                           |
|  Client --> [Cache HIT] --> response in 1ms                |
|                             (backend not touched)           |
|  Backend down? Serve stale cache! (degraded but alive)     |
|                                                             |
|  NEGATIVE: CACHE STAMPEDE:                                 |
|  Cache key expires at T=0                                  |
|  T=0.001: Request 1 --> cache MISS --> query DB            |
|  T=0.002: Request 2 --> cache MISS --> query DB            |
|  T=0.003: Request 3 --> cache MISS --> query DB            |
|  ...100 requests all hit DB simultaneously!                |
|  Fix: Cache lock (only 1 request refreshes)                |
|                                                             |
|  NEGATIVE: COLD START:                                     |
|  Deploy new instance --> empty cache                       |
|  All requests are cache MISS --> backend overwhelmed       |
|  Fix: Cache warming before adding to LB pool               |
|                                                             |
|  CONSISTENCY SPECTRUM:                                     |
|  +----------+----------+----------+----------+             |
|  | Write-   | Write-   | Write-   | Read-    |             |
|  | Through  | Behind   | Around   | Through  |             |
|  +----------+----------+----------+----------+             |
|  |Consistent|Fast write|No cache  |Lazy load |             |
|  |Slow write|Risk loss |pollution |Miss cost |             |
|  +----------+----------+----------+----------+             |
+-----------------------------------------------------------+
```

| Cache Pattern | Consistency | Performance | Reliability Risk | Use Case |
|---|---|---|---|---|
| **Write-Through** | Strong | Slow writes | Low (always in sync) | Account data |
| **Write-Behind** | Eventual | Fast writes | Data loss on crash | Analytics events |
| **Write-Around** | Eventual | Good for write-heavy | Cache misses on reads | Log data |
| **Read-Through** | Lazy | Fast reads after first | Cold start | Product catalog |
| **Cache-Aside** | Application-managed | Flexible | Stampede risk | General purpose |
| **Stale-While-Revalidate** | Eventual | Always fast | Stale data | Content delivery |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import threading
import time

@dataclass
class CacheEntry:
    key: str
    value: str
    created_at: datetime
    ttl_seconds: int
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)

@dataclass
class ReliableCache:
    max_size: int = 1000
    default_ttl: int = 300
    _store: dict = field(default_factory=dict)
    _locks: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {
        "hits": 0, "misses": 0, "stampedes_prevented": 0,
        "stale_served": 0, "evictions": 0
    })

    def get(self, key: str) -> tuple:
        """Returns (value, hit_type) where hit_type is 'hit', 'stale', or 'miss'."""
        if key in self._store:
            entry = self._store[key]
            entry.access_count += 1
            if not entry.is_expired:
                self.stats["hits"] += 1
                return entry.value, "hit"
            else:
                self.stats["stale_served"] += 1
                return entry.value, "stale"  # Serve stale while refreshing
        self.stats["misses"] += 1
        return None, "miss"

    def set(self, key: str, value: str, ttl: int = None):
        if len(self._store) >= self.max_size:
            self._evict_lru()
        self._store[key] = CacheEntry(key, value, datetime.now(), ttl or self.default_ttl)

    def get_or_fetch(self, key: str, fetch_fn, ttl: int = None):
        """Cache-aside with stampede protection."""
        value, hit_type = self.get(key)
        if hit_type == "hit":
            return value

        # Stampede protection: only one thread refreshes
        lock_key = f"lock:{key}"
        if lock_key not in self._locks:
            self._locks[lock_key] = threading.Lock()

        acquired = self._locks[lock_key].acquire(blocking=False)
        if acquired:
            try:
                fresh_value = fetch_fn()
                self.set(key, fresh_value, ttl)
                return fresh_value
            finally:
                self._locks[lock_key].release()
        else:
            self.stats["stampedes_prevented"] += 1
            return value  # Return stale while another thread refreshes

    def _evict_lru(self):
        if not self._store:
            return
        lru_key = min(self._store, key=lambda k: self._store[k].access_count)
        del self._store[lru_key]
        self.stats["evictions"] += 1

    def report(self):
        total = self.stats["hits"] + self.stats["misses"] + self.stats["stale_served"]
        hit_rate = self.stats["hits"] / total * 100 if total > 0 else 0
        print(f"Cache Report ({len(self._store)} entries)")
        print(f"  Hit Rate: {hit_rate:.1f}%")
        print(f"  Hits: {self.stats['hits']}, Misses: {self.stats['misses']}, Stale: {self.stats['stale_served']}")
        print(f"  Stampedes Prevented: {self.stats['stampedes_prevented']}")
        print(f"  Evictions: {self.stats['evictions']}")

# Simulate
cache = ReliableCache(max_size=50, default_ttl=10)
for _ in range(1000):
    key = f"product:{random.randint(1, 100)}"
    cache.get_or_fetch(key, lambda: f"data_{random.randint(1, 999)}")
cache.report()
```

**AI/ML Application:** ML systems cache **model predictions** for identical inputs (prediction caching), **feature vectors** (feature store caching), and **model artifacts** (model registry caching). Cache invalidation for ML is complex: a retrained model invalidates all prediction caches, but stale predictions may be acceptable during the transition period.

**Real-World Example:** Facebook's **Memcached** deployment caches billions of objects across multiple data centers. During demand spikes, they serve stale cached data rather than overloading backends — users see slightly outdated content rather than errors. They invented **mcrouter** for distributed caching with automatic failover across regions.

> **Interview Tip:** Focus on the **cache stampede** problem — it's a frequent interview topic. Explain the lock-based solution (only one request refreshes) and **stale-while-revalidate** (serve old data while background refresh happens). Mention cache warming as essential for deployment reliability.

---

### 22. Explain the role of rate limiting in preserving system availability . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Rate limiting** preserves availability by **preventing any single client or traffic pattern from overwhelming system resources**. It acts as a safety valve that ensures fair resource allocation, protects against DDoS attacks, prevents cascading failures from runaway clients, and maintains service quality for all users even when some exceed expected usage patterns.

- **Token Bucket**: Allows bursts while enforcing average rate — most popular algorithm
- **Sliding Window**: Smooth rate enforcement without fixed window edge effects
- **Fixed Window**: Simple but allows 2x burst at window boundaries
- **Leaky Bucket**: Smooth output rate, queue excess requests
- **Adaptive Rate Limiting**: Adjust limits based on current system load
- **Tiered Limits**: Different rates for different user tiers or API keys

```
+-----------------------------------------------------------+
|         RATE LIMITING FOR AVAILABILITY                     |
+-----------------------------------------------------------+
|                                                             |
|  TOKEN BUCKET ALGORITHM:                                   |
|  Bucket capacity: 10 tokens                                |
|  Refill rate: 2 tokens/second                              |
|                                                             |
|  [Bucket: 10 tokens]                                       |
|  Request 1: consume 1 token --> [9 tokens] --> ALLOWED     |
|  Request 2: consume 1 token --> [8 tokens] --> ALLOWED     |
|  ...burst of 10 requests...                                |
|  Request 10: consume 1 --> [0 tokens] --> ALLOWED          |
|  Request 11: no tokens --> 429 Too Many Requests           |
|  ...wait 0.5s... (1 token refilled)                        |
|  Request 12: consume 1 --> [0 tokens] --> ALLOWED          |
|                                                             |
|  RATE LIMITING LAYERS:                                     |
|  +--------+  +--------+  +--------+  +--------+           |
|  |Edge/CDN|  |API GW  |  |Service |  |Database|           |
|  |DDoS    |  |Per-API |  |Per-user|  |Per-query|          |
|  |protect |  |limits  |  |limits  |  |limits  |           |
|  +--------+  +--------+  +--------+  +--------+           |
|  Global       Per-endpoint  Per-tenant   Per-resource      |
|                                                             |
|  RESPONSE HEADERS:                                         |
|  X-RateLimit-Limit: 100                                    |
|  X-RateLimit-Remaining: 23                                 |
|  X-RateLimit-Reset: 1640995200                             |
|  Retry-After: 30                                           |
+-----------------------------------------------------------+
```

| Algorithm | Burst Handling | Smoothness | Memory | Complexity | Use Case |
|---|---|---|---|---|---|
| **Token Bucket** | Allows bursts | Good | Low (counter + timestamp) | Low | API rate limiting |
| **Leaky Bucket** | Queues bursts | Very smooth | Queue storage | Medium | Traffic shaping |
| **Fixed Window** | 2x edge burst | Poor | Low (counter) | Very low | Simple limits |
| **Sliding Window Log** | No edge burst | Good | High (per-request log) | High | Precise limiting |
| **Sliding Window Counter** | Approximate | Good | Low (2 counters) | Low | Most production use |

```python
import time
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = None
    last_refill: float = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = float(self.capacity)
        if self.last_refill is None:
            self.last_refill = time.time()

    def allow_request(self, tokens: int = 1) -> bool:
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

@dataclass
class SlidingWindowCounter:
    window_seconds: int
    max_requests: int
    _current_count: int = 0
    _previous_count: int = 0
    _window_start: float = 0.0

    def __post_init__(self):
        self._window_start = time.time()

    def allow_request(self) -> bool:
        now = time.time()
        elapsed = now - self._window_start

        if elapsed >= self.window_seconds:
            windows_passed = int(elapsed / self.window_seconds)
            if windows_passed >= 2:
                self._previous_count = 0
            else:
                self._previous_count = self._current_count
            self._current_count = 0
            self._window_start += windows_passed * self.window_seconds
            elapsed = now - self._window_start

        weight = 1 - (elapsed / self.window_seconds)
        estimated = self._previous_count * weight + self._current_count

        if estimated < self.max_requests:
            self._current_count += 1
            return True
        return False

@dataclass
class RateLimiter:
    default_limit: int = 100
    window_seconds: int = 60
    _buckets: dict = field(default_factory=dict)
    _stats: dict = field(default_factory=lambda: defaultdict(lambda: {"allowed": 0, "rejected": 0}))

    def check(self, client_id: str, cost: int = 1) -> bool:
        if client_id not in self._buckets:
            self._buckets[client_id] = TokenBucket(
                capacity=self.default_limit,
                refill_rate=self.default_limit / self.window_seconds
            )

        allowed = self._buckets[client_id].allow_request(cost)
        if allowed:
            self._stats[client_id]["allowed"] += 1
        else:
            self._stats[client_id]["rejected"] += 1
        return allowed

    def report(self):
        print("Rate Limiter Report")
        for client, stats in self._stats.items():
            total = stats["allowed"] + stats["rejected"]
            reject_rate = stats["rejected"] / total * 100 if total > 0 else 0
            print(f"  {client}: {stats['allowed']} allowed, {stats['rejected']} rejected ({reject_rate:.1f}%)")

limiter = RateLimiter(default_limit=10, window_seconds=1)

# Simulate: normal client vs abusive client
for _ in range(20):
    limiter.check("normal-client")
for _ in range(100):
    limiter.check("abusive-client")

limiter.report()
```

**AI/ML Application:** ML API endpoints require **cost-aware rate limiting** — a request that triggers a large language model costs significantly more than a simple classification. Rate limits should be based on **computational cost** (GPU-seconds) rather than just request count. Token-based pricing (like OpenAI's API) is effectively rate limiting by cost.

**Real-World Example:** GitHub's API rate limits are **5,000 requests/hour** for authenticated users and **60/hour** for unauthenticated. Stripe's API returns `429 Too Many Requests` with `Retry-After` header when limits are exceeded. Cloudflare's rate limiting protects millions of websites from DDoS attacks using distributed token bucket algorithms.

> **Interview Tip:** Discuss **where** to place rate limiters (edge/CDN, API gateway, service-level, database-level) and explain that multiple layers provide defense-in-depth. Mention the token bucket algorithm specifically and explain why it's preferred (allows bursts while enforcing average rate).

---

## Reliability in Distributed Systems

### 23. How do eventual consistency and strong consistency differ and what are the reliability implications? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Strong consistency** guarantees that all reads return the **most recent write** across all nodes, while **eventual consistency** guarantees that all replicas will **converge to the same state** given sufficient time without new writes. The reliability implications are profound: strong consistency sacrifices **availability during network partitions** (per CAP theorem), while eventual consistency sacrifices **correctness guarantees** for higher availability.

- **Strong Consistency**: All nodes see the same data at the same time; requires coordination
- **Eventual Consistency**: Replicas may return stale data temporarily; converges over time
- **Causal Consistency**: Preserves cause-effect ordering without full strong consistency
- **Read-Your-Writes**: Client always sees their own latest write (session consistency)
- **CAP Theorem Impact**: During partition, choose Consistency (CP) or Availability (AP)
- **Latency Impact**: Strong consistency adds coordination latency (consensus round-trips)

```
+-----------------------------------------------------------+
|         CONSISTENCY MODELS & RELIABILITY                   |
+-----------------------------------------------------------+
|                                                             |
|  STRONG CONSISTENCY:                                       |
|  Client writes X=5 to Node A                              |
|  Node A --> sync replicate to B, C                         |
|  All nodes: X=5 (immediately visible everywhere)           |
|                                                             |
|  Write --> [A: X=5] --sync--> [B: X=5] [C: X=5]          |
|  Read from ANY node --> X=5 (guaranteed)                   |
|  Partition? --> REFUSE writes (unavailable but consistent) |
|                                                             |
|  EVENTUAL CONSISTENCY:                                     |
|  Client writes X=5 to Node A                              |
|  Node A --> async replicate to B, C (eventually)           |
|                                                             |
|  Write --> [A: X=5] --async--> [B: X=3] [C: X=3]         |
|  Read from B --> X=3 (stale!)                              |
|  ... seconds later ...                                     |
|  [A: X=5] [B: X=5] [C: X=5] (converged)                  |
|  Partition? --> ACCEPT writes (available but inconsistent) |
|                                                             |
|  CONSISTENCY SPECTRUM:                                     |
|  Strong --> Linearizable --> Sequential --> Causal          |
|  --> Read-your-writes --> Monotonic --> Eventual            |
|  <-- More consistent          More available -->           |
+-----------------------------------------------------------+
```

| Model | Guarantee | Availability | Latency | Use Case |
|---|---|---|---|---|
| **Linearizable** | Real-time ordering, latest value | Low during partitions | High (consensus) | Financial ledger |
| **Sequential** | Total ordering across all ops | Medium | Medium | Distributed locks |
| **Causal** | Cause-effect ordering preserved | High | Low-Medium | Social media feeds |
| **Read-Your-Writes** | See your own writes | High | Low | User profiles |
| **Eventual** | Converges eventually | Very High | Very Low | DNS, shopping carts |

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import time

class ConsistencyModel(Enum):
    STRONG = "strong"
    EVENTUAL = "eventual"
    CAUSAL = "causal"

@dataclass
class ReplicaNode:
    name: str
    data: dict = field(default_factory=dict)
    version_clock: dict = field(default_factory=dict)
    is_partitioned: bool = False

    def write(self, key: str, value: str, version: int):
        self.data[key] = value
        self.version_clock[key] = version

    def read(self, key: str) -> tuple:
        return self.data.get(key), self.version_clock.get(key, 0)

@dataclass
class DistributedStore:
    model: ConsistencyModel
    nodes: list[ReplicaNode] = field(default_factory=list)
    _global_version: int = 0

    def write(self, key: str, value: str) -> dict:
        self._global_version += 1
        version = self._global_version
        primary = self.nodes[0]
        primary.write(key, value, version)

        result = {"status": "success", "node": primary.name, "version": version}

        if self.model == ConsistencyModel.STRONG:
            acks = 1
            for node in self.nodes[1:]:
                if not node.is_partitioned:
                    node.write(key, value, version)
                    acks += 1
            quorum = len(self.nodes) // 2 + 1
            if acks < quorum:
                result["status"] = "failed"
                result["reason"] = f"Only {acks}/{quorum} acks (partition)"

        elif self.model == ConsistencyModel.EVENTUAL:
            for node in self.nodes[1:]:
                if not node.is_partitioned:
                    if random.random() < 0.7:  # Simulate async lag
                        node.write(key, value, version)

        return result

    def read(self, key: str, node_index: int = None) -> dict:
        node = self.nodes[node_index] if node_index is not None else random.choice(self.nodes)
        value, version = node.read(key)
        return {"value": value, "version": version, "node": node.name}

    def check_consistency(self, key: str) -> bool:
        values = set()
        for node in self.nodes:
            v, _ = node.read(key)
            values.add(v)
        return len(values) <= 1

# Demo: Strong vs Eventual during partition
for model in [ConsistencyModel.STRONG, ConsistencyModel.EVENTUAL]:
    store = DistributedStore(model=model)
    store.nodes = [ReplicaNode(f"node-{i}") for i in range(3)]
    store.nodes[2].is_partitioned = True  # Simulate partition

    result = store.write("balance", "$1000")
    consistent = store.check_consistency("balance")
    reads = [store.read("balance", i) for i in range(3)]

    print(f"\n{model.value.upper()} CONSISTENCY (with partition):")
    print(f"  Write result: {result['status']}")
    print(f"  Reads: {[r['value'] for r in reads]}")
    print(f"  Consistent: {consistent}")
```

**AI/ML Application:** ML feature stores face a consistency choice: **strong consistency** ensures all predictions use identical features (critical for financial models), while **eventual consistency** allows faster feature serving with slightly stale features (acceptable for recommendation engines). Most ML systems use **read-your-writes** consistency for training data.

**Real-World Example:** Amazon DynamoDB defaults to **eventual consistency** (faster, cheaper reads) but offers **strongly consistent reads** at 2x the cost. Their shopping cart was famously designed for eventual consistency — two conflicting cart updates are merged rather than rejected, prioritizing availability over consistency during checkout.

> **Interview Tip:** Reference the CAP theorem: during a network partition, you must choose consistency or availability. Explain that most modern systems choose **tunable consistency** (like Cassandra's quorum reads) rather than a binary choice. The key insight is that different data types warrant different consistency models.

---

### 24. Describe the CAP theorem and its relevance to system availability . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

The **CAP theorem** (Brewer's theorem) states that a distributed system can guarantee at most **two of three properties**: **Consistency** (all nodes see the same data), **Availability** (every request gets a response), and **Partition tolerance** (system continues despite network failures). Since network partitions are **inevitable** in distributed systems, the real choice is between **CP** (consistent but may be unavailable) and **AP** (available but may be inconsistent) during partitions.

- **C (Consistency)**: Every read receives the most recent write or an error
- **A (Availability)**: Every request receives a non-error response (may be stale)
- **P (Partition Tolerance)**: System operates despite network message loss between nodes
- **CP Systems**: Refuse requests during partition to maintain consistency (ZooKeeper, HBase)
- **AP Systems**: Serve requests during partition, resolve conflicts later (Cassandra, DynamoDB)
- **CA**: Only possible in single-node systems (no partition = no distributed system)

```
+-----------------------------------------------------------+
|         CAP THEOREM                                        |
+-----------------------------------------------------------+
|                                                             |
|              Consistency                                    |
|                 /\                                          |
|                /  \                                         |
|               / CP \                                       |
|              / Zone \                                      |
|             /--------\                                     |
|            / HBase    \                                    |
|           / ZooKeeper  \                                   |
|          / MongoDB(def) \                                  |
|         /                \                                 |
|        /    CA            \                                |
|       /   (single node)   \    AP Zone                    |
|      /   RDBMS, no replic  \   Cassandra                  |
|     /                       \  DynamoDB                    |
|    /                         \ CouchDB                     |
|   /___________________________\                            |
|  Availability     Partition Tolerance                      |
|                                                             |
|  DURING NORMAL OPERATION:                                  |
|  All three properties hold! CAP only matters during        |
|  partitions.                                               |
|                                                             |
|  DURING PARTITION:                                         |
|  CP: [Node A] X partition X [Node B]                       |
|      Node B refuses reads/writes --> UNAVAILABLE           |
|      But data is consistent when partition heals           |
|                                                             |
|  AP: [Node A] X partition X [Node B]                       |
|      Both nodes serve requests --> AVAILABLE               |
|      But data may diverge --> INCONSISTENT                 |
|      Conflict resolution needed when healed                |
+-----------------------------------------------------------+
```

| System | CAP Choice | During Partition Behavior | Conflict Resolution |
|---|---|---|---|
| **ZooKeeper** | CP | Leader unavailable, reads from followers | Leader election |
| **HBase** | CP | Regions unavailable without master | Master failover |
| **Cassandra** | AP | All nodes serve reads/writes | Last-write-wins, vector clocks |
| **DynamoDB** | AP | Continues operating across partitions | Version vectors, merge |
| **MongoDB** | CP (default) | Secondaries reject writes | Primary election |
| **CockroachDB** | CP | Ranges without quorum unavailable | Raft consensus |
| **Riak** | AP | All nodes accept read/write | Sibling resolution |

```python
from dataclasses import dataclass, field
from enum import Enum

class CAPChoice(Enum):
    CP = "consistent_partition_tolerant"
    AP = "available_partition_tolerant"
    CA = "consistent_available"  # Single-node only

@dataclass
class DistributedSystem:
    name: str
    cap_choice: CAPChoice
    num_nodes: int
    replication_factor: int
    consistency_level: str = "quorum"

    @property
    def partition_behavior(self) -> str:
        if self.cap_choice == CAPChoice.CP:
            return "Refuses operations without quorum — consistent but unavailable"
        elif self.cap_choice == CAPChoice.AP:
            return "Continues serving from available nodes — available but potentially stale"
        else:
            return "N/A — single-node, no partitions"

    @property
    def quorum_size(self) -> int:
        return self.replication_factor // 2 + 1

    def availability_during_partition(self, nodes_in_partition: int) -> str:
        available_nodes = self.num_nodes - nodes_in_partition
        if self.cap_choice == CAPChoice.CP:
            if available_nodes >= self.quorum_size:
                return f"AVAILABLE (quorum met: {available_nodes} >= {self.quorum_size})"
            else:
                return f"UNAVAILABLE (no quorum: {available_nodes} < {self.quorum_size})"
        elif self.cap_choice == CAPChoice.AP:
            if available_nodes > 0:
                return f"AVAILABLE (serving from {available_nodes} nodes, may be stale)"
            return "UNAVAILABLE (all nodes partitioned)"
        return "N/A"

# Compare CP vs AP during partitions
systems = [
    DistributedSystem("ZooKeeper", CAPChoice.CP, 5, 5, "majority"),
    DistributedSystem("Cassandra", CAPChoice.AP, 5, 3, "quorum"),
    DistributedSystem("MongoDB", CAPChoice.CP, 3, 3, "majority"),
    DistributedSystem("DynamoDB", CAPChoice.AP, 3, 3, "eventual"),
]

print("CAP Theorem Analysis - Partition Scenarios")
print("=" * 70)
for sys in systems:
    print(f"\n{sys.name} ({sys.cap_choice.value}):")
    print(f"  Normal: {sys.partition_behavior}")
    for failed in [1, 2, 3]:
        if failed < sys.num_nodes:
            result = sys.availability_during_partition(failed)
            print(f"  {failed} nodes partitioned: {result}")
```

**AI/ML Application:** ML serving systems are typically **AP** — it's better to serve a slightly stale prediction than to reject the request entirely. **Feature stores** choose between CP (financial features must be exact) and AP (recommendation features can be stale). Model registries are typically CP since deploying the wrong model version is worse than a brief delay.

**Real-World Example:** Google Spanner achieves effective CA (consistency + availability) by using **GPS-synchronized atomic clocks** (TrueTime) to minimize the consistency-availability tradeoff. However, it still technically sacrifices availability during severe partitions — it just makes partitions extremely rare through high-quality network infrastructure.

> **Interview Tip:** Clarify that CAP is about behavior **during partitions only** — in normal operation, all three properties hold. Also mention **PACELC** as the extension: "if Partition, choose A or C; Else (normal), choose Latency or Consistency." This shows deeper understanding.

---

### 25. Can you discuss how quorum-based decision making in distributed systems affects reliability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Quorum-based systems** require a minimum number of nodes to **agree** before an operation is considered successful. The fundamental formula is **W + R > N** (write quorum + read quorum > total replicas), which guarantees that reads and writes always overlap on at least one node with the latest data. Quorums provide **tunable consistency** — adjusting W and R lets you trade consistency for availability and performance.

- **N** = Total number of replicas (replication factor)
- **W** = Write quorum (nodes that must acknowledge a write)
- **R** = Read quorum (nodes that must respond to a read)
- **W + R > N**: Guarantees strong consistency (read/write overlap)
- **W + R <= N**: Allows eventual consistency (faster but may read stale)
- **W = N**: All nodes must ack writes (slow writes, high durability)
- **R = 1**: Read from any single node (fast reads, may be stale)

```
+-----------------------------------------------------------+
|         QUORUM-BASED DECISION MAKING                       |
+-----------------------------------------------------------+
|                                                             |
|  N = 3 replicas, W = 2, R = 2                             |
|  W + R = 4 > 3 = N --> STRONG CONSISTENCY                 |
|                                                             |
|  WRITE (W=2):                                              |
|  Client --> [A: write ACK] [B: write ACK] [C: pending]    |
|  2 ACKs received >= W=2 --> Write SUCCESS                  |
|                                                             |
|  READ (R=2):                                               |
|  Client --> [A: v5] [B: v5] (C not needed)                 |
|  At least 1 of {A,B} has latest (overlap with write set)  |
|  Return highest version: v5                                |
|                                                             |
|  QUORUM CONFIGURATIONS:                                    |
|  +--------+-----+-----+-----------+--------------------+   |
|  | Config | W   | R   | Property  | Trade-off          |   |
|  |--------|-----|-----|-----------|---------------------|   |
|  | W2,R2  | 2   | 2   | Strong    | Balanced           |   |
|  | W3,R1  | 3   | 1   | Strong    | Fast reads         |   |
|  | W1,R3  | 1   | 3   | Strong    | Fast writes        |   |
|  | W1,R1  | 1   | 1   | Eventual  | Fastest, risky     |   |
|  | W3,R3  | 3   | 3   | Strong    | Slowest, safest    |   |
|  +--------+-----+-----+-----------+--------------------+   |
|                                                             |
|  SLOPPY QUORUM (during partition):                         |
|  If not enough replicas available, use "hint" nodes        |
|  Temporary nodes hold data until real replica recovers     |
|  --> Improves availability at cost of consistency          |
+-----------------------------------------------------------+
```

| Configuration | W + R > N? | Consistency | Write Speed | Read Speed | Failure Tolerance |
|---|---|---|---|---|---|
| **W=2, R=2, N=3** | 4 > 3 Yes | Strong | Medium | Medium | 1 node failure |
| **W=3, R=1, N=3** | 4 > 3 Yes | Strong | Slow | Fast | 0 write failures |
| **W=1, R=3, N=3** | 4 > 3 Yes | Strong | Fast | Slow | 0 read failures |
| **W=1, R=1, N=3** | 2 > 3 No | Eventual | Fastest | Fastest | 2 node failures |
| **W=2, R=2, N=5** | 4 > 5 No | Eventual | Medium | Medium | 3 node failures |
| **W=3, R=3, N=5** | 6 > 5 Yes | Strong | Slow | Slow | 2 node failures |

```python
from dataclasses import dataclass, field
import random

@dataclass
class QuorumConfig:
    n: int  # Total replicas
    w: int  # Write quorum
    r: int  # Read quorum

    @property
    def is_strongly_consistent(self) -> bool:
        return self.w + self.r > self.n

    @property
    def write_fault_tolerance(self) -> int:
        return self.n - self.w

    @property
    def read_fault_tolerance(self) -> int:
        return self.n - self.r

    @property
    def overlap(self) -> int:
        return max(0, self.w + self.r - self.n)

    def can_write(self, available_nodes: int) -> bool:
        return available_nodes >= self.w

    def can_read(self, available_nodes: int) -> bool:
        return available_nodes >= self.r

@dataclass
class QuorumSimulator:
    config: QuorumConfig
    nodes: list = field(default_factory=list)
    _versions: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = [{"id": i, "alive": True, "data": {}} for i in range(self.config.n)]

    def write(self, key: str, value: str) -> dict:
        version = self._versions.get(key, 0) + 1
        alive_nodes = [n for n in self.nodes if n["alive"]]
        random.shuffle(alive_nodes)

        acks = 0
        for node in alive_nodes[:self.config.w]:
            node["data"][key] = {"value": value, "version": version}
            acks += 1

        if acks >= self.config.w:
            self._versions[key] = version
            return {"status": "success", "acks": acks, "version": version}
        return {"status": "failed", "acks": acks, "needed": self.config.w}

    def read(self, key: str) -> dict:
        alive_nodes = [n for n in self.nodes if n["alive"]]
        random.shuffle(alive_nodes)

        responses = []
        for node in alive_nodes[:self.config.r]:
            data = node["data"].get(key, {"value": None, "version": 0})
            responses.append(data)

        if len(responses) >= self.config.r:
            best = max(responses, key=lambda r: r["version"])
            return {"status": "success", "value": best["value"], "version": best["version"]}
        return {"status": "failed", "responses": len(responses), "needed": self.config.r}

    def kill_node(self, node_id: int):
        self.nodes[node_id]["alive"] = False

    def availability_report(self):
        alive = sum(1 for n in self.nodes if n["alive"])
        print(f"Config: N={self.config.n}, W={self.config.w}, R={self.config.r}")
        print(f"Strongly consistent: {self.config.is_strongly_consistent}")
        print(f"Alive nodes: {alive}/{self.config.n}")
        print(f"Can write: {self.config.can_write(alive)}")
        print(f"Can read: {self.config.can_read(alive)}")

# Demo: different quorum configs under failure
configs = [
    ("Balanced", QuorumConfig(3, 2, 2)),
    ("Fast Reads", QuorumConfig(3, 3, 1)),
    ("Fast Writes", QuorumConfig(3, 1, 3)),
    ("Eventual", QuorumConfig(3, 1, 1)),
]

for name, config in configs:
    sim = QuorumSimulator(config)
    sim.write("key1", "hello")
    sim.kill_node(2)  # One node fails
    print(f"\n{name} (W={config.w}, R={config.r}):")
    sim.availability_report()
    result = sim.read("key1")
    print(f"Read after failure: {result}")
```

**AI/ML Application:** ML model serving can use quorum-based **prediction voting** — run the same input through N model replicas and return the **majority prediction**. This improves reliability when individual model instances may have loading issues or use slightly different model versions during rolling deployments.

**Real-World Example:** Apache Cassandra uses **tunable quorum consistency** — applications choose W and R per query. Netflix uses `LOCAL_QUORUM` (quorum within a data center) for most reads, which provides strong consistency within a region and eventual consistency across regions. This balances latency with reliability.

> **Interview Tip:** Always write the formula W + R > N and explain what each variable means. Show different configurations (W=1,R=1 for speed vs W=N,R=N for safety) and explain the tradeoffs. Mention **sloppy quorum** and **hinted handoff** as mechanisms that improve availability at the cost of strict consistency guarantees.

---

### 26. What is the role of distributed transactions in reliability , and what are the challenges associated with them? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Distributed transactions** ensure that operations spanning **multiple services or databases** either **all succeed or all fail** (atomicity), maintaining data consistency across the system. However, they introduce significant challenges: **increased latency** (coordination overhead), **reduced availability** (blocking during failures), and **complexity** (handling partial failures, network partitions, and timeout scenarios).

- **2PC (Two-Phase Commit)**: Coordinator asks all participants to prepare, then commit; blocks on failure
- **3PC (Three-Phase Commit)**: Adds pre-commit phase to reduce blocking; more complex
- **Saga Pattern**: Breaks transaction into local transactions with compensating actions; no blocking
- **TCC (Try-Confirm-Cancel)**: Reserve resources, then confirm or cancel; popular in finance
- **Challenge: Coordinator Failure**: 2PC blocks if coordinator crashes after prepare phase
- **Challenge: Network Partitions**: Participants may be unable to reach coordinator

```
+-----------------------------------------------------------+
|         DISTRIBUTED TRANSACTION PATTERNS                   |
+-----------------------------------------------------------+
|                                                             |
|  TWO-PHASE COMMIT (2PC):                                  |
|  Coordinator  Participant A  Participant B                 |
|      |              |              |                       |
|      |--PREPARE---->|              |                       |
|      |--PREPARE------------------->|                       |
|      |              |              |                       |
|      |<--VOTE YES---|              |                       |
|      |<--VOTE YES-----------------|                       |
|      |              |              |                       |
|      |--COMMIT----->|              |                       |
|      |--COMMIT-------------------->|                       |
|      |              |              |                       |
|  Problem: If coordinator crashes between PREPARE and       |
|  COMMIT, participants BLOCK indefinitely!                  |
|                                                             |
|  SAGA PATTERN (Choreography):                              |
|  [Order Service] --event--> [Payment Service]              |
|       |                          |                         |
|   T1: Create    success     T2: Charge                     |
|   Order         event       Payment                        |
|       |                          |                         |
|   C1: Cancel    failure     C2: Refund                     |
|   Order         event       Payment                        |
|                                                             |
|  Each step has a COMPENSATING action for rollback          |
|  No blocking! But eventual consistency only                |
|                                                             |
|  TCC PATTERN:                                              |
|  TRY:     Reserve $100 from account                        |
|  CONFIRM: Deduct $100 (make reservation permanent)         |
|  CANCEL:  Release $100 reservation                         |
+-----------------------------------------------------------+
```

| Pattern | Consistency | Availability | Latency | Complexity | Use Case |
|---|---|---|---|---|---|
| **2PC** | Strong (ACID) | Low (blocks on failure) | High (2 round trips) | Medium | Cross-DB transactions |
| **3PC** | Strong | Better than 2PC | Higher (3 round trips) | High | Rarely used |
| **Saga (Choreography)** | Eventual | High (non-blocking) | Low per step | Medium | Microservice workflows |
| **Saga (Orchestration)** | Eventual | High (non-blocking) | Low per step | Medium-High | Complex workflows |
| **TCC** | Strong (bounded) | Medium | Medium | High | Financial transactions |
| **Outbox Pattern** | Eventual | High | Low | Low-Medium | Event-driven systems |

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Callable, Optional

class TxStatus(Enum):
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"

@dataclass
class SagaStep:
    name: str
    action: Callable
    compensate: Callable
    status: TxStatus = TxStatus.PENDING

@dataclass
class SagaOrchestrator:
    saga_id: str
    steps: list[SagaStep] = field(default_factory=list)
    status: TxStatus = TxStatus.PENDING
    completed_steps: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def add_step(self, name: str, action: Callable, compensate: Callable):
        self.steps.append(SagaStep(name, action, compensate))

    def execute(self) -> bool:
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Saga {self.saga_id} started")

        for i, step in enumerate(self.steps):
            try:
                self.log.append(f"  Executing: {step.name}")
                step.action()
                step.status = TxStatus.COMMITTED
                self.completed_steps.append(step.name)
                self.log.append(f"  {step.name}: SUCCESS")
            except Exception as e:
                self.log.append(f"  {step.name}: FAILED ({e})")
                step.status = TxStatus.ABORTED
                self._compensate(i)
                return False

        self.status = TxStatus.COMMITTED
        self.log.append(f"Saga {self.saga_id}: COMMITTED")
        return True

    def _compensate(self, failed_step_index: int):
        self.status = TxStatus.COMPENSATING
        self.log.append(f"  Starting compensation...")

        for i in range(failed_step_index - 1, -1, -1):
            step = self.steps[i]
            try:
                self.log.append(f"  Compensating: {step.name}")
                step.compensate()
                step.status = TxStatus.COMPENSATED
                self.log.append(f"  {step.name}: COMPENSATED")
            except Exception as e:
                self.log.append(f"  {step.name}: COMPENSATION FAILED ({e}) - manual intervention needed!")

        self.status = TxStatus.COMPENSATED

    def print_log(self):
        for entry in self.log:
            print(entry)

# Simulate order processing saga
import random
saga = SagaOrchestrator("order-12345")

saga.add_step(
    "Create Order",
    action=lambda: print("    -> Order created in DB"),
    compensate=lambda: print("    -> Order cancelled in DB")
)
saga.add_step(
    "Reserve Inventory",
    action=lambda: print("    -> Inventory reserved"),
    compensate=lambda: print("    -> Inventory released")
)
saga.add_step(
    "Charge Payment",
    action=lambda: (_ for _ in ()).throw(Exception("Payment declined")),  # Simulated failure
    compensate=lambda: print("    -> Payment refunded")
)
saga.add_step(
    "Ship Order",
    action=lambda: print("    -> Shipment initiated"),
    compensate=lambda: print("    -> Shipment cancelled")
)

success = saga.execute()
print()
saga.print_log()
print(f"\nFinal Status: {saga.status.value}")
```

**AI/ML Application:** ML pipelines use saga-like patterns for **multi-step model deployment**: upload model artifact → update model registry → warm up inference instances → shift traffic. If traffic shifting reveals degraded accuracy, the **compensating action** rolls back traffic, removes instances, and restores the previous model version.

**Real-World Example:** Uber uses a **saga orchestrator** (Cadence/Temporal) for ride booking: create ride → match driver → charge passenger → pay driver. If charging fails, the saga compensates by cancelling the ride and notifying the driver. Stripe uses the **TCC pattern** for payment processing: authorize (try) → capture (confirm) → void (cancel).

> **Interview Tip:** Compare 2PC and Saga clearly: 2PC provides strong consistency but blocks during failures (poor availability), while Saga provides eventual consistency but never blocks (high availability). Mention that the **outbox pattern** is a practical way to implement reliable event publishing alongside database transactions.

---

## Recovery Strategies

### 27. What is a disaster recovery plan and how does it relate to reliability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

A **Disaster Recovery (DR) Plan** is a documented strategy for **restoring critical systems and data** after catastrophic events (natural disasters, data center failures, cyberattacks, human error). It defines **RPO** (Recovery Point Objective — maximum acceptable data loss) and **RTO** (Recovery Time Objective — maximum acceptable downtime). DR directly impacts reliability by ensuring the system can recover from worst-case scenarios.

- **RPO** (Recovery Point Objective): Maximum data loss tolerance (e.g., 1 hour = accept losing last hour of data)
- **RTO** (Recovery Time Objective): Maximum downtime tolerance (e.g., 4 hours to restore service)
- **DR Tiers**: Cold (hours-days), Warm (minutes-hours), Hot (seconds-minutes), Active-Active (zero)
- **Backup Strategy**: 3-2-1 rule (3 copies, 2 different media, 1 offsite)
- **Testing**: Regular DR drills to verify the plan works (untested plans are worthless)
- **Runbooks**: Step-by-step procedures for each disaster scenario

```
+-----------------------------------------------------------+
|         DISASTER RECOVERY TIERS                            |
+-----------------------------------------------------------+
|                                                             |
|  RPO vs RTO:                                               |
|  <------ data loss ------>|<------ downtime ------>        |
|  Last backup    Disaster  |    Service restored            |
|       |            |      |          |                      |
|       v            v      v          v                      |
|  -----+------------+------+----------+----->               |
|       |<--- RPO -->|      |<-- RTO ->|                     |
|                                                             |
|  DR TIER COMPARISON:                                       |
|  +----------+--------+--------+--------+-----------+       |
|  | Tier     | RPO    | RTO    | Cost   | Method    |       |
|  |----------|--------|--------|--------|-----------|       |
|  | Cold     | 24h    | Days   | $      | Backups   |       |
|  | Warm     | 1-4h   | Hours  | $$     | Standby   |       |
|  | Hot      | Min    | Min    | $$$    | Replicas  |       |
|  | Active-  | 0      | ~0     | $$$$   | Multi-    |       |
|  | Active   |        |        |        | region    |       |
|  +----------+--------+--------+--------+-----------+       |
|                                                             |
|  3-2-1 BACKUP RULE:                                        |
|  [Production DB]                                           |
|       |                                                     |
|       +--> [Local Replica] (copy 1, same media)            |
|       +--> [Remote Backup] (copy 2, different media)       |
|       +--> [Offsite/Cloud] (copy 3, offsite location)      |
|                                                             |
|  DR TESTING CADENCE:                                       |
|  Tabletop exercise:  Monthly (discussion-based)            |
|  Component failover: Quarterly (test single components)    |
|  Full DR drill:      Semi-annually (complete switchover)   |
|  Chaos engineering:  Continuous (automated fault injection) |
+-----------------------------------------------------------+
```

| DR Tier | RPO | RTO | Infrastructure | Cost Multiplier | Example |
|---|---|---|---|---|---|
| **Cold Site** | 24 hours | Days | Backups only, no standby | 1.1x | Small business |
| **Warm Standby** | 1-4 hours | 1-4 hours | Scaled-down replica, async replication | 1.5x | E-commerce |
| **Hot Standby** | Minutes | Minutes | Full replica, sync replication | 2x | Banking |
| **Active-Active** | Zero | Near-zero | Multi-region, both serving | 2.5x+ | Payment processing |
| **Pilot Light** | Hours | 1-2 hours | Minimal infra, scale up on trigger | 1.2x | Seasonal workloads |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class DRTier(Enum):
    COLD = "cold_site"
    PILOT_LIGHT = "pilot_light"
    WARM = "warm_standby"
    HOT = "hot_standby"
    ACTIVE_ACTIVE = "active_active"

@dataclass
class DRPlan:
    name: str
    tier: DRTier
    rpo_hours: float
    rto_hours: float
    cost_multiplier: float
    backup_frequency_hours: float = 24.0
    last_test_date: datetime = None
    test_results: list = field(default_factory=list)

    @property
    def max_data_loss_gb(self) -> float:
        """Estimate max data loss based on RPO and write rate."""
        write_rate_gb_per_hour = 10  # Example
        return self.rpo_hours * write_rate_gb_per_hour

    @property
    def annual_downtime_hours(self) -> float:
        """Expected annual downtime if 1 disaster occurs."""
        return self.rto_hours

    @property
    def days_since_last_test(self) -> int:
        if not self.last_test_date:
            return -1  # Never tested
        return (datetime.now() - self.last_test_date).days

    @property
    def test_overdue(self) -> bool:
        return self.days_since_last_test > 90 or self.days_since_last_test == -1

@dataclass
class DRAnalyzer:
    plans: list[DRPlan] = field(default_factory=list)

    def add_plan(self, plan: DRPlan):
        self.plans.append(plan)

    def cost_analysis(self, base_monthly_cost: float):
        print("Disaster Recovery Cost Analysis")
        print("=" * 65)
        for plan in self.plans:
            monthly = base_monthly_cost * plan.cost_multiplier
            annual = monthly * 12
            print(f"  {plan.tier.value:<16} RPO:{plan.rpo_hours:>5.1f}h RTO:{plan.rto_hours:>5.1f}h "
                  f"Cost:${monthly:>8,.0f}/mo (${annual:>10,.0f}/yr) "
                  f"{'UNTESTED!' if plan.test_overdue else 'Tested'}")

    def recommend(self, max_rpo_hours: float, max_rto_hours: float) -> list:
        suitable = [p for p in self.plans if p.rpo_hours <= max_rpo_hours and p.rto_hours <= max_rto_hours]
        return sorted(suitable, key=lambda p: p.cost_multiplier)

analyzer = DRAnalyzer()
analyzer.add_plan(DRPlan("Cold Backup", DRTier.COLD, 24.0, 48.0, 1.1,
                         last_test_date=datetime.now() - timedelta(days=200)))
analyzer.add_plan(DRPlan("Pilot Light", DRTier.PILOT_LIGHT, 4.0, 2.0, 1.2,
                         last_test_date=datetime.now() - timedelta(days=60)))
analyzer.add_plan(DRPlan("Warm Standby", DRTier.WARM, 1.0, 0.5, 1.5,
                         last_test_date=datetime.now() - timedelta(days=30)))
analyzer.add_plan(DRPlan("Hot Standby", DRTier.HOT, 0.1, 0.1, 2.0,
                         last_test_date=datetime.now() - timedelta(days=45)))
analyzer.add_plan(DRPlan("Active-Active", DRTier.ACTIVE_ACTIVE, 0.0, 0.0, 2.5,
                         last_test_date=datetime.now() - timedelta(days=7)))

analyzer.cost_analysis(base_monthly_cost=50000)
print("\nRecommended for RPO<1h, RTO<1h:")
for p in analyzer.recommend(1.0, 1.0):
    print(f"  {p.tier.value}: ${50000 * p.cost_multiplier:,.0f}/mo")
```

**AI/ML Application:** ML DR plans must include **model artifact recovery** (stored in S3/GCS with cross-region replication), **training data backups** (lineage-tracked datasets), and **feature pipeline recovery** (feature store snapshots). RPO for ML models depends on retraining frequency — a model trained weekly has an RPO of up to 7 days.

**Real-World Example:** AWS offers DR strategies across their services: **S3 Cross-Region Replication** for data, **Aurora Global Database** for RPO < 1 second across regions, and **Route 53 DNS failover** for traffic switching. Netflix tests their DR plan continuously with **Chaos Monkey** (instance failures) and **Chaos Kong** (entire region failover).

> **Interview Tip:** Always define RPO and RTO first, then map them to DR tiers. Emphasize that **untested DR plans are worthless** — mention the testing cadence (tabletop monthly, full drill semi-annually). The 3-2-1 backup rule is a simple but powerful framework to mention.

---

### 28. How do backup and restore operations impact system availability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Backup and restore** operations are critical for **data durability** but can negatively impact availability if not designed carefully. Full backups consume **I/O bandwidth** and **CPU resources**, potentially degrading live traffic performance. Restore operations may require **service downtime** depending on the backup method. The key is choosing strategies that minimize availability impact while meeting RPO requirements.

- **Full Backup**: Complete copy of all data — simple but slow and resource-intensive
- **Incremental Backup**: Only changes since last backup — fast but complex restore
- **Differential Backup**: Changes since last full backup — balanced approach
- **Continuous/WAL Backup**: Real-time log streaming — minimal RPO, near-zero impact
- **Snapshot**: Point-in-time copy using copy-on-write — very fast, low impact
- **Backup Window**: Schedule during low-traffic periods to minimize user impact

```
+-----------------------------------------------------------+
|         BACKUP STRATEGIES & AVAILABILITY IMPACT            |
+-----------------------------------------------------------+
|                                                             |
|  BACKUP TYPES AND RECOVERY:                                |
|                                                             |
|  Full (Sunday):     [===================] 100GB, 4 hours   |
|  Incremental (Mon): [==] 5GB, 15 min                       |
|  Incremental (Tue): [===] 7GB, 20 min                      |
|  Incremental (Wed): [==] 4GB, 12 min                       |
|                                                             |
|  Restore Wednesday: Full + Mon + Tue + Wed = 4 steps       |
|  RPO: < 24 hours (last incremental)                        |
|                                                             |
|  CONTINUOUS BACKUP (WAL streaming):                        |
|  [Primary DB] --WAL stream--> [Backup Store]               |
|  Every transaction logged in real-time                      |
|  RPO: seconds | Impact: minimal (async stream)             |
|                                                             |
|  IMPACT ON LIVE TRAFFIC:                                   |
|  Full backup running:                                      |
|  CPU: [=========>           ] 40% for backup               |
|  I/O: [============>        ] 50% for backup               |
|  Available for traffic: 50-60% capacity                    |
|                                                             |
|  SNAPSHOT (copy-on-write):                                 |
|  [Snapshot created in <1 second]                           |
|  Reads go to snapshot, writes go to new blocks             |
|  Impact: near zero during creation                         |
|  But: increased I/O during copy-on-write phase             |
+-----------------------------------------------------------+
```

| Backup Method | RPO | Backup Speed | Restore Speed | Availability Impact | Storage Cost |
|---|---|---|---|---|---|
| **Full** | Backup interval | Slow (hours) | Fast (single restore) | High during backup | High (full copies) |
| **Incremental** | Backup interval | Fast (minutes) | Slow (chain restore) | Low | Low (deltas only) |
| **Differential** | Backup interval | Medium | Medium (2 restores) | Medium | Medium |
| **WAL/Continuous** | Seconds | N/A (continuous) | Medium (replay logs) | Minimal | Medium (log storage) |
| **Snapshot** | Snapshot interval | Instant | Fast (mount snapshot) | Near zero | Medium (COW) |
| **Logical (pg_dump)** | Backup interval | Slow | Slow (re-import) | Medium-High | Low (compressed) |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"
    WAL_STREAM = "wal_stream"

@dataclass
class BackupRecord:
    backup_type: BackupType
    started_at: datetime
    completed_at: datetime
    size_gb: float
    success: bool = True
    io_impact_percent: float = 0.0

    @property
    def duration_minutes(self) -> float:
        return (self.completed_at - self.started_at).total_seconds() / 60

    @property
    def throughput_gbps(self) -> float:
        hours = self.duration_minutes / 60
        return self.size_gb / hours if hours > 0 else 0

@dataclass
class BackupScheduler:
    full_interval_days: int = 7
    incremental_interval_hours: int = 6
    retention_days: int = 30
    backups: list[BackupRecord] = field(default_factory=list)

    def record_backup(self, backup: BackupRecord):
        self.backups.append(backup)

    @property
    def total_storage_gb(self) -> float:
        return sum(b.size_gb for b in self.backups)

    def rpo_current(self) -> float:
        """Current RPO based on most recent backup."""
        if not self.backups:
            return float('inf')
        latest = max(self.backups, key=lambda b: b.completed_at)
        hours_since = (datetime.now() - latest.completed_at).total_seconds() / 3600
        return hours_since

    def restore_time_estimate(self) -> dict:
        """Estimate restore time based on backup chain."""
        fulls = [b for b in self.backups if b.backup_type == BackupType.FULL and b.success]
        incrementals = [b for b in self.backups if b.backup_type == BackupType.INCREMENTAL and b.success]

        if not fulls:
            return {"strategy": "none", "time_hours": float('inf')}

        latest_full = max(fulls, key=lambda b: b.completed_at)
        chain = [b for b in incrementals if b.completed_at > latest_full.completed_at]

        total_gb = latest_full.size_gb + sum(b.size_gb for b in chain)
        restore_speed_gbps = 50  # Estimated restore throughput
        time_hours = total_gb / restore_speed_gbps

        return {
            "strategy": f"1 full + {len(chain)} incrementals",
            "total_data_gb": total_gb,
            "estimated_time_hours": time_hours,
            "steps": 1 + len(chain),
        }

    def report(self):
        print(f"Backup Report")
        print(f"  Total backups: {len(self.backups)}")
        print(f"  Total storage: {self.total_storage_gb:.1f} GB")
        print(f"  Current RPO: {self.rpo_current():.2f} hours")

        success_rate = sum(1 for b in self.backups if b.success) / len(self.backups) * 100 if self.backups else 0
        print(f"  Success rate: {success_rate:.1f}%")

        restore = self.restore_time_estimate()
        print(f"  Restore estimate: {restore['strategy']} ({restore.get('estimated_time_hours', 'N/A'):.2f} hours)")

# Demo
scheduler = BackupScheduler()
now = datetime.now()

# Full backup (Sunday)
scheduler.record_backup(BackupRecord(
    BackupType.FULL, now - timedelta(days=3), now - timedelta(days=3, hours=-4), 500, True, 40))

# Daily incrementals
for day in range(3):
    scheduler.record_backup(BackupRecord(
        BackupType.INCREMENTAL, now - timedelta(days=2-day), now - timedelta(days=2-day, hours=-0.25),
        15 + day * 5, True, 10))

scheduler.report()
```

**AI/ML Application:** ML backups include **model checkpoints** (saved during training every N epochs), **feature store snapshots** (point-in-time view of all features), and **training dataset versioning** (DVC or similar tools). Model checkpoints are critical — a 3-day GPU training run failing on day 2 without checkpoints wastes significant compute resources.

**Real-World Example:** PostgreSQL's **WAL archiving** + **pg_basebackup** enables Point-in-Time Recovery (PITR) with RPO in seconds. A full base backup is taken weekly, and WAL segments are continuously archived to S3. Restoration replays the base backup + WAL segments to any point in time. Netflix uses this for their PostgreSQL instances with automated backup verification.

> **Interview Tip:** Discuss the **tradeoff between backup frequency and system impact** — more frequent backups reduce RPO but increase I/O load. Mention snapshots (near-zero impact) and WAL streaming (continuous, minimal impact) as modern approaches. Always mention **backup testing** — backups that can't be restored are useless.

---

### 29. Discuss the importance and challenges of data replication in a highly available system. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Data replication** is the process of maintaining **identical copies of data** across multiple nodes or regions to enable **failover**, **read scaling**, and **geographic locality**. It is the foundation of high availability for stateful systems. Challenges include **replication lag**, **conflict resolution**, **split-brain scenarios**, **bandwidth costs**, and the fundamental tension between **consistency and latency** defined by the CAP theorem.

- **Synchronous Replication**: Primary waits for replica acknowledgment — strong consistency, higher latency
- **Asynchronous Replication**: Primary doesn't wait — low latency but data loss risk
- **Semi-Synchronous**: Wait for at least one replica — balance of durability and performance
- **Multi-Master**: All nodes accept writes — highest availability but complex conflict resolution
- **Conflict Resolution**: Last-writer-wins (LWW), vector clocks, CRDTs, application-level merge
- **Replication Lag**: Time delay between primary write and replica visibility — critical SLI

```
+-----------------------------------------------------------+
|         DATA REPLICATION PATTERNS                          |
+-----------------------------------------------------------+
|                                                             |
|  SYNCHRONOUS:                                              |
|  Client --> Primary --sync--> Replica 1                    |
|                    --sync--> Replica 2                     |
|             Wait for ALL acks before confirming write      |
|  Latency: High | Data Loss: Zero | Availability: Lower    |
|                                                             |
|  ASYNCHRONOUS:                                             |
|  Client --> Primary --async--> Replica 1 (eventually)      |
|             Confirm immediately                            |
|  Latency: Low | Data Loss: Possible | Availability: Higher|
|                                                             |
|  SEMI-SYNCHRONOUS:                                         |
|  Client --> Primary --sync--> Replica 1 (wait for 1)       |
|                    --async--> Replica 2 (eventually)       |
|  Latency: Medium | Data Loss: Minimal | Good Balance      |
|                                                             |
|  SPLIT-BRAIN SCENARIO:                                     |
|  [Primary A] X partition X [Primary B]                     |
|  Both accept writes --> DATA DIVERGES!                     |
|  Fix: Fencing (STONITH), quorum, leader lease              |
|                                                             |
|  REPLICATION LAG:                                          |
|  Primary: write at T=0                                     |
|  Replica: sees write at T=100ms (100ms lag)                |
|  Client reads from replica at T=50ms --> STALE DATA!       |
|                                                             |
|  MULTI-REGION:                                             |
|  US-East [Primary] --WAN replication--> EU-West [Replica]  |
|  Latency: 70-100ms cross-Atlantic                          |
|  Sync: too slow | Async: RPO = replication lag             |
+-----------------------------------------------------------+
```

| Replication Type | Consistency | Latency | Data Loss Risk | Write Throughput | Complexity |
|---|---|---|---|---|---|
| **Sync (all replicas)** | Strong | High | Zero | Low (bottleneck) | Low |
| **Semi-Sync (1 replica)** | Strong | Medium | Minimal | Medium | Medium |
| **Async** | Eventual | Low | Possible (lag window) | High | Low |
| **Multi-Master** | Eventual | Low | Conflict risk | Highest | Very High |
| **Chain Replication** | Strong | Medium | Zero | Medium | Medium |
| **CRDT-Based** | Strong eventual | Low | None (merge) | High | High |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
from statistics import mean, percentile_approx

class ReplicationType(Enum):
    SYNC = "synchronous"
    ASYNC = "asynchronous"
    SEMI_SYNC = "semi_synchronous"

@dataclass
class ReplicaNode:
    name: str
    region: str
    is_primary: bool = False
    replication_lag_ms: float = 0.0
    writes_applied: int = 0
    is_healthy: bool = True

@dataclass
class ReplicationCluster:
    name: str
    replication_type: ReplicationType
    nodes: list[ReplicaNode] = field(default_factory=list)
    _lag_history: list = field(default_factory=list)

    def add_node(self, node: ReplicaNode):
        self.nodes.append(node)

    @property
    def primary(self) -> ReplicaNode:
        return next(n for n in self.nodes if n.is_primary)

    @property
    def replicas(self) -> list:
        return [n for n in self.nodes if not n.is_primary]

    def simulate_write(self) -> dict:
        primary = self.primary
        result = {"primary": primary.name, "replicated_to": [], "lag_ms": {}}

        for replica in self.replicas:
            if not replica.is_healthy:
                continue

            base_lag = random.uniform(1, 10) if replica.region == primary.region else random.uniform(50, 150)

            if self.replication_type == ReplicationType.SYNC:
                replica.replication_lag_ms = base_lag
                result["replicated_to"].append(replica.name)
                result["lag_ms"][replica.name] = base_lag
            elif self.replication_type == ReplicationType.ASYNC:
                lag = base_lag * random.uniform(1, 10)
                replica.replication_lag_ms = lag
                if random.random() < 0.9:
                    result["replicated_to"].append(replica.name)
                    result["lag_ms"][replica.name] = lag
            elif self.replication_type == ReplicationType.SEMI_SYNC:
                replica.replication_lag_ms = base_lag
                result["replicated_to"].append(replica.name)
                result["lag_ms"][replica.name] = base_lag

            self._lag_history.append(replica.replication_lag_ms)

        return result

    def report(self):
        print(f"\nReplication Cluster: {self.name} ({self.replication_type.value})")
        print(f"  Primary: {self.primary.name} ({self.primary.region})")
        for r in self.replicas:
            status = "healthy" if r.is_healthy else "DOWN"
            print(f"  Replica: {r.name} ({r.region}) lag={r.replication_lag_ms:.1f}ms [{status}]")

        if self._lag_history:
            sorted_lags = sorted(self._lag_history)
            p50 = sorted_lags[len(sorted_lags)//2]
            p99 = sorted_lags[int(len(sorted_lags)*0.99)]
            print(f"  Lag P50: {p50:.1f}ms, P99: {p99:.1f}ms, Max: {max(sorted_lags):.1f}ms")

# Demo
cluster = ReplicationCluster("orders-db", ReplicationType.SEMI_SYNC)
cluster.add_node(ReplicaNode("primary-us-east", "us-east", is_primary=True))
cluster.add_node(ReplicaNode("replica-us-east", "us-east"))
cluster.add_node(ReplicaNode("replica-eu-west", "eu-west"))
cluster.add_node(ReplicaNode("replica-ap-south", "ap-south", is_healthy=False))

for _ in range(100):
    cluster.simulate_write()

cluster.report()
```

**AI/ML Application:** ML model registries replicate model artifacts across regions for **low-latency model loading**. Feature stores use **cross-region replication** so inference endpoints in each region can read features locally. The challenge is that ML features update frequently (real-time features), making replication lag directly impact **prediction freshness**.

**Real-World Example:** MySQL at Meta uses **semi-synchronous replication** within AZs (strong consistency for writes) and **asynchronous replication** across regions (eventual consistency for global reads). Their system handles billions of writes per day with replication lag typically under 100ms. During the 2021 outage, BGP configuration changes caused a cascading DNS failure, but data remained consistent across replicas.

> **Interview Tip:** Discuss the **CAP tradeoff in replication**: sync replication gives CP, async gives AP. Mention **split-brain prevention** (STONITH fencing, quorum, leader leases) as a critical reliability concern. Explain why **replication lag is an important SLI** to monitor — it directly determines your effective RPO.

---

### 30. Explain how you would plan for a failover strategy in a multi-region deployment to ensure reliability . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

A **multi-region failover strategy** ensures service continuity when an entire region becomes unavailable. Planning involves choosing between **active-active** (all regions serve traffic simultaneously) and **active-passive** (standby region activates on failure), defining **failover triggers**, testing through **chaos engineering**, and managing **data consistency** across regions. The strategy must address DNS propagation, data synchronization, and client reconnection.

- **Active-Active**: All regions serve traffic — fastest failover, highest cost and complexity
- **Active-Passive**: Primary serves, secondary standby — simpler, but failover takes minutes
- **DNS Failover**: Route 53 health checks → automatic DNS record updates (60-300s TTL)
- **Global Load Balancer**: Anycast or GeoDNS routes users to nearest healthy region
- **Data Synchronization**: Cross-region replication with conflict resolution strategy
- **Failover Testing**: Regular Chaos Kong-style exercises (simulate entire region failure)

```
+-----------------------------------------------------------+
|         MULTI-REGION FAILOVER ARCHITECTURE                 |
+-----------------------------------------------------------+
|                                                             |
|  ACTIVE-ACTIVE MULTI-REGION:                               |
|                                                             |
|  Users --> [Global DNS / Anycast LB]                       |
|                |               |                            |
|         +------+------+ +-----+------+                     |
|         | US-EAST     | | EU-WEST    |                     |
|         | +---------+ | | +---------+|                     |
|         | |App Tier | | | |App Tier ||                     |
|         | +---------+ | | +---------+|                     |
|         | +---------+ | | +---------+|                     |
|         | |DB Primary| | | |DB Primary||                    |
|         | +---------+ | | +---------+|                     |
|         +------+------+ +-----+------+                     |
|                |               |                            |
|         Cross-region async replication                      |
|                                                             |
|  FAILOVER SEQUENCE (active-passive):                       |
|  1. Health check detects us-east failure                   |
|  2. DNS TTL expires (60s) or instant with Anycast          |
|  3. Traffic routes to eu-west                              |
|  4. eu-west replica promoted to primary                    |
|  5. Application scales up to handle full load              |
|  6. Monitor for data consistency                           |
|                                                             |
|  FAILBACK SEQUENCE:                                        |
|  1. us-east recovered and verified                         |
|  2. Re-establish replication from eu-west --> us-east      |
|  3. Wait for full sync (replication lag = 0)               |
|  4. Gradually shift traffic back (10% -> 50% -> 100%)     |
|  5. Restore original primary/secondary roles               |
+-----------------------------------------------------------+
```

| Failover Aspect | Active-Active | Active-Passive | Consideration |
|---|---|---|---|
| **Normal Traffic** | Split across regions | Primary region only | Cost vs utilization |
| **Failover Time** | Near-zero (already serving) | Minutes (DNS + promotion) | RTOs requirement |
| **Data Consistency** | Conflict resolution needed | Simple (single writer) | CAP tradeoff |
| **Cost** | 2x+ (full infra everywhere) | 1.3-1.8x (standby capacity) | Budget constraint |
| **Complexity** | Very high | Medium | Team capability |
| **Testing** | Continuous (both always active) | Requires explicit drills | Verification needed |
| **Failback** | N/A (both always active) | Complex (re-sync + switch) | Often overlooked |

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

class RegionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

class FailoverMode(Enum):
    ACTIVE_ACTIVE = "active_active"
    ACTIVE_PASSIVE = "active_passive"

@dataclass
class Region:
    name: str
    status: RegionStatus = RegionStatus.HEALTHY
    traffic_weight: float = 0.0
    is_primary: bool = False
    replication_lag_ms: float = 0.0
    capacity_percent: float = 100.0

@dataclass
class MultiRegionFailover:
    mode: FailoverMode
    regions: list[Region] = field(default_factory=list)
    dns_ttl_seconds: int = 60
    health_check_interval_seconds: int = 10
    failover_threshold: int = 3  # Consecutive failures before failover
    events: list[str] = field(default_factory=list)

    def add_region(self, region: Region):
        self.regions.append(region)

    def detect_failure(self, region_name: str):
        region = self._get_region(region_name)
        region.status = RegionStatus.FAILED
        self.events.append(f"[DETECTED] {region_name} failed")
        self._execute_failover(region)

    def _execute_failover(self, failed_region: Region):
        healthy = [r for r in self.regions if r.status == RegionStatus.HEALTHY]

        if not healthy:
            self.events.append("[CRITICAL] No healthy regions available!")
            return

        if self.mode == FailoverMode.ACTIVE_ACTIVE:
            failed_region.traffic_weight = 0.0
            total_weight = sum(r.traffic_weight for r in healthy)
            for r in healthy:
                r.traffic_weight = r.traffic_weight / total_weight if total_weight > 0 else 1.0 / len(healthy)
            self.events.append(f"[FAILOVER] Redistributed traffic to {[r.name for r in healthy]}")

        elif self.mode == FailoverMode.ACTIVE_PASSIVE:
            failed_region.traffic_weight = 0.0
            failed_region.is_primary = False
            new_primary = healthy[0]
            new_primary.is_primary = True
            new_primary.traffic_weight = 1.0
            self.events.append(f"[FAILOVER] Promoted {new_primary.name} to primary")
            self.events.append(f"[DNS] Updating records (TTL: {self.dns_ttl_seconds}s)")

    def recover_region(self, region_name: str):
        region = self._get_region(region_name)
        region.status = RegionStatus.RECOVERING
        self.events.append(f"[RECOVERY] {region_name} starting recovery")
        region.status = RegionStatus.HEALTHY

        if self.mode == FailoverMode.ACTIVE_ACTIVE:
            n = len([r for r in self.regions if r.status == RegionStatus.HEALTHY])
            for r in self.regions:
                if r.status == RegionStatus.HEALTHY:
                    r.traffic_weight = 1.0 / n
            self.events.append(f"[RECOVERY] {region_name} added back to rotation")

    def _get_region(self, name: str) -> Region:
        return next(r for r in self.regions if r.name == name)

    def status_report(self):
        print(f"\nMulti-Region Status ({self.mode.value})")
        print("=" * 60)
        for r in self.regions:
            print(f"  {r.name:<15} [{r.status.value:>11}] "
                  f"traffic={r.traffic_weight*100:.0f}% "
                  f"primary={r.is_primary} "
                  f"lag={r.replication_lag_ms:.0f}ms")
        print(f"\nEvents:")
        for e in self.events:
            print(f"  {e}")

# Simulate active-passive failover
system = MultiRegionFailover(FailoverMode.ACTIVE_PASSIVE, dns_ttl_seconds=60)
system.add_region(Region("us-east-1", traffic_weight=1.0, is_primary=True))
system.add_region(Region("eu-west-1", traffic_weight=0.0, capacity_percent=80))
system.add_region(Region("ap-south-1", traffic_weight=0.0, capacity_percent=60))

print("=== Initial State ===")
system.status_report()

print("\n=== After us-east-1 Failure ===")
system.detect_failure("us-east-1")
system.status_report()

print("\n=== After us-east-1 Recovery ===")
system.recover_region("us-east-1")
system.status_report()
```

**AI/ML Application:** Multi-region ML serving requires **model artifact replication** (S3/GCS cross-region), **regional feature caches** (to avoid cross-region feature lookups), and **traffic-weighted model deployments** (canary a new model in one region first). GPU availability varies by region, so failover plans must account for different GPU instance types between regions.

**Real-World Example:** Netflix uses **active-active** across three AWS regions (US-East, US-West, EU-West). Each region is provisioned to handle 100% of traffic. Their **Chaos Kong** exercise regularly simulates an entire region failure by redirecting all traffic to the remaining two regions. This continuous testing ensures their failover actually works in production.

> **Interview Tip:** Walk through the fullsequence: detection → DNS update → traffic shift → database promotion → capacity scaling → verification. Don't forget **failback** — recovering the failed region and restoring normal operations is often harder than the initial failover. Mention that DNS TTL is a critical factor in failover speed.

---
