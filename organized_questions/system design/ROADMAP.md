# 🗺️ System Design Roadmap — Zero to Advanced

> **18 Topics · 825 Questions · All Theory**
> Study these in order. Each stage builds on the previous one.
> Estimated timeline: 2–3 months (2–3 topics per week)

---

## How to Use This Roadmap

- **Stages** go from foundational → intermediate → advanced
- **Within each stage**, topics are grouped — study them together as they reinforce each other
- ✅ Check off topics as you complete them
- 🔢 Numbers in parentheses = total questions available for that topic
- 💡 "Why now" explains the dependency and reasoning

---

## Stage 1 — Core Foundations (Start Here)

> 🎯 **Goal:** Understand the building blocks every system is built on — databases, APIs, and general architecture principles.
> ⏱️ **Time:** Week 1–3

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 1 | [Software Architecture](Software%20Architecture.md) | 85 | The big picture — monolith vs distributed, layered architecture, design principles (SOLID, DRY) |
| 2 | [Databases](Databases.md) | 50 | Every system needs persistent storage — ACID, indexing, normalization, SQL fundamentals |
| 3 | [API Design](API%20Design.md) | 50 | Services communicate via APIs — REST, GraphQL, gRPC, versioning, status codes |

**📌 Checkpoint:** You can sketch a basic 3-tier architecture (client → API → database), explain CRUD operations, REST principles, and database indexing.

---

## Stage 2 — Scalability Essentials

> 🎯 **Goal:** Learn the core techniques for making systems handle more users and data.
> ⏱️ **Time:** Week 4–6

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 4 | [Caching](Caching.md) | 50 | First scalability lever — reduce DB load; Redis, Memcached, cache invalidation strategies |
| 5 | [Load Balancing](Load%20Balancing.md) | 50 | Distribute traffic across servers; round-robin, consistent hashing, health checks |
| 6 | [CAP Theorem](CAP%20Theorem.md) | 15 | Critical theory — Consistency vs Availability vs Partition Tolerance tradeoffs |
| 7 | [NoSQL](NoSQL.md) | 35 | When to choose NoSQL over SQL; document, key-value, column-family, graph databases |

**📌 Checkpoint:** You can explain why a system needs caching (and where), design a load-balanced API tier, and reason about CAP tradeoffs when choosing databases.

---

## Stage 3 — Reliability & Data Architecture

> 🎯 **Goal:** Ensure systems don't go down and data stays consistent.
> ⏱️ **Time:** Week 7–8

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 8 | [Availability & Reliability](Availability%20%26%20Reliability.md) | 30 | SLAs, redundancy, failover, disaster recovery, circuit breakers |
| 9 | [Concurrency](Concurrency.md) | 40 | Race conditions, locks, deadlocks, thread safety — critical for multi-user systems |
| 10 | [Cryptography](Cryptography.md) | 55 | Authentication, encryption (TLS/SSL), hashing, JWT, OAuth — security is non-negotiable |

**📌 Checkpoint:** You can design for 99.9% uptime, explain how to prevent race conditions, and secure API endpoints with proper auth.

---

## Stage 4 — Distributed Architecture Patterns

> 🎯 **Goal:** Learn the architectural patterns used in modern large-scale systems.
> ⏱️ **Time:** Week 9–11

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 11 | [Microservices](Microservices.md) | 60 | Breaking monoliths into services; requires understanding APIs, databases, caching from previous stages |
| 12 | [SOA](SOA.md) | 35 | Service-Oriented Architecture — predecessor to microservices; ESB, service contracts |
| 13 | [Layering & Middleware](Layering%20%26%20Middleware.md) | 35 | Presentation → Business → Data layers; middleware (message queues, API gateways) |
| 14 | [Domain Driven Design](Domain%20Driven%20Design.md) | 40 | Bounded contexts, aggregates, entities, value objects — how to model complex business domains |

**📌 Checkpoint:** You can decompose a monolith into microservices, define bounded contexts, explain event-driven communication, and design an API gateway.

---

## Stage 5 — Infrastructure & Deployment

> 🎯 **Goal:** Understand how systems are deployed and orchestrated in production.
> ⏱️ **Time:** Week 12–13

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 15 | [Docker](Docker.md) | 55 | Containerization — package services with dependencies; Dockerfile, images, volumes |
| 16 | [Kubernetes](Kubernetes.md) | 42 | Orchestrate containers at scale; pods, services, deployments, scaling, self-healing |

**📌 Checkpoint:** You can containerize a microservice, write a Kubernetes deployment, and explain how K8s handles scaling and failover.

---

## Stage 6 — Advanced & Specialized

> 🎯 **Goal:** Round out your knowledge with advanced patterns and niche but powerful topics.
> ⏱️ **Time:** Week 14–15

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 17 | [Reactive Systems](Reactive%20Systems.md) | 32 | Event-driven, resilient, elastic, message-driven — the Reactive Manifesto |
| 18 | [XML](XML.md) | 66 | Data interchange format; SOAP, XSLT, XPath — still used in enterprise/banking systems |

**📌 Checkpoint:** You can compare reactive vs imperative architectures, explain backpressure, and parse/validate XML schemas.

---

## 📊 Summary — Study Order at a Glance

```
STAGE 1 (Foundation)      → Software Architecture → Databases → API Design
     ↓
STAGE 2 (Scalability)     → Caching → Load Balancing → CAP Theorem → NoSQL
     ↓
STAGE 3 (Reliability)     → Availability & Reliability → Concurrency → Cryptography
     ↓
STAGE 4 (Distributed)     → Microservices → SOA → Layering & Middleware → DDD
     ↓
STAGE 5 (Infrastructure)  → Docker → Kubernetes
     ↓
STAGE 6 (Advanced)        → Reactive Systems → XML
```

---

## 🎯 Grouping by Interview Priority

### 🔴 Must-Know (Top 5 — covers 80% of system design interviews)
1. Software Architecture
2. Databases
3. Caching
4. Load Balancing
5. API Design

### 🟡 Important (Next 5 — frequently asked)
6. Microservices
7. CAP Theorem
8. Availability & Reliability
9. Concurrency
10. NoSQL

### 🟢 Good to Know (Next 5 — shows depth)
11. Docker
12. Kubernetes
13. Cryptography
14. Domain Driven Design
15. Layering & Middleware

### ⚪ Niche (Last 3 — role-specific)
16. SOA
17. Reactive Systems
18. XML

---

## 🔄 Prerequisite Map

```
Software Architecture ──────────────────► (foundation for everything)
        │
        ├── Databases ──┬── NoSQL
        │               └── CAP Theorem
        │
        ├── API Design ──── Microservices
        │                       │
        │                       ├── SOA
        │                       ├── Domain Driven Design
        │                       └── Docker ──── Kubernetes
        │
        ├── Caching
        │
        ├── Load Balancing
        │
        └── Layering & Middleware

Availability & Reliability ──── (cross-cutting: applies to all)
Concurrency ──── (cross-cutting: applies to all)
Cryptography ──── (cross-cutting: applies to all)

Microservices + Kubernetes + Reactive Systems = Modern Architecture
```

---

## 🧩 System Design Interview Template

When solving a system design question, apply topics in this order:

```
1. Requirements         → (no specific topic — clarify with interviewer)
2. API Design           → Define endpoints, request/response
3. Data Model           → Databases, NoSQL, CAP Theorem
4. High-Level Design    → Software Architecture, Layering
5. Scalability          → Load Balancing, Caching
6. Reliability          → Availability, Concurrency
7. Security             → Cryptography
8. Deployment           → Docker, Kubernetes
9. Deep Dive            → Microservices, DDD, Reactive Systems
```
