# 35 NoSQL interview questions

> Source: [https://devinterview.io/questions/software-architecture-and-system-design/nosql-interview-questions/](https://devinterview.io/questions/software-architecture-and-system-design/nosql-interview-questions/)
> Scraped: 2026-02-20 00:41
> Total Questions: 35

---

## Table of Contents

1. [NoSQL Fundamentals](#nosql-fundamentals) (7 questions)
2. [Key-Value Stores](#key-value-stores) (4 questions)
3. [Document-Oriented Databases](#document-oriented-databases) (5 questions)
4. [Column-Family Stores](#column-family-stores) (3 questions)
5. [Graph Databases](#graph-databases) (4 questions)
6. [Data Consistency & Replication](#data-consistency-replication) (3 questions)
7. [Scalability & Performance](#scalability-performance) (4 questions)
8. [Querying and Data Manipulation](#querying-and-data-manipulation) (3 questions)
9. [Indexing and Storage Strategies](#indexing-and-storage-strategies) (2 questions)

---

## NoSQL Fundamentals

### 1. What are the different types of NoSQL databases , with an example of each?

**Type:** 📝 Question

**Answer:**

There are **four main types** of NoSQL databases, each optimized for different data models and access patterns: **Key-Value**, **Document**, **Column-Family**, and **Graph**. Each sacrifices relational structure for **horizontal scalability**, **flexible schemas**, and **performance** at specific workloads.

```
  FOUR TYPES OF NoSQL DATABASES:
  
  1. KEY-VALUE STORE
  +--------+----------+
  | Key    | Value    |
  +--------+----------+
  | user:1 | {blob}   |
  | sess:X | {blob}   |
  +--------+----------+
  Examples: Redis, DynamoDB, Memcached
  
  2. DOCUMENT STORE
  +----------------------------------+
  | { "_id": "user1",               |
  |   "name": "Alice",              |
  |   "orders": [                   |
  |     {"item": "Book", "qty": 2}  |
  |   ]                             |
  | }                               |
  +----------------------------------+
  Examples: MongoDB, CouchDB, Firestore
  
  3. COLUMN-FAMILY STORE
  Row Key | col1:name | col2:email | col3:age
  --------+-----------+------------+---------
  user1   | Alice     | a@b.com    | 30
  user2   | Bob       |            | 25
  (sparse: missing columns OK)
  Examples: Cassandra, HBase, ScyllaDB
  
  4. GRAPH DATABASE
      (Alice)---[FOLLOWS]--->(Bob)
         |                     |
    [LIKES]               [WROTE]
         |                     |
      (Post1)              (Post2)
  Examples: Neo4j, Amazon Neptune, JanusGraph
```

| Type | Data Model | Best For | Query Pattern | Example |
|------|-----------|---------|---------------|---------|
| **Key-Value** | Hash map | Caching, sessions | Get/Put by key | Redis, DynamoDB |
| **Document** | JSON/BSON docs | Content mgmt, catalogs | Query by fields | MongoDB, CouchDB |
| **Column-Family** | Wide columns | Time-series, analytics | Range scans by row | Cassandra, HBase |
| **Graph** | Nodes + edges | Social, recommendations | Traversals, paths | Neo4j, Neptune |

```python
# Example operations across all 4 NoSQL types

# 1. KEY-VALUE (Redis-like)
class KeyValueStore:
    def __init__(self):
        self.store = {}
    def get(self, key): return self.store.get(key)
    def set(self, key, value): self.store[key] = value
    def delete(self, key): self.store.pop(key, None)

# 2. DOCUMENT (MongoDB-like)
class DocumentStore:
    def __init__(self):
        self.collections = {}
    def insert(self, collection, doc):
        if collection not in self.collections:
            self.collections[collection] = []
        self.collections[collection].append(doc)
    def find(self, collection, query):
        return [d for d in self.collections.get(collection, [])
                if all(d.get(k) == v for k, v in query.items())]

# 3. COLUMN-FAMILY (Cassandra-like)
class ColumnFamilyStore:
    def __init__(self):
        self.rows = {}  # row_key -> {col_family: {col: value}}
    def put(self, row_key, col_family, column, value):
        self.rows.setdefault(row_key, {}).setdefault(col_family, {})[column] = value
    def get(self, row_key, col_family):
        return self.rows.get(row_key, {}).get(col_family, {})

# 4. GRAPH (Neo4j-like)
class GraphDB:
    def __init__(self):
        self.nodes = {}
        self.edges = []
    def add_node(self, node_id, properties):
        self.nodes[node_id] = properties
    def add_edge(self, from_id, to_id, relation):
        self.edges.append({"from": from_id, "to": to_id, "type": relation})
    def traverse(self, start, relation):
        return [e["to"] for e in self.edges if e["from"] == start and e["type"] == relation]
```

**AI/ML Application:**
- **Key-Value (Redis):** Feature stores cache ML features for real-time inference. Redis serves precomputed embeddings at sub-millisecond latency.
- **Document (MongoDB):** ML experiment tracking (MLflow) stores experiment metadata, parameters, and metrics as JSON documents with flexible schemas.
- **Column-Family (Cassandra):** Time-series ML training metrics, IoT sensor data for anomaly detection models.
- **Graph (Neo4j):** Knowledge graphs for NLP, social network analysis for recommendation engines, fraud detection via graph patterns.

**Real-World Example:**
Netflix uses all four: Redis for session caching, Cassandra for user activity data (column-family), MongoDB-like document stores for content metadata, and a custom graph database for content recommendations. Uber uses Redis for geospatial caching, Cassandra for trip data, and a graph DB for route optimization.

> **Interview Tip:** "Name all four types with one example each: Key-Value (Redis), Document (MongoDB), Column-Family (Cassandra), Graph (Neo4j). Then match to use cases: KV = caching/sessions, Document = content/catalogs, Column = time-series/analytics, Graph = relationships/recommendations. Show you understand WHY each type exists, not just WHAT they are."

---

### 2. Explain eventual consistency and its role in NoSQL databases .

**Type:** 📝 Question

**Answer:**

**Eventual consistency** guarantees that if no new writes occur, all replicas will **converge to the same value** over time. Most NoSQL databases use eventual consistency as their **default** because it enables **high availability** and **low latency** by allowing reads from any replica without waiting for full synchronization. This is the **AP** side of the CAP theorem applied in practice.

```
  EVENTUAL CONSISTENCY IN NoSQL:
  
  Write at t=0:  SET user:1 = "Alice"
  
  Time:   t=0      t=1      t=2      t=3
  Node A: "Alice"  "Alice"  "Alice"  "Alice"
  Node B: "Bob"    "Alice"  "Alice"  "Alice"  (updated via replication)
  Node C: "Bob"    "Bob"    "Alice"  "Alice"  (updated later)
          |        |                 |
          Write    Repl.             All converged!
          here     starts            
                   |<-- inconsistency window -->|
  
  DYNAMODB EXAMPLE:
  +--------+  PutItem   +--------+
  | Client |----------->| Leader |
  +--------+            +--------+
                             |
                      Async replication
                         /       \
                   +--------+ +--------+
                   |Replica1| |Replica2|
                   +--------+ +--------+
  
  GetItem(ConsistentRead=false) --> May hit any replica (fast, maybe stale)
  GetItem(ConsistentRead=true)  --> Always from Leader (slower, fresh)
  
  CASSANDRA CONSISTENCY LEVELS:
  +----------+--------+---------------------------+
  | Level    | Nodes  | Trade-off                 |
  +----------+--------+---------------------------+
  | ONE      | 1      | Fastest, most stale risk  |
  | QUORUM   | N/2+1  | Balanced                  |
  | ALL      | N      | Slowest, strongest        |
  +----------+--------+---------------------------+
```

| Aspect | Eventual Consistency | Strong Consistency |
|--------|---------------------|-------------------|
| **Read latency** | Low (any replica) | Higher (leader/quorum) |
| **Write latency** | Low (ack from one) | Higher (ack from majority) |
| **Staleness** | Possible (ms to sec) | Never |
| **Availability** | High (AP) | Lower during partitions (CP) |
| **Use case** | Social feeds, analytics | Banking, inventory |

```python
# Eventual consistency with anti-entropy repair
import time
import threading
from collections import defaultdict

class EventuallyConsistentNoSQL:
    """NoSQL store with eventual consistency and repair mechanisms."""

    def __init__(self, num_replicas=3, replication_delay=0.5):
        self.replicas = {f"r{i}": {} for i in range(num_replicas)}
        self.primary = "r0"
        self.delay = replication_delay
        self.read_repair_enabled = True

    def write(self, key, value):
        """Write to primary, async replicate."""
        ts = time.time()
        self.replicas[self.primary][key] = {"value": value, "ts": ts}
        # Async background replication
        def replicate():
            time.sleep(self.delay)
            for rid in self.replicas:
                if rid != self.primary:
                    self.replicas[rid][key] = {"value": value, "ts": ts}
        threading.Thread(target=replicate, daemon=True).start()
        return {"status": "ok"}

    def read(self, key, consistency="eventual"):
        """Read with optional read-repair."""
        if consistency == "strong":
            return self._read_quorum(key)
        # Eventual: read from any replica
        import random
        replica = random.choice(list(self.replicas.keys()))
        entry = self.replicas[replica].get(key)
        if self.read_repair_enabled:
            self._trigger_read_repair(key)
        return {"value": entry["value"] if entry else None, "from": replica}

    def _read_quorum(self, key):
        responses = []
        for rid, store in self.replicas.items():
            if key in store:
                responses.append(store[key])
        if responses:
            latest = max(responses, key=lambda x: x["ts"])
            return {"value": latest["value"], "consistent": True}
        return {"value": None}

    def _trigger_read_repair(self, key):
        """Background repair: sync replicas to latest version."""
        entries = [(rid, store.get(key)) for rid, store in self.replicas.items() if key in store]
        if entries:
            latest = max(entries, key=lambda x: x[1]["ts"])
            for rid, _ in entries:
                self.replicas[rid][key] = latest[1]
```

**AI/ML Application:**
- **Feature store eventual consistency:** ML feature stores serve features from local replicas (eventual). During the inconsistency window, some predictions use slightly stale features — acceptable for recommendations but not for fraud detection.
- **Model metric dashboards:** ML monitoring metrics (accuracy, latency, throughput) use eventual consistency — seeing metrics delayed by 1-2 seconds doesn't affect decision-making.
- **Training data ingestion:** Clickstream data ingested into Cassandra for ML training is eventually consistent — training doesn't need real-time latest data.

**Real-World Example:**
Amazon DynamoDB's default reads are eventually consistent (~4ms) vs strongly consistent (~8ms, double the cost). Cassandra's default `ONE` consistency provides fastest reads with eventual guarantees. MongoDB replica set reads from secondaries are eventually consistent. DNS is the most familiar eventually consistent system — TTL-based cache expiration.

> **Interview Tip:** "Eventual consistency doesn't mean 'unreliable.' It means there's a brief window (usually milliseconds) where replicas may return stale data. It's crucial for availability: AP systems like Cassandra and DynamoDB use it to keep 99.99% uptime. The inconsistency window is tunable. For NoSQL, always mention the specific tuning: Cassandra consistency levels, DynamoDB ConsistentRead flag."

---

### 3. How is data modeling in NoSQL databases distinct from that in relational databases ?

**Type:** 📝 Question

**Answer:**

NoSQL data modeling is **query-driven** (design around access patterns) while relational modeling is **data-driven** (normalize data, then query flexibly). NoSQL uses **denormalization**, **embedding**, and **duplication** to optimize read performance, while relational uses **normalization** and **joins** to minimize redundancy.

```
  RELATIONAL (Normalized):
  
  Users table:        Orders table:         Products table:
  +----+-------+      +----+-------+-----+  +----+--------+
  | id | name  |      | id | uid   | pid |  | id | name   |
  +----+-------+      +----+-------+-----+  +----+--------+
  | 1  | Alice |      | 1  | 1     | 10  |  | 10 | Book   |
  | 2  | Bob   |      | 2  | 1     | 20  |  | 20 | Phone  |
  +----+-------+      +----+-------+-----+  +----+--------+
  
  Query: SELECT * FROM users 
         JOIN orders ON users.id = orders.uid
         JOIN products ON orders.pid = products.id
  (3 tables, 2 joins)
  
  NoSQL (Denormalized - MongoDB):
  
  {
    "_id": "user1",
    "name": "Alice",
    "orders": [
      {"id": 1, "product": {"id": 10, "name": "Book"}},
      {"id": 2, "product": {"id": 20, "name": "Phone"}}
    ]
  }
  
  Query: db.users.findOne({_id: "user1"})
  (1 read, 0 joins, all data embedded!)

  MODELING APPROACHES:
  +-----------------+--------------------+--------------------+
  | Principle       | Relational         | NoSQL              |
  +-----------------+--------------------+--------------------+
  | Design by       | Data structure     | Query patterns     |
  | Normalization   | Yes (3NF, BCNF)   | No (denormalize!)  |
  | Joins           | Heavy use          | Avoid at all costs |
  | Duplication     | Minimized          | Embraced           |
  | Schema          | Fixed (ALTER)      | Flexible/schemaless|
  | Relationships   | Foreign keys       | Embedded docs/refs |
  +-----------------+--------------------+--------------------+
```

| Aspect | Relational | NoSQL |
|--------|-----------|-------|
| **Design approach** | Normalize first, query later | Know queries first, model data around them |
| **Redundancy** | Eliminated via normalization | Embraced for read performance |
| **Joins** | Native, optimized | Avoided (embed/denormalize instead) |
| **Schema** | Rigid, enforced | Flexible, schema-on-read |
| **Relationships** | Foreign keys + joins | Embedding, referencing, or graph edges |
| **Write vs Read** | Write-optimized (no duplication) | Read-optimized (pre-joined data) |
| **Update cost** | Low (single source of truth) | Higher (update all copies) |

```python
# Relational vs NoSQL data modeling comparison

# RELATIONAL approach (normalized)
relational_schema = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"}
    ],
    "orders": [
        {"id": 1, "user_id": 1, "product_id": 10, "qty": 2},
        {"id": 2, "user_id": 1, "product_id": 20, "qty": 1}
    ],
    "products": [
        {"id": 10, "name": "Book", "price": 15.99},
        {"id": 20, "name": "Phone", "price": 699.99}
    ]
}

def relational_query(user_id):
    """Simulates JOIN across 3 tables."""
    user = next(u for u in relational_schema["users"] if u["id"] == user_id)
    orders = [o for o in relational_schema["orders"] if o["user_id"] == user_id]
    result = {**user, "orders": []}
    for order in orders:
        product = next(p for p in relational_schema["products"] if p["id"] == order["product_id"])
        result["orders"].append({**order, "product": product})
    return result  # Requires 3 table scans + joins

# NoSQL approach (denormalized)
nosql_document = {
    "_id": "user_1",
    "name": "Alice",
    "email": "alice@example.com",
    "orders": [
        {"id": 1, "product": {"id": 10, "name": "Book", "price": 15.99}, "qty": 2},
        {"id": 2, "product": {"id": 20, "name": "Phone", "price": 699.99}, "qty": 1}
    ]
}

def nosql_query(user_id):
    """Single document read - no joins needed."""
    return nosql_document  # One read, all data embedded!
```

**AI/ML Application:**
- **Feature store modeling:** ML feature stores use NoSQL modeling principles — denormalize features per entity (user, item) for single-key lookup during inference. Avoid joins at serving time.
- **Training data denormalization:** ML training pipelines preprocess and denormalize data into wide feature tables, similar to NoSQL document modeling — each training example is a self-contained document.
- **Embedding storage:** Vector databases model data like NoSQL documents — each entity is a document with metadata + embedding vector. No joins, single-lookup retrieval.

**Real-World Example:**
MongoDB's documentation advises: "Model your data to match your most common queries." eBay migrated from relational joins to Cassandra denormalized tables — reducing read latency from 50ms to 2ms for product listings. Instagram stores user feeds as denormalized documents in Cassandra — each feed entry has all data needed for display (user name, photo URL, likes count) embedded in one row.

> **Interview Tip:** "The #1 difference: relational is data-driven (normalize, then query), NoSQL is query-driven (know your queries, then model). In NoSQL, start with 'What queries does my app need?' and design your data model to answer each query in a single read. This means data duplication is intentional and expected — it's the price of fast reads."

---

### 4. What advantages do NoSQL databases offer for managing large volumes of data?

**Type:** 📝 Question

**Answer:**

NoSQL databases excel at **large-scale data** through **horizontal scalability** (add commodity servers), **flexible schemas** (handle unstructured/semi-structured data), **high write throughput** (append-optimized storage engines), and **distributed architecture** (automatic sharding). They handle **petabytes** of data that would be impractical or extremely expensive with traditional RDBMS.

```
  SCALABILITY: RDBMS vs NoSQL
  
  RDBMS (Vertical Scaling):
  +------------------+
  | Single Server    |   Cost: $$$$$
  | 256 GB RAM       |   Scaling: Limited
  | 64 cores         |   Single point of failure
  | 100 TB SSD       |
  +------------------+
  
  NoSQL (Horizontal Scaling):
  +------+ +------+ +------+ +------+ +------+
  |Node 1| |Node 2| |Node 3| |Node 4| |Node 5|
  | 32GB | | 32GB | | 32GB | | 32GB | | 32GB |
  +------+ +------+ +------+ +------+ +------+
  Cost: $$  Scaling: Add more nodes  No SPOF
  
  AUTO-SHARDING:
  +-------------------------------------------+
  |              Data: 100 TB                 |
  +-------------------------------------------+
  | Hash(key) mod 5:                          |
  |  0 -> Node 1 (20TB)                      |
  |  1 -> Node 2 (20TB)                      |
  |  2 -> Node 3 (20TB)                      |
  |  3 -> Node 4 (20TB)                      |
  |  4 -> Node 5 (20TB)                      |
  +-------------------------------------------+
  Need more capacity? Add Node 6, auto-rebalance!

  WRITE PERFORMANCE (LSM-Tree):
  Write --> Memtable (RAM) --> Flush --> SSTable (Disk)
  |                                         |
  +-- Sequential writes only (fast!)        |
                                            |
  vs RDBMS: Random I/O for B-tree updates (slow at scale)
```

| Advantage | How NoSQL Achieves It | RDBMS Limitation |
|-----------|----------------------|-----------------|
| **Horizontal scaling** | Auto-sharding across nodes | Vertical only (mostly) |
| **Schema flexibility** | Schema-on-read, dynamic fields | ALTER TABLE is expensive |
| **Write throughput** | LSM-trees, append-only logs | B-tree random I/O bottleneck |
| **Data variety** | JSON, binary, graph, columns | Fixed tabular rows |
| **Cost at scale** | Commodity hardware | Expensive enterprise servers |
| **Geo-distribution** | Built-in multi-region | Complex replication setup |

```python
# Demonstrating NoSQL advantages for large data
import hashlib
import time

class ShardedNoSQL:
    """Auto-sharding NoSQL store for horizontal scalability."""

    def __init__(self, num_shards=4):
        self.shards = {i: {} for i in range(num_shards)}
        self.num_shards = num_shards

    def _get_shard(self, key):
        """Consistent hashing to determine shard."""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def write(self, key, value):
        shard_id = self._get_shard(key)
        self.shards[shard_id][key] = value
        return {"shard": shard_id, "status": "ok"}

    def read(self, key):
        shard_id = self._get_shard(key)
        return self.shards[shard_id].get(key)

    def add_shard(self):
        """Scale out: add new shard and rebalance."""
        new_id = self.num_shards
        self.shards[new_id] = {}
        self.num_shards += 1
        self._rebalance()
        return f"Added shard {new_id}, now {self.num_shards} shards"

    def _rebalance(self):
        all_data = {}
        for shard in self.shards.values():
            all_data.update(shard)
            shard.clear()
        for key, value in all_data.items():
            shard_id = self._get_shard(key)
            self.shards[shard_id][key] = value

    def stats(self):
        return {f"shard_{i}": len(data) for i, data in self.shards.items()}
```

**AI/ML Application:**
- **Training data at scale:** ML models trained on billions of examples (GPT, recommendation models) store training data in NoSQL — Cassandra handles TB of clickstream, HBase stores web crawl data.
- **Feature storage:** Feature stores like Feast use Redis/DynamoDB to serve billions of feature vectors. Horizontal scaling lets you add more feature entities without schema changes.
- **Embedding databases:** Vector databases (Milvus, Pinecone) use NoSQL-like sharding to store and query billions of high-dimensional embeddings across distributed nodes.

**Real-World Example:**
Apple runs 75,000+ Cassandra nodes storing 10+ PB. Netflix stores billions of viewing events daily in Cassandra. Discord migrated from MongoDB to Cassandra to handle trillions of messages — the horizontal scaling and write performance were critical. Uber's Schemaless (built on MySQL shards) handles 1M+ writes/sec for trip data.

> **Interview Tip:** "Lead with horizontal scalability — it's the #1 advantage. NoSQL databases shard automatically across commodity servers, while RDBMS typically scales vertically (bigger servers). Then mention flexible schemas (handle changing data structures without migrations), high write throughput (LSM-trees), and built-in geo-distribution. Always tie to specific numbers: 'Cassandra can handle 1M writes/sec across 100 nodes.'"

---

### 5. When would you choose a NoSQL database over a relational database ?

**Type:** 📝 Question

**Answer:**

Choose NoSQL when: **data is unstructured/semi-structured**, **schema evolves frequently**, **horizontal scaling is required**, **high write throughput matters**, or **specific access patterns** (key lookup, graph traversal, time-series) dominate. Choose relational when: **complex queries with joins**, **ACID transactions**, **data integrity constraints**, or **ad-hoc reporting** are priorities.

```
  DECISION FRAMEWORK:
  
  Does your data have complex relationships
  with many-to-many JOINs?
  |
  +-- Yes --> RDBMS (PostgreSQL, MySQL)
  |           or Graph DB (Neo4j) for traversals
  |
  +-- No
       |
       Do you need ACID transactions across entities?
       |
       +-- Yes --> RDBMS or NewSQL (CockroachDB, Spanner)
       |
       +-- No
            |
            What's your primary access pattern?
            |
            +-- Key lookup          --> Key-Value (Redis, DynamoDB)
            +-- Flexible documents  --> Document (MongoDB)
            +-- Time-series/wide    --> Column-Family (Cassandra)
            +-- Relationships       --> Graph (Neo4j)
            +-- Full-text search    --> Search engine (Elasticsearch)

  WHEN TO USE EACH:
  +----------------+---------------------------+---------------------------+
  | Choose...      | When You Need             | NOT When You Need         |
  +----------------+---------------------------+---------------------------+
  | RDBMS          | ACID, complex JOINs,      | Massive scale, flexible   |
  |                | strong schema, reporting   | schema, high write volume |
  +----------------+---------------------------+---------------------------+
  | Key-Value      | Simple get/set, caching,  | Complex queries, ranges,  |
  |                | sessions, sub-ms latency  | relationships             |
  +----------------+---------------------------+---------------------------+
  | Document       | Flexible schema, nested   | Complex JOINs, strict     |
  |                | objects, content mgmt     | schema enforcement        |
  +----------------+---------------------------+---------------------------+
  | Column-Family  | Time-series, high write   | Ad-hoc queries, complex   |
  |                | throughput, wide rows     | transactions              |
  +----------------+---------------------------+---------------------------+
  | Graph          | Relationship traversals,  | Simple CRUD, bulk writes, |
  |                | social networks, fraud    | tabular data              |
  +----------------+---------------------------+---------------------------+
```

| Scenario | Best Choice | Why |
|----------|------------|-----|
| **E-commerce catalog** | Document (MongoDB) | Varied product attributes, flexible schema |
| **Shopping cart** | Key-Value (DynamoDB) | Simple put/get by user_id, high availability |
| **Banking ledger** | RDBMS (PostgreSQL) | ACID transactions, double-entry accounting |
| **Social feed** | Column-Family (Cassandra) | High write volume, time-ordered, denormalized |
| **Friend recommendations** | Graph (Neo4j) | Traverse friend-of-friend relationships |
| **Session store** | Key-Value (Redis) | Sub-ms access, TTL expiration |
| **IoT sensor data** | Column-Family (Cassandra) | Time-series, high write volume |

```python
# Database selection advisor
class DatabaseSelector:
    """Recommends database type based on requirements."""

    def recommend(self, requirements):
        scores = {"rdbms": 0, "key_value": 0, "document": 0,
                  "column_family": 0, "graph": 0}

        if requirements.get("acid_transactions"):
            scores["rdbms"] += 3
        if requirements.get("complex_joins"):
            scores["rdbms"] += 2
        if requirements.get("flexible_schema"):
            scores["document"] += 3
            scores["key_value"] += 1
        if requirements.get("high_write_throughput"):
            scores["column_family"] += 3
            scores["key_value"] += 2
        if requirements.get("horizontal_scale"):
            scores["column_family"] += 2
            scores["document"] += 2
            scores["key_value"] += 2
        if requirements.get("relationship_queries"):
            scores["graph"] += 3
        if requirements.get("simple_key_lookup"):
            scores["key_value"] += 3
        if requirements.get("time_series"):
            scores["column_family"] += 3
        if requirements.get("nested_documents"):
            scores["document"] += 2

        best = max(scores, key=scores.get)
        return {"recommendation": best, "scores": scores}

# Usage
selector = DatabaseSelector()
result = selector.recommend({
    "flexible_schema": True,
    "horizontal_scale": True,
    "high_write_throughput": True,
    "time_series": True
})
# Result: column_family (Cassandra) with highest score
```

**AI/ML Application:**
- **ML metadata store:** Choose Document DB (MongoDB) for experiment tracking — experiments have varied parameters, metrics, and artifacts that don't fit a fixed schema.
- **Real-time features:** Choose Key-Value (Redis) for serving ML features at inference time — single-key lookup with sub-ms latency is the primary access pattern.
- **Knowledge graphs:** Choose Graph DB (Neo4j) for relationship-heavy ML applications — recommendation engines, entity resolution, drug discovery graphs.

**Real-World Example:**
Airbnb uses both: PostgreSQL for bookings (ACID transactions for payments) and Elasticsearch for search. Twitter migrated from MySQL to Manhattan (custom NoSQL) for timeline storage — the write volume exceeded what RDBMS could handle. Uber uses PostgreSQL for trip billing (ACID) and Cassandra for driver location tracking (high write throughput, geo-distribution).

> **Interview Tip:** "Never say 'always use NoSQL' or 'always use SQL.' The correct answer starts with: 'It depends on the access patterns.' Then walk through: (1) Do I need transactions? → SQL. (2) Is my schema flexible? → Document. (3) Do I need massive write throughput? → Column-family. (4) Is my data relationship-heavy? → Graph. (5) Just key-value lookups? → KV store. Many real systems use polyglot persistence — multiple databases for different needs."

---

### 6. Describe the various consistency models in NoSQL databases and how they handle transactions and conflict resolution .

**Type:** 📝 Question

**Answer:**

NoSQL databases offer a **spectrum of consistency models**: **strong (linearizable)**, **causal**, **session**, **read-your-writes**, and **eventual**. Transactions range from **single-document ACID** (MongoDB) to **lightweight transactions** (Cassandra LWT) to **no transactions** (basic KV stores). Conflict resolution uses **last-write-wins (LWW)**, **vector clocks**, **CRDTs**, or **application-level merge**.

```
  CONSISTENCY SPECTRUM IN NoSQL:
  
  Strong                                          Eventual
  |-----|-----------|-----------|-----------|---------|
  |     |           |           |           |         |
  Linear Sequential  Causal     Session     Eventual
  izable             
  
  MongoDB   CockroachDB  MongoDB    DynamoDB    Cassandra
  (single)  (multi-row)  (causal    (read-your  (ONE)
                          sessions)  -writes)

  CONFLICT RESOLUTION STRATEGIES:
  +------------------+----------------------------------+
  | Strategy         | How It Works                     |
  +------------------+----------------------------------+
  | Last-Write-Wins  | Timestamp comparison; latest     |
  | (LWW)            | write wins. Simple but loses     |
  |                  | concurrent writes.               |
  +------------------+----------------------------------+
  | Vector Clocks    | Track causal history per node.   |
  |                  | Detect concurrent writes.        |
  |                  | App resolves.                    |
  +------------------+----------------------------------+
  | CRDTs            | Mathematically guaranteed merge.  |
  |                  | No conflicts possible.           |
  |                  | (counters, sets, registers)      |
  +------------------+----------------------------------+
  | App-Level Merge  | Application logic decides which  |
  |                  | version wins (e.g., merge carts) |
  +------------------+----------------------------------+

  TRANSACTION SUPPORT:
  +-------------------+-------------------------+
  | Database          | Transaction Support     |
  +-------------------+-------------------------+
  | MongoDB 4.0+      | Multi-document ACID     |
  | DynamoDB          | TransactWriteItems (25) |
  | Cassandra         | Lightweight (Paxos LWT) |
  | Redis             | MULTI/EXEC (single-node)|
  | CockroachDB       | Full distributed ACID   |
  +-------------------+-------------------------+
```

| Model | Guarantee | NoSQL Example | Latency |
|-------|----------|---------------|---------|
| **Linearizable** | Latest write visible immediately | MongoDB (single-doc) | Highest |
| **Causal** | Cause-before-effect ordering | MongoDB causal sessions | Medium |
| **Session** | Client sees own writes | DynamoDB sessions | Medium |
| **Read-your-writes** | Writer reads own writes | Cassandra LOCAL_QUORUM | Low-Medium |
| **Eventual** | All replicas converge eventually | Cassandra ONE | Lowest |

```python
# NoSQL consistency models and conflict resolution
import time

# Last-Write-Wins conflict resolution
class LWWRegister:
    """Last-Write-Wins register for conflict resolution."""
    def __init__(self):
        self.value = None
        self.timestamp = 0

    def write(self, value):
        ts = time.time()
        if ts > self.timestamp:
            self.value = value
            self.timestamp = ts
        return self.value

    def merge(self, other):
        """LWW merge: highest timestamp wins."""
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp

# Vector clock for detecting concurrent writes
class VectorClock:
    """Tracks causality between distributed events."""
    def __init__(self, node_id, num_nodes):
        self.node_id = node_id
        self.clock = {f"n{i}": 0 for i in range(num_nodes)}

    def increment(self):
        self.clock[self.node_id] += 1
        return dict(self.clock)

    def merge(self, other_clock):
        for node, ts in other_clock.items():
            self.clock[node] = max(self.clock.get(node, 0), ts)
        self.increment()

    def is_concurrent(self, other_clock):
        """True if neither clock dominates the other."""
        self_gte = all(self.clock.get(n, 0) >= t for n, t in other_clock.items())
        other_gte = all(t >= self.clock.get(n, 0) for n, t in other_clock.items())
        return not self_gte and not other_gte

# MongoDB-like multi-document transaction
class MongoTransaction:
    def __init__(self, store):
        self.store = store
        self.writes = {}
        self.committed = False

    def insert(self, collection, doc):
        self.writes.setdefault(collection, []).append(doc)

    def commit(self):
        """Atomic: all or nothing."""
        for collection, docs in self.writes.items():
            self.store.setdefault(collection, []).extend(docs)
        self.committed = True
        return {"status": "committed"}

    def abort(self):
        self.writes.clear()
        return {"status": "aborted"}
```

**AI/ML Application:**
- **Experiment tracking consistency:** MLflow uses MongoDB for experiment metadata. Multi-document transactions ensure that experiment parameters + metrics + artifacts are stored atomically.
- **Feature store versioning:** Feature stores need versioned features with causal consistency — a model must see features from the same time window, not a mix of old and new.
- **Model A/B test results:** A/B test results stored in Cassandra use LWW for metric aggregation — the latest aggregated result wins.

**Real-World Example:**
DynamoDB uses LWW with timestamps for conflict resolution in Global Tables. Amazon's original Dynamo used vector clocks and surfaced conflicts to the application (shopping cart merge). MongoDB 4.0+ supports multi-document ACID transactions with snapshot isolation. Cassandra's LWT (Lightweight Transactions) uses Paxos for compare-and-set operations — useful for unique constraints.

> **Interview Tip:** "Cover the spectrum: eventual (Cassandra ONE), session (DynamoDB), causal (MongoDB sessions), strong (MongoDB single-doc). For conflict resolution, name three strategies: LWW (simplest, may lose data), vector clocks (detect conflicts, app resolves), CRDTs (no conflicts by design). For transactions, mention MongoDB's multi-document ACID and Cassandra's LWT as real-world examples."

---

### 7. List some NoSQL databases and the primary use cases they address.

**Type:** 📝 Question

**Answer:**

Each NoSQL database is optimized for specific **access patterns**, **scale requirements**, and **data models**. Choosing the right one depends on your workload characteristics.

```
  NoSQL DATABASE LANDSCAPE:
  
  KEY-VALUE:
  +----------+--------------------------------------+
  | Redis    | Caching, sessions, leaderboards,     |
  |          | pub/sub, rate limiting               |
  +----------+--------------------------------------+
  | DynamoDB | Serverless apps, gaming, IoT,        |
  |          | e-commerce, single-digit ms latency  |
  +----------+--------------------------------------+
  | Memcached| Simple caching (no persistence)      |
  +----------+--------------------------------------+
  
  DOCUMENT:
  +----------+--------------------------------------+
  | MongoDB  | Content mgmt, catalogs, user         |
  |          | profiles, mobile apps                |
  +----------+--------------------------------------+
  | CouchDB  | Offline-first mobile, sync/replicate |
  +----------+--------------------------------------+
  | Firestore| Real-time apps, serverless, mobile   |
  +----------+--------------------------------------+
  
  COLUMN-FAMILY:
  +----------+--------------------------------------+
  | Cassandra| Time-series, messaging, IoT,         |
  |          | social feeds, high write volume      |
  +----------+--------------------------------------+
  | HBase    | Hadoop ecosystem, big data analytics,|
  |          | sequential large scans               |
  +----------+--------------------------------------+
  | ScyllaDB | Drop-in Cassandra replacement,       |
  |          | C++ performance, lower latency       |
  +----------+--------------------------------------+
  
  GRAPH:
  +----------+--------------------------------------+
  | Neo4j    | Social networks, fraud detection,    |
  |          | knowledge graphs, recommendations    |
  +----------+--------------------------------------+
  | Neptune  | Managed graph, identity graphs,      |
  |          | network/IT operations                |
  +----------+--------------------------------------+
  
  SEARCH:
  +----------+--------------------------------------+
  | Elastic  | Full-text search, log analytics,     |
  | search   | observability (ELK stack)            |
  +----------+--------------------------------------+
```

| Database | Type | Key Feature | Scale | Best For |
|----------|------|------------|-------|---------|
| **Redis** | Key-Value | In-memory, sub-ms | TB | Caching, sessions |
| **DynamoDB** | Key-Value | Serverless, auto-scale | Unlimited | Web/mobile backends |
| **MongoDB** | Document | Flexible schema, rich queries | PB | Content management |
| **Cassandra** | Column-Family | Write throughput, multi-DC | PB | Time-series, messaging |
| **HBase** | Column-Family | Hadoop integration, scans | PB | Big data analytics |
| **Neo4j** | Graph | Cypher query, traversals | TB | Social, fraud |
| **Elasticsearch** | Search | Full-text, aggregations | PB | Search, logging |
| **ScyllaDB** | Column-Family | C++ Cassandra-compatible | PB | Low-latency Cassandra |

```python
# NoSQL database selector with real-world examples
NOSQL_DB_GUIDE = {
    "redis": {
        "type": "Key-Value",
        "strengths": ["Sub-ms latency", "Rich data structures", "Pub/Sub"],
        "use_cases": ["Caching", "Session store", "Leaderboards", "Rate limiting"],
        "users": ["Twitter (timeline cache)", "GitHub (job queues)", "StackOverflow (caching)"],
        "max_data": "~100GB per node (memory-bound)",
        "consistency": "Strong (single-node), Eventual (cluster)"
    },
    "dynamodb": {
        "type": "Key-Value/Document",
        "strengths": ["Serverless", "Auto-scaling", "Single-digit ms"],
        "use_cases": ["E-commerce", "Gaming", "IoT", "Serverless apps"],
        "users": ["Amazon.com", "Lyft", "Airbnb", "Samsung"],
        "max_data": "Unlimited (managed)",
        "consistency": "Eventual (default), Strong (per-read option)"
    },
    "mongodb": {
        "type": "Document",
        "strengths": ["Rich queries", "Aggregation pipeline", "Change streams"],
        "use_cases": ["Content mgmt", "User profiles", "Product catalogs"],
        "users": ["eBay", "Forbes", "Adobe", "Toyota"],
        "max_data": "PB (sharded clusters)",
        "consistency": "Strong (single-doc), Eventual (replicas)"
    },
    "cassandra": {
        "type": "Column-Family",
        "strengths": ["Write throughput", "Multi-DC", "Linear scalability"],
        "use_cases": ["Time-series", "Messaging", "IoT", "Social feeds"],
        "users": ["Apple (75K+ nodes)", "Netflix", "Discord", "Instagram"],
        "max_data": "PB (linear scale)",
        "consistency": "Tunable (ONE to ALL per-query)"
    },
    "neo4j": {
        "type": "Graph",
        "strengths": ["Relationship traversals", "Cypher query language", "ACID"],
        "use_cases": ["Social networks", "Fraud detection", "Recommendations"],
        "users": ["NASA", "Walmart", "eBay", "UBS"],
        "max_data": "TB (single cluster)",
        "consistency": "Strong (single master), Causal (cluster)"
    }
}

def recommend_db(primary_need):
    mapping = {
        "caching": "redis", "sessions": "redis",
        "serverless": "dynamodb", "e-commerce": "dynamodb",
        "content": "mongodb", "catalogs": "mongodb",
        "time_series": "cassandra", "messaging": "cassandra",
        "social_graph": "neo4j", "fraud_detection": "neo4j"
    }
    db = mapping.get(primary_need)
    return NOSQL_DB_GUIDE.get(db, "Unknown need") if db else "No match"
```

**AI/ML Application:**
- **Redis:** ML feature serving (sub-ms latency), model prediction caching, real-time feature computation (sorted sets for top-K).
- **MongoDB:** ML experiment tracking (MLflow, Weights & Biases), model metadata storage, flexible schema for varying model architectures.
- **Cassandra:** ML training data ingestion (high write throughput), user behavior event streams for recommendation training.
- **Neo4j:** Knowledge graph construction for NLP, customer 360 graphs for personalization, molecular graphs for drug discovery.
- **Elasticsearch:** ML model observability, log-based anomaly detection, semantic search with vector embeddings (8.0+).

**Real-World Example:**
Spotify uses Cassandra for user listening history (write-heavy) and PostgreSQL for playlists (ACID). Pinterest uses Redis for caching, HBase for pin data at scale, and MySQL for user relationships. LinkedIn uses Espresso (document store) for profiles, Voldemort (KV) for caching, and custom graph infrastructure for the connection graph.

> **Interview Tip:** "Map databases to companies: Redis (Twitter caching), DynamoDB (Amazon.com), MongoDB (eBay catalogs), Cassandra (Apple 75K nodes, Netflix, Discord), Neo4j (fraud detection at eBay). Then explain WHY each company chose that database based on their specific access patterns and scale requirements."

---

## Key-Value Stores

### 8. How does a key-value store operate, and can you give a use-case example?

**Type:** 📝 Question

**Answer:**

A **key-value store** operates like a **distributed hash map** — data is stored and retrieved using a **unique key** that maps to an opaque **value** (string, JSON, binary blob). Operations are limited to `GET(key)`, `PUT(key, value)`, `DELETE(key)`. This simplicity enables **O(1) lookups**, **horizontal scaling** via consistent hashing, and **sub-millisecond latency**.

```
  KEY-VALUE STORE INTERNALS:
  
  Client: PUT("user:1001", {name: "Alice", age: 30})
  
  Step 1: Hash the key
  hash("user:1001") = 0x7A3F --> Shard 2
  
  Step 2: Store on correct shard
  +--------+  +--------+  +--------+  +--------+
  |Shard 0 |  |Shard 1 |  |Shard 2 |  |Shard 3 |
  |        |  |        |  |user:1001|  |        |
  |        |  |        |  | = blob  |  |        |
  +--------+  +--------+  +--------+  +--------+
  
  Client: GET("user:1001")
  hash("user:1001") = 0x7A3F --> Shard 2 --> Return blob
  
  USE CASE: SESSION STORE
  +--------+         +---------+
  | Browser|--cookie-| Web App |
  | session|  "abc"  |         |
  | id=abc |         +----+----+
  +--------+              |
                     GET("sess:abc")
                          |
                    +-----v-----+
                    |   Redis   |
                    | sess:abc  |
                    | = {user,  |
                    |   cart,   |
                    |   prefs}  |
                    +-----------+
                    Response: 0.2ms
```

| Operation | Time Complexity | Description |
|-----------|----------------|-------------|
| **GET(key)** | O(1) | Retrieve value by key |
| **PUT(key, value)** | O(1) | Insert or update |
| **DELETE(key)** | O(1) | Remove key-value pair |
| **SCAN** | O(n) | Iterate all keys (expensive!) |
| **TTL** | O(1) | Auto-expire after timeout |

```python
# Key-Value store with TTL and consistent hashing
import hashlib
import time

class KeyValueStore:
    """Redis-like key-value store with TTL support."""

    def __init__(self):
        self.store = {}
        self.expiry = {}

    def put(self, key, value, ttl=None):
        """Store key-value pair with optional TTL (seconds)."""
        self.store[key] = value
        if ttl:
            self.expiry[key] = time.time() + ttl
        return "OK"

    def get(self, key):
        """Retrieve value, respecting TTL."""
        if key in self.expiry and time.time() > self.expiry[key]:
            del self.store[key]
            del self.expiry[key]
            return None
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)
        self.expiry.pop(key, None)
        return "OK"

# Session store use case
session_store = KeyValueStore()

# Login: create session
session_store.put("sess:abc123", {
    "user_id": 1001,
    "username": "alice",
    "cart": ["item_1", "item_2"],
    "preferences": {"theme": "dark"}
}, ttl=3600)  # Expire in 1 hour

# Every request: validate session (sub-ms)
session = session_store.get("sess:abc123")
if session:
    print(f"Welcome back, {session['username']}!")

# Distributed KV with consistent hashing
class DistributedKV:
    def __init__(self, nodes):
        self.ring = {}
        for node in nodes:
            for i in range(150):  # Virtual nodes
                h = self._hash(f"{node}:{i}")
                self.ring[h] = node
        self.sorted_keys = sorted(self.ring.keys())
        self.node_stores = {n: {} for n in nodes}

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _get_node(self, key):
        h = self._hash(key)
        for k in self.sorted_keys:
            if k >= h:
                return self.ring[k]
        return self.ring[self.sorted_keys[0]]

    def put(self, key, value):
        node = self._get_node(key)
        self.node_stores[node][key] = value

    def get(self, key):
        node = self._get_node(key)
        return self.node_stores[node].get(key)
```

**AI/ML Application:**
- **Feature serving:** Redis stores precomputed ML features (embeddings, statistics) keyed by entity_id. Inference reads features in <1ms: `GET("features:user:1001")`.
- **Prediction caching:** Cache expensive model predictions keyed by input hash. Same input returns cached prediction without re-running the model.
- **Rate limiting:** Redis key-value with TTL implements rate limiting for ML API endpoints: `INCR("ratelimit:user:1001")` with 60-second TTL.

**Real-World Example:**
Twitter uses Redis as a caching layer for timelines — each user's timeline is a key-value entry (user_id → sorted list of tweet IDs). GitHub uses Redis for background job queues (Resque/Sidekiq). Instagram uses Redis for storing like counts and comment counts (key = media_id, value = count). DynamoDB powers Amazon.com's shopping cart — `cart:{user_id}` stores the entire cart as a value.

> **Interview Tip:** "Key-value stores are the simplest NoSQL type: O(1) get/put using hash-based lookup. Best use cases: sessions, caching, rate limiting, leaderboards. Redis is the go-to example — mention its data structures (strings, lists, sets, sorted sets, hashes). DynamoDB extends KV with secondary indexes and conditional writes. Always note the limitation: you can ONLY query by key — no complex queries."

---

### 9. What strategies can be used to scale key-value stores for high demand or large data volumes?

**Type:** 📝 Question

**Answer:**

Key-value stores scale through **horizontal partitioning (sharding)**, **replication**, **consistent hashing**, **read replicas**, **caching tiers**, and **cluster auto-scaling**. The key insight: since data access is by key, partitioning is trivial — hash the key to determine the node.

```
  SCALING STRATEGIES:
  
  1. CONSISTENT HASHING (Partitioning):
  
       Node A           Node B
      (0-90)           (91-180)
        \                 /
         \   Hash Ring   /
          +------+------+
          |      |      |
          | 270  | 180  |
          |      |      |
          +------+------+
         /               \
       Node D           Node C
      (271-360)        (181-270)
  
  key "user:1" hashes to 45 --> Node A
  key "user:2" hashes to 200 --> Node C
  Add Node E at 135 --> only keys 91-135 move (minimal disruption!)
  
  2. REPLICATION (Read scaling):
  +--------+    Write    +--------+
  | Client |------------>| Primary|
  +--------+             +----+---+
       |                      |
       | Read            Replicate
       v                      |
  +--------+             +----v---+
  | Client |<------------|Replica1|
  +--------+   Read      +--------+
                          +--------+
                          |Replica2|
                          +--------+
  Write: 1 node  |  Read: 3 nodes (3x read capacity)
  
  3. TIERED CACHING:
  Client --> L1 (Local) --> L2 (Redis) --> L3 (DynamoDB)
             ~0.01ms       ~0.5ms         ~5ms
```

| Strategy | Benefit | Trade-off | Example |
|----------|--------|-----------|---------|
| **Consistent hashing** | Minimal data movement on scale | Virtual nodes complexity | DynamoDB, Cassandra |
| **Read replicas** | Multiply read throughput | Replication lag (eventual) | Redis Cluster, ElastiCache |
| **Write sharding** | Distribute write load | Cross-shard queries impossible | Redis Cluster |
| **Tiered caching** | Reduce backend load | Cache invalidation complexity | L1→Redis→DynamoDB |
| **Auto-scaling** | Match demand dynamically | Cold start latency | DynamoDB on-demand |
| **Data compression** | Reduce memory/storage | CPU overhead | Redis with LZ4 |

```python
# Consistent hashing for KV store scaling
import hashlib
import bisect

class ConsistentHashRing:
    """Consistent hash ring for distributed key-value store."""

    def __init__(self, virtual_nodes=150):
        self.ring = {}
        self.sorted_hashes = []
        self.virtual_nodes = virtual_nodes
        self.nodes = set()

    def _hash(self, key):
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    def add_node(self, node):
        """Add node with virtual nodes for even distribution."""
        self.nodes.add(node)
        for i in range(self.virtual_nodes):
            h = self._hash(f"{node}:vn{i}")
            self.ring[h] = node
            bisect.insort(self.sorted_hashes, h)

    def remove_node(self, node):
        """Remove node - only its keys need redistribution."""
        self.nodes.discard(node)
        self.sorted_hashes = [h for h in self.sorted_hashes if self.ring.get(h) != node]
        self.ring = {h: n for h, n in self.ring.items() if n != node}

    def get_node(self, key):
        """Find which node owns this key."""
        h = self._hash(key)
        idx = bisect.bisect_right(self.sorted_hashes, h)
        if idx == len(self.sorted_hashes):
            idx = 0
        return self.ring[self.sorted_hashes[idx]]

# Auto-scaling key-value cluster
class AutoScalingKVCluster:
    def __init__(self, initial_nodes=3, max_nodes=10):
        self.ring = ConsistentHashRing()
        self.stores = {}
        self.max_nodes = max_nodes
        for i in range(initial_nodes):
            self._add_node(f"node_{i}")

    def _add_node(self, node_id):
        self.ring.add_node(node_id)
        self.stores[node_id] = {}

    def put(self, key, value):
        node = self.ring.get_node(key)
        self.stores[node][key] = value
        if self._should_scale_out(node):
            self._scale_out()

    def get(self, key):
        node = self.ring.get_node(key)
        return self.stores[node].get(key)

    def _should_scale_out(self, node):
        return len(self.stores[node]) > 10000  # Threshold

    def _scale_out(self):
        if len(self.stores) < self.max_nodes:
            new_id = f"node_{len(self.stores)}"
            self._add_node(new_id)
```

**AI/ML Application:**
- **Feature store scaling:** ML feature stores use consistent hashing to shard features across Redis nodes. As the number of entity features grows, adding nodes scales linearly with minimal data migration.
- **Embedding cache scaling:** Large language model embedding caches (billions of vectors) use tiered caching: L1 in-process → L2 Redis cluster → L3 DynamoDB. Read replicas scale read-heavy inference loads.
- **Model prediction cache:** High-QPS ML services cache predictions in Redis Cluster. Consistent hashing distributes cache entries evenly, and read replicas handle prediction cache reads for popular items.

**Real-World Example:**
Redis Cluster supports up to 1,000 nodes with 16,384 hash slots distributed across them. DynamoDB uses consistent hashing internally and auto-scales without downtime. Amazon ElastiCache offers auto-scaling Redis clusters that add/remove nodes based on CPU and memory utilization. Discord scaled from 1 Redis node to a Redis Cluster handling 40M+ requests/sec.

> **Interview Tip:** "Three key strategies: (1) Consistent hashing for partitioning — minimal data movement when adding/removing nodes. (2) Read replicas for read-heavy workloads — Redis Cluster replicates each primary to 1-2 replicas. (3) Tiered caching — local cache + distributed cache + persistent store. Always mention 'virtual nodes' when discussing consistent hashing — they ensure even data distribution."

---

### 10. What are some drawbacks of key-value stores compared to other NoSQL types?

**Type:** 📝 Question

**Answer:**

Key-value stores sacrifice **query flexibility** for **performance simplicity**. They cannot perform **range queries**, **secondary indexes**, **joins**, **aggregations**, or **full-text search** natively. The value is an **opaque blob** — the database cannot inspect or filter by its contents.

```
  WHAT KEY-VALUE STORES CANNOT DO:
  
  1. NO SECONDARY INDEXES
  Store: {user:1: {name:"Alice", age:30}, user:2: {name:"Bob", age:25}}
  
  GET("user:1")           --> Works! O(1)
  FIND(age > 25)          --> IMPOSSIBLE (must scan ALL keys!)
  
  2. NO RANGE QUERIES
  GET("user:1")           --> Works!
  GET_RANGE("user:1".."user:100")  --> Not supported in pure KV
  (Column-family stores like Cassandra CAN do this)
  
  3. NO JOINS
  User: {id:1, order_ids:[10,20]}
  Order: {id:10, product:"Book"}
  
  "Get user 1 with all order details"
  --> Requires multiple GET calls from application code
  --> Document DB would embed orders in user document
  
  4. NO AGGREGATIONS
  "Count users by country"    --> Must scan ALL values
  "Average order value"       --> Must read and compute in app
  
  5. NO FILTERING ON VALUES
  "Find all users named Alice" --> Full scan required

  COMPARISON:
  +------------------+----+-----+--------+-------+
  | Capability       | KV | Doc | Column | Graph |
  +------------------+----+-----+--------+-------+
  | Key lookup       | ++ | ++  | ++     | +     |
  | Range queries    | -- | +   | ++     | +     |
  | Secondary index  | -- | ++  | +      | +     |
  | Joins/traversals | -- | -   | -      | ++    |
  | Aggregations     | -- | ++  | +      | +     |
  | Full-text search | -- | +   | -      | -     |
  +------------------+----+-----+--------+-------+
```

| Drawback | Impact | Workaround |
|----------|--------|-----------|
| **No secondary indexes** | Can't query by value fields | Maintain separate index keys manually |
| **No range queries** | Can't scan key ranges | Use sorted sets (Redis) or move to column-family |
| **Opaque values** | DB can't filter internals | Application-side filtering |
| **No joins** | Multi-entity queries need app logic | Denormalize or use document DB |
| **No aggregations** | Analytics requires full scan | Pre-compute aggregates, use analytics DB |
| **Limited data modeling** | Complex domains hard to represent | Use document or graph DB |

```python
# Demonstrating KV limitations and workarounds

class KeyValueLimitations:
    """Shows what's hard with pure key-value stores."""

    def __init__(self):
        self.store = {}

    def put(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    # LIMITATION: Finding by value requires full scan!
    def find_by_field(self, field, value):
        """O(n) - must scan ALL entries."""
        results = []
        for k, v in self.store.items():
            if isinstance(v, dict) and v.get(field) == value:
                results.append((k, v))
        return results  # SLOW at scale!

    # WORKAROUND: Maintain manual secondary index
    def put_with_index(self, key, value, index_fields):
        self.store[key] = value
        for field in index_fields:
            if field in value:
                idx_key = f"idx:{field}:{value[field]}"
                existing = self.store.get(idx_key, [])
                existing.append(key)
                self.store[idx_key] = existing

    def query_by_index(self, field, value):
        """O(1) lookup using secondary index."""
        idx_key = f"idx:{field}:{value}"
        keys = self.store.get(idx_key, [])
        return [(k, self.store.get(k)) for k in keys]

# Demo
kv = KeyValueLimitations()
kv.put_with_index("user:1", {"name": "Alice", "country": "US"}, ["country"])
kv.put_with_index("user:2", {"name": "Bob", "country": "UK"}, ["country"])
kv.put_with_index("user:3", {"name": "Charlie", "country": "US"}, ["country"])

# Fast index lookup
us_users = kv.query_by_index("country", "US")  # O(1) instead of O(n)
```

**AI/ML Application:**
- **Feature query limitations:** Key-value feature stores (Redis) can serve features by entity_id fast, but cannot answer "which users have feature X > threshold?" — that requires a separate analytics query against the batch store.
- **Model metadata search:** Storing ML model metadata in a KV store makes it impossible to search "find all models with accuracy > 0.95" — use MongoDB or a search engine instead.
- **Embedding retrieval:** KV stores can retrieve a known embedding by ID, but cannot perform nearest-neighbor search — that requires a specialized vector database (Pinecone, Milvus).

**Real-World Example:**
Redis partially addresses KV limitations with data structures: sorted sets enable range queries, hash types allow partial field updates. DynamoDB adds Global Secondary Indexes (GSIs) to query by non-key attributes. However, pure KV stores like Memcached have none of these — truly opaque values. This is why many systems use KV for caching + a document/relational DB for queries.

> **Interview Tip:** "The main drawback: you can ONLY query by key. No filtering, no ranges, no joins, no aggregations on values. Workarounds exist: Redis sorted sets for ranges, DynamoDB GSIs for secondary queries, but these add complexity and cost. If you need to query by any field, use a document DB (MongoDB) or relational DB instead. KV shines when access is always by a known key."

---

### 11. Name a scenario where a key-value store might not be the best fit.

**Type:** 📝 Question

**Answer:**

Key-value stores are a **poor fit** for **complex querying**, **relational data**, **analytics/reporting**, and **search** use cases. Any scenario requiring **queries beyond simple key lookup** — such as filtering by attributes, joining entities, aggregation, or full-text search — is better served by other database types.

```
  POOR FIT SCENARIOS FOR KV STORES:
  
  1. E-COMMERCE PRODUCT SEARCH
  "Find all laptops under $1000 with 16GB RAM sorted by rating"
  --> Requires: filtering, range queries, sorting, facets
  --> Better: Elasticsearch, MongoDB
  
  2. FINANCIAL REPORTING
  "Sum all transactions for Q4 2024 grouped by category"
  --> Requires: aggregation, grouping, date ranges
  --> Better: PostgreSQL, ClickHouse
  
  3. SOCIAL NETWORK
  "Find friends of friends who also like photography"
  --> Requires: graph traversal, relationship queries
  --> Better: Neo4j, Amazon Neptune
  
  4. CONTENT MANAGEMENT SYSTEM
  "Find all articles tagged 'python' published this week"
  --> Requires: secondary indexes, filtering, sorting
  --> Better: MongoDB, PostgreSQL
  
  5. ANALYTICS DASHBOARD
  "Average response time per endpoint per day for last 30 days"
  --> Requires: time-series aggregation, GROUP BY
  --> Better: ClickHouse, TimescaleDB, Cassandra

  DECISION MATRIX:
  +---------------------------+--------+------------------+
  | Scenario                  | KV Fit?| Better Option    |
  +---------------------------+--------+------------------+
  | Session storage           | YES    | Redis            |
  | Product search            | NO     | Elasticsearch    |
  | Shopping cart              | YES    | DynamoDB         |
  | Financial reporting       | NO     | PostgreSQL       |
  | User profile by ID        | YES    | Redis/DynamoDB   |
  | Friend recommendations    | NO     | Neo4j            |
  | Real-time leaderboard     | YES    | Redis sorted set |
  | Log analytics             | NO     | Elasticsearch    |
  | Multi-field filtering     | NO     | MongoDB          |
  +---------------------------+--------+------------------+
```

| Anti-Pattern | Why KV Fails | Better Alternative |
|-------------|-------------|-------------------|
| **Multi-criteria search** | No field-level indexing | Elasticsearch, MongoDB |
| **Relational queries** | No joins, no FK relationships | PostgreSQL, MySQL |
| **Aggregation/reporting** | Must scan all values | ClickHouse, BigQuery |
| **Graph traversals** | No relationship modeling | Neo4j, Neptune |
| **Full-text search** | No text indexing/ranking | Elasticsearch, Solr |
| **Time-series analytics** | No range scans (pure KV) | TimescaleDB, Cassandra |

```python
# Anti-pattern: Using KV store for complex queries (DON'T DO THIS)

class KVAntiPattern:
    """Shows why using KV for complex queries is a bad idea."""

    def __init__(self):
        self.products = {}  # product_id -> product_data

    def add_product(self, product_id, product):
        self.products[product_id] = product

    def search_products(self, category=None, max_price=None, min_rating=None):
        """O(n) scan - TERRIBLE for large catalogs!"""
        results = []
        for pid, product in self.products.items():  # Full scan!
            if category and product.get("category") != category:
                continue
            if max_price and product.get("price", 0) > max_price:
                continue
            if min_rating and product.get("rating", 0) < min_rating:
                continue
            results.append(product)
        return sorted(results, key=lambda p: -p.get("rating", 0))

# BETTER: Use a document DB with indexes
class DocumentDBApproach:
    """MongoDB-like approach with proper indexing."""

    def __init__(self):
        self.products = []
        self.indexes = {}  # field -> {value -> [doc_indices]}

    def create_index(self, field):
        self.indexes[field] = {}
        for i, doc in enumerate(self.products):
            val = doc.get(field)
            self.indexes[field].setdefault(val, []).append(i)

    def search_products(self, **criteria):
        """Uses indexes for efficient filtering."""
        candidate_sets = []
        for field, value in criteria.items():
            if field in self.indexes and value in self.indexes[field]:
                candidate_sets.append(set(self.indexes[field][value]))
        if candidate_sets:
            result_indices = set.intersection(*candidate_sets)
            return [self.products[i] for i in result_indices]
        return []
```

**AI/ML Application:**
- **ML model comparison:** "Find all models with accuracy > 0.9 and latency < 50ms" — this multi-criteria query is impossible in a KV store. Use MongoDB or MLflow's tracking database.
- **Feature discovery:** "Which features are most correlated with the target?" requires analytical queries across all features — use a data warehouse (BigQuery), not a KV store.
- **Training data filtering:** "Get all training examples with label=positive from last month" requires indexing and filtering — use a document DB or data lake.

**Real-World Example:**
Slack initially stored messages in a KV-like format but needed rich search ("find all messages mentioning 'deploy' in #engineering from last week") — they added Elasticsearch. Uber uses a KV store for driver location (by driver_id) but a separate geospatial index (H3) for "find drivers near this location" queries. Amazon uses DynamoDB for cart items (by cart_id) but Elasticsearch for product search.

> **Interview Tip:** "If the interviewer asks 'when NOT to use KV,' describe the product search scenario: an e-commerce site where users filter by category, price range, rating, and sort by relevance. A KV store requires scanning millions of products for every query — O(n). A document DB with indexes or Elasticsearch handles this in milliseconds. The rule: if you need to query by anything other than the key, KV is the wrong choice."

---

## Document-Oriented Databases

### 12. What makes a document in a NoSQL database different from a row in a relational database ?

**Type:** 📝 Question

**Answer:**

A **document** is a **self-contained**, **hierarchically nested** data structure (JSON/BSON) with a **flexible schema**, while a relational **row** is a **flat** record conforming to a **fixed schema**. Documents can contain **nested objects**, **arrays**, and **varying fields** per document in the same collection — no schema migration needed.

```
  RELATIONAL ROW (Fixed Schema):
  +----+-------+-----+-----------+
  | id | name  | age | email     |
  +----+-------+-----+-----------+
  | 1  | Alice | 30  | a@b.com   |  <-- Every row: same columns
  | 2  | Bob   | 25  | b@c.com   |  <-- NULL for missing
  | 3  | Carol | 28  | NULL      |  <-- Must have all cols
  +----+-------+-----+-----------+
  
  DOCUMENT (Flexible Schema):
  // Document 1 - has orders embedded
  {
    "_id": "user1",
    "name": "Alice",
    "age": 30,
    "email": "a@b.com",
    "orders": [
      {"item": "Book", "price": 15.99},
      {"item": "Phone", "price": 699.99}
    ],
    "preferences": {"theme": "dark", "lang": "en"}
  }
  
  // Document 2 - different fields!
  {
    "_id": "user2",
    "name": "Bob",
    "age": 25,
    "social": {"twitter": "@bob", "github": "bob123"}
  }
  // No "email" field, no "orders" - that's OK!
  // Extra "social" field - that's OK too!

  KEY DIFFERENCES:
  +------------------+--------------------+
  | Relational Row   | NoSQL Document     |
  +------------------+--------------------+
  | Flat (1 level)   | Nested (N levels)  |
  | Fixed schema     | Flexible schema    |
  | NULL for missing | Field simply absent|
  | Joins for related| Embedded sub-docs  |
  | ALTER to change  | Just add new fields|
  | Normalized       | Denormalized       |
  +------------------+--------------------+
```

| Aspect | Relational Row | NoSQL Document |
|--------|---------------|---------------|
| **Structure** | Flat columns, single level | Nested objects, arrays, N levels |
| **Schema** | Fixed, enforced by DB | Flexible, enforced by app (optional) |
| **Related data** | Separate table + JOIN | Embedded sub-documents |
| **Missing fields** | NULL values | Field simply absent |
| **Schema changes** | ALTER TABLE (expensive) | Just write new shape |
| **Storage format** | Binary rows | JSON, BSON, or binary |

```python
# Relational row vs Document comparison

# RELATIONAL: Flat rows, separate tables
class RelationalDB:
    def __init__(self):
        self.users = []    # Fixed schema: [id, name, age, email]
        self.orders = []   # Fixed schema: [id, user_id, item, price]
        self.schema = {
            "users": ["id", "name", "age", "email"],
            "orders": ["id", "user_id", "item", "price"]
        }

    def insert_user(self, user):
        # Enforce schema: all fields must be present
        row = {col: user.get(col) for col in self.schema["users"]}
        self.users.append(row)
        # Extra fields silently ignored!

    def get_user_with_orders(self, user_id):
        user = next((u for u in self.users if u["id"] == user_id), None)
        orders = [o for o in self.orders if o["user_id"] == user_id]
        return {**user, "orders": orders}  # Manual JOIN

# DOCUMENT: Flexible, nested
class DocumentDB:
    def __init__(self):
        self.collections = {}

    def insert(self, collection, doc):
        # No schema enforcement — any shape accepted
        self.collections.setdefault(collection, []).append(doc)

    def find_one(self, collection, query):
        for doc in self.collections.get(collection, []):
            if all(doc.get(k) == v for k, v in query.items()):
                return doc  # Complete document with all nested data!

# Document: one read gets everything
doc_db = DocumentDB()
doc_db.insert("users", {
    "_id": "user1",
    "name": "Alice",
    "orders": [{"item": "Book", "price": 15.99}],  # Embedded!
    "preferences": {"theme": "dark"}                 # Nested!
})
user = doc_db.find_one("users", {"_id": "user1"})
# One read: user + orders + preferences (no JOINs)
```

**AI/ML Application:**
- **ML experiment metadata:** Each ML experiment has different hyperparameters (learning_rate, batch_size, num_layers, dropout, etc.). Documents handle this naturally — each experiment document has different fields. A relational row would need columns for every possible hyperparameter.
- **Model versioning:** Model artifacts have varying metadata (architecture, framework, input/output shapes). Document DBs store each model as a self-describing document.
- **Training data catalogs:** Data catalogs describe datasets with varying attributes — documents naturally represent datasets with different schemas.

**Real-World Example:**
MongoDB stores eBay's product catalog — each product category (electronics, clothing, food) has completely different attributes. A relational table would need thousands of nullable columns. With MongoDB, each product document only stores relevant fields. Contentful (headless CMS) uses a document model because content types vary wildly between organizations.

> **Interview Tip:** "Key difference: a relational row is flat and fixed-schema; a document is nested and flexible-schema. Documents embed related data as sub-documents (no joins needed), support varying fields per document (no NULLs for missing data), and can be evolved without migrations (just add new fields). The trade-off: less data integrity enforcement — the application must validate data instead of the database."

---

### 13. How does indexing work in document-oriented databases ?

**Type:** 📝 Question

**Answer:**

Document databases support **rich indexing** on any field within documents, including **nested fields**, **array elements**, and **compound fields**. They use **B-tree indexes** (default), **text indexes** (full-text search), **geospatial indexes** (2dsphere), **hashed indexes** (sharding), and **wildcard indexes** (dynamic schemas). Unlike KV stores, document DBs can index **inside** the value.

```
  MONGODB INDEXING EXAMPLE:
  
  Document:
  {
    "_id": "order1",
    "customer": {
      "name": "Alice",
      "email": "alice@example.com"
    },
    "items": [
      {"product": "Book", "price": 15.99},
      {"product": "Phone", "price": 699.99}
    ],
    "total": 715.98,
    "created": "2024-01-15",
    "location": {"type": "Point", "coordinates": [-73.9, 40.7]}
  }
  
  INDEXES:
  1. _id              --> Default (unique B-tree)
  2. customer.email   --> Nested field index (dot notation)
  3. items.price      --> Multi-key (indexes each array element)
  4. total            --> Single field B-tree
  5. {total, created} --> Compound index
  6. location         --> 2dsphere (geospatial)
  7. customer.name    --> Text index (full-text search)
  
  QUERY PERFORMANCE WITH INDEXES:
  +-------------------------------------+---------+
  | Query                               | Indexed?|
  +-------------------------------------+---------+
  | find({_id: "order1"})               | YES O(log n) |
  | find({"customer.email": "a@b.com"}) | YES O(log n) |
  | find({"items.price": {$gt: 500}})   | YES O(log n) |
  | find({total: {$gt: 100}})           | YES O(log n) |
  | find({"customer.phone": "123"})     | NO  O(n)     |
  +-------------------------------------+---------+
  
  B-TREE INDEX STRUCTURE:
           [500]
          /     \
      [100]     [700]
      /   \     /   \
   [50] [200] [600] [900]
    |     |     |     |
   docs  docs  docs  docs
```

| Index Type | Use Case | MongoDB Syntax | Performance |
|-----------|---------|---------------|-------------|
| **Single field** | Equality/range on one field | `{field: 1}` | O(log n) |
| **Compound** | Multi-field queries | `{field1: 1, field2: -1}` | O(log n) |
| **Multi-key** | Array element queries | `{array.field: 1}` | O(log n) per element |
| **Text** | Full-text search | `{field: "text"}` | Inverted index |
| **Geospatial** | Location queries | `{loc: "2dsphere"}` | R-tree |
| **Hashed** | Equality + sharding | `{field: "hashed"}` | O(1) |
| **Wildcard** | Dynamic schema fields | `{"$**": 1}` | Flexible but costly |

```python
# Document DB indexing simulation
class IndexedDocumentDB:
    """Document store with B-tree-like indexing."""

    def __init__(self):
        self.documents = []
        self.indexes = {}  # field_path -> sorted_index

    def create_index(self, field_path):
        """Create B-tree-like index on a field (supports dot notation)."""
        index = {}
        for i, doc in enumerate(self.documents):
            value = self._get_nested(doc, field_path)
            if value is not None:
                if isinstance(value, list):  # Multi-key index
                    for item in value:
                        index.setdefault(item, []).append(i)
                else:
                    index.setdefault(value, []).append(i)
        self.indexes[field_path] = index

    def _get_nested(self, doc, path):
        """Navigate dot-notation path in document."""
        parts = path.split(".")
        current = doc
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                return [item.get(part) for item in current if isinstance(item, dict)]
            else:
                return None
        return current

    def find(self, query):
        """Query using indexes when available."""
        for field, value in query.items():
            if field in self.indexes:  # Use index: O(log n)
                doc_indices = self.indexes[field].get(value, [])
                return [self.documents[i] for i in doc_indices]
        # Full scan fallback: O(n)
        return [d for d in self.documents
                if all(self._get_nested(d, k) == v for k, v in query.items())]

    def insert(self, doc):
        idx = len(self.documents)
        self.documents.append(doc)
        # Update all indexes
        for field_path, index in self.indexes.items():
            value = self._get_nested(doc, field_path)
            if value is not None:
                index.setdefault(value, []).append(idx)
```

**AI/ML Application:**
- **Experiment search:** Index `model_type`, `accuracy`, `created_at` in experiment documents to quickly find "all CNN experiments with accuracy > 0.95 from last week."
- **Feature metadata:** Index feature names, data types, and owners in feature catalog documents for fast feature discovery.
- **Model registry queries:** Compound indexes on `{model_name: 1, version: -1}` enable fast retrieval of "latest version of production model."

**Real-World Example:**
MongoDB's query planner automatically selects the best index for each query. Atlas Search integrates Lucene-based full-text search indexes alongside traditional B-tree indexes. CouchDB uses MapReduce views as indexes — you define a JavaScript map function that emits index keys. Firestore automatically indexes every field (no manual index creation for simple queries).

> **Interview Tip:** "Document DBs index INSIDE documents using dot notation (e.g., 'customer.email'). Key types: single-field, compound, multi-key (arrays), text, geospatial. Compound indexes follow the ESR rule (Equality, Sort, Range) for field ordering. Always mention the trade-off: indexes speed reads but slow writes (must update index on every insert/update) and consume memory."

---

### 14. Give an example of a query in a document-oriented database .

**Type:** 📝 Question

**Answer:**

Document databases use **rich query languages** to filter, project, sort, and aggregate documents. MongoDB's query language supports **comparison operators**, **logical operators**, **array queries**, **nested field queries**, and **aggregation pipelines**. This is far more expressive than key-value GET/PUT.

```
  MONGODB QUERY EXAMPLES:
  
  Collection: orders
  [
    {
      "_id": "o1", "customer": "Alice", "status": "shipped",
      "total": 150.00, "items": [
        {"product": "Book", "qty": 2, "price": 15},
        {"product": "Phone", "qty": 1, "price": 120}
      ],
      "date": "2024-03-15", "tags": ["electronics", "books"]
    },
    {
      "_id": "o2", "customer": "Bob", "status": "pending",
      "total": 45.00, "items": [
        {"product": "Book", "qty": 3, "price": 15}
      ],
      "date": "2024-03-20", "tags": ["books"]
    }
  ]
  
  QUERIES:
  
  1. Simple equality:
     db.orders.find({customer: "Alice"})
  
  2. Comparison:
     db.orders.find({total: {$gt: 100}})
  
  3. Nested field:
     db.orders.find({"items.product": "Phone"})
  
  4. Array contains:
     db.orders.find({tags: {$in: ["electronics"]}})
  
  5. Aggregation pipeline:
     db.orders.aggregate([
       {$match: {status: "shipped"}},
       {$unwind: "$items"},
       {$group: {_id: "$items.product",
                 total_sold: {$sum: "$items.qty"},
                 revenue: {$sum: {$multiply: ["$items.qty","$items.price"]}}
       }},
       {$sort: {revenue: -1}}
     ])
```

| Query Type | MongoDB Syntax | SQL Equivalent |
|-----------|---------------|---------------|
| **Equality** | `find({status: "shipped"})` | `WHERE status = 'shipped'` |
| **Range** | `find({total: {$gt: 100}})` | `WHERE total > 100` |
| **Nested** | `find({"addr.city": "NYC"})` | `JOIN + WHERE` |
| **Array** | `find({tags: "electronics"})` | Separate junction table |
| **Regex** | `find({name: /^Ali/})` | `WHERE name LIKE 'Ali%'` |
| **Aggregation** | `aggregate([{$group:...}])` | `GROUP BY ... HAVING` |
| **Projection** | `find({}, {name:1, total:1})` | `SELECT name, total` |

```python
# MongoDB-like query engine simulation
import re
from datetime import datetime

class QueryEngine:
    """Simulates MongoDB query operations."""

    def __init__(self, documents):
        self.docs = documents

    def find(self, query, projection=None):
        """Filter documents matching query criteria."""
        results = [d for d in self.docs if self._matches(d, query)]
        if projection:
            results = [self._project(d, projection) for d in results]
        return results

    def _matches(self, doc, query):
        for field, condition in query.items():
            value = self._get_field(doc, field)
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$gt" and not (value and value > target): return False
                    if op == "$lt" and not (value and value < target): return False
                    if op == "$gte" and not (value and value >= target): return False
                    if op == "$in" and value not in target: return False
                    if op == "$regex" and not re.search(target, str(value)): return False
            elif isinstance(value, list):
                if condition not in value: return False
            elif value != condition:
                return False
        return True

    def _get_field(self, doc, path):
        parts = path.split(".")
        current = doc
        for p in parts:
            if isinstance(current, dict):
                current = current.get(p)
            else:
                return None
        return current

    def _project(self, doc, fields):
        return {f: doc.get(f) for f in fields if f in doc}

    def aggregate(self, pipeline):
        """Simple aggregation pipeline."""
        result = list(self.docs)
        for stage in pipeline:
            if "$match" in stage:
                result = [d for d in result if self._matches(d, stage["$match"])]
            elif "$sort" in stage:
                field = list(stage["$sort"].keys())[0]
                reverse = stage["$sort"][field] == -1
                result.sort(key=lambda d: d.get(field, 0), reverse=reverse)
            elif "$limit" in stage:
                result = result[:stage["$limit"]]
        return result

# Usage
orders = [
    {"customer": "Alice", "total": 150, "status": "shipped", "tags": ["electronics"]},
    {"customer": "Bob", "total": 45, "status": "pending", "tags": ["books"]},
    {"customer": "Carol", "total": 320, "status": "shipped", "tags": ["electronics"]}
]
qe = QueryEngine(orders)

# Find shipped orders over $100 sorted by total
results = qe.aggregate([
    {"$match": {"status": "shipped", "total": {"$gt": 100}}},
    {"$sort": {"total": -1}}
])
```

**AI/ML Application:**
- **Experiment queries:** `db.experiments.find({"model": "transformer", "metrics.accuracy": {$gt: 0.95}}).sort({"metrics.f1": -1})` — find best transformer experiments.
- **Feature lineage:** Aggregation pipelines trace feature dependencies: `aggregate([{$match: {downstream_model: "prod_v2"}}, {$unwind: "$input_features"}, {$group: {_id: "$input_features", count: {$sum: 1}}}])`.
- **Dataset discovery:** `db.datasets.find({tags: {$in: ["nlp", "classification"]}, size_gb: {$lt: 10}})` — find small NLP classification datasets.

**Real-World Example:**
MongoDB's aggregation pipeline powers Forbes' real-time statistics dashboard — multiple $match, $group, and $project stages transform raw page-view documents into analytics. eBay uses MongoDB aggregation to generate product listing statistics. Expedia aggregates hotel pricing documents to find the best deals with complex multi-field queries.

> **Interview Tip:** "Show a concrete query: `db.orders.find({status: 'shipped', total: {$gt: 100}}).sort({date: -1}).limit(10)`. Then explain the aggregation pipeline as a series of stages ($match → $group → $sort → $project) similar to SQL's WHERE → GROUP BY → ORDER BY → SELECT. Mention that document DBs can query nested fields with dot notation and array elements with $elemMatch."

---

### 15. Suggest a typical application for a document-oriented database .

**Type:** 📝 Question

**Answer:**

**Content Management Systems (CMS)** are the ideal application for document databases. Each content type (article, video, product, user profile) has **different fields**, **nested metadata**, and **evolving schemas** — perfectly matching the document model's flexibility.

```
  CMS WITH DOCUMENT DB (MongoDB):
  
  Collection: content
  
  Article document:
  {
    "_id": "article_001",
    "type": "article",
    "title": "Intro to ML",
    "body": "Machine learning is...",
    "author": {"name": "Alice", "avatar": "/img/alice.png"},
    "tags": ["ML", "AI", "tutorial"],
    "published": true,
    "publish_date": "2024-03-15",
    "comments": [
      {"user": "Bob", "text": "Great article!", "date": "2024-03-16"},
      {"user": "Carol", "text": "Very helpful", "date": "2024-03-17"}
    ],
    "seo": {"meta_title": "ML Guide", "meta_desc": "Learn ML basics"}
  }
  
  Video document (DIFFERENT FIELDS!):
  {
    "_id": "video_001",
    "type": "video",
    "title": "ML Demo",
    "url": "https://cdn.example.com/ml-demo.mp4",
    "duration_sec": 600,
    "resolution": "1080p",
    "thumbnail": "/img/ml-demo-thumb.jpg",
    "transcript": "In this video..."
  }
  
  WHY DOCUMENT DB FITS:
  - Article and Video have DIFFERENT fields
  - Comments embedded (no JOIN for display)
  - Tags as array (multi-value, indexed)
  - Author embedded (denormalized, fast)
  - Schema evolves: add "likes_count" anytime

  OTHER IDEAL APPLICATIONS:
  +-------------------+-----------------------------------+
  | Application       | Why Document DB?                  |
  +-------------------+-----------------------------------+
  | E-commerce catalog| Products have varied attributes   |
  | User profiles     | Preferences, settings vary        |
  | Event logging     | Events have different payloads    |
  | Mobile backends   | Flexible API responses            |
  | IoT metadata      | Device configs vary by type       |
  +-------------------+-----------------------------------+
```

| Application | Why Document DB | Alternative | Document DB Advantage |
|-------------|----------------|-------------|----------------------|
| **CMS** | Varying content types | RDBMS + EAV pattern | No schema migration for new types |
| **Product catalog** | Different attributes per category | RDBMS + JSON columns | Rich queries on nested fields |
| **User profiles** | Preferences vary per user | RDBMS + KV | Queryable + flexible |
| **Event sourcing** | Events have different payloads | RDBMS + JSON | Native nested event data |
| **Mobile backend** | Rapid iteration, schema changes | RDBMS | Faster development cycle |

```python
# CMS built on document database
class CMSDocumentDB:
    """Content Management System using document-oriented storage."""

    def __init__(self):
        self.collections = {"content": [], "users": []}
        self.indexes = {}

    def create_content(self, content_type, data):
        """Create any content type without schema changes."""
        doc = {
            "_id": f"{content_type}_{len(self.collections['content'])}",
            "type": content_type,
            "created_at": __import__('datetime').datetime.now().isoformat(),
            "published": False,
            **data
        }
        self.collections["content"].append(doc)
        return doc["_id"]

    def publish(self, content_id):
        for doc in self.collections["content"]:
            if doc["_id"] == content_id:
                doc["published"] = True
                doc["publish_date"] = __import__('datetime').datetime.now().isoformat()

    def find_by_type(self, content_type, published_only=True):
        return [d for d in self.collections["content"]
                if d["type"] == content_type
                and (not published_only or d.get("published"))]

    def find_by_tag(self, tag):
        return [d for d in self.collections["content"]
                if tag in d.get("tags", [])]

    def add_comment(self, content_id, comment):
        for doc in self.collections["content"]:
            if doc["_id"] == content_id:
                doc.setdefault("comments", []).append(comment)

# Usage
cms = CMSDocumentDB()
# Create article (fields: title, body, tags, author)
cms.create_content("article", {
    "title": "Intro to ML",
    "body": "Machine learning is...",
    "tags": ["ML", "AI"],
    "author": {"name": "Alice"}
})
# Create video (COMPLETELY DIFFERENT fields!)
cms.create_content("video", {
    "title": "ML Demo",
    "url": "https://cdn.example.com/video.mp4",
    "duration_sec": 600,
    "resolution": "1080p"
})
# No schema changes needed! Both stored in same collection.
```

**AI/ML Application:**
- **ML experiment tracking:** MLflow stores experiment metadata in MongoDB — each experiment has different hyperparameters, metrics, and artifacts, making the document model ideal.
- **Model registry:** Each model version has different metadata (architecture, framework, input shapes). Document DB stores model cards as flexible documents.
- **Dataset catalog:** Datasets have varying schemas, sizes, and annotations. A document per dataset with embedded metadata enables flexible discovery.

**Real-World Example:**
Forbes uses MongoDB to power their CMS — article types (news, opinion, lists, videos) have different fields and embedding requirements. The Washington Post uses document storage for their Arc Publishing CMS. Adobe Experience Manager uses a document-like content repository. Shopify uses document-oriented storage for product catalogs where a T-shirt and a laptop have completely different attributes.

> **Interview Tip:** "The textbook answer is CMS, but also mention e-commerce catalogs (products with different attributes per category), user profiles (varying preferences per user), and event logging (events with different payloads). Emphasize the 'schema flexibility' advantage: with a document DB, adding a new content type (podcast, newsletter) requires zero schema migration — just start inserting documents with new fields."

---

### 16. How do document databases handle schema changes and migration ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Document databases handle schema changes through **lazy migration** (update-on-read), **eager migration** (batch update), and **schema versioning**. Unlike RDBMS `ALTER TABLE`, document DBs don't require downtime because documents are **schema-flexible** — old and new shapes coexist in the same collection.

```
  SCHEMA MIGRATION STRATEGIES:
  
  1. LAZY MIGRATION (Update on Read):
  +----------------+      +------------------+
  | Read document  | ---> | Check version    |
  +----------------+      +------------------+
                                |
                   +------------+------------+
                   |                         |
              v1 (old)                  v2 (current)
                   |                         |
           +-------v--------+        +------v------+
           | Transform to v2|        | Return as-is|
           | Write back     |        +-------------+
           +----------------+
  
  2. EAGER MIGRATION (Batch Update):
  +------------------+      +------------------+
  | Background job   | ---> | Read all v1 docs |
  +------------------+      +------------------+
                                    |
                             +------v------+
                             | Transform   |
                             | to v2       |
                             +------+------+
                                    |
                             +------v------+
                             | Write back  |
                             | in batches  |
                             +-------------+
  
  3. SCHEMA VERSIONING:
  // v1 document
  {
    "_id": "user1",
    "_schema_version": 1,
    "name": "Alice Smith"
  }
  
  // v2 document (name split)
  {
    "_id": "user2",
    "_schema_version": 2,
    "first_name": "Bob",
    "last_name": "Jones",
    "full_name": "Bob Jones"
  }
  
  // Application code handles BOTH versions
```

| Strategy | When to Use | Downtime | Data Consistency | Speed |
|----------|-----------|----------|------------------|-------|
| **Lazy migration** | Gradual rollout, low-risk | None | Eventually consistent | Slow (on-demand) |
| **Eager migration** | Must update all docs | None (background) | Fully consistent after | Medium |
| **Dual writes** | Feature flag rollout | None | Both formats available | Write overhead |
| **Expand-contract** | Breaking changes | None | Safe rollback possible | Two deployments |
| **RDBMS ALTER TABLE** | Comparison baseline | Possible (lock) | Immediate | Fast (small) / Slow (large) |

```python
# Schema migration strategies for document databases
class SchemaMigrator:
    """Handles document schema evolution."""

    def __init__(self, db):
        self.db = db
        self.migrations = {}  # version -> transform_fn
        self.current_version = 1

    def register_migration(self, from_version, to_version, transform_fn):
        self.migrations[(from_version, to_version)] = transform_fn
        self.current_version = max(self.current_version, to_version)

    def lazy_migrate(self, doc):
        """Migrate on read — transform when accessed."""
        doc_version = doc.get("_schema_version", 1)
        while doc_version < self.current_version:
            next_version = doc_version + 1
            transform = self.migrations.get((doc_version, next_version))
            if transform:
                doc = transform(doc)
                doc["_schema_version"] = next_version
                # Write back migrated document
                self.db.update(doc["_id"], doc)
            doc_version = next_version
        return doc

    def eager_migrate(self, batch_size=100):
        """Batch migrate all old documents."""
        old_docs = self.db.find({"_schema_version": {"$lt": self.current_version}})
        for i in range(0, len(old_docs), batch_size):
            batch = old_docs[i:i + batch_size]
            for doc in batch:
                migrated = self.lazy_migrate(doc)
                self.db.update(migrated["_id"], migrated)
            print(f"Migrated batch {i // batch_size + 1}")

# Define migrations
def v1_to_v2(doc):
    """Split 'name' into 'first_name' and 'last_name'."""
    name = doc.pop("name", "")
    parts = name.split(" ", 1)
    doc["first_name"] = parts[0]
    doc["last_name"] = parts[1] if len(parts) > 1 else ""
    doc["full_name"] = name
    return doc

def v2_to_v3(doc):
    """Add 'email_verified' field with default."""
    doc.setdefault("email_verified", False)
    doc.setdefault("created_at", "1970-01-01")
    return doc
```

**AI/ML Application:**
- **ML pipeline evolution:** As feature engineering evolves, new features are added to training data documents. Lazy migration ensures old experiment records get the new schema when re-analyzed, while new experiments use the latest schema.
- **Model metadata versioning:** Model cards evolve (adding fairness metrics, carbon footprint). Schema versioning lets old and new model documents coexist.

**Real-World Example:**
MongoDB's `$jsonSchema` validator supports optional enforcement — you can warn on invalid documents without rejecting them during migration. Mongoose (Node.js ODM) handles schema versioning with discriminators and middleware hooks. Contentful CMS manages content model migrations through their Migration API — each migration is a versioned script that transforms content types.

> **Interview Tip:** "Explain the expand-contract pattern: Phase 1 (expand) — add new fields, write to both old and new format. Phase 2 — migrate readers to use new fields. Phase 3 (contract) — remove old fields. This avoids downtime and allows safe rollback. Contrast with RDBMS where ALTER TABLE on a billion-row table can lock the table for minutes."

---

## Column-Family Stores

### 17. Describe the data structure in a column-family store and how it supports certain query types . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

A **column-family store** organizes data into **row keys**, **column families**, and **columns** (qualifiers). Each row can have **different columns** within a column family. Data is stored **column-wise** on disk, enabling efficient **range scans**, **time-series queries**, and **wide-row access** patterns.

```
  COLUMN-FAMILY DATA STRUCTURE (Cassandra/HBase):
  
  Keyspace: social_media
  Table: user_posts
  
  ROW KEY    | COLUMN FAMILY: post_data          | COLUMN FAMILY: metadata
  -----------+-----------------------------------+------------------------
  user:alice | post:001 -> "Hello!"              | name -> "Alice"
             | post:002 -> "NoSQL rocks"         | joined -> "2024-01"
             | post:003 -> "Learning Cassandra"  | followers -> 1500
  -----------+-----------------------------------+------------------------
  user:bob   | post:001 -> "First post"          | name -> "Bob"
             |                                   | joined -> "2024-03"
             |                                   | location -> "NYC"
  -----------+-----------------------------------+------------------------
  
  KEY CONCEPTS:
  - Row Key: Primary lookup key (partitioned across nodes)
  - Column Family: Logical grouping (like a table)
  - Column: Key-value pair within a family
  - Each row can have DIFFERENT columns (sparse)
  - Columns sorted within each family (enables range scans)
  
  STORAGE ON DISK (Column-Oriented):
  +------------------+     +------------------+
  | post_data SSTable|     | metadata SSTable  |
  +------------------+     +------------------+
  | alice:post:001   |     | alice:name       |
  | alice:post:002   |     | alice:joined     |
  | alice:post:003   |     | alice:followers  |
  | bob:post:001     |     | bob:name         |
  +------------------+     | bob:joined       |
                           | bob:location     |
                           +------------------+
  
  QUERY PATTERNS SUPPORTED:
  1. Get all posts by user:   Scan row key "user:alice", CF "post_data"
  2. Get recent posts:        Range scan post:090..post:100 (sorted!)
  3. Get user metadata:       Point read row "user:alice", CF "metadata"
  4. Time-range scan:         Columns sorted by timestamp
```

| Feature | Column-Family Store | RDBMS | Document DB |
|---------|-------------------|-------|-------------|
| **Row structure** | Sparse, variable columns | Fixed columns | Nested JSON |
| **Storage** | Column-oriented SSTables | Row-oriented pages | Document BSON |
| **Best query** | Range scans, time-series | Complex JOINs | Single-doc reads |
| **Write pattern** | Append-only (LSM-tree) | In-place update (B-tree) | In-place or append |
| **Sorting** | Columns sorted per row | Requires index | Requires index |
| **Scalability** | Linear horizontal | Vertical + read replicas | Horizontal (sharding) |

```python
# Column-family store simulation
from collections import defaultdict, OrderedDict

class ColumnFamilyStore:
    """Simulates Cassandra/HBase column-family data model."""

    def __init__(self):
        # Structure: {row_key: {column_family: OrderedDict({column: value})}}
        self.data = defaultdict(lambda: defaultdict(OrderedDict))

    def put(self, row_key, column_family, column, value, timestamp=None):
        """Write a single cell (row_key, cf, column) -> value."""
        ts = timestamp or __import__('time').time()
        self.data[row_key][column_family][column] = (value, ts)

    def get(self, row_key, column_family, column=None):
        """Point read: single cell or entire column family."""
        cf = self.data.get(row_key, {}).get(column_family, {})
        if column:
            cell = cf.get(column)
            return cell[0] if cell else None
        return {k: v[0] for k, v in cf.items()}

    def range_scan(self, row_key, column_family, start_col, end_col):
        """Range scan: get columns in sorted range (time-series!)."""
        cf = self.data.get(row_key, {}).get(column_family, OrderedDict())
        return {k: v[0] for k, v in cf.items() if start_col <= k <= end_col}

    def scan_rows(self, start_key, end_key, column_family):
        """Scan multiple rows in key range."""
        return {rk: {k: v[0] for k, v in cfs.get(column_family, {}).items()}
                for rk, cfs in self.data.items()
                if start_key <= rk <= end_key}

# Time-series use case
store = ColumnFamilyStore()
store.put("sensor:temp_01", "readings", "2024-03-15T10:00", 22.5)
store.put("sensor:temp_01", "readings", "2024-03-15T10:05", 22.8)
store.put("sensor:temp_01", "readings", "2024-03-15T10:10", 23.1)
store.put("sensor:temp_01", "metadata", "location", "Building A")

# Range scan: get readings from 10:00 to 10:10
readings = store.range_scan("sensor:temp_01", "readings",
                            "2024-03-15T10:00", "2024-03-15T10:10")
```

**AI/ML Application:**
- **Feature store (time-series):** Store ML features keyed by entity ID with timestamped columns. Cassandra enables fast "get features for user X at time T" queries for point-in-time training data.
- **Training metrics logging:** Row per experiment, columns per epoch — `epoch:001 -> {loss: 0.5, acc: 0.8}`. Range scan retrieves all metrics for a training run.
- **Event streams:** User activity events stored as wide rows — efficient for building user behavior feature vectors.

**Real-World Example:**
Apache Cassandra powers Netflix's viewing history — each user is a row key, each viewing event is a timestamped column (wide row). Instagram uses Cassandra for feed storage. HBase (Hadoop column-family) is the backend for Facebook Messenger's message storage — messages stored as timestamped columns per conversation row key.

> **Interview Tip:** "Column-family stores excel at two patterns: (1) wide-row scans — one row key with thousands of sorted columns (time-series, user activity), and (2) high-throughput writes — LSM-tree append-only storage is faster than B-tree in-place updates. The key design principle is 'query-driven modeling' — design your column families around your access patterns, not your entity relationships."

---

### 18. Name a popular column-family store and its key features that contribute to its performance. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

**Apache Cassandra** is the most popular column-family store. Its performance comes from **masterless architecture** (no single point of failure), **LSM-tree storage** (fast writes), **tunable consistency** (per-query trade-offs), **linear scalability** (add nodes for throughput), and **token-based partitioning** (consistent hashing).

```
  CASSANDRA ARCHITECTURE:
  
  +--------+    +--------+    +--------+
  | Node 1 |<-->| Node 2 |<-->| Node 3 |
  | Token:  |    | Token:  |    | Token:  |
  | 0-100   |    | 101-200 |    | 201-300 |
  +---+----+    +---+----+    +---+----+
      |    \        |   \        |    \
      |     +-------+    +------+     |
      +------------>|<-----------+
                    |
          +--------+--------+
          | Gossip Protocol  |
          | (Peer Discovery) |
          +-----------------+
  
  WRITE PATH (LSM-Tree):
  Client Write
      |
      v
  +---------------+
  | Commit Log    |  <-- Durability (WAL)
  | (append-only) |
  +------+--------+
         |
         v
  +---------------+
  | MemTable      |  <-- In-memory (fast!)
  | (sorted map)  |
  +------+--------+
         | (flush when full)
         v
  +---------------+
  | SSTable       |  <-- On-disk (immutable)
  | (sorted file) |
  +------+--------+
         | (background)
         v
  +---------------+
  | Compaction     |  <-- Merge SSTables
  | (merge sort)   |
  +----------------+
  
  READ PATH:
  Client Read --> MemTable (cache) --> Bloom Filter --> SSTable
                                        |
                                   Skip if key
                                   not in SSTable
  
  CONSISTENCY LEVELS:
  +----------+----------------------------+
  | Level    | Guarantees                 |
  +----------+----------------------------+
  | ONE      | 1 replica ACK (fastest)    |
  | QUORUM   | N/2+1 ACK (balanced)       |
  | ALL      | All replicas ACK (safest)  |
  | LOCAL_Q  | Quorum in local DC         |
  +----------+----------------------------+
```

| Feature | Cassandra | HBase | ScyllaDB |
|---------|----------|-------|----------|
| **Architecture** | Masterless (ring) | Master-slave | Masterless (ring) |
| **Language** | Java | Java | C++ |
| **Write speed** | ~100K/s per node | ~10K/s per node | ~1M/s per node |
| **Consistency** | Tunable (ONE to ALL) | Strong (CP) | Tunable (ONE to ALL) |
| **Query language** | CQL (SQL-like) | Java API / Phoenix | CQL compatible |
| **Use case** | High-write, multi-DC | Hadoop ecosystem, analytics | Drop-in Cassandra replacement |

```python
# Cassandra-like write path simulation
import time
import hashlib
from collections import OrderedDict

class CassandraNode:
    """Simulates Cassandra's write path (LSM-tree)."""

    def __init__(self, node_id, memtable_limit=4):
        self.node_id = node_id
        self.commit_log = []       # WAL for durability
        self.memtable = OrderedDict()  # In-memory sorted map
        self.sstables = []         # Immutable on-disk files
        self.memtable_limit = memtable_limit
        self.bloom_filters = []    # Probabilistic key existence

    def write(self, key, value):
        """Write path: commit log -> memtable -> (flush to SSTable)."""
        # Step 1: Append to commit log (durability)
        self.commit_log.append((key, value, time.time()))
        # Step 2: Write to memtable (in-memory, fast!)
        self.memtable[key] = value
        # Step 3: Flush to SSTable if memtable full
        if len(self.memtable) >= self.memtable_limit:
            self._flush_memtable()

    def read(self, key):
        """Read path: memtable -> bloom filters -> SSTables."""
        # Check memtable first (most recent writes)
        if key in self.memtable:
            return self.memtable[key]
        # Check SSTables (newest first) with bloom filter
        for i, (sstable, bloom) in enumerate(
                zip(reversed(self.sstables), reversed(self.bloom_filters))):
            if key in bloom:  # Bloom filter: might exist
                if key in sstable:
                    return sstable[key]
        return None

    def _flush_memtable(self):
        """Flush memtable to immutable SSTable."""
        sstable = OrderedDict(sorted(self.memtable.items()))
        bloom = set(sstable.keys())  # Simplified bloom filter
        self.sstables.append(sstable)
        self.bloom_filters.append(bloom)
        self.memtable.clear()

class CassandraCluster:
    """Simulates Cassandra ring with consistent hashing."""

    def __init__(self, num_nodes=3, replication_factor=2):
        self.nodes = [CassandraNode(i) for i in range(num_nodes)]
        self.rf = replication_factor

    def _get_replicas(self, key):
        """Consistent hashing: determine which nodes store a key."""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        primary = hash_val % len(self.nodes)
        return [self.nodes[(primary + i) % len(self.nodes)]
                for i in range(self.rf)]

    def write(self, key, value, consistency="QUORUM"):
        replicas = self._get_replicas(key)
        acks = 0
        for node in replicas:
            node.write(key, value)
            acks += 1
        required = {"ONE": 1, "QUORUM": len(replicas)//2+1, "ALL": len(replicas)}
        return acks >= required[consistency]
```

**AI/ML Application:**
- **Real-time feature serving:** Cassandra's sub-millisecond reads serve ML features at prediction time (Uber uses Cassandra for real-time feature serving in their ML platform).
- **Event logging for training:** High write throughput captures millions of user events/sec for building training datasets.
- **Multi-DC ML pipelines:** Tunable consistency enables training in one DC while serving predictions from another.

**Real-World Example:**
Apple operates one of the largest Cassandra deployments globally (over 150,000 nodes) for iCloud services. Netflix uses Cassandra for subscriber data, viewing history, and bookmarks — chosen for its masterless architecture (no SPOF) and multi-DC replication. Discord uses Cassandra for message storage (billions of messages) — chosen for fast writes and time-range queries.

> **Interview Tip:** "Cassandra's key performance features: (1) LSM-tree writes are append-only (faster than B-tree in-place updates), (2) bloom filters skip irrelevant SSTables during reads, (3) consistent hashing distributes data evenly, (4) masterless gossip protocol means no SPOF. The trade-off: reads can be slower than writes (must check memtable + multiple SSTables), and compaction causes periodic I/O spikes."

---

### 19. How are data partitioning and distribution handled in column-family stores ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Column-family stores use **consistent hashing** to partition data across nodes. Each node owns a **token range** on a hash ring. The **partition key** (first part of the primary key) determines which node stores the data. **Virtual nodes (vnodes)** improve load distribution. **Replication** copies data to adjacent nodes for fault tolerance.

```
  CONSISTENT HASH RING (Cassandra):
  
          Token: 0
            |
     +------+------+
     |  Node A     |
     |  Tokens:    |
     |  0-63       |
     +------+------+
            |
  Token:    |          Token: 64
  192 ------+------ Node B
     |             Tokens:
  Node D           64-127
  Tokens:             |
  192-255      +------+------+
     |         |  Node C     |
     +---------+  Tokens:    |
               |  128-191    |
               +-------------+
                    Token: 128
  
  PARTITION KEY DETERMINES NODE:
  INSERT INTO users (user_id, name)
  VALUES ('alice', 'Alice Smith');
  
  hash('alice') = 42  --> Token range 0-63 --> Node A
  hash('bob')   = 150 --> Token range 128-191 --> Node C
  
  VIRTUAL NODES (vnodes):
  Without vnodes:     With vnodes (256/node):
  +---+---+---+---+   +--+--+--+--+--+--+--+--+
  | A | B | C | D |   |A |C |B |D |A |B |D |C |
  +---+---+---+---+   +--+--+--+--+--+--+--+--+
  Uneven if node dies  Even: vnodes redistributed
  
  REPLICATION (RF=3):
  hash('alice') = 42
  Primary: Node A (owns token 42)
  Replica 1: Node B (next on ring)
  Replica 2: Node C (next after B)
  
  +-------+     +-------+     +-------+
  | Node A| --> | Node B| --> | Node C|
  |Primary|     |Replica|     |Replica|
  | alice |     | alice |     | alice |
  +-------+     +-------+     +-------+
  
  COMPOUND PRIMARY KEY:
  PRIMARY KEY ((partition_key), clustering_col1, clustering_col2)
               ^                ^
               |                |
        Determines NODE    Determines ORDER within partition
```

| Aspect | Consistent Hashing | Range Partitioning | Hash Partitioning |
|--------|-------------------|--------------------|-------------------|
| **Used by** | Cassandra, DynamoDB | HBase, Bigtable | MongoDB (hashed) |
| **Distribution** | Even (with vnodes) | Can be uneven (hotspots) | Even |
| **Range queries** | Only within partition | Efficient across partitions | Not supported |
| **Rebalancing** | Minimal data movement | Region split/merge | Full reshard |
| **Hot spots** | Possible (bad partition key) | Likely (sequential keys) | Rare |

```python
# Data partitioning and distribution simulation
import hashlib

class ConsistentHashRing:
    """Cassandra-style consistent hashing with virtual nodes."""

    def __init__(self, nodes, vnodes_per_node=256, replication_factor=3):
        self.rf = replication_factor
        self.ring = {}          # token -> physical_node
        self.sorted_tokens = []
        self.node_data = {n: {} for n in nodes}  # node -> data

        # Assign virtual nodes
        for node in nodes:
            for i in range(vnodes_per_node):
                token = self._hash(f"{node}:vnode{i}")
                self.ring[token] = node
        self.sorted_tokens = sorted(self.ring.keys())

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)

    def get_replicas(self, key):
        """Find RF distinct physical nodes for a partition key."""
        token = self._hash(key)
        replicas = []
        idx = self._find_token(token)
        seen = set()
        while len(replicas) < self.rf and len(seen) < len(set(self.ring.values())):
            node = self.ring[self.sorted_tokens[idx % len(self.sorted_tokens)]]
            if node not in seen:
                replicas.append(node)
                seen.add(node)
            idx += 1
        return replicas

    def _find_token(self, token):
        """Binary search for the token's position on ring."""
        lo, hi = 0, len(self.sorted_tokens) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.sorted_tokens[mid] >= token:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo % len(self.sorted_tokens)

    def write(self, partition_key, data):
        """Write to all replicas."""
        replicas = self.get_replicas(partition_key)
        for node in replicas:
            self.node_data[node][partition_key] = data
        return replicas

    def add_node(self, new_node):
        """Add node — only adjacent data moves (minimal disruption)."""
        self.node_data[new_node] = {}
        moved_keys = 0
        for i in range(256):  # Add vnodes
            token = self._hash(f"{new_node}:vnode{i}")
            self.ring[token] = new_node
        self.sorted_tokens = sorted(self.ring.keys())
        return moved_keys

# Demo
ring = ConsistentHashRing(["node_A", "node_B", "node_C"], vnodes_per_node=64)
replicas = ring.write("user:alice", {"name": "Alice", "age": 30})
print(f"'alice' stored on: {replicas}")  # 3 nodes (RF=3)
```

**AI/ML Application:**
- **Distributed feature store:** Partition features by entity ID across nodes. Each ML model request hits the correct partition to retrieve user features for prediction.
- **Distributed training data:** Partition training samples by hash — each worker node processes local data (data parallelism). Consistent hashing ensures balanced distribution.
- **Model shard routing:** Large models sharded across GPU nodes. Consistent hashing routes inference requests to the correct model shard.

**Real-World Example:**
Cassandra's vnodes (virtual nodes, default 256 per physical node) ensure that when a node fails, its load is distributed evenly across all remaining nodes — not just the next node on the ring. DynamoDB uses consistent hashing with partition splits when a partition exceeds 10GB or 3,000 RCU/1,000 WCU. HBase uses range-based partitioning with automatic region splits — a region server manages ranges of contiguous row keys.

> **Interview Tip:** "Always mention the partition key design: a bad partition key creates hot spots. For a social media app, partition by user_id (even distribution), NOT by date (all writes hit one partition). Compound primary keys: partition key determines WHICH node, clustering columns determine ORDER within that partition. When adding a node, consistent hashing moves only K/N keys (K = total keys, N = nodes) — far better than hash-mod which moves most keys."

---

## Graph Databases

### 20. Explain the data representation in a graph database . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Graph databases represent data as **nodes** (vertices/entities), **edges** (relationships), and **properties** (key-value attributes on both). This **property graph model** stores data as a network of connected entities with named, directed, typed relationships. This structure enables **index-free adjacency** — each node directly references its neighbors, making traversal O(1) per hop (vs. O(log n) for JOINs).

```
  PROPERTY GRAPH MODEL:
  
         +------------------+      FOLLOWS      +------------------+
         |    Node: User    |  <------------>   |    Node: User    |
         +------------------+     (since:2023)  +------------------+
         | id: "alice"      |                   | id: "bob"        |
         | name: "Alice"    |                   | name: "Bob"      |
         | age: 30          |                   | age: 25          |
         +--------+---------+                   +--------+---------+
                  |                                      |
                  | WROTE                                | LIKES
                  | (date: "2024-03")                    | (rating: 5)
                  v                                      v
         +------------------+                   +------------------+
         |   Node: Article  |                   |   Node: Article  |
         +------------------+                   +------------------+
         | id: "art_001"    |                   | id: "art_001"    |
         | title: "ML Intro"|  <--- Same node! ---+               |
         | tags: ["ML","AI"]|                                      |
         +------------------+                                      |
  
  INDEX-FREE ADJACENCY:
  
  Relational (JOIN):
  SELECT * FROM users u
  JOIN follows f ON u.id = f.follower_id  -- O(log n) index lookup
  JOIN users u2 ON f.following_id = u2.id -- O(log n) again
  -- Each hop: O(log n)
  
  Graph (Traversal):
  alice.outEdges("FOLLOWS")  -- O(1) pointer follow
    --> bob                  -- O(1) per hop!
  -- 6-hop traversal: O(6) vs O(6 * log n)
  
  GRAPH vs RELATIONAL STORAGE:
  +-------------------+---------------------------+
  | Relational        | Graph                     |
  +-------------------+---------------------------+
  | Users table       | User nodes + properties   |
  | Follows table     | FOLLOWS edges + properties|
  | JOIN on read      | Direct pointer traversal  |
  | O(log n) per hop  | O(1) per hop              |
  +-------------------+---------------------------+
```

| Concept | Definition | Example | Properties |
|---------|-----------|---------|-----------|
| **Node** | Entity/vertex | User, Product, Article | id, name, age, email |
| **Edge** | Relationship (directed) | FOLLOWS, WROTE, LIKES | since, weight, type |
| **Label** | Node/edge type | :User, :Article, :FOLLOWS | Enables type filtering |
| **Property** | Key-value on node/edge | `name: "Alice"` | Any primitive or array |
| **Path** | Sequence of nodes+edges | Alice->FOLLOWS->Bob->WROTE->Article | Length = # edges |

```python
# Property graph implementation
class Node:
    def __init__(self, node_id, label, **properties):
        self.id = node_id
        self.label = label
        self.properties = properties
        self.out_edges = []  # Direct pointers (index-free adjacency!)
        self.in_edges = []

    def __repr__(self):
        return f"({self.label}:{self.id})"

class Edge:
    def __init__(self, edge_type, source, target, **properties):
        self.type = edge_type
        self.source = source
        self.target = target
        self.properties = properties
        # Wire up index-free adjacency
        source.out_edges.append(self)
        target.in_edges.append(self)

    def __repr__(self):
        return f"{self.source}-[{self.type}]->{self.target}"

class GraphDB:
    """Property graph database with traversal operations."""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, label, **props):
        node = Node(node_id, label, **props)
        self.nodes[node_id] = node
        return node

    def add_edge(self, edge_type, source_id, target_id, **props):
        edge = Edge(edge_type, self.nodes[source_id],
                    self.nodes[target_id], **props)
        self.edges.append(edge)
        return edge

    def traverse(self, start_id, edge_type, depth=1):
        """BFS traversal following edges of given type."""
        visited = set()
        current = [self.nodes[start_id]]
        results = []
        for _ in range(depth):
            next_level = []
            for node in current:
                for edge in node.out_edges:
                    if edge.type == edge_type and edge.target.id not in visited:
                        visited.add(edge.target.id)
                        next_level.append(edge.target)
                        results.append(edge.target)
            current = next_level
        return results

    def shortest_path(self, start_id, end_id):
        """BFS shortest path between two nodes."""
        from collections import deque
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        while queue:
            current, path = queue.popleft()
            if current == end_id:
                return [self.nodes[nid] for nid in path]
            for edge in self.nodes[current].out_edges:
                if edge.target.id not in visited:
                    visited.add(edge.target.id)
                    queue.append((edge.target.id, path + [edge.target.id]))
        return None

# Build social graph
db = GraphDB()
db.add_node("alice", "User", name="Alice")
db.add_node("bob", "User", name="Bob")
db.add_node("carol", "User", name="Carol")
db.add_edge("FOLLOWS", "alice", "bob", since="2024-01")
db.add_edge("FOLLOWS", "bob", "carol", since="2024-02")
db.add_edge("FOLLOWS", "carol", "alice", since="2024-03")

# Traverse: who does Alice follow (depth 2)?
friends_of_friends = db.traverse("alice", "FOLLOWS", depth=2)
```

**AI/ML Application:**
- **Knowledge graphs:** Google's Knowledge Graph and Wikidata use graph representation for entity relationships — enabling ML-based question answering over structured knowledge.
- **Graph Neural Networks (GNNs):** Property graphs form the input to GCN/GAT models that learn node embeddings from neighborhood structure — used for drug discovery (molecule graphs), recommendation (user-item graphs), and fraud detection.
- **Feature engineering:** Graph features (PageRank, degree centrality, community membership) are powerful ML features.

**Real-World Example:**
Neo4j powers LinkedIn's knowledge graph for skill-job matching. NASA uses Neo4j to track component relationships across spacecraft systems. Panama Papers investigation used Neo4j to discover hidden corporate ownership networks — graph traversal revealed connections invisible in tabular data.

> **Interview Tip:** "Key distinction: graph databases store relationships as first-class citizens (edges with properties), not as foreign keys in junction tables. Index-free adjacency means traversal is O(1) per hop — on a billion-node graph, finding friends-of-friends 6 hops away is O(6), not O(6 * log(1B)). This makes graph DBs ideal for social networks, recommendation engines, and fraud detection."

---

### 21. Compare querying in graph databases with other NoSQL database types. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Graph database queries express **traversal patterns** (paths, neighbors, shortest path) declaratively, while other NoSQL types require **application-level graph traversal** using multiple queries. **Cypher** (Neo4j) and **Gremlin** (Apache TinkerPop) are purpose-built graph query languages that express complex relationship patterns in a few lines.

```
  QUERY COMPARISON - "Find Alice's friends who also like Jazz":
  
  1. GRAPH DB (Cypher / Neo4j):
     MATCH (alice:User {name:"Alice"})-[:FOLLOWS]->(friend)
           -[:LIKES]->(g:Genre {name:"Jazz"})
     RETURN friend.name
     // 3 lines, single query, O(k) where k = matched paths
  
  2. DOCUMENT DB (MongoDB):
     // Step 1: Find Alice
     alice = db.users.findOne({name: "Alice"})
     // Step 2: Find friends (separate collection or embedded)
     friends = db.users.find({_id: {$in: alice.following}})
     // Step 3: Filter friends who like Jazz
     jazz_friends = [f for f in friends if "Jazz" in f.get("likes",[])]
     // 3 queries, application-level joins, O(n) scan
  
  3. KEY-VALUE STORE (Redis):
     // Step 1: Get Alice's friend list
     friends = SMEMBERS "user:alice:following"
     // Step 2: For EACH friend, check likes
     for friend_id in friends:
         likes = SMEMBERS f"user:{friend_id}:likes"
         if "Jazz" in likes: yield friend_id
     // N+1 queries! Application-level logic
  
  4. COLUMN-FAMILY (Cassandra):
     // Must denormalize at write time!
     // Table: friends_who_like_genre
     // Partition: (user_id, genre)
     SELECT friend_name FROM friends_who_like_genre
     WHERE user_id = 'alice' AND genre = 'Jazz';
     // Fast read, but massive write amplification
  
  5. RELATIONAL (SQL):
     SELECT u2.name FROM users u1
     JOIN follows f ON u1.id = f.user_id
     JOIN users u2 ON f.friend_id = u2.id
     JOIN user_likes ul ON u2.id = ul.user_id
     JOIN genres g ON ul.genre_id = g.id
     WHERE u1.name = 'Alice' AND g.name = 'Jazz';
     // 4 JOINs, each O(log n) index lookup
```

| Query Type | Graph DB | Document DB | KV Store | Column-Family | RDBMS |
|-----------|---------|-------------|----------|---------------|-------|
| **Traversal (N hops)** | O(1)/hop native | N queries | N queries | Denormalize | N JOINs O(log n) |
| **Shortest path** | Built-in | Not supported | Not supported | Not supported | Recursive CTE |
| **Pattern match** | MATCH clause | Aggregation | Not supported | Not supported | Complex JOINs |
| **Aggregation** | Supported | Strong (pipeline) | Limited | Strong (CQL) | Strong (GROUP BY) |
| **Full-text search** | Plugin | Native (Atlas) | RediSearch | Solr integration | LIKE / FTS |
| **Range scan** | On properties | On indexed fields | Not supported | Primary use case | WHERE + index |

```python
# Comparing query approaches across NoSQL types
class QueryComparison:
    """Shows how the same query differs across database types."""

    @staticmethod
    def graph_query(graph_db):
        """Graph DB: Single traversal, O(friends) per hop."""
        # Cypher equivalent: MATCH (a)-[:FOLLOWS]->(f)-[:LIKES]->(g {name:"Jazz"})
        alice = graph_db.nodes["alice"]
        results = []
        for edge in alice.out_edges:
            if edge.type == "FOLLOWS":
                friend = edge.target
                for like_edge in friend.out_edges:
                    if like_edge.type == "LIKES" and \
                       like_edge.target.properties.get("name") == "Jazz":
                        results.append(friend)
        return results  # Direct pointer traversal!

    @staticmethod
    def document_query(doc_db):
        """Document DB: Multiple queries + app-level join."""
        alice = doc_db.find_one("users", {"name": "Alice"})
        friend_ids = alice.get("following", [])
        friends = doc_db.find("users", {"_id": {"$in": friend_ids}})
        return [f for f in friends if "Jazz" in f.get("likes", [])]

    @staticmethod
    def kv_query(kv_store):
        """KV Store: N+1 queries pattern."""
        friend_ids = kv_store.get("user:alice:following")  # Query 1
        results = []
        for fid in friend_ids:  # N queries
            likes = kv_store.get(f"user:{fid}:likes")
            if "Jazz" in (likes or []):
                results.append(fid)
        return results

    @staticmethod
    def compare_complexity():
        """Complexity comparison for depth-D traversal."""
        return {
            "graph_db": "O(D) per path — index-free adjacency",
            "document_db": "O(D * N) — D queries, each scans N docs",
            "kv_store": "O(D * branching_factor) — D rounds of gets",
            "column_family": "O(1) read — but O(D!) denormalized writes",
            "rdbms": "O(D * log N) — D JOINs with index lookups"
        }
```

**AI/ML Application:**
- **Recommendation queries:** "Find products liked by users who are similar to User X" — a 3-hop graph traversal in Cypher, vs. 3 separate queries + application-level joins in MongoDB.
- **GNN neighbor sampling:** Graph databases natively support "sample K random neighbors of each node" — a core operation for graph neural network training (GraphSAGE, GAT).
- **Knowledge graph QA:** SPARQL/Cypher queries over knowledge graphs answer complex questions like "Which proteins interact with drug X AND are expressed in liver tissue?"

**Real-World Example:**
Facebook's TAO (graph-like system) handles 1B+ reads/sec for social graph queries — "show friends who liked this post" is a multi-hop traversal. Airbnb uses a graph approach for search ranking — host trust, guest reviews, and booking history form a graph that informs ranking. Twitter's "Who to Follow" uses graph traversal (friends-of-friends) combined with ML to recommend users.

> **Interview Tip:** "Frame the comparison around the N+1 query problem: document and KV stores require N+1 queries to traverse relationships (1 to get the entity, N to fetch related entities). Graph databases eliminate this with index-free adjacency — each node directly points to its neighbors. However, graph DBs aren't optimal for bulk analytics or simple CRUD — use the right tool for each pattern."

---

### 22. Suggest a real-world problem that fits a graph database solution. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

**Fraud detection in financial networks** is the ideal graph database problem. Fraudulent transactions form **rings**, **chains**, and **hub-and-spoke patterns** that are invisible in tabular data but immediately visible as graph structures. Graph traversal detects these patterns in real-time.

```
  FRAUD DETECTION GRAPH:
  
  Normal transaction:
  (Alice) --[$500]--> (Bob)
  
  Fraud ring (circular money movement):
  (Shell_Co_A) --[$10K]--> (Shell_Co_B) --[$9.8K]--> (Shell_Co_C)
       ^                                                    |
       |                  $9.5K                             |
       +----------------------------------------------------+
  
  Money laundering layering:
  (Criminal) --[$1M]--> (Offshore_1) --[$500K]--> (Corp_A)
             --[$1M]--> (Offshore_2) --[$500K]--> (Corp_B)
                                     --[$500K]--> (Corp_C)
  
  Alert: Cycle detected!    Alert: Fan-out pattern!
  
  CYPHER QUERY TO DETECT FRAUD RING:
  MATCH path = (a:Account)-[:TRANSFER*3..6]->(a)
  WHERE ALL(r IN relationships(path) WHERE
    r.amount > 5000 AND
    r.timestamp > datetime() - duration('P7D'))
  RETURN path
  // Finds circular transfers > $5K in last 7 days
  
  SUPPLEMENTARY PATTERNS:
  +--------------------+-------------------------------+
  | Pattern            | Graph Query                   |
  +--------------------+-------------------------------+
  | Fraud ring         | Cycle detection (depth 3-6)   |
  | Money laundering   | Fan-out/fan-in traversal      |
  | Identity theft     | Shared attributes (SSN, addr) |
  | Account takeover   | Device/IP node reuse          |
  | Synthetic identity | Similar attributes cluster    |
  +--------------------+-------------------------------+
  
  OTHER IDEAL GRAPH PROBLEMS:
  +---------------------+-----------------------------------+
  | Problem             | Why Graph?                        |
  +---------------------+-----------------------------------+
  | Social networks     | Friend/follow relationships       |
  | Recommendation      | User-Product-Category connections |
  | Supply chain        | Supplier -> Manufacturer -> Retail|
  | Network topology    | Router connectivity, failure paths|
  | Knowledge graphs    | Entity-Relationship reasoning     |
  +---------------------+-----------------------------------+
```

| Graph Problem | Key Query | Depth | Relational Alternative |
|--------------|-----------|-------|----------------------|
| **Fraud rings** | Cycle detection | 3-6 hops | Recursive CTE (slow) |
| **Social recommendations** | Friends-of-friends | 2-3 hops | Multiple JOINs |
| **Shortest supply route** | Shortest path | N hops | Dijkstra in app code |
| **Impact analysis** | Subgraph reachability | N hops | Recursive CTE |
| **Access control** | Role-permission traversal | 2-4 hops | Multiple JOINs |

```python
# Fraud detection with graph database
from collections import deque

class FraudDetector:
    """Graph-based fraud detection system."""

    def __init__(self, graph_db):
        self.graph = graph_db
        self.alerts = []

    def detect_cycles(self, max_depth=6, min_amount=5000):
        """Detect fraud rings: circular money flows."""
        cycles = []
        for node_id, node in self.graph.nodes.items():
            if node.label != "Account":
                continue
            # DFS for cycles starting and ending at this node
            found = self._find_cycles(node_id, max_depth, min_amount)
            cycles.extend(found)
        return cycles

    def _find_cycles(self, start_id, max_depth, min_amount):
        """DFS cycle detection from a starting account."""
        cycles = []
        stack = [(start_id, [start_id], 0)]
        while stack:
            current_id, path, depth = stack.pop()
            if depth >= max_depth:
                continue
            node = self.graph.nodes[current_id]
            for edge in node.out_edges:
                if edge.type != "TRANSFER":
                    continue
                if edge.properties.get("amount", 0) < min_amount:
                    continue
                next_id = edge.target.id
                if next_id == start_id and depth >= 2:
                    cycles.append(list(path))  # Found cycle!
                elif next_id not in path:
                    stack.append((next_id, path + [next_id], depth + 1))
        return cycles

    def detect_fan_pattern(self, threshold=5):
        """Detect fan-out/fan-in money laundering."""
        alerts = []
        for node_id, node in self.graph.nodes.items():
            out_transfers = [e for e in node.out_edges if e.type == "TRANSFER"]
            in_transfers = [e for e in node.in_edges if e.type == "TRANSFER"]
            if len(out_transfers) >= threshold:
                alerts.append({"type": "fan_out", "account": node_id,
                               "count": len(out_transfers)})
            if len(in_transfers) >= threshold:
                alerts.append({"type": "fan_in", "account": node_id,
                               "count": len(in_transfers)})
        return alerts

    def shared_identity_check(self, attribute="ssn"):
        """Find accounts sharing suspicious attributes."""
        attr_map = {}
        for node_id, node in self.graph.nodes.items():
            val = node.properties.get(attribute)
            if val:
                attr_map.setdefault(val, []).append(node_id)
        return {k: v for k, v in attr_map.items() if len(v) > 1}
```

**AI/ML Application:**
- **Graph-based fraud scoring:** Train a GNN on the transaction graph — node features (account age, balance) + edge features (transfer amount, frequency) + graph structure (cycles, centrality) produce a fraud probability score per transaction.
- **Anomaly detection:** Graph embedding (Node2Vec, DeepWalk) converts graph structure into vectors. Anomalous patterns (sudden new connections, cycle formation) show as outliers in embedding space.
- **Dynamic fraud detection:** Temporal graph neural networks (T-GNN) track how the transaction graph evolves — detecting emerging fraud rings before they complete.

**Real-World Example:**
PayPal uses graph analysis to detect fraud rings — their system processes 1.1B+ transactions and identified $1.4B+ in potential fraud using graph algorithms. HSBC and Citibank use Neo4j for anti-money laundering compliance. The Panama Papers investigation used Neo4j to analyze 2.6TB of data (11.5M documents) revealing hidden corporate networks used for tax evasion.

> **Interview Tip:** "Lead with the specific Cypher query: `MATCH path = (a:Account)-[:TRANSFER*3..6]->(a) RETURN path` — this finds cyclic transfers in one query. In SQL, this requires recursive CTEs with depth limits, which are orders of magnitude slower on large datasets. Mention that graph databases also detect patterns impossible in tabular analysis: shared device fingerprints, common IP addresses, and geographic impossibilities."

---

### 23. Define the role of an edge in a graph database and its relationship with nodes . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

An **edge** (relationship) is a **first-class entity** connecting two nodes with a **type**, **direction**, and **properties**. Edges are the defining feature of graph databases — they are stored and indexed independently, making relationship traversal O(1). Edges transform isolated data points into a connected network of meaning.

```
  EDGE ANATOMY:
  
  (Source Node) --[EDGE_TYPE {properties}]--> (Target Node)
  
  Example:
  (alice:User) --[:FOLLOWS {since: "2024-01", close_friend: true}]--> (bob:User)
  
  Components:
  +-------------------+
  | Edge              |
  +-------------------+
  | type: "FOLLOWS"   |  <-- Named relationship type
  | direction: -->    |  <-- Source to target
  | source: alice     |  <-- Start node pointer
  | target: bob       |  <-- End node pointer
  | since: "2024-01"  |  <-- Edge property
  | close_friend: true|  <-- Edge property
  +-------------------+
  
  EDGE TYPES AND PATTERNS:
  
  1. DIRECTED (one-way):
     (Alice) --[:FOLLOWS]--> (Bob)
     Alice follows Bob, but Bob may not follow Alice
  
  2. BIDIRECTIONAL (modeled as two edges):
     (Alice) --[:FRIENDS]--> (Bob)
     (Bob)   --[:FRIENDS]--> (Alice)
  
  3. SELF-REFERENCING:
     (Company) --[:SUBSIDIARY_OF]--> (Company)
  
  4. WEIGHTED:
     (A) --[:ROUTE {distance: 100, cost: 50}]--> (B)
  
  5. TEMPORAL:
     (Alice) --[:EMPLOYED_AT {start: 2020, end: 2023}]--> (Google)
     (Alice) --[:EMPLOYED_AT {start: 2023, end: null}]--> (Meta)
  
  STORAGE: INDEX-FREE ADJACENCY
  +--------+    +--------+    +--------+
  | Node A |    | Edge 1 |    | Node B |
  | out: --|----->source:A|    | in:  --|--+
  |        |    | target:B|----->       |  |
  +--------+    | type:FOL|    +--------+  |
                +--------+                 |
                +--------+                 |
                | Edge 2 |<----------------+
                | source:B|
                | target:C|--> ...
                +--------+
  Each node has a linked list of edges (O(1) access)
```

| Edge Aspect | Description | Example |
|------------|------------|---------|
| **Type/Label** | Names the relationship | :FOLLOWS, :PURCHASED, :MANAGES |
| **Direction** | Source → Target | (Manager) → (Employee) |
| **Properties** | Metadata on relationship | since, weight, confidence |
| **Cardinality** | How many edges per node | 1:1, 1:N, N:M all natural |
| **Traversal** | Following edges hop-by-hop | Path, shortest path, pattern match |

```python
# Edge as first-class entity with rich properties
class GraphEdge:
    """Rich edge with type, direction, properties, and temporal tracking."""

    def __init__(self, edge_id, edge_type, source, target, **properties):
        self.id = edge_id
        self.type = edge_type
        self.source = source  # Node reference (pointer)
        self.target = target  # Node reference (pointer)
        self.properties = properties
        self.created_at = __import__('datetime').datetime.now()
        # Wire up: index-free adjacency
        source.out_edges.append(self)
        target.in_edges.append(self)

    def reverse(self):
        """Traverse edge in reverse direction."""
        return self.source  # From target, go back to source

class GraphNode:
    def __init__(self, node_id, label, **properties):
        self.id = node_id
        self.label = label
        self.properties = properties
        self.out_edges = []  # Edges where this node is source
        self.in_edges = []   # Edges where this node is target

    def neighbors(self, edge_type=None, direction="out"):
        """Get connected nodes filtered by edge type and direction."""
        edges = self.out_edges if direction == "out" else self.in_edges
        if edge_type:
            edges = [e for e in edges if e.type == edge_type]
        return [e.target if direction == "out" else e.source for e in edges]

    def degree(self, edge_type=None):
        """Count connections (important graph metric)."""
        out = len([e for e in self.out_edges if not edge_type or e.type == edge_type])
        in_ = len([e for e in self.in_edges if not edge_type or e.type == edge_type])
        return {"in": in_, "out": out, "total": in_ + out}

    def has_relationship(self, target_id, edge_type=None):
        """Check if direct relationship exists (O(degree) check)."""
        for edge in self.out_edges:
            if edge.target.id == target_id:
                if not edge_type or edge.type == edge_type:
                    return True
        return False

# Usage: Social network with rich edges
alice = GraphNode("alice", "User", name="Alice")
bob = GraphNode("bob", "User", name="Bob")
article = GraphNode("art1", "Article", title="Graph DBs 101")

# Edges carry meaning and properties
GraphEdge("e1", "FOLLOWS", alice, bob, since="2024-01", close_friend=True)
GraphEdge("e2", "WROTE", alice, article, date="2024-03", draft=False)
GraphEdge("e3", "LIKES", bob, article, rating=5)

# Traverse: who does Alice follow?
alice_follows = alice.neighbors("FOLLOWS")  # [bob] — O(1)!
```

**AI/ML Application:**
- **Link prediction:** Given nodes A and B with no edge, predict the probability of a future relationship. GNNs learn from edge features (type, properties, creation patterns) to predict missing edges — used for recommending connections on LinkedIn.
- **Edge classification:** Classify edges as benign or fraudulent based on edge properties (amount, frequency, time) and surrounding graph structure.
- **Weighted graph algorithms:** Edge weights (confidence scores, distances) enable PageRank, betweenness centrality, and community detection — core features for ML models.

**Real-World Example:**
LinkedIn's graph has 800M+ member nodes connected by billions of edges (connections, follows, endorsements) — each edge type has different properties. Google's Knowledge Graph uses typed edges to represent facts: `(Paris)-[:CAPITAL_OF]->(France)`. Uber Eats uses weighted edges (distance, ETA, rating) between restaurant, driver, and customer nodes for optimal delivery routing.

> **Interview Tip:** "Emphasize that edges are first-class citizens — they have their own properties, indexes, and query patterns. In RDBMS, relationships are implicit (foreign keys in junction tables). In graph DBs, relationships are explicit, named, and queryable. This means you can ask 'show all FOLLOWS relationships created this month' directly — in SQL that's a query on a junction table with no semantic meaning."

---

## Data Consistency & Replication

### 24. What replication strategies are often used in NoSQL databases ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

NoSQL databases use **master-slave** (single writer, multiple readers), **multi-master** (any node writes), **masterless/peer-to-peer** (all nodes equal), and **chain replication** strategies. Each trades off between **write throughput**, **consistency guarantees**, and **availability** during failures.

```
  REPLICATION STRATEGIES:
  
  1. MASTER-SLAVE (Single Leader):
     +--------+      +--------+      +--------+
     | Master | ---> | Slave 1| ---> | Slave 2|
     | (Write)|      | (Read) |      | (Read) |
     +--------+      +--------+      +--------+
     Writes: Master only
     Reads: Any node
     Failover: Elect new master
     Used by: MongoDB, Redis
  
  2. MULTI-MASTER (Multiple Leaders):
     +--------+      +--------+
     | Master | <--> | Master |
     | DC-East|      | DC-West|
     +---+----+      +---+----+
         |                |
     +---+----+      +---+----+
     | Slave  |      | Slave  |
     +--------+      +--------+
     Writes: Any master
     Conflict: Last-Write-Wins / CRDTs
     Used by: CouchDB, Galera
  
  3. MASTERLESS / PEER-TO-PEER:
     +------+    +------+    +------+
     |Node 1|<-->|Node 2|<-->|Node 3|
     +--+---+    +--+---+    +--+---+
        |           |           |
        +-----+-----+-----+----+
              |           |
           +--+---+    +--+---+
           |Node 4|<-->|Node 5|
           +------+    +------+
     Writes: ANY node (with quorum)
     Reads: Quorum (R+W > N)
     Anti-entropy: Merkle trees, read repair
     Used by: Cassandra, DynamoDB, Riak
  
  4. CHAIN REPLICATION:
     Client --> [Head] --> [Middle] --> [Tail] --> Client
     Write: Head        Read: Tail
     Strong consistency, high throughput
     Used by: HDFS, Azure Storage
```

| Strategy | Consistency | Write Throughput | Availability | Conflict Handling |
|----------|-----------|-----------------|-------------|-------------------|
| **Master-Slave** | Strong (sync) or eventual (async) | Limited (single writer) | Read: High, Write: leader SPOF | No conflicts (single writer) |
| **Multi-Master** | Eventual | High (multiple writers) | High (any master writes) | LWW, vector clocks, CRDTs |
| **Masterless** | Tunable (quorum) | High (any node writes) | Highest (no SPOF) | Read repair, anti-entropy |
| **Chain** | Strong | High (pipelined) | Moderate (reconfig on failure) | No conflicts (ordered chain) |

```python
# Replication strategy implementations
import time
import threading
from collections import defaultdict

class MasterSlaveReplication:
    """Single-leader replication with async followers."""

    def __init__(self, num_replicas=2):
        self.master = {"data": {}, "wal": []}
        self.replicas = [{"data": {}, "offset": 0} for _ in range(num_replicas)]

    def write(self, key, value):
        """All writes go to master."""
        entry = {"key": key, "value": value, "ts": time.time()}
        self.master["wal"].append(entry)
        self.master["data"][key] = value
        self._async_replicate(entry)

    def _async_replicate(self, entry):
        """Async replication to followers (eventual consistency)."""
        for replica in self.replicas:
            replica["data"][entry["key"]] = entry["value"]
            replica["offset"] += 1

    def read(self, key, from_replica=False):
        if from_replica:
            return self.replicas[0]["data"].get(key)  # May be stale!
        return self.master["data"].get(key)

class MasterlessReplication:
    """Cassandra-style quorum-based replication."""

    def __init__(self, num_nodes=5, replication_factor=3):
        self.nodes = [{} for _ in range(num_nodes)]
        self.rf = replication_factor
        self.n = num_nodes

    def _get_replicas(self, key):
        primary = hash(key) % self.n
        return [(primary + i) % self.n for i in range(self.rf)]

    def write(self, key, value, consistency="QUORUM"):
        """Write to RF nodes, wait for consistency level ACKs."""
        replicas = self._get_replicas(key)
        ts = time.time()
        acks = 0
        for node_idx in replicas:
            self.nodes[node_idx][key] = {"value": value, "ts": ts}
            acks += 1
        required = {"ONE": 1, "QUORUM": self.rf // 2 + 1, "ALL": self.rf}
        return acks >= required[consistency]

    def read(self, key, consistency="QUORUM"):
        """Read from multiple replicas, return latest."""
        replicas = self._get_replicas(key)
        responses = []
        for node_idx in replicas:
            data = self.nodes[node_idx].get(key)
            if data:
                responses.append(data)
        if not responses:
            return None
        # Return value with highest timestamp (read repair)
        latest = max(responses, key=lambda r: r["ts"])
        return latest["value"]
```

**AI/ML Application:**
- **Feature store replication:** ML feature stores replicate features across regions so prediction services read from local replicas (low latency). Master-slave ensures training writes go to one source of truth.
- **Model registry sync:** Multi-master replication syncs model versions across data centers — each team deploys models locally while staying eventually consistent.
- **Training data pipelines:** Chain replication ensures training data flows through preprocessing stages with strong ordering guarantees.

**Real-World Example:**
MongoDB uses primary-secondary (master-slave) replication with automatic failover via Raft consensus. Cassandra uses masterless replication with tunable consistency — Netflix runs multi-DC Cassandra clusters with LOCAL_QUORUM for low-latency reads. DynamoDB uses leaderless replication internally with sloppy quorum and hinted handoff. CockroachDB uses Raft-based synchronous replication for strong consistency across replicas.

> **Interview Tip:** "Frame replication as a spectrum: master-slave gives simplicity and strong consistency but has a write bottleneck (single leader). Masterless gives maximum availability and write throughput but requires conflict resolution. The key formula: R + W > N guarantees read-your-writes consistency in quorum systems. Always mention the CAP theorem trade-off when discussing replication strategies."

---

### 25. What are quorum reads/writes and their impact on consistency in NoSQL databases ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

A **quorum** requires a **majority** of replicas to acknowledge an operation. For N replicas, **W** (write quorum) + **R** (read quorum) > **N** guarantees **strong consistency** — every read sees the latest write. **Tunable consistency** adjusts W and R per query to trade off between latency and consistency.

```
  QUORUM FORMULA:
  W + R > N  -->  Guarantees overlap between write and read sets
  
  N = 3 replicas, RF = 3
  
  CASE 1: W=2, R=2 (QUORUM/QUORUM) --> Strong consistency
  Write: Node 1 [ACK], Node 2 [ACK], Node 3 [pending]
  Read:  Node 1 [v2],  Node 2 [v2],  Node 3 [v1-stale]
                  ^              ^
                  +-- Overlap! At least 1 node has latest --+
  
  CASE 2: W=1, R=3 (ONE/ALL) --> Fast writes, strong reads
  Write: Node 1 [ACK], Node 2 [later], Node 3 [later]
  Read:  Node 1 [v2],  Node 2 [v1],    Node 3 [v1]
         Return v2 (most recent timestamp)
  
  CASE 3: W=3, R=1 (ALL/ONE) --> Strong writes, fast reads
  Write: Node 1 [ACK], Node 2 [ACK], Node 3 [ACK]
  Read:  Node 1 [v2]  <-- Any single node has latest
  
  CASE 4: W=1, R=1 (ONE/ONE) --> Eventual consistency!
  W + R = 2 which is NOT > 3
  Write: Node 1 [ACK]
  Read:  Node 2 [v1-STALE!]  <-- No guaranteed overlap
  
  VISUAL: WRITE & READ OVERLAP
  
  Nodes:  [1]  [2]  [3]  [4]  [5]    (N=5, RF=5)
  
  W=3:    [W]  [W]  [W]  [ ]  [ ]
  R=3:    [ ]  [ ]  [R]  [R]  [R]
                      ^
                      +-- Overlap guaranteed (W+R=6 > 5)
  
  W=2:    [W]  [W]  [ ]  [ ]  [ ]
  R=2:    [ ]  [ ]  [ ]  [R]  [R]
  W+R=4 < 5 --> NO overlap guaranteed --> Eventual!
```

| Configuration | W | R | Consistency | Write Latency | Read Latency | Availability |
|--------------|---|---|-------------|---------------|--------------|-------------|
| **QUORUM/QUORUM** | N/2+1 | N/2+1 | Strong | Medium | Medium | Balanced |
| **ONE/ALL** | 1 | N | Strong | Fastest write | Slowest read | Write-favored |
| **ALL/ONE** | N | 1 | Strong | Slowest write | Fastest read | Read-favored |
| **ONE/ONE** | 1 | 1 | Eventual | Fastest | Fastest | Maximum |
| **LOCAL_QUORUM** | DC majority | DC majority | Strong (local) | Low (no cross-DC) | Low | Per-DC |

```python
# Quorum-based consistency simulation
import time
import random

class QuorumSystem:
    """Simulates quorum reads/writes with tunable consistency."""

    def __init__(self, num_nodes=5, replication_factor=3):
        self.nodes = [{"data": {}, "alive": True} for _ in range(num_nodes)]
        self.rf = replication_factor
        self.num_nodes = num_nodes

    def write(self, key, value, w=None):
        """Write with configurable quorum level."""
        if w is None:
            w = self.rf // 2 + 1  # Default: QUORUM
        replicas = self._get_replicas(key)
        ts = time.time()
        acks = 0
        for idx in replicas:
            if self.nodes[idx]["alive"]:
                self.nodes[idx]["data"][key] = {"value": value, "ts": ts}
                acks += 1
                if acks >= w:
                    return True  # Write quorum met
        return acks >= w

    def read(self, key, r=None):
        """Read with configurable quorum + read repair."""
        if r is None:
            r = self.rf // 2 + 1  # Default: QUORUM
        replicas = self._get_replicas(key)
        responses = []
        for idx in replicas:
            if self.nodes[idx]["alive"]:
                data = self.nodes[idx]["data"].get(key)
                if data:
                    responses.append((idx, data))
                if len(responses) >= r:
                    break
        if not responses:
            return None
        # Return latest value (highest timestamp)
        latest = max(responses, key=lambda x: x[1]["ts"])
        # Read repair: update stale replicas
        for idx, data in responses:
            if data["ts"] < latest[1]["ts"]:
                self.nodes[idx]["data"][key] = latest[1]
        return latest[1]["value"]

    def _get_replicas(self, key):
        primary = hash(key) % self.num_nodes
        return [(primary + i) % self.num_nodes for i in range(self.rf)]

    def simulate_partition(self, node_idx):
        """Simulate node failure."""
        self.nodes[node_idx]["alive"] = False

    def check_consistency(self, key):
        """Verify all replicas agree (for testing)."""
        replicas = self._get_replicas(key)
        values = set()
        for idx in replicas:
            data = self.nodes[idx]["data"].get(key)
            if data:
                values.add(data["value"])
        return len(values) <= 1  # True if consistent

# Demo: quorum guarantees
qs = QuorumSystem(num_nodes=5, replication_factor=3)
qs.write("user:1", "Alice", w=2)  # Write quorum = 2
result = qs.read("user:1", r=2)    # Read quorum = 2 (W+R=4 > 3: strong!)
```

**AI/ML Application:**
- **Feature serving consistency:** Use QUORUM reads for critical ML features (fraud score) to ensure models always see the latest values. Use ONE reads for non-critical features (recommendation scores) for faster inference latency.
- **A/B test assignment:** Quorum writes ensure experiment assignments are consistent — a user always sees the same variant regardless of which replica serves the request.
- **Model version coordination:** QUORUM writes when deploying a new model version ensure majority of serving nodes have the new model before routing traffic.

**Real-World Example:**
Cassandra allows per-query consistency: `SELECT * FROM features USING CONSISTENCY QUORUM`. DynamoDB uses quorum internally for strongly consistent reads (doubles read cost). Riak's `n_val`, `r`, and `w` parameters let operators tune consistency per bucket. Apache ZooKeeper uses strict quorum (majority) for all operations — no eventual consistency option.

> **Interview Tip:** "Always draw the W+R>N diagram showing overlapping sets. Key insight: QUORUM/QUORUM (W=2, R=2 with N=3) is the default because it tolerates 1 node failure for both reads and writes. Mention sloppy quorum (DynamoDB's hinted handoff) — when a designated replica is down, writes go to a temporary node that forwards data when the replica recovers. Sloppy quorum increases availability but weakens consistency guarantees."

---

### 26. Describe how NoSQL databases maintain consistency during network issues. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

During network partitions, NoSQL databases use **hinted handoff**, **read repair**, **anti-entropy** (Merkle trees), **conflict resolution** (LWW, vector clocks, CRDTs), and **gossip protocols** to detect and recover from inconsistencies. AP systems prioritize availability during partitions; CP systems reject writes to maintain consistency.

```
  NETWORK PARTITION SCENARIO:
  
  Normal:  [A] <--> [B] <--> [C]
  
  Partition: [A] <--> [B]  X  [C]  (C isolated)
  
  AP SYSTEM (Cassandra) RESPONSE:
  - Writes to A,B succeed (quorum: 2/3)
  - Writes to C succeed locally (hinted handoff)
  - After partition heals: anti-entropy syncs all nodes
  
  CP SYSTEM (MongoDB) RESPONSE:
  - If C was primary: election in {A,B} partition
  - Minority partition (C alone): rejects writes
  - Majority partition (A,B): elects new primary
  
  CONSISTENCY RECOVERY MECHANISMS:
  
  1. HINTED HANDOFF:
  Write for Node C (down) --> Node A stores hint
  +--------+                  +--------+
  | Node A |  hint: {         | Node C |
  | (alive)|  key: "x",      | (down) |
  |        |  value: "v2",   |   X    |
  |        |  for: "C"       |        |
  +--------+  }              +--------+
  When C returns: A forwards hint to C
  
  2. READ REPAIR:
  Client reads key "x" from 3 replicas:
  Node A: value="v2", ts=100   <-- Latest
  Node B: value="v2", ts=100
  Node C: value="v1", ts=90    <-- Stale!
  --> Coordinator sends v2 to Node C (repair)
  
  3. ANTI-ENTROPY (Merkle Trees):
  Node A tree:         Node C tree:
      [hash_root]          [hash_root]
      /         \          /         \
  [hash_L]  [hash_R]  [hash_L]  [hash_R']
    /\         /\        /\         /\
  [1][2]   [3][4]    [1][2]   [3][4']
                                    ^
  Compare roots --> Different!      |
  Drill down --> Right subtree -->  Key 4 differs
  Sync only key 4 (minimal data transfer)
  
  4. VECTOR CLOCKS (Conflict Detection):
  Node A writes: x = "hello"  VC: {A:1}
  Node B writes: x = "world"  VC: {B:1}
  --> Concurrent! VC: {A:1} || {B:1}
  --> Conflict resolution needed (app or LWW)
```

| Mechanism | When Used | How It Works | Consistency Impact |
|-----------|----------|-------------|-------------------|
| **Hinted handoff** | Node temporarily down | Neighbor stores writes, forwards on recovery | Speeds convergence |
| **Read repair** | Stale replica detected | Coordinator updates stale node during read | Passive consistency |
| **Anti-entropy** | Background sync | Merkle tree comparison identifies differences | Full consistency |
| **Vector clocks** | Concurrent writes | Track causal history, detect conflicts | Conflict detection |
| **CRDTs** | Concurrent updates | Mathematically mergeable data types | Automatic resolution |
| **Gossip protocol** | Failure detection | Nodes exchange state periodically | Cluster awareness |

```python
# Consistency recovery mechanisms
import hashlib
from collections import defaultdict

class HintedHandoff:
    """Stores hints for unavailable nodes."""

    def __init__(self):
        self.hints = defaultdict(list)  # target_node -> [hints]

    def store_hint(self, target_node, key, value, timestamp):
        """Store write intended for an unavailable node."""
        self.hints[target_node].append({
            "key": key, "value": value, "ts": timestamp
        })

    def deliver_hints(self, target_node, node_store):
        """Replay hints when node recovers."""
        for hint in self.hints.pop(target_node, []):
            existing = node_store.get(hint["key"])
            if not existing or existing["ts"] < hint["ts"]:
                node_store[hint["key"]] = {"value": hint["value"], "ts": hint["ts"]}

class MerkleTree:
    """Anti-entropy: efficient replica comparison."""

    def __init__(self, data):
        self.data = data
        self.tree = self._build(sorted(data.items()))

    def _build(self, items):
        if len(items) <= 1:
            if items:
                key, val = items[0]
                return {"hash": self._hash(f"{key}:{val}"),
                        "keys": [key], "children": None}
            return {"hash": "", "keys": [], "children": None}
        mid = len(items) // 2
        left = self._build(items[:mid])
        right = self._build(items[mid:])
        return {
            "hash": self._hash(left["hash"] + right["hash"]),
            "keys": left["keys"] + right["keys"],
            "children": (left, right)
        }

    def _hash(self, s):
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def diff(self, other_tree):
        """Find keys that differ between two replicas."""
        return self._diff_recursive(self.tree, other_tree.tree)

    def _diff_recursive(self, node_a, node_b):
        if node_a["hash"] == node_b["hash"]:
            return []  # Subtrees match!
        if not node_a["children"]:
            return node_a["keys"]  # Leaf differ
        diffs = []
        la, ra = node_a["children"]
        lb, rb = node_b["children"]
        diffs.extend(self._diff_recursive(la, lb))
        diffs.extend(self._diff_recursive(ra, rb))
        return diffs

class VectorClock:
    """Detect concurrent writes for conflict resolution."""

    def __init__(self):
        self.clock = defaultdict(int)

    def increment(self, node_id):
        self.clock[node_id] += 1
        return dict(self.clock)

    def merge(self, other_clock):
        for node, ts in other_clock.items():
            self.clock[node] = max(self.clock[node], ts)

    def is_concurrent(self, other):
        """True if neither clock dominates the other."""
        self_ahead = any(self.clock.get(k, 0) > other.clock.get(k, 0)
                         for k in set(self.clock) | set(other.clock))
        other_ahead = any(other.clock.get(k, 0) > self.clock.get(k, 0)
                          for k in set(self.clock) | set(other.clock))
        return self_ahead and other_ahead  # Both have unseen updates!
```

**AI/ML Application:**
- **Feature consistency during outages:** Hinted handoff ensures ML feature updates aren't lost during node failures. Read repair corrects stale features at prediction time.
- **Model rollback safety:** Vector clocks track model version history — if two teams deploy conflicting model versions concurrently, the conflict is detected rather than silently overwriting.
- **Distributed training checkpoints:** Anti-entropy with Merkle trees verifies checkpoint consistency across storage replicas — critical for resuming distributed training after failures.

**Real-World Example:**
Cassandra combines all four mechanisms: gossip (failure detection), hinted handoff (temporary storage), read repair (passive fix), and anti-entropy repair (background full sync). DynamoDB uses vector clocks for conflict detection and application-level resolution. Riak uses CRDTs (counters, sets, maps) for automatic conflict-free merging. MongoDB uses Raft consensus — during partitions, the majority partition elects a new primary while the minority partition becomes read-only.

> **Interview Tip:** "Draw the timeline: (1) Partition occurs → gossip detects failure, (2) Hinted handoff stores writes locally, (3) Partition heals → hints replayed, (4) Background anti-entropy (Merkle tree) verifies full consistency. Mention the performance impact: Merkle tree comparison transfers only differing keys — on a 1TB dataset with 3 stale keys, only those 3 keys are synced, not the entire dataset."

---

## Scalability & Performance

### 27. Explain common data sharding strategies in NoSQL databases and their impact on system performance . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Sharding (horizontal partitioning) distributes data across nodes. Common strategies: **hash-based** (even distribution, no range queries), **range-based** (sequential data locality, hotspot risk), **directory-based** (lookup table routing), and **geo-based** (data near users). The shard key choice fundamentally determines query performance and cluster balance.

```
  SHARDING STRATEGIES:
  
  1. HASH-BASED SHARDING:
  shard = hash(key) % num_shards
  
  Key: "alice" --> hash("alice")=42 --> 42 % 4 = Shard 2
  Key: "bob"   --> hash("bob")=87   --> 87 % 4 = Shard 3
  
  +----------+  +----------+  +----------+  +----------+
  | Shard 0  |  | Shard 1  |  | Shard 2  |  | Shard 3  |
  | carol    |  | dave     |  | alice    |  | bob      |
  | frank    |  | grace    |  | eve      |  | henry    |
  +----------+  +----------+  +----------+  +----------+
  Pro: Even distribution    Con: No range queries
  
  2. RANGE-BASED SHARDING:
  Key ranges assigned to shards
  
  +----------+  +----------+  +----------+  +----------+
  | Shard 0  |  | Shard 1  |  | Shard 2  |  | Shard 3  |
  | A-F      |  | G-L      |  | M-R      |  | S-Z      |
  | alice    |  | grace    |  | mary     |  | steve    |
  | bob      |  | henry    |  | nick     |  | tina     |
  +----------+  +----------+  +----------+  +----------+
  Pro: Range queries efficient  Con: Hotspots (e.g., all "S" names)
  
  3. DIRECTORY-BASED SHARDING:
  +------------------+
  | Lookup Service   |
  | alice -> Shard 2 |
  | bob   -> Shard 1 |
  | carol -> Shard 3 |
  +------------------+
       |
  +----------+  +----------+  +----------+
  | Shard 1  |  | Shard 2  |  | Shard 3  |
  | bob      |  | alice    |  | carol    |
  +----------+  +----------+  +----------+
  Pro: Flexible routing  Con: Lookup is SPOF/bottleneck
  
  4. COMPOUND SHARD KEY:
  Shard by (tenant_id, date)
  --> All data for one tenant in same shard
  --> Time-range queries within tenant are local
```

| Strategy | Distribution | Range Queries | Rebalancing | Hot Spots |
|----------|-------------|---------------|-------------|-----------|
| **Hash-based** | Excellent (uniform) | Not supported | Full reshard on add/remove | Rare |
| **Consistent hash** | Good (with vnodes) | Not supported | Minimal data movement | Rare (with vnodes) |
| **Range-based** | Variable | Efficient | Region split/merge | Common |
| **Directory** | Configurable | Depends on routing | Lookup update only | Configurable |
| **Compound key** | Good | Within partition | Depends on key design | Depends on key |

```python
# Sharding strategy implementations
import hashlib

class HashSharding:
    """Hash-based sharding with consistent distribution."""

    def __init__(self, num_shards):
        self.num_shards = num_shards
        self.shards = [[] for _ in range(num_shards)]

    def get_shard(self, key):
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def insert(self, key, value):
        shard_id = self.get_shard(key)
        self.shards[shard_id].append({"key": key, "value": value})
        return shard_id

    def get(self, key):
        shard_id = self.get_shard(key)
        for item in self.shards[shard_id]:
            if item["key"] == key:
                return item["value"]
        return None

    def range_query(self, start_key, end_key):
        """Hash sharding: must scatter-gather ALL shards."""
        results = []
        for shard in self.shards:
            results.extend([item for item in shard
                          if start_key <= item["key"] <= end_key])
        return results  # O(N) — touches every shard!

class RangeSharding:
    """Range-based sharding with sequential locality."""

    def __init__(self, boundaries):
        self.boundaries = sorted(boundaries)  # e.g., ['G', 'M', 'S']
        self.shards = [[] for _ in range(len(boundaries) + 1)]

    def get_shard(self, key):
        for i, boundary in enumerate(self.boundaries):
            if key < boundary:
                return i
        return len(self.boundaries)

    def range_query(self, start_key, end_key):
        """Range sharding: only scan relevant shards."""
        start_shard = self.get_shard(start_key)
        end_shard = self.get_shard(end_key)
        results = []
        for shard_id in range(start_shard, end_shard + 1):
            results.extend([item for item in self.shards[shard_id]
                          if start_key <= item["key"] <= end_key])
        return results  # Only touches 1-2 shards!

# Comparison
hash_shard = HashSharding(4)
range_shard = RangeSharding(['G', 'M', 'S'])
# Hash: range query touches all 4 shards
# Range: range query "A"-"F" touches only shard 0
```

**AI/ML Application:**
- **Feature store sharding:** Shard by entity_id (user, item) so all features for one entity are co-located — single-shard reads for prediction-time feature retrieval.
- **Training data partitioning:** Range-shard by timestamp for time-series ML — each worker processes a time range without cross-shard communication.
- **Embedding index sharding:** Vector databases shard embedding indexes by hash — each shard runs an approximate nearest neighbor search, results merged by a coordinator.

**Real-World Example:**
MongoDB supports both hash and range shard keys — the choice depends on query patterns. DynamoDB automatically partitions by hash of partition key and splits partitions that exceed 10GB or throughput limits. Cassandra uses consistent hashing with virtual nodes (256 per physical node by default). Vitess (YouTube's MySQL sharding layer) uses range-based sharding with automatic resharding.

> **Interview Tip:** "The shard key is the most critical design decision: a bad shard key creates hot spots (e.g., sharding by date puts all today's writes on one shard). Good shard keys have high cardinality, even distribution, and match query patterns. Mention compound shard keys: `{tenant_id, timestamp}` gives per-tenant isolation AND time-range queries within a tenant."

---

### 28. How is load balancing achieved in NoSQL database environments? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

NoSQL load balancing operates at **data level** (even shard distribution), **query level** (routing reads across replicas), and **cluster level** (rebalancing when nodes join/leave). Mechanisms include **consistent hashing**, **virtual nodes**, **token-aware routing**, **read-replica load spreading**, and **automatic rebalancing**.

```
  LOAD BALANCING LAYERS:
  
  1. CLIENT-SIDE (Token-Aware Routing):
  +--------+     Token map: key -> node
  | Client | --> hash("alice") = 42 --> Node B owns token 42
  +--------+     Direct connection (skip coordinator!)
       |
       +------> Node B (single hop, lowest latency)
  
  2. COORDINATOR-BASED:
  +--------+     +-------------+     +--------+
  | Client | --> | Coordinator | --> | Node B |
  +--------+     | (any node)  |     | (owner)|
                 +-------------+     +--------+
                       |
                  Forwards to owner (extra hop)
  
  3. READ LOAD SPREADING:
  +--------+    +--------+    +--------+
  | Node A |    | Node B |    | Node C |
  |Primary |    |Replica |    |Replica |
  +--------+    +--------+    +--------+
      ^              ^              ^
      |              |              |
  Write only    Read 33%       Read 33%
                Round-robin across replicas
  
  4. AUTOMATIC REBALANCING:
  Before:   [A: 40%] [B: 40%] [C: 20%]  (uneven)
  Detect:   Node C underloaded
  Action:   Move vnodes from A,B to C
  After:    [A: 33%] [B: 33%] [C: 33%]  (balanced)
  
  5. HOT PARTITION DETECTION:
  Monitor:  Shard 3 receives 80% of traffic
  Action:   Split Shard 3 into 3a and 3b
  Result:   Even distribution restored
  
  +----------+  +----------+  +----------+
  | Shard 1  |  | Shard 2  |  |Shard 3   |
  | 20% load |  | 20% load |  |80% load! |
  +----------+  +----------+  +-----+----+
                                     |
                               +-----+-----+
                               |           |
                          +----+---+  +----+---+
                          |Shard 3a|  |Shard 3b|
                          | 40%    |  | 40%    |
                          +--------+  +--------+
```

| Mechanism | Level | How It Works | Database Example |
|-----------|-------|-------------|-----------------|
| **Token-aware routing** | Client | Client directly contacts owning node | Cassandra (driver) |
| **Virtual nodes** | Data | 256 vnodes/node → even distribution | Cassandra |
| **Read replicas** | Query | Spread reads across replicas | MongoDB secondaryPreferred |
| **Auto-split** | Partition | Split hot partitions automatically | DynamoDB, HBase |
| **Rack-aware placement** | Cluster | Spread replicas across racks/AZs | Cassandra, MongoDB |
| **Connection pooling** | Client | Limit connections per node | All drivers |

```python
# NoSQL load balancing simulation
import random
import hashlib

class LoadBalancer:
    """Multi-level NoSQL load balancer."""

    def __init__(self, nodes):
        self.nodes = {n: {"load": 0, "alive": True, "data_size": 0}
                      for n in nodes}
        self.token_ring = self._build_ring(nodes)

    def _build_ring(self, nodes, vnodes=64):
        ring = {}
        for node in nodes:
            for i in range(vnodes):
                token = int(hashlib.md5(f"{node}:{i}".encode()).hexdigest(), 16)
                ring[token] = node
        return dict(sorted(ring.items()))

    def token_aware_route(self, key):
        """Route directly to owning node (skip coordinator)."""
        token = int(hashlib.md5(key.encode()).hexdigest(), 16)
        tokens = list(self.token_ring.keys())
        for t in tokens:
            if t >= token:
                node = self.token_ring[t]
                self.nodes[node]["load"] += 1
                return node
        node = self.token_ring[tokens[0]]
        self.nodes[node]["load"] += 1
        return node

    def read_load_spread(self, key, replicas):
        """Spread reads across healthy replicas."""
        healthy = [r for r in replicas if self.nodes[r]["alive"]]
        if not healthy:
            raise Exception("No healthy replicas!")
        # Pick least-loaded replica
        chosen = min(healthy, key=lambda r: self.nodes[r]["load"])
        self.nodes[chosen]["load"] += 1
        return chosen

    def detect_hot_partitions(self, threshold_ratio=3.0):
        """Detect nodes with disproportionate load."""
        avg_load = sum(n["load"] for n in self.nodes.values()) / len(self.nodes)
        hot = []
        for node, stats in self.nodes.items():
            if avg_load > 0 and stats["load"] / avg_load > threshold_ratio:
                hot.append(node)
        return hot

    def rebalance(self):
        """Redistribute vnodes from overloaded to underloaded nodes."""
        loads = {n: s["load"] for n, s in self.nodes.items()}
        avg = sum(loads.values()) / len(loads)
        overloaded = [n for n, l in loads.items() if l > avg * 1.5]
        underloaded = [n for n, l in loads.items() if l < avg * 0.5]
        moves = []
        for over in overloaded:
            for under in underloaded:
                moves.append(f"Move vnodes: {over} -> {under}")
        return moves
```

**AI/ML Application:**
- **Prediction serving:** Token-aware routing sends feature lookups directly to the node that owns the entity's partition — minimizing latency for real-time ML inference.
- **Training data locality:** Rack-aware placement ensures training data replicas are spread across availability zones — ML training jobs read from the closest replica.
- **Model serving auto-scaling:** Hot partition detection triggers automatic shard splitting when a popular model version receives disproportionate traffic.

**Real-World Example:**
Cassandra's Java/Python drivers use token-aware routing — the driver maintains a copy of the token ring and sends requests directly to the owning node (no coordinator hop). DynamoDB's adaptive capacity automatically shifts throughput from less-active partitions to hot partitions. MongoDB's balancer process migrates chunks between shards when size imbalance exceeds a threshold. ScyllaDB (C++ Cassandra) achieves 10x better per-node throughput through shard-per-core architecture.

> **Interview Tip:** "Describe three levels: (1) Data level — consistent hashing + vnodes for even distribution, (2) Query level — token-aware routing sends reads/writes directly to the owning node, (3) Cluster level — automatic rebalancing when nodes join/leave. Mention that client-side load balancing (token-aware) eliminates the coordinator hop, reducing latency by ~50%. DynamoDB does this automatically; Cassandra requires a smart driver."

---

### 29. Discuss the role of caching in NoSQL database performance enhancement. 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Caching enhances NoSQL performance at **internal** (built-in buffer pools, row caches, key caches) and **external** (Redis, Memcached as read-through/write-through layers) levels. Caching reduces disk I/O, lowers read latency from milliseconds to microseconds, and absorbs traffic spikes that would overwhelm the database.

```
  CACHING LAYERS FOR NoSQL:
  
  +--------+
  | Client |
  +---+----+
      |
  +---v--------------+
  | L1: Client Cache  |  <-- In-process (HashMap, LRU)
  | Latency: ~0.1ms   |      Nearest, fastest, smallest
  +---+---------------+
      |
  +---v--------------+
  | L2: Redis/Memcached|  <-- Distributed cache
  | Latency: ~1ms      |      Shared across app instances
  +---+---------------+
      |
  +---v--------------+
  | L3: DB Internal   |  <-- Row cache, block cache
  | Latency: ~5ms     |      Managed by DB engine
  +---+---------------+
      |
  +---v--------------+
  | L4: Disk (SSTable)|  <-- Actual storage
  | Latency: ~10-50ms |      SSD or HDD
  +-------------------+
  
  CACHING PATTERNS:
  
  1. CACHE-ASIDE (Lazy Loading):
  App --> Cache? --> Hit --> Return
                --> Miss --> DB --> Write to cache --> Return
  
  2. READ-THROUGH:
  App --> Cache --> Miss --> Cache reads from DB --> Return
  (Cache manages DB reads)
  
  3. WRITE-THROUGH:
  App --> Cache --> DB (synchronous)
  Every write updates cache AND DB
  
  4. WRITE-BEHIND (Write-Back):
  App --> Cache --> Return (immediate)
             \--> DB (async, batched)
  
  CASSANDRA INTERNAL CACHES:
  +------------------+     +------------------+
  | Key Cache        |     | Row Cache        |
  | (partition keys) |     | (hot rows)       |
  | Saves 1 disk seek|     | Saves entire read|
  +--------+---------+     +--------+---------+
           |                        |
  +--------v------------------------v---------+
  | Bloom Filter (per SSTable)                |
  | Skip SSTables that don't contain the key  |
  +-------------------------------------------+
```

| Cache Type | Location | Hit Latency | Capacity | Eviction |
|-----------|---------|-------------|----------|----------|
| **Client-side (L1)** | App process | ~0.1ms | Small (MB) | TTL + LRU |
| **Redis/Memcached (L2)** | External | ~1ms | Large (GB-TB) | LRU, LFU, TTL |
| **DB row cache** | DB memory | ~0.5ms | Configurable | LRU |
| **DB key cache** | DB memory | ~0.1ms | Configurable | LRU |
| **OS page cache** | OS kernel | ~0.5ms | Available RAM | LRU |
| **Bloom filter** | Per SSTable | ~0.01ms | Bits per key | None (immutable) |

```python
# Multi-layer caching for NoSQL
import time
from collections import OrderedDict

class LRUCache:
    """Least Recently Used cache with TTL."""

    def __init__(self, capacity, ttl_seconds=60):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] > self.ttl:
                self._evict(key)
                self.misses += 1
                return None
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            self._evict(oldest)
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def _evict(self, key):
        del self.cache[key]
        del self.timestamps[key]

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

class CacheAsidePattern:
    """Cache-aside with external cache + NoSQL backend."""

    def __init__(self, db, cache_capacity=1000, ttl=300):
        self.db = db
        self.cache = LRUCache(cache_capacity, ttl)

    def read(self, key):
        # Try cache first
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # Cache hit!
        # Cache miss: read from DB
        value = self.db.get(key)
        if value is not None:
            self.cache.put(key, value)  # Populate cache
        return value

    def write(self, key, value):
        # Write to DB first (source of truth)
        self.db.put(key, value)
        # Invalidate cache (not update — avoids race conditions)
        if key in self.cache.cache:
            self.cache._evict(key)

    def write_through(self, key, value):
        """Write to cache AND DB simultaneously."""
        self.db.put(key, value)
        self.cache.put(key, value)
```

**AI/ML Application:**
- **Feature serving cache:** Cache hot ML features in Redis — prediction-time lookups for popular users/items hit cache (sub-ms) instead of Cassandra (~5ms). DoorDash caches top-1000 restaurant features.
- **Model inference cache:** Cache prediction results for repeated inputs (same user, same context) — eliminates GPU inference for duplicate requests.
- **Embedding cache:** Cache frequently queried vector embeddings in Redis to avoid repeated ANN index lookups. Spotify caches hot track embeddings for recommendation.

**Real-World Example:**
Instagram uses Memcached in front of Cassandra — caching profile data, follower counts, and feed entries. Their cache hit rate exceeds 99%. Facebook's TAO cache serves 99.8% of social graph reads from cache. DynamoDB DAX (DynamoDB Accelerator) provides a fully managed in-memory cache that reduces read latency from single-digit milliseconds to microseconds. MongoDB's WiredTiger engine uses an internal cache (default 50% of RAM) for hot documents.

> **Interview Tip:** "Mention the cache invalidation problem: 'There are only two hard things in computer science: cache invalidation and naming things.' For NoSQL, prefer cache-aside with invalidation (delete key on write) over write-through (update key on write) — invalidation avoids race conditions where two concurrent writes leave the cache with a stale value. TTL is your safety net."

---

### 30. What approaches do NoSQL databases use to ensure high availability ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

NoSQL databases achieve high availability through **replication** (data copies on multiple nodes), **automatic failover** (leader election on failure), **masterless architecture** (no single point of failure), **multi-datacenter deployment** (survive regional outages), and **graceful degradation** (serve stale data instead of failing).

```
  HIGH AVAILABILITY MECHANISMS:
  
  1. REPLICATION (Multiple Copies):
  Data "alice" copied to 3 nodes:
  +--------+    +--------+    +--------+
  | Node 1 |    | Node 2 |    | Node 3 |
  | alice:v2|    | alice:v2|    | alice:v2|
  +--------+    +--------+    +--------+
  Node 1 dies --> Read from Node 2 or 3
  
  2. AUTOMATIC FAILOVER (Leader Election):
  +--------+    +--------+    +--------+
  | Primary| X  |Secondary|    |Secondary|
  | (dies) |    | (detects|    |         |
  +--------+    |  via    |    |         |
                | heartbeat)   |         |
                +---+----+    +----+----+
                    |              |
                    v              v
               [ELECTION via Raft/Paxos]
                    |
                +---v----+
                | NEW     |
                | Primary |  <-- Automatic, ~10s
                +---------+
  
  3. MASTERLESS (No SPOF):
  +------+    +------+    +------+
  |Node 1|<-->|Node 2|<-->|Node 3|
  +------+    +------+    +------+
  Any node dies --> Others continue serving
  No election needed, no downtime
  
  4. MULTI-DATACENTER:
  +--[DC-East]--+     +--[DC-West]--+
  | Node1 Node2 | <-> | Node4 Node5 |
  | Node3       |     | Node6       |
  +-------------+     +-------------+
  DC-East destroyed --> DC-West serves all traffic
  
  5. GRACEFUL DEGRADATION:
  Normal:   Read from quorum (strong consistency)
  Degraded: Read from ANY available (eventual consistency)
  Offline:  Serve from local cache
  
  AVAILABILITY MATH:
  Single node: 99.9% = 8.76 hours downtime/year
  3 replicas:  1-(0.001)^3 = 99.9999999% (independent failures)
  With correlated failures (realistic): ~99.99% = 52 min/year
```

| Approach | Handles | Downtime | Trade-off | Example DB |
|----------|---------|----------|-----------|-----------|
| **Replication (RF=3)** | Node failure | ~0 (auto-redirect) | 3x storage cost | All NoSQL |
| **Auto-failover** | Leader failure | ~10-30s (election) | Brief write unavailability | MongoDB, Redis Sentinel |
| **Masterless** | Any node failure | 0 (no election) | Complex conflict resolution | Cassandra, Riak |
| **Multi-DC** | Regional outage | ~0 (traffic shift) | Cross-DC latency, cost | Cassandra, CockroachDB |
| **Read replicas** | Read load spikes | 0 (add replicas) | Replication lag | MongoDB, DynamoDB |

```python
# High availability system simulation
import time
import random
from enum import Enum

class NodeState(Enum):
    HEALTHY = "healthy"
    SUSPECTED = "suspected"
    DEAD = "dead"

class HACluster:
    """NoSQL high availability cluster with failover."""

    def __init__(self, nodes, replication_factor=3):
        self.rf = replication_factor
        self.nodes = {
            n: {"state": NodeState.HEALTHY, "data": {},
                "last_heartbeat": time.time(), "is_primary": i == 0}
            for i, n in enumerate(nodes)
        }
        self.primary = nodes[0]

    def heartbeat_check(self, timeout=5.0):
        """Detect failed nodes via heartbeat timeout."""
        now = time.time()
        for node, info in self.nodes.items():
            if node == "self":
                continue
            elapsed = now - info["last_heartbeat"]
            if elapsed > timeout and info["state"] == NodeState.HEALTHY:
                info["state"] = NodeState.SUSPECTED
                if info["is_primary"]:
                    self._trigger_failover(node)

    def _trigger_failover(self, failed_primary):
        """Elect new primary when leader fails."""
        self.nodes[failed_primary]["is_primary"] = False
        self.nodes[failed_primary]["state"] = NodeState.DEAD
        # Elect healthiest secondary (most up-to-date)
        candidates = [n for n, info in self.nodes.items()
                     if info["state"] == NodeState.HEALTHY and not info["is_primary"]]
        if candidates:
            new_primary = candidates[0]
            self.nodes[new_primary]["is_primary"] = True
            self.primary = new_primary
            return new_primary
        return None

    def write(self, key, value):
        """Write to primary, replicate to secondaries."""
        primary_info = self.nodes.get(self.primary)
        if not primary_info or primary_info["state"] != NodeState.HEALTHY:
            raise Exception("Primary unavailable!")
        primary_info["data"][key] = value
        # Replicate
        secondaries = [n for n, i in self.nodes.items()
                      if i["state"] == NodeState.HEALTHY and n != self.primary]
        for sec in secondaries[:self.rf - 1]:
            self.nodes[sec]["data"][key] = value

    def read(self, key, prefer_secondary=True):
        """Read with automatic failover."""
        if prefer_secondary:
            secondaries = [n for n, i in self.nodes.items()
                          if i["state"] == NodeState.HEALTHY and not i["is_primary"]]
            for sec in secondaries:
                val = self.nodes[sec]["data"].get(key)
                if val is not None:
                    return val
        return self.nodes[self.primary]["data"].get(key)
```

**AI/ML Application:**
- **Always-on prediction serving:** Replication + automatic failover ensures ML models are always available for inference. A/B test results are never lost due to node failures.
- **Multi-region ML serving:** Multi-DC deployment serves predictions from the nearest datacenter — <50ms latency globally for real-time scoring.
- **Graceful ML degradation:** When the primary model store is unavailable, fall back to a cached model version — served from the read replica with last-known-good predictions rather than returning errors.

**Real-World Example:**
Netflix runs Cassandra across 3+ AWS regions — if an entire region fails, traffic automatically routes to surviving regions with zero downtime. MongoDB replica sets use Raft-based election: when the primary fails, a new primary is elected in 10-12 seconds. DynamoDB's SLA guarantees 99.999% availability for global tables (multi-region). Redis Sentinel monitors masters and automatically promotes a slave to master on failure.

> **Interview Tip:** "Calculate availability: RF=3 with independent node failures gives 1-(1-p)^3 = theoretical 99.9999999% (p=99.9%). But real HA requires handling correlated failures (power outage, network partition). Multi-DC deployment handles correlated failures within a DC. The gold standard is Cassandra-style: masterless + multi-DC + tunable consistency = highest availability with configurable consistency trade-offs."

---

## Querying and Data Manipulation

### 31. How do NoSQL databases manage complex queries , such as those requiring join operations ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

NoSQL databases handle joins through **denormalization** (embed related data), **application-level joins** (multiple queries + client-side merge), **materialized views** (pre-computed join results), and **database-level lookups** (MongoDB's `$lookup`, Cassandra's materialized views). The core principle: **optimize for reads by denormalizing at write time**.

```
  JOIN STRATEGIES IN NoSQL:
  
  1. DENORMALIZATION (Embed Related Data):
  // Instead of: users + orders tables with FK
  // Store everything together:
  {
    "_id": "user1",
    "name": "Alice",
    "orders": [                    <-- Embedded!
      {"id": "o1", "amount": 150, "date": "2024-03"},
      {"id": "o2", "amount": 75,  "date": "2024-04"}
    ]
  }
  // ONE read gets user + all orders (no join!)
  
  2. APPLICATION-LEVEL JOIN:
  // Query 1: Get user
  user = db.users.findOne({_id: "user1"})
  // Query 2: Get orders for user
  orders = db.orders.find({user_id: "user1"})
  // Application code merges results
  user["orders"] = list(orders)
  
  3. MONGODB $lookup (Server-Side Join):
  db.users.aggregate([
    { $lookup: {
        from: "orders",
        localField: "_id",
        foreignField: "user_id",
        as: "user_orders"
      }
    }
  ])
  // Server-side join (slower than denormalization)
  
  4. MATERIALIZED VIEW (Pre-Computed):
  // Write time: update both collections
  // users: {_id: "user1", name: "Alice"}
  // user_orders_view: {_id: "user1", name: "Alice",
  //                    total_orders: 2, total_spent: 225}
  
  DECISION MATRIX:
  +---------------------+------------------+-------------------+
  | Access Pattern      | Strategy         | Trade-off         |
  +---------------------+------------------+-------------------+
  | Always read together| Embed (denorm)   | Duplicate data    |
  | Rarely read together| App-level join   | Multiple queries  |
  | Analytics/reporting | $lookup / MV     | Slower, more CPU  |
  | High read frequency | Materialized view| Write amplification|
  +---------------------+------------------+-------------------+
```

| Strategy | Read Performance | Write Complexity | Data Consistency | Best For |
|----------|-----------------|-----------------|------------------|----------|
| **Denormalization** | O(1) single read | Update all copies | App-managed | Read-heavy, co-accessed data |
| **Application join** | N+1 queries | Simple writes | Normalized | Rarely co-accessed data |
| **$lookup** | O(n*m) server-side | Simple writes | Real-time | Ad-hoc analytics |
| **Materialized view** | O(1) pre-computed | Write amplification | Eventual | Dashboard/reporting |
| **Change streams** | Async propagation | Event-driven | Eventually consistent | Real-time sync |

```python
# NoSQL join strategies implementation
class DenormalizedStore:
    """Strategy 1: Embed related data (most common)."""

    def __init__(self):
        self.collection = {}

    def create_user_with_orders(self, user_id, name, orders):
        """Embed orders inside user document."""
        self.collection[user_id] = {
            "name": name,
            "orders": orders,  # Embedded array
            "order_count": len(orders),
            "total_spent": sum(o["amount"] for o in orders)
        }

    def get_user_with_orders(self, user_id):
        """Single read — no join needed!"""
        return self.collection.get(user_id)  # O(1)

    def add_order(self, user_id, order):
        """Write is more complex (update embedding)."""
        user = self.collection.get(user_id)
        if user:
            user["orders"].append(order)
            user["order_count"] += 1
            user["total_spent"] += order["amount"]

class ApplicationJoinStore:
    """Strategy 2: Separate collections + app-level merge."""

    def __init__(self):
        self.users = {}
        self.orders = {}

    def get_user_with_orders(self, user_id):
        """N+1 queries: 1 for user + 1 for orders."""
        user = self.users.get(user_id)          # Query 1
        user_orders = [o for o in self.orders.values()
                      if o["user_id"] == user_id]  # Query 2
        if user:
            return {**user, "orders": user_orders}  # App-level merge

class LookupJoinStore:
    """Strategy 3: Server-side $lookup (MongoDB aggregate)."""

    def __init__(self):
        self.collections = {}

    def lookup(self, from_coll, lookup_coll, local_field, foreign_field, as_field):
        """Simulates MongoDB $lookup (nested loop join)."""
        results = []
        for doc in self.collections.get(from_coll, []):
            matches = [
                foreign_doc for foreign_doc
                in self.collections.get(lookup_coll, [])
                if foreign_doc.get(foreign_field) == doc.get(local_field)
            ]
            result = {**doc, as_field: matches}
            results.append(result)
        return results  # O(n * m) — use sparingly!

# Compare: denormalized vs app-level
denorm = DenormalizedStore()
denorm.create_user_with_orders("u1", "Alice", [
    {"id": "o1", "amount": 150},
    {"id": "o2", "amount": 75}
])
result = denorm.get_user_with_orders("u1")  # 1 read, O(1)
```

**AI/ML Application:**
- **Feature store denormalization:** Pre-compute and embed user features (demographics + behavior + purchase history) in a single document — one read at prediction time instead of joining 5 feature tables.
- **Training data materialization:** Materialize training features by joining event logs with user profiles at write time — the ML training pipeline reads pre-joined documents directly.
- **Recommendation denormalization:** Embed "users who liked this also liked" lists directly in product documents — enables O(1) recommendation lookups without joining user-item interaction tables.

**Real-World Example:**
MongoDB's `$lookup` is used by eBay for ad-hoc analytics joining product and seller collections. Cassandra materializes denormalized views at write time — Instagram stores both "posts by user" and "posts by hashtag" tables (same data, different partition key). DynamoDB uses GSI (Global Secondary Index) as a materialized view — a different partition key for the same data. Facebook's TAO denormalizes the social graph: each user's friend list is stored directly, not computed by joining a friendship table.

> **Interview Tip:** "The #1 NoSQL design principle: 'model your data for your queries, not your entities.' If you always read user+orders together, embed orders in the user document (denormalize). If you sometimes need just users, keep them separate and do an application-level join. `$lookup` exists but is a last resort — it's an O(n*m) nested loop join, orders of magnitude slower than a pre-denormalized read."

---

### 32. What is map-reduce , and how is it utilized within NoSQL databases ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

**MapReduce** is a **distributed data processing paradigm** with two phases: **Map** (transform each document into key-value pairs) and **Reduce** (aggregate values by key). NoSQL databases use it for **batch analytics**, **aggregation**, and **data transformation** across distributed datasets. It enables parallel processing across shards.

```
  MAPREDUCE FLOW:
  
  Input Documents:
  [{"product": "Book", "price": 15, "qty": 2},
   {"product": "Phone", "price": 700, "qty": 1},
   {"product": "Book", "price": 12, "qty": 3},
   {"product": "Phone", "price": 650, "qty": 2}]
  
  1. MAP PHASE (per document, parallel):
  doc1 --> emit("Book",  {revenue: 30})
  doc2 --> emit("Phone", {revenue: 700})
  doc3 --> emit("Book",  {revenue: 36})
  doc4 --> emit("Phone", {revenue: 1300})
  
  2. SHUFFLE (group by key):
  "Book"  --> [{revenue: 30}, {revenue: 36}]
  "Phone" --> [{revenue: 700}, {revenue: 1300}]
  
  3. REDUCE PHASE (per key, parallel):
  "Book"  --> sum([30, 36])   = 66
  "Phone" --> sum([700, 1300]) = 2000
  
  Output: {"Book": 66, "Phone": 2000}
  
  DISTRIBUTED EXECUTION:
  +----------+   MAP    +----------+   SHUFFLE   +----------+
  | Shard 1  | -------> | Mapper 1 | ---------> | Reducer 1|
  | doc1,doc3|          | Book:30  |            | Book: 66 |
  +----------+          | Book:36  |            +----------+
                        +----------+
  +----------+   MAP    +----------+   SHUFFLE   +----------+
  | Shard 2  | -------> | Mapper 2 | ---------> | Reducer 2|
  | doc2,doc4|          | Phone:700|            |Phone:2000|
  +----------+          |Phone:1300|            +----------+
                        +----------+
  
  MAPREDUCE vs AGGREGATION PIPELINE:
  +------------------+----------------------------+
  | MapReduce        | Aggregation Pipeline       |
  +------------------+----------------------------+
  | Custom JS funcs  | Declarative stages         |
  | Flexible         | Optimized by engine        |
  | Slower (JS eval) | Faster (native C++)        |
  | Deprecated (Mongo)| Preferred (Mongo 5.0+)   |
  | Hadoop ecosystem | Database-native            |
  +------------------+----------------------------+
```

| Aspect | MapReduce | Aggregation Pipeline | SQL GROUP BY |
|--------|----------|---------------------|-------------|
| **Parallelism** | Across shards | Across shards | Single node or parallel |
| **Custom logic** | Any JavaScript/Python | Limited to stages | SQL functions |
| **Performance** | Moderate (JS overhead) | Fast (native engine) | Fast (optimized) |
| **Incremental** | Possible | Not built-in | Not built-in |
| **Use case** | Complex transformations | Analytics, reporting | Traditional analytics |

```python
# MapReduce implementation for NoSQL
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class MapReduceEngine:
    """Distributed MapReduce framework for NoSQL data."""

    def __init__(self, num_workers=4):
        self.num_workers = num_workers

    def execute(self, documents, map_fn, reduce_fn):
        """Run a complete MapReduce job."""
        # Phase 1: Map (parallel across documents)
        mapped = self._map_phase(documents, map_fn)
        # Phase 2: Shuffle (group by key)
        shuffled = self._shuffle(mapped)
        # Phase 3: Reduce (parallel across keys)
        reduced = self._reduce_phase(shuffled, reduce_fn)
        return reduced

    def _map_phase(self, documents, map_fn):
        """Apply map function to each document."""
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            for doc_results in pool.map(map_fn, documents):
                results.extend(doc_results)
        return results  # List of (key, value) pairs

    def _shuffle(self, mapped):
        """Group mapped results by key."""
        groups = defaultdict(list)
        for key, value in mapped:
            groups[key].append(value)
        return dict(groups)

    def _reduce_phase(self, shuffled, reduce_fn):
        """Apply reduce function to each group."""
        results = {}
        for key, values in shuffled.items():
            results[key] = reduce_fn(key, values)
        return results

# Example: Revenue per product
def map_revenue(doc):
    """Map: emit (product, revenue) for each order item."""
    return [(doc["product"], doc["price"] * doc["qty"])]

def reduce_sum(key, values):
    """Reduce: sum revenues per product."""
    return {"total_revenue": sum(values), "num_orders": len(values)}

# More complex: word count across documents
def map_words(doc):
    """Map: emit (word, 1) for each word in text."""
    words = doc.get("text", "").lower().split()
    return [(word, 1) for word in words]

def reduce_count(key, values):
    """Reduce: count occurrences."""
    return sum(values)

engine = MapReduceEngine(num_workers=4)
orders = [
    {"product": "Book", "price": 15, "qty": 2},
    {"product": "Phone", "price": 700, "qty": 1},
    {"product": "Book", "price": 12, "qty": 3},
]
result = engine.execute(orders, map_revenue, reduce_sum)
# {"Book": {"total_revenue": 66, "num_orders": 2},
#  "Phone": {"total_revenue": 700, "num_orders": 1}}
```

**AI/ML Application:**
- **Feature aggregation:** MapReduce computes aggregate features across billions of events: map each click event to `(user_id, {clicks: 1, duration: X})`, reduce to `{total_clicks, avg_duration}` per user — core input features for recommendation models.
- **Distributed preprocessing:** MapReduce normalizes and transforms raw training data across shards — text tokenization (map each doc to tokens), TF-IDF computation (reduce term frequencies across corpus).
- **Model evaluation:** Map: run model prediction on each test sample (parallel). Reduce: aggregate metrics (accuracy, precision, recall) across all samples.

**Real-World Example:**
CouchDB uses MapReduce views as its primary query mechanism — define a JavaScript map function, CouchDB incrementally maintains the view. MongoDB deprecated MapReduce in favor of the aggregation pipeline (5x-10x faster for most use cases). Apache HBase integrates with Hadoop MapReduce for batch analytics on column-family data. Google's original MapReduce paper (2004) processed 20 PB/day at Google — this inspired Hadoop, Spark, and all modern big data processing.

> **Interview Tip:** "MapReduce is deprecated in modern MongoDB (use aggregation pipeline instead), but the concept remains fundamental. Explain the three phases: Map (parallel transform per document), Shuffle (group by key), Reduce (aggregate per group). Key insight: MapReduce enables embarrassingly parallel processing — each mapper is independent, so you scale linearly by adding nodes. Mention Spark as the modern replacement (10-100x faster due to in-memory processing vs. MapReduce's disk-based shuffle)."

---

### 33. Are transactions supported by NoSQL databases , and if so, how are they implemented? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Modern NoSQL databases support transactions at varying levels: **single-document ACID** (MongoDB, DynamoDB), **multi-document ACID** (MongoDB 4.0+, CockroachDB), **lightweight transactions** (Cassandra's COMPARE-AND-SET), and **saga patterns** (application-level distributed transactions). Full distributed ACID is rare due to performance overhead.

```
  TRANSACTION SUPPORT SPECTRUM:
  
  No Transactions          Full Distributed ACID
  |                                            |
  Redis  Cassandra  MongoDB  CockroachDB  Spanner
  KV ops   LWT     Multi-doc    ACID      Global ACID
  
  1. SINGLE-DOCUMENT ACID (MongoDB):
  // Atomic: update all fields in ONE document
  db.accounts.updateOne(
    {_id: "alice"},
    {$inc: {balance: -100}, $push: {history: {type: "debit", amount: 100}}}
  )
  // Entire update is atomic — no partial state visible
  
  2. MULTI-DOCUMENT TRANSACTION (MongoDB 4.0+):
  session = client.start_session()
  session.start_transaction()
  try:
      db.accounts.update({_id: "alice"}, {$inc: {balance: -100}}, session=session)
      db.accounts.update({_id: "bob"},   {$inc: {balance: +100}}, session=session)
      session.commit_transaction()  // All-or-nothing
  except:
      session.abort_transaction()   // Rollback
  
  3. CASSANDRA LWT (Compare-And-Set):
  INSERT INTO users (id, email)
  VALUES ('user1', 'alice@example.com')
  IF NOT EXISTS;  // Lightweight transaction (Paxos)
  // Only succeeds if no existing row for id='user1'
  
  4. SAGA PATTERN (Application Level):
  +--------+    +--------+    +--------+
  | Step 1 |    | Step 2 |    | Step 3 |
  |Debit   |--->|Ship    |--->|Email   |
  |Account |    |Product |    |Receipt |
  +---+----+    +---+----+    +--------+
      |             |
  [Compensate]  [Compensate]
  |Credit back| |Cancel ship|
  +-----------+ +-----------+
  If Step 2 fails: run compensating actions backward
  
  ISOLATION LEVELS COMPARISON:
  +------------------+----------+----------+----------+
  | Database         | Isolation| Scope    | Protocol |
  +------------------+----------+----------+----------+
  | MongoDB          | Snapshot | Multi-doc| WiredTiger|
  | DynamoDB         | Serializ.| 25 items | 2PC      |
  | Cassandra LWT    | Lineariz.| Single   | Paxos    |
  | CockroachDB      | Serializ.| Distrib. | Raft+MVCC|
  | Google Spanner   | External | Global   | TrueTime |
  +------------------+----------+----------+----------+
```

| Database | Transaction Type | Scope | Max Items | Protocol | Performance Impact |
|----------|-----------------|-------|-----------|----------|-------------------|
| **MongoDB 4.0+** | Multi-document ACID | Replica set / sharded | Unlimited | WiredTiger MVCC | 2-5x slower |
| **DynamoDB** | ACID | Single region | 25 items | 2PC | 2x cost (CU) |
| **Cassandra** | LWT (CAS) | Single partition | 1 row | Paxos | 4-10x slower |
| **CockroachDB** | Serializable ACID | Distributed | Unlimited | Raft + timestamp ordering | Moderate |
| **Google Spanner** | External consistency | Global | Unlimited | TrueTime + 2PC | GPS/atomic clock |
| **Redis** | MULTI/EXEC | Single node | Unlimited | Single-threaded | Minimal |

```python
# NoSQL transaction implementations
import time
import threading

class MongoLikeTransaction:
    """Multi-document ACID transaction simulation."""

    def __init__(self, db):
        self.db = db
        self.write_set = {}       # Buffered writes
        self.read_set = {}        # Snapshot reads
        self.snapshot_time = None
        self.committed = False

    def start(self):
        self.snapshot_time = time.time()
        self.write_set = {}
        self.read_set = {}

    def read(self, collection, key):
        """Snapshot read: see data as of transaction start."""
        if key in self.write_set.get(collection, {}):
            return self.write_set[collection][key]  # Read your writes
        value = self.db.get(collection, key)
        self.read_set.setdefault(collection, {})[key] = value
        return value

    def write(self, collection, key, value):
        """Buffer write until commit."""
        self.write_set.setdefault(collection, {})[key] = value

    def commit(self):
        """Atomic commit: apply all writes or none."""
        # Check for conflicts (optimistic concurrency)
        for coll, keys in self.read_set.items():
            for key, old_val in keys.items():
                current = self.db.get(coll, key)
                if current != old_val:
                    raise Exception(f"Conflict on {coll}.{key}! Abort.")
        # Apply all writes atomically
        for coll, writes in self.write_set.items():
            for key, value in writes.items():
                self.db.put(coll, key, value)
        self.committed = True

    def abort(self):
        self.write_set.clear()
        self.read_set.clear()

class SagaOrchestrator:
    """Saga pattern for distributed pseudo-transactions."""

    def __init__(self):
        self.steps = []
        self.compensations = []
        self.completed = []

    def add_step(self, action, compensation):
        self.steps.append(action)
        self.compensations.append(compensation)

    def execute(self):
        """Execute steps; on failure, run compensating actions."""
        for i, step in enumerate(self.steps):
            try:
                step()
                self.completed.append(i)
            except Exception as e:
                print(f"Step {i} failed: {e}. Running compensations...")
                self._compensate()
                return False
        return True

    def _compensate(self):
        """Run compensating actions in reverse order."""
        for i in reversed(self.completed):
            try:
                self.compensations[i]()
            except Exception as e:
                print(f"Compensation {i} failed: {e}")

# Saga usage: transfer money between services
saga = SagaOrchestrator()
saga.add_step(
    action=lambda: print("Debit $100 from Alice"),
    compensation=lambda: print("Credit $100 back to Alice")
)
saga.add_step(
    action=lambda: print("Credit $100 to Bob"),
    compensation=lambda: print("Debit $100 from Bob")
)
saga.execute()
```

**AI/ML Application:**
- **Atomic model deployment:** Multi-document transactions ensure model version + routing config + feature schema are updated atomically — no partial deployments where the model expects features that aren't yet available.
- **Training data consistency:** Transactions ensure training labels and features are updated together — prevents training on mismatched data (feature from version V1, label from V2).
- **A/B test assignment:** DynamoDB transactions assign users to experiment variants atomically — prevents a user from being in two conflicting experiments simultaneously.

**Real-World Example:**
MongoDB's multi-document transactions enabled Goldman Sachs to migrate from Oracle — financial operations require atomic updates across accounts, positions, and ledgers. DynamoDB transactions power Amazon's checkout (update inventory + create order + charge payment atomically, max 25 items per transaction). Cassandra's LWT is used by Apple for iCloud account operations (IF NOT EXISTS prevents duplicate account creation).

> **Interview Tip:** "NoSQL transactions exist but with trade-offs: MongoDB supports full ACID but transactions are 2-5x slower than non-transactional operations. Cassandra's LWT uses Paxos (4 round-trips) — 4-10x slower than normal writes. The best practice: design schemas to minimize cross-document transactions. If an operation always updates related data together, embed it in ONE document (single-document writes are always atomic in MongoDB)."

---

## Indexing and Storage Strategies

### 34. Compare indexing strategies between NoSQL and traditional relational databases . 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

RDBMS uses **B-tree indexes** (default) on fixed-schema columns. NoSQL databases use **diverse indexing** depending on data model: **B-trees** (MongoDB), **LSM-tree SSTables** (Cassandra), **inverted indexes** (Elasticsearch), **spatial indexes** (MongoDB 2dsphere), and **graph indexes** (Neo4j). NoSQL indexes must handle **flexible schemas**, **nested fields**, and **distributed data**.

```
  INDEXING COMPARISON:
  
  RDBMS B-TREE INDEX:
  CREATE INDEX idx_email ON users(email);
  
         [M]                     <-- Root
        /   \
     [D]     [S]                 <-- Internal
    / | \   / | \
  [A][F][K][N][R][Z]            <-- Leaf (pointers to rows)
   |  |  |  |  |  |
  rows rows rows rows rows rows
  
  Fixed columns, single level, UPDATE in-place
  
  MONGODB B-TREE ON NESTED FIELDS:
  db.users.createIndex({"address.city": 1})
  
  Index on nested field "address.city":
         ["N"]
        /     \
  ["Austin"] ["NYC"]
      |          |
   doc_ids    doc_ids
  
  Supports: dot notation, arrays (multi-key), wildcard
  
  CASSANDRA LSM-TREE (Log-Structured Merge):
  Write --> MemTable (sorted, in-memory)
                |
                v (flush)
         SSTable L0 (sorted, immutable, on disk)
                |
                v (compaction)
         SSTable L1 (merged, larger)
                |
                v (compaction)
         SSTable L2 (merged, largest)
  
  Read: MemTable --> Bloom Filter --> SSTable scan
  
  ELASTICSEARCH INVERTED INDEX:
  Document 1: "quick brown fox"
  Document 2: "quick red fox"
  
  Inverted Index:
  +-------+----------+
  | Term  | Doc IDs  |
  +-------+----------+
  | brown | [1]      |
  | fox   | [1, 2]   |
  | quick | [1, 2]   |
  | red   | [2]      |
  +-------+----------+
  
  Query: "quick fox" --> intersection([1,2], [1,2]) = [1, 2]
  
  NEO4J GRAPH INDEX:
  Node Label Index: :User(name) --> B-tree for property lookup
  Relationship: Index-free adjacency (pointer following)
```

| Feature | RDBMS B-Tree | MongoDB B-Tree | Cassandra LSM | Elasticsearch Inverted | Neo4j |
|---------|-------------|---------------|---------------|----------------------|-------|
| **Write speed** | Moderate (in-place) | Moderate (in-place) | Fast (append-only) | Moderate (segment merge) | Moderate |
| **Read speed** | Fast O(log n) | Fast O(log n) | Slower (multi-SSTable) | Fast (term lookup) | O(1) per hop |
| **Nested fields** | Not supported | Dot notation | Limited | Flattened/nested | Properties |
| **Array indexing** | Requires junction table | Multi-key (native) | Collection columns | Array fields | Not applicable |
| **Full-text** | Separate FTS engine | Atlas Search (Lucene) | Solr/Lucene integration | Native (core feature) | Plugin |
| **Schema requirement** | Column must exist | Field can be absent | Column defined in CQL | Mapping (optional) | Optional |

```python
# Comparing indexing strategies
import hashlib
from collections import defaultdict

class BTreeIndex:
    """RDBMS/MongoDB-style B-tree index."""

    def __init__(self):
        self.tree = {}  # Simplified: dict acts as sorted map

    def insert(self, key, doc_id):
        self.tree.setdefault(key, []).append(doc_id)

    def exact_lookup(self, key):
        """O(log n) exact match."""
        return self.tree.get(key, [])

    def range_scan(self, start, end):
        """O(log n + k) range query."""
        return {k: v for k, v in self.tree.items() if start <= k <= end}

class LSMTreeIndex:
    """Cassandra-style LSM-tree index."""

    def __init__(self, memtable_limit=4):
        self.memtable = {}         # In-memory (current writes)
        self.sstables = []          # On-disk (immutable, sorted)
        self.bloom_filters = []     # Per-SSTable bloom filter
        self.memtable_limit = memtable_limit

    def write(self, key, value):
        """Append-only write (fast!)."""
        self.memtable[key] = value
        if len(self.memtable) >= self.memtable_limit:
            self._flush()

    def _flush(self):
        sstable = dict(sorted(self.memtable.items()))
        bloom = set(sstable.keys())
        self.sstables.append(sstable)
        self.bloom_filters.append(bloom)
        self.memtable.clear()

    def read(self, key):
        """Check memtable, then SSTables with bloom filter."""
        if key in self.memtable:
            return self.memtable[key]
        for sstable, bloom in zip(reversed(self.sstables),
                                   reversed(self.bloom_filters)):
            if key in bloom:  # Bloom: O(1) probabilistic check
                if key in sstable:
                    return sstable[key]
        return None

class InvertedIndex:
    """Elasticsearch-style inverted index for full-text search."""

    def __init__(self):
        self.index = defaultdict(set)  # term -> set of doc_ids
        self.documents = {}

    def index_document(self, doc_id, text):
        """Tokenize and index each term."""
        self.documents[doc_id] = text
        tokens = text.lower().split()
        for token in tokens:
            self.index[token].add(doc_id)

    def search(self, query):
        """AND search: return docs containing ALL query terms."""
        terms = query.lower().split()
        if not terms:
            return set()
        result = self.index.get(terms[0], set())
        for term in terms[1:]:
            result = result & self.index.get(term, set())
        return result

    def search_or(self, query):
        """OR search: return docs containing ANY query term."""
        terms = query.lower().split()
        result = set()
        for term in terms:
            result |= self.index.get(term, set())
        return result

# Compare
btree = BTreeIndex()
btree.insert("alice@example.com", "doc1")
btree.insert("bob@example.com", "doc2")
print(btree.exact_lookup("alice@example.com"))  # ["doc1"] — O(log n)

inv = InvertedIndex()
inv.index_document("d1", "quick brown fox")
inv.index_document("d2", "quick red fox")
print(inv.search("quick fox"))  # {"d1", "d2"}
```

**AI/ML Application:**
- **Embedding indexes:** Vector databases (Pinecone, Milvus) use specialized indexes — HNSW (Hierarchical Navigable Small World) for approximate nearest neighbor search on ML embeddings, fundamentally different from B-tree/LSM indexes.
- **Feature discovery:** Inverted indexes on feature metadata enable "find all features related to user engagement" — full-text search across the feature catalog.
- **Experiment search:** Compound B-tree indexes on `{model_type, metric, timestamp}` enable efficient experiment queries: "find all transformer experiments with accuracy > 0.9 from this month."

**Real-World Example:**
MongoDB's WiredTiger uses B-tree indexes with page compression. Cassandra uses LSM-trees (bloom filter hit rate >99% reduces unnecessary SSTable reads). Elasticsearch powers Wikipedia search (inverted index over millions of articles). Neo4j uses B-tree for property lookups but index-free adjacency for traversal — this hybrid approach gives O(log n) for "find user by name" and O(1) per hop for "traverse friends."

> **Interview Tip:** "Key distinction: B-tree indexes optimize reads (in-place updates maintain sorted order), LSM-tree indexes optimize writes (append-only, batch-merge). Bloom filters are the secret weapon for LSM-trees — they answer 'is this key definitely NOT in this SSTable?' with O(1) and ~1% false positive rate, avoiding expensive disk reads. Inverted indexes are optimal for text search (term → document_ids mapping). Modern databases combine multiple index types — MongoDB uses B-trees + inverted (Atlas Search) + geospatial."

---

### 35. How do different storage formats , like JSON , BSON , or binary, affect the performance and flexibility of NoSQL databases ? 🔒

**Type:** 📝 Question
**Access:** 🔒 Premium

**Answer:**

Storage format determines **serialization speed**, **storage efficiency**, **query performance**, and **data type support**. **JSON** is human-readable but slow to parse. **BSON** (Binary JSON) adds types and is faster to traverse. **Protocol Buffers**, **MessagePack**, and **Avro** are compact binary formats optimized for performance. The format choice fundamentally affects read/write throughput.

```
  STORAGE FORMAT COMPARISON:
  
  SAME DATA IN DIFFERENT FORMATS:
  
  1. JSON (Text, human-readable):
  {"name": "Alice", "age": 30, "score": 95.5}
  Size: 46 bytes
  Parse: String scanning, number conversion
  
  2. BSON (Binary JSON, MongoDB):
  \x32\x00\x00\x00          // Document size (50 bytes)
  \x02 name\x00             // Type: string
  \x06\x00\x00\x00 Alice\x00  // String length + value
  \x10 age\x00              // Type: int32
  \x1e\x00\x00\x00          // Value: 30
  \x01 score\x00            // Type: double
  \x00\x00\x00\x00\x00\xe0\x57\x40  // Value: 95.5
  \x00                      // Document end
  Size: ~50 bytes (slightly larger than JSON!)
  Parse: Direct offset access (no scanning)
  
  3. MESSAGEPACK (Binary, compact):
  \x83                     // Map with 3 entries
  \xa4 name \xa5 Alice     // Fixed-size headers
  \xa3 age \x1e            // Integer encoded efficiently
  \xa5 score \xcb...       // Double
  Size: ~30 bytes (35% smaller than JSON)
  
  4. PROTOCOL BUFFERS (Schema-based binary):
  Schema: message User {
    string name = 1;
    int32 age = 2;
    double score = 3;
  }
  Size: ~20 bytes (56% smaller than JSON)
  Parse: Fastest (schema-driven, no field names in wire)
  
  PERFORMANCE COMPARISON:
  +----------+---------+---------+---------+--------+
  | Format   | Size    | Encode  | Decode  | Types  |
  +----------+---------+---------+---------+--------+
  | JSON     | Large   | Fast    | Slow    | 6      |
  | BSON     | Medium+ | Fast    | Fast*   | 18     |
  | MsgPack  | Small   | Fast    | Fast    | ~15    |
  | Protobuf | Smallest| Fastest | Fastest | Custom |
  | Avro     | Small   | Fast    | Fast    | Rich   |
  +----------+---------+---------+---------+--------+
  *BSON: fast traversal (offset access, not scanning)
  
  BSON ADVANTAGE: TYPED TRAVERSAL
  JSON:  {"age": 30}  --> Parse "age" as string, find ":",
                          parse "30" as string, convert to int
  BSON:  [type:int32][name:"age"][value:30_as_4_bytes]
         --> Skip to field by offset, read 4 bytes as int
         --> No string parsing needed!
```

| Format | Size | Human Readable | Type System | Used By | Best For |
|--------|------|---------------|-------------|---------|----------|
| **JSON** | Largest | Yes | 6 types (string, number, bool, null, array, object) | CouchDB, many APIs | Readability, debugging |
| **BSON** | ~JSON+5% | No (binary) | 18 types (Date, Decimal128, Binary, ObjectId, RegExp) | MongoDB | Rich types + fast in-doc traversal |
| **MessagePack** | ~30% smaller | No | ~15 types | Redis modules, Fluentd | Compact + schemaless |
| **Protocol Buffers** | Smallest | No | Custom schema | Google Cloud, gRPC | Maximum performance |
| **Avro** | Small | No | Rich + schema evolution | Kafka, Hadoop | Schema evolution |
| **Parquet** | Column-optimal | No | Typed columns | Analytics (Spark, BigQuery) | Columnar analytics |

```python
# Storage format comparison
import json
import struct
import time

class JSONStorage:
    """JSON-based storage (CouchDB-style)."""

    def serialize(self, doc):
        return json.dumps(doc).encode("utf-8")

    def deserialize(self, data):
        return json.loads(data.decode("utf-8"))

    def get_field(self, data, field):
        """Must parse entire document to get one field."""
        doc = self.deserialize(data)
        return doc.get(field)

class BSONLikeStorage:
    """Simplified BSON-like binary format with offset access."""

    TYPE_STRING = 0x02
    TYPE_INT32 = 0x10
    TYPE_DOUBLE = 0x01

    def serialize(self, doc):
        parts = []
        index = {}  # field_name -> offset (enables skip-to access)
        offset = 4  # Skip document length header
        for key, value in doc.items():
            index[key] = offset
            if isinstance(value, str):
                encoded = self._encode_string(key, value)
            elif isinstance(value, int):
                encoded = self._encode_int(key, value)
            elif isinstance(value, float):
                encoded = self._encode_double(key, value)
            else:
                continue
            parts.append(encoded)
            offset += len(encoded)
        body = b"".join(parts)
        return struct.pack("<I", len(body) + 4) + body, index

    def _encode_string(self, key, value):
        key_bytes = key.encode("utf-8") + b"\x00"
        val_bytes = value.encode("utf-8") + b"\x00"
        return bytes([self.TYPE_STRING]) + key_bytes + struct.pack("<I", len(val_bytes)) + val_bytes

    def _encode_int(self, key, value):
        key_bytes = key.encode("utf-8") + b"\x00"
        return bytes([self.TYPE_INT32]) + key_bytes + struct.pack("<i", value)

    def _encode_double(self, key, value):
        key_bytes = key.encode("utf-8") + b"\x00"
        return bytes([self.TYPE_DOUBLE]) + key_bytes + struct.pack("<d", value)

class MessagePackLikeStorage:
    """Simplified MessagePack-like compact binary format."""

    def serialize(self, doc):
        # Minimal encoding: fixmap + fixstr + values
        parts = [bytes([0x80 | len(doc)])]  # fixmap header
        for key, value in doc.items():
            # Encode key as fixstr
            key_bytes = key.encode("utf-8")
            parts.append(bytes([0xa0 | len(key_bytes)]) + key_bytes)
            # Encode value
            if isinstance(value, str):
                val_bytes = value.encode("utf-8")
                parts.append(bytes([0xa0 | len(val_bytes)]) + val_bytes)
            elif isinstance(value, int) and 0 <= value < 128:
                parts.append(bytes([value]))  # Positive fixint
            elif isinstance(value, float):
                parts.append(b"\xcb" + struct.pack(">d", value))
        return b"".join(parts)

# Benchmark
def benchmark_formats():
    doc = {"name": "Alice", "age": 30, "score": 95.5, "active": True}
    json_store = JSONStorage()
    bson_store = BSONLikeStorage()
    msgpack_store = MessagePackLikeStorage()

    json_data = json_store.serialize(doc)
    bson_data, _ = bson_store.serialize({"name": "Alice", "age": 30, "score": 95.5})
    msgpack_data = msgpack_store.serialize({"name": "Alice", "age": 30, "score": 95.5})

    return {
        "json_size": len(json_data),
        "bson_size": len(bson_data),
        "msgpack_size": len(msgpack_data),
    }
```

**AI/ML Application:**
- **Model serialization:** Protocol Buffers (TensorFlow's SavedModel) and FlatBuffers (TFLite) store ML models in compact binary formats — 10x faster to load than JSON model configs.
- **Feature encoding:** Avro encodes training features with schema evolution — add new features without breaking existing training pipelines. Parquet stores features in columnar format for efficient batch reads.
- **Inference latency:** MessagePack encoding for prediction requests reduces serialization overhead vs JSON — critical for real-time ML serving where every millisecond counts.

**Real-World Example:**
MongoDB uses BSON internally for its rich type system (Date, Decimal128, ObjectId) — the wire protocol sends BSON between driver and server. CouchDB stores documents as JSON with optional binary attachments. DynamoDB uses a custom binary format internally but exposes JSON-like document API. Apache Kafka uses Avro with Schema Registry for schema-evolving event streams. Google's Bigtable stores raw bytes — the application defines the encoding.

> **Interview Tip:** "BSON's advantage over JSON isn't size (BSON is actually slightly larger due to type headers) — it's traversal speed. BSON encodes field lengths, so you can skip to a specific field without parsing the entire document. JSON must scan character-by-character to find a field. For a 1MB document, BSON field access is O(1) via offset; JSON field access is O(n) via scanning. Also mention Parquet for analytics: columnar storage means reading one column from a billion-row table only touches that column's bytes, not the entire dataset."

---
