# Apache Spark Interview Questions - Theory Questions

## Question 1

**What is Apache Spark and what are its main components?**

### Answer

**Definition**: Apache Spark is an open-source, distributed computing engine designed for fast, large-scale data processing with in-memory computing capabilities.

### Core Components

| Component | Purpose |
|-----------|---------|
| **Spark Core** | Task scheduling, memory management, RDD API |
| **Spark SQL** | Structured data processing with DataFrame/Dataset |
| **Spark Streaming** | Real-time stream processing |
| **MLlib** | Machine learning library |
| **GraphX** | Graph processing |

### Key Features

| Feature | Description |
|---------|-------------|
| In-memory computing | 100x faster than MapReduce |
| Lazy evaluation | Optimizes execution plan |
| DAG execution | Efficient task scheduling |
| Fault tolerance | RDD lineage for recovery |

### Python Code Example
```python
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("SparkDemo") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Check Spark version
print(f"Spark version: {spark.version}")

# Create DataFrame
data = [("Alice", 30), ("Bob", 25), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])
df.show()

# Stop session
spark.stop()
```

---

## Question 2

**Explain RDD, DataFrame, and Dataset in Spark.**

### Answer

### Comparison

| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| **Type safety** | Compile-time | Runtime | Compile-time |
| **Schema** | No | Yes | Yes |
| **Optimization** | None | Catalyst | Catalyst |
| **API** | Low-level | High-level | High-level |
| **Language** | All | All | Scala/Java |

### When to Use

| Use Case | Best Choice |
|----------|-------------|
| Unstructured data | RDD |
| Structured/semi-structured | DataFrame |
| Type safety needed | Dataset |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("RDDvsDF").getOrCreate()

# RDD: Low-level, functional transformations
rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
result_rdd = rdd.filter(lambda x: x > 2).map(lambda x: x * 2)
print(f"RDD result: {result_rdd.collect()}")

# DataFrame: High-level, optimized
data = [(1, "a"), (2, "b"), (3, "c")]
df = spark.createDataFrame(data, ["id", "value"])
result_df = df.filter(col("id") > 1).select("value")
result_df.show()
```

---

## Question 3

**What is lazy evaluation in Spark?**

### Answer

**Definition**: Spark delays execution of transformations until an action is called, allowing it to optimize the entire computation plan.

### Transformation vs Action

| Transformation | Action |
|----------------|--------|
| `map`, `filter`, `select` | `collect`, `count`, `show` |
| `groupBy`, `join` | `write`, `take` |
| Returns new RDD/DF | Returns value/writes data |
| Lazy | Triggers execution |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("LazyEval").getOrCreate()

df = spark.range(1000000)

# Transformations (LAZY - nothing executes yet)
transformed = df \
    .filter(col("id") > 500000) \
    .select((col("id") * 2).alias("doubled"))

print("Transformations defined - no execution yet")

# Action (TRIGGERS execution)
count = transformed.count()
print(f"Count: {count}")

# View execution plan
transformed.explain(True)
```

---

## Question 4

**Explain Spark's DAG and execution model.**

### Answer

**Definition**: DAG (Directed Acyclic Graph) represents the logical execution plan showing dependencies between RDDs/operations.

### Execution Flow

| Step | Description |
|------|-------------|
| 1. DAG creation | Transformations create DAG |
| 2. Stage division | Split at shuffle boundaries |
| 3. Task creation | Tasks per partition per stage |
| 4. Execution | Tasks run on executors |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Job** | Action triggers a job |
| **Stage** | Set of tasks with no shuffle |
| **Task** | Unit of work on one partition |
| **Shuffle** | Data redistribution across nodes |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DAGExample").getOrCreate()
sc = spark.sparkContext

rdd = sc.parallelize(range(1000), 4)

# Transformations create DAG
mapped = rdd.map(lambda x: (x % 10, x))      # Stage 1
grouped = mapped.groupByKey()                 # Shuffle boundary
summed = grouped.mapValues(sum)               # Stage 2

# View DAG
print(summed.toDebugString())

# Action triggers execution
result = summed.collect()
```

---

## Question 5

**What are narrow and wide transformations?**

### Answer

### Comparison

| Narrow Transformation | Wide Transformation |
|-----------------------|---------------------|
| Data stays in partition | Data shuffles across partitions |
| No network transfer | Network transfer required |
| `map`, `filter`, `flatMap` | `groupBy`, `reduceByKey`, `join` |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Transformations").getOrCreate()
sc = spark.sparkContext

rdd = sc.parallelize([(1, 'a'), (2, 'b'), (3, 'a'), (4, 'b')], 2)

# Narrow transformations (no shuffle)
narrow_1 = rdd.map(lambda x: (x[0] * 2, x[1]))
narrow_2 = rdd.filter(lambda x: x[0] > 1)

# Wide transformations (shuffle required)
wide_1 = rdd.groupByKey()
wide_2 = rdd.reduceByKey(lambda a, b: a + b)

print(narrow_1.toDebugString())
print(wide_1.toDebugString())
```

### Interview Tip
Prefer `reduceByKey` over `groupByKey` as it performs local aggregation before shuffle.

---

## Question 6

**How does Spark handle fault tolerance?**

### Answer

### Fault Tolerance Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **RDD Lineage** | Recompute lost partitions from source |
| **Checkpointing** | Save RDD to reliable storage |
| **Task retry** | Re-execute failed tasks |

### Python Code Example
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FaultTolerance").getOrCreate()
sc = spark.sparkContext

# Set checkpoint directory
sc.setCheckpointDir("hdfs:///checkpoints")

rdd = sc.parallelize(range(1000))
rdd1 = rdd.map(lambda x: x * 2)
rdd2 = rdd1.filter(lambda x: x > 100)

# View lineage
print(rdd2.toDebugString())

# Checkpoint breaks lineage
rdd2.checkpoint()
rdd2.count()

print("After checkpoint:")
print(rdd2.toDebugString())
```

---

## Question 7

**What is the Catalyst optimizer?**

### Answer

**Definition**: Catalyst is Spark SQL's query optimizer that transforms logical plans into optimized physical execution plans.

### Optimization Phases

| Phase | Description |
|-------|-------------|
| **Analysis** | Resolve references, validate |
| **Logical Optimization** | Predicate pushdown, constant folding |
| **Physical Planning** | Generate execution strategies |
| **Code Generation** | Generate optimized bytecode |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Catalyst").getOrCreate()

df = spark.range(1000000).withColumn("value", col("id") * 2)

result = df \
    .filter(col("id") > 500000) \
    .filter(col("value") < 1500000) \
    .select("id")

# View optimization plans
result.explain(mode="extended")
```

---

## Question 8

**Explain Spark memory management.**

### Answer

### Memory Regions

| Region | Purpose | Default |
|--------|---------|---------|
| **Execution** | Shuffles, joins, sorts | 50% |
| **Storage** | Cache, broadcast | 50% |
| **Reserved** | System overhead | 300MB |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark import StorageLevel

spark = SparkSession.builder \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.5") \
    .appName("MemoryManagement").getOrCreate()

df = spark.range(1000000)

# Different storage levels
df.cache()  # MEMORY_ONLY
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.MEMORY_ONLY_SER)

# Unpersist when done
df.unpersist()
```

---

## Question 9

**What is data shuffling in Spark?**

### Answer

**Definition**: Shuffling is the process of redistributing data across partitions, typically required for wide transformations.

### Shuffle Operations

| Operation | Requires Shuffle |
|-----------|------------------|
| `groupByKey` | Yes |
| `reduceByKey` | Yes (with combiner) |
| `join` | Yes (unless co-partitioned) |
| `map`, `filter` | No |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

spark = SparkSession.builder \
    .config("spark.sql.shuffle.partitions", "200") \
    .appName("ShuffleExample").getOrCreate()

# BAD: groupByKey shuffles all data
rdd = spark.sparkContext.parallelize([(i % 100, i) for i in range(10000)])
grouped = rdd.groupByKey().mapValues(sum)

# BETTER: reduceByKey does local combine first
reduced = rdd.reduceByKey(lambda a, b: a + b)

# Broadcast join (avoid shuffle)
large_df = spark.range(1000000).withColumnRenamed("id", "key")
small_df = spark.createDataFrame([(i, f"v{i}") for i in range(100)], ["key", "value"])
broadcast_join = large_df.join(broadcast(small_df), "key")
```

---

## Question 10

**Explain Spark Streaming vs Structured Streaming.**

### Answer

### Comparison

| Feature | DStream (old) | Structured Streaming |
|---------|---------------|----------------------|
| **API** | RDD-based | DataFrame-based |
| **Exactly-once** | Difficult | Built-in |
| **Event time** | Manual | Native support |

### Python Code Example
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Streaming").getOrCreate()

# Read from Kafka
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "events") \
    .load()

# Windowed aggregation with watermark
schema = "user_id STRING, event_time TIMESTAMP, action STRING"
events = stream_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

result = events \
    .withWatermark("event_time", "10 minutes") \
    .groupBy(window("event_time", "5 minutes"), "user_id") \
    .count()

# Write to console
query = result.writeStream \
    .format("console") \
    .outputMode("update") \
    .start()
```

