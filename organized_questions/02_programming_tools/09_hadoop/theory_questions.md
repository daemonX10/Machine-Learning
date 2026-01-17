# Hadoop Interview Questions - Theory Questions

## Question 1

**What is Hadoop and what are its core components?**

### Answer

**Definition**: Hadoop is an open-source framework for distributed storage and processing of large datasets across clusters of computers using simple programming models.

### Core Components

| Component | Description |
|-----------|-------------|
| **HDFS** | Hadoop Distributed File System - storage layer |
| **MapReduce** | Processing engine for batch processing |
| **YARN** | Yet Another Resource Negotiator - resource management |
| **Hadoop Common** | Utilities and libraries |

### HDFS Architecture

| Component | Role |
|-----------|------|
| NameNode | Master - manages metadata |
| DataNode | Slave - stores actual data |
| Secondary NameNode | Checkpoint for NameNode |

### Python Code Example (using hdfs library)
```python
from hdfs import InsecureClient

# Connect to HDFS
client = InsecureClient('http://namenode:50070', user='hadoop')

# List directory contents
files = client.list('/')
print(f"Root files: {files}")

# Upload file
client.upload('/user/data/', 'local_file.txt')

# Download file
client.download('/user/data/file.txt', 'local_copy.txt')

# Read file content
with client.read('/user/data/file.txt') as reader:
    content = reader.read()
```

---

## Question 2

**Explain the HDFS architecture in detail.**

### Answer

### HDFS Components

| Component | Function |
|-----------|----------|
| **NameNode** | Stores metadata (file names, block locations) |
| **DataNode** | Stores actual data blocks |
| **Block** | Default 128MB chunks |
| **Replication** | Default 3 copies for fault tolerance |

### Data Flow

| Operation | Flow |
|-----------|------|
| Write | Client → NameNode → DataNodes (pipeline) |
| Read | Client → NameNode → DataNode |

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Check file status
status = client.status('/user/data/large_file.csv')
print(f"File size: {status['length']} bytes")
print(f"Replication: {status['replication']}")
print(f"Block size: {status['blockSize']}")

# Create directory
client.makedirs('/user/new_directory')

# Delete file
client.delete('/user/old_file.txt')

# Rename/Move
client.rename('/user/old_path', '/user/new_path')

# Set replication factor
client.set_replication('/user/important_file.txt', replication=5)
```

### HDFS Design Principles
- **Write-once, read-many**: Optimized for batch processing
- **Large files**: Designed for GB/TB size files
- **Streaming access**: Sequential reads preferred
- **Commodity hardware**: Fault tolerance through replication

---

## Question 3

**What is MapReduce and how does it work?**

### Answer

**Definition**: MapReduce is a programming model for processing large datasets in parallel across a Hadoop cluster.

### Phases

| Phase | Description |
|-------|-------------|
| **Map** | Process input, emit key-value pairs |
| **Shuffle** | Group by key, sort |
| **Reduce** | Aggregate values for each key |

### Python Code Example (using mrjob)
```python
from mrjob.job import MRJob
from mrjob.step import MRStep

# Word Count Example
class WordCount(MRJob):
    
    def mapper(self, _, line):
        """Map phase: emit (word, 1) for each word"""
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer(self, word, counts):
        """Reduce phase: sum counts for each word"""
        yield word, sum(counts)

# Run: python wordcount.py input.txt

# Advanced: Multi-step MapReduce
class TopWords(MRJob):
    
    def steps(self):
        return [
            MRStep(mapper=self.mapper_count,
                   reducer=self.reducer_count),
            MRStep(reducer=self.reducer_top)
        ]
    
    def mapper_count(self, _, line):
        for word in line.strip().split():
            yield word.lower(), 1
    
    def reducer_count(self, word, counts):
        yield None, (sum(counts), word)
    
    def reducer_top(self, _, word_counts):
        # Get top 10 words
        top_10 = sorted(word_counts, reverse=True)[:10]
        for count, word in top_10:
            yield word, count

if __name__ == '__main__':
    WordCount.run()
```

---

## Question 4

**What is YARN and what are its components?**

### Answer

**Definition**: YARN (Yet Another Resource Negotiator) is Hadoop's resource management layer that separates resource management from processing.

### Components

| Component | Description |
|-----------|-------------|
| **ResourceManager** | Master - allocates resources |
| **NodeManager** | Slave - manages containers on each node |
| **ApplicationMaster** | Per-app coordinator |
| **Container** | Resource allocation unit |

### YARN vs MapReduce 1.0

| Aspect | MapReduce 1.0 | YARN |
|--------|---------------|------|
| Resource Management | JobTracker | ResourceManager |
| Processing | MapReduce only | Multiple frameworks |
| Scalability | ~4000 nodes | ~10000 nodes |

### Python Code Example
```python
# Using YARN REST API
import requests
import json

YARN_RM_URL = "http://resourcemanager:8088"

# Get cluster info
def get_cluster_info():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/info")
    return response.json()

# Get cluster metrics
def get_cluster_metrics():
    response = requests.get(f"{YARN_RM_URL}/ws/v1/cluster/metrics")
    metrics = response.json()['clusterMetrics']
    return {
        'activeNodes': metrics['activeNodes'],
        'totalMemory': metrics['totalMB'],
        'availableMemory': metrics['availableMB'],
        'appsRunning': metrics['appsRunning']
    }

# List applications
def list_applications(state='RUNNING'):
    response = requests.get(
        f"{YARN_RM_URL}/ws/v1/cluster/apps",
        params={'state': state}
    )
    return response.json()

# Kill application
def kill_application(app_id):
    response = requests.put(
        f"{YARN_RM_URL}/ws/v1/cluster/apps/{app_id}/state",
        json={'state': 'KILLED'}
    )
    return response.status_code == 200

print(get_cluster_metrics())
```

---

## Question 5

**What is the difference between HDFS and traditional file systems?**

### Answer

### Comparison

| Aspect | HDFS | Traditional FS |
|--------|------|----------------|
| **File Size** | GB to TB | KB to GB |
| **Access Pattern** | Write-once, read-many | Random read/write |
| **Block Size** | 128MB default | 4KB typical |
| **Replication** | Built-in (3x) | RAID or manual |
| **Fault Tolerance** | Automatic | Hardware dependent |
| **Scalability** | Thousands of nodes | Single machine |

### When to Use HDFS

| Use Case | HDFS Suitable |
|----------|---------------|
| Log analysis | ✅ Yes |
| Random access | ❌ No |
| Large batch processing | ✅ Yes |
| Low-latency queries | ❌ No |
| Data archival | ✅ Yes |

### Python Code Example
```python
from hdfs import InsecureClient
import os

# HDFS operations vs local file system
class FileSystemComparison:
    def __init__(self, hdfs_url, hdfs_user):
        self.hdfs = InsecureClient(hdfs_url, user=hdfs_user)
    
    def upload_to_hdfs(self, local_path, hdfs_path):
        """Upload local file to HDFS"""
        self.hdfs.upload(hdfs_path, local_path)
        
        # Verify
        status = self.hdfs.status(hdfs_path)
        print(f"Uploaded: {status['length']} bytes")
        print(f"Block size: {status['blockSize']} bytes")
        print(f"Replication: {status['replication']}")
    
    def compare_read_performance(self, hdfs_path, local_path):
        """Compare read performance"""
        import time
        
        # HDFS read
        start = time.time()
        with self.hdfs.read(hdfs_path) as reader:
            hdfs_data = reader.read()
        hdfs_time = time.time() - start
        
        # Local read
        start = time.time()
        with open(local_path, 'rb') as f:
            local_data = f.read()
        local_time = time.time() - start
        
        print(f"HDFS read: {hdfs_time:.2f}s")
        print(f"Local read: {local_time:.2f}s")
```

---

## Question 6

**Explain data replication in HDFS.**

### Answer

### Replication Strategy

| Factor | Default | Description |
|--------|---------|-------------|
| Replication Factor | 3 | Number of copies |
| Rack Awareness | Enabled | Distributes across racks |
| Block Placement | Optimized | 1 local, 2 remote |

### Block Placement Policy
1. First replica: Same node as writer (or random if external)
2. Second replica: Different rack
3. Third replica: Same rack as second, different node

### Python Code Example
```python
from hdfs import InsecureClient

client = InsecureClient('http://namenode:50070', user='hadoop')

# Set replication for specific file
def set_replication(path, factor):
    """Change replication factor"""
    client.set_replication(path, replication=factor)
    status = client.status(path)
    print(f"New replication: {status['replication']}")

# Check replication status
def check_replication(path):
    """Check if file is under-replicated"""
    status = client.status(path)
    current = status['replication']
    
    # Using WebHDFS to get block info
    import requests
    response = requests.get(
        f'http://namenode:50070/webhdfs/v1{path}?op=GETFILEBLOCKLOCATIONS'
    )
    blocks = response.json()
    
    for block in blocks.get('BlockLocations', {}).get('BlockLocation', []):
        actual_replicas = len(block.get('hosts', []))
        if actual_replicas < current:
            print(f"Block under-replicated: {actual_replicas}/{current}")

# Different replication for different data types
def upload_with_replication(local_path, hdfs_path, data_type):
    """Upload with appropriate replication"""
    replication_config = {
        'critical': 5,
        'normal': 3,
        'temporary': 1
    }
    
    client.upload(hdfs_path, local_path)
    client.set_replication(hdfs_path, replication_config.get(data_type, 3))
```

---

## Question 7

**What is NameNode federation and High Availability?**

### Answer

### NameNode High Availability (HA)

| Component | Role |
|-----------|------|
| Active NameNode | Handles all client operations |
| Standby NameNode | Ready to take over |
| JournalNodes | Store edit logs (quorum) |
| ZooKeeper | Automatic failover |

### Federation

| Feature | Description |
|---------|-------------|
| Multiple NameNodes | Each manages namespace portion |
| Block Pools | Each NameNode has its own |
| Scalability | Horizontal namespace scaling |

### Python Code Example
```python
import requests
from kazoo.client import KazooClient

class HDFSHAClient:
    def __init__(self, namenodes, zk_hosts):
        self.namenodes = namenodes  # List of NameNode URLs
        self.zk = KazooClient(hosts=zk_hosts)
        self.zk.start()
    
    def get_active_namenode(self):
        """Find active NameNode"""
        for nn in self.namenodes:
            try:
                response = requests.get(f"{nn}/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus")
                status = response.json()['beans'][0]['State']
                if status == 'active':
                    return nn
            except:
                continue
        return None
    
    def check_health(self):
        """Check cluster health"""
        active = self.get_active_namenode()
        if not active:
            return {'status': 'CRITICAL', 'message': 'No active NameNode'}
        
        response = requests.get(f"{active}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystem")
        metrics = response.json()['beans'][0]
        
        return {
            'status': 'OK',
            'activeNN': active,
            'totalBlocks': metrics['BlocksTotal'],
            'missingBlocks': metrics['MissingBlocks'],
            'underReplicatedBlocks': metrics['UnderReplicatedBlocks']
        }
    
    def trigger_failover(self):
        """Manual failover (use with caution)"""
        # This would typically use hdfs haadmin command
        import subprocess
        result = subprocess.run(
            ['hdfs', 'haadmin', '-failover', 'nn1', 'nn2'],
            capture_output=True, text=True
        )
        return result.returncode == 0
```

---

## Question 8

**What are the different input formats in Hadoop?**

### Answer

### Common Input Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| TextInputFormat | Line-by-line text | Log files |
| KeyValueTextInputFormat | Tab-separated K-V | Structured text |
| SequenceFileInputFormat | Binary K-V | Intermediate data |
| NLineInputFormat | N lines per split | Fixed-size splits |
| CombineFileInputFormat | Combine small files | Many small files |

### Python Code Example (mrjob)
```python
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol, RawValueProtocol
import json
import csv

# Processing different input formats

# 1. JSON Lines format
class ProcessJSON(MRJob):
    INPUT_PROTOCOL = RawValueProtocol
    
    def mapper(self, _, line):
        record = json.loads(line)
        yield record['category'], record['amount']
    
    def reducer(self, category, amounts):
        yield category, sum(amounts)

# 2. CSV format
class ProcessCSV(MRJob):
    
    def mapper(self, _, line):
        row = next(csv.reader([line]))
        if len(row) >= 3:
            yield row[0], float(row[2])  # category, amount
    
    def reducer(self, key, values):
        yield key, sum(values)

# 3. Custom delimiter
class ProcessCustomDelimiter(MRJob):
    
    def mapper(self, _, line):
        fields = line.split('|')  # Pipe delimiter
        yield fields[0], 1
    
    def reducer(self, key, counts):
        yield key, sum(counts)

# 4. Multi-line records (using combiner)
class ProcessMultiLine(MRJob):
    
    def mapper_init(self):
        self.buffer = []
    
    def mapper(self, _, line):
        if line.startswith('---'):  # Record separator
            if self.buffer:
                record = '\n'.join(self.buffer)
                yield 'record', record
                self.buffer = []
        else:
            self.buffer.append(line)
    
    def mapper_final(self):
        if self.buffer:
            yield 'record', '\n'.join(self.buffer)

if __name__ == '__main__':
    ProcessJSON.run()
```

---

## Question 9

**Explain the Hadoop ecosystem components.**

### Answer

### Ecosystem Overview

| Component | Category | Purpose |
|-----------|----------|---------|
| **Hive** | SQL | SQL-like queries |
| **Pig** | Scripting | Data flow language |
| **HBase** | NoSQL | Real-time random access |
| **Spark** | Processing | In-memory processing |
| **Sqoop** | Data Transfer | RDBMS ↔ HDFS |
| **Flume** | Ingestion | Log collection |
| **Kafka** | Streaming | Message queue |
| **Oozie** | Workflow | Job scheduling |
| **ZooKeeper** | Coordination | Distributed coordination |

### Python Code Example
```python
# Working with Hadoop ecosystem

# 1. Hive with PyHive
from pyhive import hive

conn = hive.Connection(host='hiveserver', port=10000, database='default')
cursor = conn.cursor()

cursor.execute('SELECT * FROM sales LIMIT 10')
for row in cursor.fetchall():
    print(row)

# 2. HBase with happybase
import happybase

connection = happybase.Connection('hbase-master')
table = connection.table('users')

# Put data
table.put(b'user1', {b'info:name': b'John', b'info:age': b'30'})

# Get data
row = table.row(b'user1')
print(row)

# Scan
for key, data in table.scan(row_prefix=b'user'):
    print(key, data)

# 3. Sqoop-like data transfer
import subprocess

def sqoop_import(jdbc_url, table, target_dir, username, password):
    """Import from RDBMS to HDFS"""
    cmd = [
        'sqoop', 'import',
        '--connect', jdbc_url,
        '--username', username,
        '--password', password,
        '--table', table,
        '--target-dir', target_dir,
        '--num-mappers', '4'
    ]
    subprocess.run(cmd, check=True)

# 4. Oozie workflow submission
def submit_oozie_workflow(oozie_url, properties):
    """Submit Oozie workflow"""
    import requests
    
    response = requests.post(
        f"{oozie_url}/v1/jobs",
        params={'action': 'start'},
        data=properties
    )
    return response.json()['id']
```

---

## Question 10

**What is speculative execution in Hadoop?**

### Answer

**Definition**: Speculative execution launches backup copies of slow-running tasks to reduce job completion time.

### How It Works

| Step | Description |
|------|-------------|
| Monitor | Track task progress |
| Detect | Identify slow tasks |
| Launch | Start backup task |
| Kill | Terminate slower one |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mapreduce.map.speculative` | true | Enable for mappers |
| `mapreduce.reduce.speculative` | true | Enable for reducers |

### Python Code Example
```python
from mrjob.job import MRJob
from mrjob.step import MRStep
import time
import random

class JobWithSpeculative(MRJob):
    """Example showing when speculative execution helps"""
    
    JOBCONF = {
        'mapreduce.map.speculative': 'true',
        'mapreduce.reduce.speculative': 'true',
        'mapreduce.job.speculative.slowtaskthreshold': '1.0',  # 100% slower than average
    }
    
    def mapper(self, _, line):
        # Simulate variable processing time
        # In real scenarios, this could be due to:
        # - Bad hardware
        # - Network issues
        # - Data skew
        
        if random.random() < 0.01:  # 1% chance of being slow
            time.sleep(10)  # Slow task
        
        for word in line.split():
            yield word, 1
    
    def reducer(self, word, counts):
        yield word, sum(counts)

# Monitoring speculative execution
class SpeculativeMonitor:
    def __init__(self, yarn_url):
        self.yarn_url = yarn_url
    
    def get_speculative_tasks(self, app_id):
        """Get info about speculative task attempts"""
        import requests
        
        response = requests.get(
            f"{self.yarn_url}/ws/v1/history/mapreduce/jobs/{app_id}/tasks"
        )
        tasks = response.json()['tasks']['task']
        
        speculative = []
        for task in tasks:
            if task.get('successfulAttempt') != task.get('id') + '_0':
                # Original attempt wasn't successful
                speculative.append({
                    'taskId': task['id'],
                    'type': task['type'],
                    'elapsedTime': task['elapsedTime']
                })
        
        return speculative

# When to disable speculative execution
"""
Disable when:
1. Tasks have side effects (writing to external systems)
2. Tasks are expensive to restart
3. Cluster is heavily loaded
4. Tasks are non-idempotent
"""

if __name__ == '__main__':
    JobWithSpeculative.run()
```

### Best Practices
- Enable for CPU-bound tasks
- Disable for tasks with external side effects
- Monitor cluster resources
- Consider data locality impact
