# Ml Design Patterns Interview Questions - Scenario_Based Questions

## Question 1

**Discuss the purpose of the 'Replay' design pattern in machine learning.**

**Answer:**

### 1. Definition
The Replay pattern stores historical data/events and replays them to retrain models, test new algorithms, or simulate past scenarios. It enables reproducibility and experimentation on historical data.

### 2. Core Concepts
- **Event Logging**: Store all input data with timestamps
- **Time Travel**: Recreate exact state at any past moment
- **A/B Backtesting**: Test new models on historical data before deployment
- **Debugging**: Reproduce bugs by replaying exact inputs

### 3. Scenario Application
**Problem**: New recommendation model deployed, metrics dropped.

**Using Replay Pattern**:
1. Retrieve historical request logs from past week
2. Replay same requests through old model and new model
3. Compare predictions side-by-side
4. Identify which request patterns caused degradation
5. Fix model, replay again to validate

### 4. When to Use
- Model debugging and root cause analysis
- Backtesting before production deployment
- Training on exact historical scenarios
- Regulatory audits requiring decision reconstruction

### 5. Python Code Example

```python
import json
from datetime import datetime
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)
    
    def log_event(self, features, prediction, timestamp=None):
        event = {
            'timestamp': timestamp or datetime.utcnow().isoformat(),
            'features': features,
            'prediction': prediction
        }
        self.buffer.append(event)
    
    def replay(self, model, start_time=None, end_time=None):
        """Replay historical events through a model"""
        results = []
        for event in self.buffer:
            if start_time and event['timestamp'] < start_time:
                continue
            if end_time and event['timestamp'] > end_time:
                continue
            
            new_pred = model.predict(event['features'])
            results.append({
                'original': event['prediction'],
                'replayed': new_pred,
                'match': event['prediction'] == new_pred
            })
        return results

# Usage
buffer = ReplayBuffer()
buffer.log_event({'user_id': 1, 'item_id': 100}, prediction='buy')
buffer.log_event({'user_id': 2, 'item_id': 200}, prediction='skip')

class NewModel:
    def predict(self, features):
        return 'buy' if features['item_id'] > 150 else 'skip'

results = buffer.replay(NewModel())
print(f"Replay results: {results}")
```

### 6. Interview Tips
- Replay enables offline evaluation without production risk
- Combine with A/B testing for comprehensive validation
- Storage cost is a trade-off - sample if full logging is expensive

---

## Question 2

**Discuss the 'Microservice' design pattern in deploying ML models.**

**Answer:**

### 1. Definition
The Microservice pattern deploys ML models as independent, loosely-coupled services that communicate via APIs. Each model/functionality runs in its own container with separate scaling, deployment, and lifecycle.

### 2. Core Concepts
- **Single Responsibility**: Each service does one thing well
- **Independent Deployment**: Update one model without affecting others
- **Technology Agnostic**: Python service, Java caller - doesn't matter
- **Fault Isolation**: One service failure doesn't crash entire system
- **Independent Scaling**: Scale recommendation service more than fraud service

### 3. Architecture Example
```
API Gateway
    |
    +-- User Service (authentication)
    +-- Feature Service (feature engineering)
    +-- Model Service A (recommendations)
    +-- Model Service B (fraud detection)
    +-- Model Service C (pricing)
```

### 4. Scenario Application
**E-commerce Platform**:
- **Recommendation Service**: Product suggestions, scales during peak hours
- **Fraud Service**: Transaction validation, always-on critical service
- **Search Ranking Service**: Query understanding + ranking
- Each team owns their service, deploys independently

### 5. Advantages vs Monolith

| Aspect | Microservice | Monolith |
|--------|--------------|----------|
| Deployment | Independent | All-or-nothing |
| Scaling | Per-service | Entire app |
| Failure | Isolated | Cascading |
| Tech stack | Flexible | Uniform |
| Complexity | Higher | Lower |

### 6. Python Code Example

```python
# Microservice architecture example

# Service 1: Feature Service
from flask import Flask, request, jsonify

feature_app = Flask('feature_service')

@feature_app.route('/features/<user_id>', methods=['GET'])
def get_features(user_id):
    features = {'user_id': user_id, 'age': 25, 'segment': 'premium'}
    return jsonify(features)

# Service 2: Recommendation Service
import requests

recommendation_app = Flask('recommendation_service')

@recommendation_app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data['user_id']
    
    # Call Feature Service (inter-service communication)
    features = requests.get(f'http://feature-service:5001/features/{user_id}').json()
    
    # Generate recommendations based on features
    recommendations = ['item_1', 'item_2', 'item_3']
    return jsonify({'user_id': user_id, 'items': recommendations})

# Docker Compose for orchestration
docker_compose = """
version: '3'
services:
  feature-service:
    build: ./feature_service
    ports: ["5001:5001"]
  recommendation-service:
    build: ./recommendation_service
    ports: ["5002:5002"]
    depends_on: [feature-service]
"""
```

### 7. Common Pitfalls
- **Network Latency**: Multiple service calls add latency
- **Distributed Tracing**: Hard to debug across services
- **Data Consistency**: Each service may have stale data
- **Operational Overhead**: More services = more to manage

### 8. Interview Tips
- Discuss trade-offs: simplicity of monolith vs flexibility of microservices
- Mention service mesh (Istio), API gateways, container orchestration (K8s)

---

## Question 3

**Can you discuss the 'Warm Start' pattern in machine learning model training?**

**Answer:**

### 1. Definition
Warm Start initializes model training with pre-trained weights instead of random initialization. This accelerates convergence and often improves final performance, especially with limited data.

### 2. Core Concepts
- **Weight Initialization**: Start from pre-trained parameters
- **Transfer Learning**: Leverage knowledge from related tasks
- **Incremental Training**: Continue from previous checkpoint
- **Faster Convergence**: Skip early training phases

### 3. Types of Warm Start
- **Self Warm Start**: Continue training from own checkpoint
- **Transfer Warm Start**: Initialize from model trained on different task
- **Partial Warm Start**: Initialize some layers, randomize others

### 4. Scenario Application
**Scenario**: Monthly model retraining for credit scoring.

**Without Warm Start**:
- Train from scratch each month
- 10 hours training time
- Model may not converge to same quality

**With Warm Start**:
- Initialize from last month's model
- 2 hours training (80% reduction)
- Stable performance across retrains
- New patterns learned on top of existing knowledge

### 5. When to Use
- Periodic retraining with similar data distribution
- Limited compute budget
- Fine-tuning pre-trained models (BERT, ResNet)
- Online learning scenarios

### 6. Python Code Example

```python
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib

# Scenario: Monthly model retraining with warm start

# Month 1: Train from scratch
X_month1 = np.random.randn(1000, 10)
y_month1 = (X_month1.sum(axis=1) > 0).astype(int)

model = SGDClassifier(warm_start=True, max_iter=1000)
model.fit(X_month1, y_month1)
print(f"Month 1 Score: {model.score(X_month1, y_month1):.4f}")

# Save model checkpoint
joblib.dump(model, 'model_month1.pkl')

# Month 2: Warm start from previous model
model_warm = joblib.load('model_month1.pkl')

X_month2 = np.random.randn(500, 10)  # New data
y_month2 = (X_month2.sum(axis=1) > 0).astype(int)

# Continue training (warm_start=True allows this)
model_warm.fit(X_month2, y_month2)
print(f"Month 2 Score: {model_warm.score(X_month2, y_month2):.4f}")

# Compare with cold start
model_cold = SGDClassifier(max_iter=1000)
model_cold.fit(X_month2, y_month2)
print(f"Cold Start Score: {model_cold.score(X_month2, y_month2):.4f}")

# PyTorch warm start example
import torch

def warm_start_pytorch(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Optionally load optimizer state for exact resume
    return model
```

### 7. Common Pitfalls
- **Distribution Shift**: Warm start may hurt if new data is very different
- **Overfitting**: Model may be too tuned to old patterns
- **Catastrophic Forgetting**: New training overwrites old knowledge

### 8. Interview Tips
- Discuss when NOT to use warm start (major data distribution change)
- Mention learning rate adjustment - often use lower LR for fine-tuning
- Warm start + early stopping is powerful combination

---

## Question 4

**Discuss the 'Rebalancing' design pattern and its importance in training datasets.**

**Answer:**

### 1. Definition
Rebalancing addresses class imbalance in training data by adjusting sample weights or modifying the dataset to ensure the model learns effectively from minority classes.

### 2. Core Concepts
- **Class Imbalance**: 99% negative, 1% positive (fraud detection)
- **Oversampling**: Duplicate minority class samples (SMOTE)
- **Undersampling**: Remove majority class samples
- **Class Weights**: Penalize misclassification of minority class more
- **Threshold Adjustment**: Tune decision threshold post-training

### 3. Rebalancing Techniques

| Technique | Method | Pros | Cons |
|-----------|--------|------|------|
| Random Oversampling | Duplicate minority | Simple | Overfitting |
| SMOTE | Synthetic samples | Better generalization | Noisy for high-dim |
| Random Undersampling | Remove majority | Faster training | Lose information |
| Class Weights | Weight loss function | No data change | May not be enough |

### 4. Scenario Application
**Fraud Detection**: 100,000 transactions, only 200 frauds (0.2%)

**Problem**: Model predicts "no fraud" for everything - 99.8% accuracy but useless.

**Rebalancing Solution**:
1. Apply SMOTE to generate synthetic fraud samples
2. Balance to 50-50 or 70-30 ratio
3. Use class_weight='balanced' in model
4. Evaluate with precision/recall, not accuracy

### 5. Python Code Example

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Create imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=20, 
                           weights=[0.99, 0.01], random_state=42)
print(f"Class distribution: {np.bincount(y)}")  # [9900, 100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Method 1: No rebalancing (baseline)
model_baseline = RandomForestClassifier()
model_baseline.fit(X_train, y_train)
print("\n=== Baseline (No Rebalancing) ===")
print(classification_report(y_test, model_baseline.predict(X_test)))

# Method 2: Class Weights
model_weighted = RandomForestClassifier(class_weight='balanced')
model_weighted.fit(X_train, y_train)
print("\n=== Class Weights ===")
print(classification_report(y_test, model_weighted.predict(X_test)))

# Method 3: SMOTE Oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {np.bincount(y_resampled)}")

model_smote = RandomForestClassifier()
model_smote.fit(X_resampled, y_resampled)
print("\n=== SMOTE ===")
print(classification_report(y_test, model_smote.predict(X_test)))

# Method 4: Undersampling
under = RandomUnderSampler(random_state=42)
X_under, y_under = under.fit_resample(X_train, y_train)
print(f"\nAfter Undersampling: {np.bincount(y_under)}")
```

### 6. When NOT to Rebalance
- When class imbalance reflects true prior probability
- When false positive cost differs from false negative cost (use cost-sensitive learning)
- Very small minority class - consider anomaly detection instead

### 7. Interview Tips
- Always evaluate with appropriate metrics (F1, AUC-PR, not accuracy)
- Discuss SMOTE variants (Borderline-SMOTE, ADASYN)
- Mention threshold tuning as alternative to resampling

---

## Question 5

**How would you scale a machine learning pipeline according to the 'Horizontal Scaling' design pattern?**

**Answer:**

### 1. Definition
Horizontal Scaling adds more machines/instances to handle increased load, rather than upgrading a single machine (vertical scaling). Each instance handles a portion of requests, enabling linear throughput growth.

### 2. Core Concepts
- **Stateless Services**: No session state stored on instance
- **Load Balancer**: Distributes requests across instances
- **Auto-scaling**: Add/remove instances based on metrics
- **Containerization**: Package service for easy replication
- **Shared Storage**: Models stored in central location (S3, NFS)

### 3. Scaling Strategy

| Component | Horizontal Scaling Approach |
|-----------|---------------------------|
| Data Ingestion | Kafka partitions, multiple consumers |
| Feature Engineering | Spark workers, distributed compute |
| Model Training | Data parallelism across GPUs |
| Model Serving | Multiple API replicas behind LB |
| Batch Inference | Parallel job executors |

### 4. Scenario Application
**Problem**: Recommendation API gets 1000 QPS during sale, but only 100 QPS normally.

**Horizontal Scaling Solution**:
1. Containerize model service (Docker)
2. Deploy to Kubernetes with auto-scaling
3. Set scaling policy: if CPU > 70% for 2 min, add replica
4. Load balancer distributes requests
5. During sale: 10 replicas; normal: 2 replicas
6. Cost-effective: pay only for what you use

### 5. Python Code Example

```python
# Architecture for horizontally scalable ML pipeline

# Step 1: Stateless model service
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model from shared storage (not local)
MODEL_PATH = os.environ.get('MODEL_PATH', 's3://models/latest.pkl')

class ModelService:
    def __init__(self):
        # Load from shared storage - all replicas get same model
        self.model = self._load_model()
    
    def _load_model(self):
        # In production: download from S3/GCS
        return lambda x: sum(x) > 0
    
    def predict(self, features):
        return self.model(features)

model_service = ModelService()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    result = model_service.predict(data['features'])
    return jsonify({'prediction': result})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

# Step 2: Kubernetes deployment config
k8s_deployment = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
spec:
  replicas: 3  # Start with 3 instances
  selector:
    matchLabels:
      app: ml-service
  template:
    spec:
      containers:
      - name: ml-service
        image: ml-service:latest
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
        env:
        - name: MODEL_PATH
          value: "s3://models/latest.pkl"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
print("Kubernetes config for horizontal scaling:")
print(k8s_deployment)
```

### 6. Key Requirements for Horizontal Scaling
- **Stateless**: No local state, use external cache (Redis)
- **Shared Model**: All replicas load same model version
- **Health Checks**: Load balancer routes to healthy instances only
- **Graceful Shutdown**: Complete in-flight requests before terminating

### 7. Interview Tips
- Discuss vertical vs horizontal trade-offs
- Mention statelessness as prerequisite
- Talk about cold start when scaling up (pre-warming)

---

## Question 6

**Discuss the 'Model Decay' design pattern and strategies to overcome it.**

**Answer:**

### 1. Definition
Model Decay (also called model staleness or model rot) refers to the degradation of model performance over time as the real-world data distribution shifts away from the training data distribution.

### 2. Core Concepts
- **Data Drift**: Input feature distributions change
- **Concept Drift**: Relationship between features and target changes
- **Performance Degradation**: Accuracy, precision, recall decline over time
- **Decay Rate**: How fast performance drops (varies by domain)

### 3. Causes of Model Decay

| Cause | Example |
|-------|---------|
| User behavior change | COVID changed shopping patterns |
| Seasonality | Holiday vs normal periods |
| Market dynamics | New competitors, regulations |
| Data quality issues | Upstream pipeline changes |
| Feature drift | Third-party data source changes |

### 4. Scenario Application
**E-commerce Recommendation Model**:
- Deployed: January 2024, CTR = 5%
- March 2024: CTR drops to 3.5%
- Diagnosis: New product categories added, user preferences shifted

**Resolution Strategy**:
1. Detect: Monitor CTR daily, alert on 20% drop
2. Diagnose: Compare feature distributions train vs production
3. Retrain: Weekly retraining on recent 30-day data
4. Validate: A/B test before full deployment
5. Prevent: Set up continuous training pipeline

### 5. Mitigation Strategies

```python
import numpy as np
from datetime import datetime, timedelta

class ModelDecayMonitor:
    def __init__(self, baseline_metric, decay_threshold=0.1):
        self.baseline = baseline_metric
        self.threshold = decay_threshold
        self.history = []
    
    def log_metric(self, metric_value, timestamp=None):
        self.history.append({
            'timestamp': timestamp or datetime.now(),
            'metric': metric_value,
            'decay': (self.baseline - metric_value) / self.baseline
        })
    
    def check_decay(self):
        if not self.history:
            return False, 0
        
        recent = self.history[-1]
        if recent['decay'] > self.threshold:
            return True, recent['decay']
        return False, recent['decay']
    
    def should_retrain(self):
        decayed, decay_rate = self.check_decay()
        if decayed:
            print(f"ALERT: Model decayed by {decay_rate:.1%}. Retrain recommended.")
            return True
        return False

# Mitigation strategies implementation
class DecayMitigation:
    @staticmethod
    def scheduled_retraining(model, data_loader, frequency_days=7):
        """Retrain on schedule regardless of performance"""
        # Triggered by scheduler (Airflow, cron)
        new_data = data_loader.get_recent_data(days=30)
        model.fit(new_data)
        return model
    
    @staticmethod
    def triggered_retraining(model, data_loader, monitor):
        """Retrain only when decay detected"""
        if monitor.should_retrain():
            new_data = data_loader.get_recent_data(days=30)
            model.fit(new_data)
        return model
    
    @staticmethod
    def online_learning(model, new_sample):
        """Continuously update with each new sample"""
        model.partial_fit(new_sample)
        return model
    
    @staticmethod
    def ensemble_with_recent(old_model, recent_model, alpha=0.7):
        """Blend old and recent model predictions"""
        def predict(x):
            old_pred = old_model.predict(x)
            recent_pred = recent_model.predict(x)
            return alpha * recent_pred + (1 - alpha) * old_pred
        return predict

# Usage
monitor = ModelDecayMonitor(baseline_metric=0.85, decay_threshold=0.1)
monitor.log_metric(0.82)  # Slight drop
monitor.log_metric(0.75)  # Significant drop

if monitor.should_retrain():
    print("Initiating retraining pipeline...")
```

### 6. Prevention Best Practices
- **Monitor continuously**: Don't wait for complaints
- **Automate retraining**: Scheduled or trigger-based
- **Version everything**: Data, model, features
- **A/B test updates**: Never blind deploy

### 7. Interview Tips
- Decay is inevitable - question is how fast and how to detect
- Different domains decay at different rates (finance: fast, medical imaging: slow)
- Balance stability vs freshness in retraining frequency

---

## Question 7

**Discuss how 'Continuous Evaluation' helps in maintaining model quality.**

**Answer:**

### 1. Definition
Continuous Evaluation monitors model performance in production using live predictions and outcomes, enabling early detection of degradation before significant business impact.

### 2. Core Concepts
- **Ground Truth Collection**: Gather actual outcomes for predictions
- **Delayed Labels**: Handle lag between prediction and outcome
- **Proxy Metrics**: Use leading indicators when labels unavailable
- **Sliding Window**: Evaluate on recent time window, not all-time
- **Statistical Testing**: Detect significant performance changes

### 3. Evaluation Components

| Component | Purpose | Example |
|-----------|---------|---------|
| Real-time metrics | Immediate feedback | Latency, error rate |
| Business metrics | Outcome impact | CTR, revenue, conversions |
| Model metrics | ML performance | Accuracy, AUC (when labels arrive) |
| Data quality | Input health | Missing values, schema violations |

### 4. Scenario Application
**Fraud Detection System**:
- **T+0 (immediate)**: Monitor prediction latency, API errors
- **T+1 hour**: Track approval rate, flagged transaction rate
- **T+7 days**: Confirmed fraud labels arrive, compute precision/recall
- **T+30 days**: Chargeback data, compute false negative rate

**Action Triggers**:
- Latency > 100ms: Alert oncall
- Flagged rate > 5%: Review model threshold
- Precision < 80%: Schedule retraining
- False negative spike: Emergency model update

### 5. Python Code Example

```python
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

class ContinuousEvaluator:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
        self.prediction_log = []
        self.outcome_log = []
    
    def log_prediction(self, prediction_id, prediction, timestamp=None):
        self.prediction_log.append({
            'id': prediction_id,
            'prediction': prediction,
            'timestamp': timestamp or datetime.now()
        })
    
    def log_outcome(self, prediction_id, actual):
        """Called when ground truth becomes available"""
        self.outcome_log.append({
            'id': prediction_id,
            'actual': actual,
            'timestamp': datetime.now()
        })
    
    def compute_metrics(self, window_days=7):
        """Compute metrics on recent window"""
        cutoff = datetime.now() - timedelta(days=window_days)
        
        # Join predictions with outcomes
        pred_dict = {p['id']: p for p in self.prediction_log}
        matched = []
        for outcome in self.outcome_log:
            if outcome['id'] in pred_dict:
                pred = pred_dict[outcome['id']]
                if pred['timestamp'] > cutoff:
                    matched.append({
                        'prediction': pred['prediction'],
                        'actual': outcome['actual']
                    })
        
        if not matched:
            return None
        
        preds = [m['prediction'] for m in matched]
        actuals = [m['actual'] for m in matched]
        
        accuracy = sum(p == a for p, a in zip(preds, actuals)) / len(matched)
        return {'accuracy': accuracy, 'sample_size': len(matched)}
    
    def check_degradation(self):
        """Detect statistically significant performance drop"""
        current = self.compute_metrics(window_days=7)
        if not current or current['sample_size'] < 100:
            return False, "Insufficient data"
        
        baseline_acc = self.baseline.get('accuracy', 0.9)
        current_acc = current['accuracy']
        
        # Binomial test for significance
        n = current['sample_size']
        k = int(current_acc * n)
        p_value = stats.binom_test(k, n, baseline_acc, alternative='less')
        
        if p_value < 0.05:
            return True, f"Significant drop: {baseline_acc:.2%} -> {current_acc:.2%}"
        return False, f"No significant change: {current_acc:.2%}"

# Usage
evaluator = ContinuousEvaluator(baseline_metrics={'accuracy': 0.92})

# Log predictions (real-time)
for i in range(200):
    pred = np.random.choice([0, 1])
    evaluator.log_prediction(f'req_{i}', pred)

# Log outcomes (delayed by days)
for i in range(200):
    actual = np.random.choice([0, 1], p=[0.6, 0.4])  # Some degradation
    evaluator.log_outcome(f'req_{i}', actual)

# Evaluate
degraded, message = evaluator.check_degradation()
print(f"Degradation detected: {degraded}")
print(f"Message: {message}")
```

### 6. Proxy Metrics When Labels Delayed
- **Recommendation**: Click rate (immediate) vs purchase rate (delayed)
- **Fraud**: Flagged rate (immediate) vs confirmed fraud (delayed)
- **Search**: Click-through rate (immediate) vs task completion (delayed)

### 7. Interview Tips
- Emphasize the importance of feedback loops
- Discuss sample size requirements for statistical significance
- Mention A/B testing integration for model comparisons

---

## Question 8

**How would you set up a 'Champion/Challenger' model deployment architecture?**

**Answer:**

### 1. Definition
Champion/Challenger is a deployment pattern where the current production model (Champion) runs alongside candidate models (Challengers), with traffic split to compare performance safely before full rollout.

### 2. Core Concepts
- **Champion**: Current production model (proven, stable)
- **Challenger**: New candidate model (to be evaluated)
- **Traffic Splitting**: Route percentage of requests to each
- **Shadow Mode**: Challenger runs but doesn't affect users
- **Gradual Rollout**: Increase challenger traffic as confidence grows

### 3. Deployment Stages

| Stage | Champion | Challenger | Risk |
|-------|----------|------------|------|
| Shadow | 100% served | 0% served (logged) | Zero |
| Canary | 95% served | 5% served | Low |
| Gradual | 70% served | 30% served | Medium |
| Switchover | 0% served | 100% served | Evaluated |

### 4. Scenario Application
**Search Ranking Model Update**:

**Week 1 - Shadow Mode**:
- Champion serves all traffic
- Challenger runs on same queries, predictions logged (not shown)
- Compare offline: latency, coverage, diversity

**Week 2 - Canary (5%)**:
- 5% users see Challenger results
- Monitor CTR, session duration, bounce rate
- Alert if metrics drop >10%

**Week 3 - Gradual (30%)**:
- Expand to 30% if Week 2 successful
- A/B test significance builds

**Week 4 - Decision**:
- If Challenger wins: Promote to Champion
- If Champion wins: Discard Challenger
- If inconclusive: Extend test

### 5. Python Code Example

```python
import random
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Model:
    name: str
    version: str
    
    def predict(self, features: Dict) -> Any:
        # Mock prediction
        return f"prediction_from_{self.name}"

class ChampionChallengerRouter:
    def __init__(self, champion: Model):
        self.champion = champion
        self.challengers = {}  # name -> (model, traffic_percentage)
        self.metrics = {'champion': [], 'challengers': {}}
    
    def add_challenger(self, name: str, model: Model, traffic_pct: float):
        """Add challenger with traffic percentage (0-1)"""
        self.challengers[name] = {'model': model, 'traffic': traffic_pct}
        self.metrics['challengers'][name] = []
    
    def route(self, request_id: str, features: Dict):
        """Route request to champion or challenger"""
        roll = random.random()
        cumulative = 0
        
        # Check if request goes to any challenger
        for name, config in self.challengers.items():
            cumulative += config['traffic']
            if roll < cumulative:
                prediction = config['model'].predict(features)
                self._log(name, request_id, prediction, is_champion=False)
                return prediction, name
        
        # Default to champion
        prediction = self.champion.predict(features)
        self._log('champion', request_id, prediction, is_champion=True)
        return prediction, 'champion'
    
    def shadow_evaluate(self, features: Dict):
        """Run challenger in shadow mode (no traffic impact)"""
        champion_pred = self.champion.predict(features)
        
        shadow_results = {'champion': champion_pred}
        for name, config in self.challengers.items():
            shadow_results[name] = config['model'].predict(features)
        
        return shadow_results
    
    def _log(self, model_name, request_id, prediction, is_champion):
        # Log for analysis
        pass
    
    def promote_challenger(self, challenger_name: str):
        """Promote challenger to champion"""
        if challenger_name in self.challengers:
            self.champion = self.challengers[challenger_name]['model']
            del self.challengers[challenger_name]
            print(f"Promoted {challenger_name} to champion")

# Usage
champion = Model(name="model_v1", version="1.0")
challenger = Model(name="model_v2", version="2.0")

router = ChampionChallengerRouter(champion)
router.add_challenger("v2_candidate", challenger, traffic_pct=0.1)  # 10%

# Route requests
for i in range(10):
    features = {'user_id': i}
    prediction, served_by = router.route(f"req_{i}", features)
    print(f"Request {i}: served by {served_by}")

# Shadow evaluation (no user impact)
shadow = router.shadow_evaluate({'user_id': 123})
print(f"\nShadow comparison: {shadow}")
```

### 6. Key Metrics to Compare
- **Business**: CTR, conversion, revenue per user
- **Model**: Accuracy, latency, coverage
- **System**: Error rate, p99 latency

### 7. Interview Tips
- Always have rollback plan (instant switch back to champion)
- Statistical significance requires sufficient sample size
- Consider segment-based routing (test on specific user cohorts first)

---

## Question 9

**Propose a system design using 'Static Model' and 'Dynamic Model' patterns to handle different needs.**

**Answer:**

### 1. Definitions
**Static Model**: Pre-trained, fixed weights, deployed once, updated infrequently (batch updates).

**Dynamic Model**: Updates continuously with new data, adapts in real-time or near-real-time.

### 2. Comparison

| Aspect | Static Model | Dynamic Model |
|--------|--------------|---------------|
| Update frequency | Weekly/Monthly | Hourly/Real-time |
| Stability | High | Variable |
| Compute cost | Low (inference only) | High (continuous training) |
| Latency | Predictable | May vary |
| Use case | Stable patterns | Rapidly changing data |

### 3. Hybrid System Design Scenario
**News Recommendation Platform**:

**Static Model (Long-term preferences)**:
- User topic preferences learned from months of history
- Category affinity, reading level, device preferences
- Retrained weekly, very stable
- Handles cold-start with demographics

**Dynamic Model (Trending content)**:
- Captures breaking news, viral articles
- Updated every 15 minutes with recent clicks
- High volatility, captures recency

**Fusion Layer**:
- Combine scores: `final_score = 0.6 * static_score + 0.4 * dynamic_score`
- Dynamic weight increases for news/finance, decreases for evergreen content

### 4. Architecture

```
User Request
     |
     v
+--------------------+
|   Feature Service  |
+--------------------+
     |
     +--------+--------+
     |                 |
     v                 v
+-----------+   +-----------+
| Static    |   | Dynamic   |
| Model     |   | Model     |
| (Weekly)  |   | (15 min)  |
+-----------+   +-----------+
     |                 |
     v                 v
+--------------------+
|   Score Fusion     |
+--------------------+
     |
     v
  Final Ranking
```

### 5. Python Code Example

```python
import numpy as np
from datetime import datetime

class StaticModel:
    """Trained weekly on historical data"""
    def __init__(self):
        self.user_preferences = {}  # Loaded from batch training
        self.last_update = datetime.now()
    
    def predict(self, user_id, items):
        # Stable, long-term preferences
        prefs = self.user_preferences.get(user_id, np.zeros(len(items)))
        return np.random.rand(len(items))  # Mock scores

class DynamicModel:
    """Updated continuously with streaming data"""
    def __init__(self):
        self.trending_scores = {}
        self.recent_clicks = []
    
    def update(self, click_event):
        """Called on each new click (streaming update)"""
        self.recent_clicks.append(click_event)
        # Update trending scores based on recent activity
        item_id = click_event['item_id']
        self.trending_scores[item_id] = self.trending_scores.get(item_id, 0) + 1
    
    def predict(self, user_id, items):
        # Return trending/recency scores
        scores = [self.trending_scores.get(item, 0) for item in items]
        return np.array(scores) / (max(scores) + 1)  # Normalize

class HybridRecommender:
    def __init__(self, static_weight=0.6):
        self.static_model = StaticModel()
        self.dynamic_model = DynamicModel()
        self.static_weight = static_weight
    
    def recommend(self, user_id, candidate_items, context=None):
        # Get scores from both models
        static_scores = self.static_model.predict(user_id, candidate_items)
        dynamic_scores = self.dynamic_model.predict(user_id, candidate_items)
        
        # Adjust weights based on context
        weight = self.static_weight
        if context and context.get('content_type') == 'news':
            weight = 0.3  # More weight on dynamic for news
        
        # Fuse scores
        final_scores = weight * static_scores + (1 - weight) * dynamic_scores
        
        # Rank items
        ranked_indices = np.argsort(final_scores)[::-1]
        return [(candidate_items[i], final_scores[i]) for i in ranked_indices[:10]]
    
    def on_click(self, user_id, item_id):
        """Handle streaming click event"""
        self.dynamic_model.update({'user_id': user_id, 'item_id': item_id})

# Usage
recommender = HybridRecommender(static_weight=0.6)

# Simulate clicks (updates dynamic model)
for item in ['article_1', 'article_2', 'article_1', 'article_3']:
    recommender.on_click('user_123', item)

# Get recommendations
items = ['article_1', 'article_2', 'article_3', 'article_4']
recommendations = recommender.recommend('user_123', items, context={'content_type': 'news'})
print("Recommendations:", recommendations)
```

### 6. When to Use Each

| Use Static | Use Dynamic | Use Hybrid |
|------------|-------------|------------|
| Stable domains | Fast-changing data | Best of both |
| Limited compute | Need real-time adaptation | News + preferences |
| Compliance needs | Personalization | E-commerce |

### 7. Interview Tips
- Discuss trade-offs: stability vs freshness
- Mention that hybrid often outperforms either alone
- Dynamic models need careful monitoring for instability

---

## Question 10

**Discuss how you'd use the 'Servant' design pattern to ensure your models are easily reusable.**

**Answer:**

### 1. Definition
The Servant design pattern creates a helper class that provides common functionality to multiple model classes without those models inheriting from a common base. It promotes code reuse and separation of concerns.

### 2. Core Concepts
- **Servant Class**: Provides shared operations (preprocessing, logging, validation)
- **Served Classes**: Models that use servant's functionality
- **Decoupling**: Models don't know about each other, only the servant
- **Single Responsibility**: Each servant handles one cross-cutting concern

### 3. Common Servants in ML

| Servant | Responsibility | Used By |
|---------|---------------|---------|
| Preprocessor | Feature transformation | All models |
| Validator | Input validation | All models |
| Logger | Prediction logging | All models |
| MetricsCollector | Performance tracking | All models |
| Explainer | Model explanations | All models |

### 4. Scenario Application
**Multi-Model Platform**:
- Fraud model, recommendation model, pricing model
- All need: preprocessing, logging, validation, monitoring
- Instead of duplicating code, create servants

### 5. Python Code Example

```python
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

# Servant 1: Preprocessor (serves all models)
class PreprocessorServant:
    def __init__(self, config):
        self.config = config
    
    def serve(self, model, raw_features):
        """Preprocess features for any model"""
        features = raw_features.copy()
        
        # Normalize numerical features
        if 'numerical_cols' in self.config:
            for col in self.config['numerical_cols']:
                if col in features:
                    features[col] = (features[col] - self.config['means'][col]) / self.config['stds'][col]
        
        return features

# Servant 2: Validator (serves all models)
class ValidatorServant:
    def __init__(self, schema):
        self.schema = schema
    
    def serve(self, model, features):
        """Validate input for any model"""
        errors = []
        for field, rules in self.schema.items():
            if rules.get('required') and field not in features:
                errors.append(f"Missing required field: {field}")
            if field in features and 'type' in rules:
                if not isinstance(features[field], rules['type']):
                    errors.append(f"Invalid type for {field}")
        
        if errors:
            raise ValueError(f"Validation failed: {errors}")
        return True

# Servant 3: Logger (serves all models)
class LoggerServant:
    def __init__(self):
        self.logs = []
    
    def serve(self, model, features, prediction):
        """Log prediction for any model"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model.__class__.__name__,
            'features': features,
            'prediction': prediction
        }
        self.logs.append(log_entry)
        print(f"[LOG] {model.__class__.__name__}: {prediction}")

# Base model interface
class BaseModel(ABC):
    def __init__(self, servants=None):
        self.servants = servants or {}
    
    @abstractmethod
    def _predict(self, features):
        pass
    
    def predict(self, raw_features):
        # Use preprocessor servant
        if 'preprocessor' in self.servants:
            features = self.servants['preprocessor'].serve(self, raw_features)
        else:
            features = raw_features
        
        # Use validator servant
        if 'validator' in self.servants:
            self.servants['validator'].serve(self, features)
        
        # Model-specific prediction
        prediction = self._predict(features)
        
        # Use logger servant
        if 'logger' in self.servants:
            self.servants['logger'].serve(self, features, prediction)
        
        return prediction

# Concrete models (served by servants)
class FraudModel(BaseModel):
    def _predict(self, features):
        score = np.random.random()
        return {'is_fraud': score > 0.5, 'score': score}

class RecommendationModel(BaseModel):
    def _predict(self, features):
        items = ['item_1', 'item_2', 'item_3']
        return {'recommendations': items[:features.get('limit', 3)]}

# Setup servants (shared across models)
config = {'numerical_cols': ['amount'], 'means': {'amount': 100}, 'stds': {'amount': 50}}
schema = {'user_id': {'required': True, 'type': (int, str)}}

servants = {
    'preprocessor': PreprocessorServant(config),
    'validator': ValidatorServant(schema),
    'logger': LoggerServant()
}

# Models use same servants
fraud_model = FraudModel(servants=servants)
rec_model = RecommendationModel(servants=servants)

# Both models benefit from shared functionality
print("Fraud prediction:")
fraud_model.predict({'user_id': 123, 'amount': 150})

print("\nRecommendation:")
rec_model.predict({'user_id': 456, 'limit': 5})
```

### 6. Benefits
- **Code Reuse**: Write preprocessing once, use everywhere
- **Consistency**: All models validated/logged the same way
- **Maintainability**: Update servant, all models benefit
- **Testability**: Test servants independently

### 7. Interview Tips
- Distinguish from inheritance (servant is composition)
- Mention related patterns: Strategy, Decorator
- Real-world: sklearn Pipeline, MLflow logging wrappers

---

## Question 11

**How would you leverage 'Transfer Learning' in a case where labeled data is scarce?**

**Answer:**

### 1. Definition
Transfer Learning uses knowledge from a model trained on a large source dataset to improve learning on a target task with limited labeled data. The pre-trained model's learned representations transfer to the new domain.

### 2. Core Concepts
- **Source Domain**: Large dataset where model was pre-trained (ImageNet, Wikipedia)
- **Target Domain**: Your task with limited labels
- **Feature Extraction**: Use pre-trained layers as fixed feature extractors
- **Fine-tuning**: Unfreeze some layers and train on target data
- **Domain Adaptation**: Bridge gap between source and target distributions

### 3. Transfer Learning Strategies

| Strategy | Approach | When to Use |
|----------|----------|-------------|
| Feature Extraction | Freeze all layers, train classifier | Very few labels (<100) |
| Fine-tune Top Layers | Unfreeze last few layers | Moderate labels (100-1000) |
| Full Fine-tuning | Unfreeze all with low LR | More labels (1000+) |
| Domain Adaptation | Align distributions | Domain shift exists |

### 4. Scenario Application
**Medical Image Classification**: 500 labeled X-ray images for rare disease.

**Solution**:
1. Use ResNet pre-trained on ImageNet (millions of images)
2. Remove classification head
3. Add new head for your disease classes
4. Strategy based on data size:
   - 100 images: Feature extraction only
   - 500 images: Fine-tune last 2 blocks
   - 5000 images: Full fine-tuning with low LR

### 5. Python Code Example

```python
import torch
import torch.nn as nn
from torchvision import models

# Scenario: Medical image classification with 500 labeled samples

# Step 1: Load pre-trained model
def create_transfer_model(num_classes, strategy='feature_extraction'):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    if strategy == 'feature_extraction':
        # Freeze ALL layers (fastest, least data needed)
        for param in model.parameters():
            param.requires_grad = False
    
    elif strategy == 'fine_tune_top':
        # Freeze early layers, unfreeze last block
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 (last block)
        for param in model.layer4.parameters():
            param.requires_grad = True
    
    elif strategy == 'full_fine_tune':
        # Unfreeze all but use lower learning rate
        for param in model.parameters():
            param.requires_grad = True
    
    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model

# Step 2: Select strategy based on data size
def select_strategy(num_samples):
    if num_samples < 200:
        return 'feature_extraction'
    elif num_samples < 2000:
        return 'fine_tune_top'
    else:
        return 'full_fine_tune'

# Step 3: Configure optimizer with layer-wise learning rates
def get_optimizer(model, strategy, base_lr=0.001):
    if strategy == 'feature_extraction':
        # Only train classifier head
        params = model.fc.parameters()
        return torch.optim.Adam(params, lr=base_lr)
    
    elif strategy == 'fine_tune_top':
        # Different LR for different layers
        params = [
            {'params': model.layer4.parameters(), 'lr': base_lr * 0.1},
            {'params': model.fc.parameters(), 'lr': base_lr}
        ]
        return torch.optim.Adam(params)
    
    else:
        # Full fine-tune with low LR
        return torch.optim.Adam(model.parameters(), lr=base_lr * 0.01)

# Usage
num_samples = 500
num_classes = 3  # Disease types

strategy = select_strategy(num_samples)
print(f"Strategy for {num_samples} samples: {strategy}")

model = create_transfer_model(num_classes, strategy)
optimizer = get_optimizer(model, strategy)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

# Training loop (standard)
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

**Output:**
```
Strategy for 500 samples: fine_tune_top
Trainable params: 8,399,875 / 25,557,032 (32.9%)
```

### 6. Best Practices
- **Data Augmentation**: Critical with small datasets
- **Learning Rate**: Lower for pre-trained layers than new layers
- **Regularization**: Dropout, weight decay to prevent overfitting
- **Gradual Unfreezing**: Unfreeze progressively during training

### 7. Interview Tips
- Transfer works best when source and target domains are related
- Mention domain-specific pre-trained models (BioBERT, ClinicalBERT)
- Discuss negative transfer when domains are too different

---

## Question 12

**Discuss any recent research that effectively uses the 'Repeatable Process' design pattern.**

**Answer:**

### 1. Definition
The Repeatable Process pattern ensures ML experiments, training, and deployments can be exactly reproduced. This includes versioning data, code, environment, hyperparameters, and random seeds.

### 2. Core Components
- **Data Versioning**: Track exact dataset used (DVC, Delta Lake)
- **Code Versioning**: Git commit hash for training code
- **Environment**: Docker, conda-lock for dependencies
- **Configuration**: YAML/JSON for all hyperparameters
- **Seed Control**: Fixed random seeds across all sources

### 3. Recent Research Examples

**a) MLflow + DVC Pipelines (Industry Standard)**
- Track experiments with full lineage
- Reproduce any past run with single command
- Adopted by Databricks, Microsoft, and major companies

**b) Hugging Face Transformers**
- Training scripts with full reproducibility
- Fixed seeds, deterministic operations
- Model cards with training configuration

**c) Google's ML Metadata (MLMD)**
- Lineage tracking for TFX pipelines
- Used in Vertex AI for production ML
- Research: "Towards ML Engineering" papers

**d) Papers With Code + OpenML**
- Research reproducibility initiatives
- Standardized benchmarks, datasets, code
- Community verification of results

### 4. Implementation Example

```python
import random
import numpy as np
import torch
import hashlib
import json
import os
from datetime import datetime

class RepeatableExperiment:
    """Ensures full reproducibility of ML experiments"""
    
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.experiment_id = self._generate_id()
        self._set_seeds()
        self._log_environment()
    
    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def _generate_id(self):
        """Generate unique experiment ID from config hash"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _set_seeds(self):
        """Set all random seeds for reproducibility"""
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _log_environment(self):
        """Log environment for reproducibility"""
        self.environment = {
            'python_version': os.popen('python --version').read().strip(),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'git_commit': os.popen('git rev-parse HEAD').read().strip(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_experiment(self, model, metrics, output_dir):
        """Save all artifacts for reproducibility"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save environment
        with open(f'{output_dir}/environment.json', 'w') as f:
            json.dump(self.environment, f, indent=2)
        
        # Save metrics
        with open(f'{output_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model
        torch.save(model.state_dict(), f'{output_dir}/model.pt')
        
        print(f"Experiment {self.experiment_id} saved to {output_dir}")
    
    @classmethod
    def reproduce(cls, experiment_dir):
        """Reproduce experiment from saved artifacts"""
        config = json.load(open(f'{experiment_dir}/config.json'))
        env = json.load(open(f'{experiment_dir}/environment.json'))
        
        print(f"Reproducing experiment from {env['timestamp']}")
        print(f"Original git commit: {env['git_commit']}")
        
        # User should checkout that commit and run
        return config

# Example config
config = {
    "seed": 42,
    "model": "resnet18",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "data_version": "v1.2.3"
}

# Save config
with open('experiment_config.json', 'w') as f:
    json.dump(config, f)

# Run repeatable experiment
experiment = RepeatableExperiment('experiment_config.json')
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Environment: {experiment.environment}")
```

### 5. Key Research Contributions
- **MLflow**: "Managing ML Experiments" (Zaharia et al.)
- **DVC**: "Data Version Control" open-source project
- **Weights & Biases**: Experiment tracking at scale
- **Neptune.ai**: ML metadata management

### 6. Interview Tips
- Reproducibility is crucial for debugging and compliance
- Mention specific tools: MLflow, DVC, Weights & Biases
- Discuss challenges: non-deterministic GPU operations, floating-point precision

---

## Question 13

**Discuss the potential impact of AI Ethics and Fairness considerations on ML design patterns.**

**Answer:**

### 1. Definition
AI Ethics and Fairness in ML design patterns addresses bias detection, mitigation, transparency, and accountability throughout the ML lifecycle, ensuring models don't discriminate against protected groups.

### 2. Core Concepts
- **Fairness Metrics**: Demographic parity, equalized odds, calibration
- **Bias Detection**: Identify disparate impact across groups
- **Bias Mitigation**: Pre-processing, in-processing, post-processing techniques
- **Explainability**: Model decisions must be interpretable
- **Accountability**: Audit trails, human oversight

### 3. Impact on Design Patterns

| Pattern | Ethics Consideration |
|---------|---------------------|
| Data Ingestion | Check for representation bias, protected attributes |
| Feature Engineering | Avoid proxies for protected attributes |
| Model Training | Fairness constraints, bias-aware algorithms |
| Evaluation | Disaggregated metrics by demographic groups |
| Serving | Explanation with each prediction |
| Monitoring | Track fairness metrics in production |

### 4. Scenario Application
**Credit Scoring Model**:

**Ethical Concerns**:
- Historical data reflects past discrimination
- Zip code may proxy for race
- Age discrimination in lending

**Design Pattern Modifications**:
1. **Data Pattern**: Audit training data for demographic imbalance
2. **Feature Pattern**: Remove or de-bias proxy features
3. **Training Pattern**: Add fairness constraints to loss function
4. **Evaluation Pattern**: Report metrics per demographic group
5. **Serving Pattern**: Provide rejection reasons (Right to Explanation)
6. **Monitoring Pattern**: Alert on disparate impact

### 5. Python Code Example

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class FairMLPipeline:
    """ML pipeline with fairness considerations built in"""
    
    def __init__(self, protected_attribute):
        self.protected_attr = protected_attribute
        self.model = LogisticRegression()
        self.fairness_metrics = {}
    
    def check_data_bias(self, X, y, sensitive_col):
        """Pre-training: Check for representation bias"""
        positive_rate_by_group = {}
        
        for group in X[sensitive_col].unique():
            mask = X[sensitive_col] == group
            positive_rate = y[mask].mean()
            positive_rate_by_group[group] = positive_rate
            print(f"Group {group}: {positive_rate:.2%} positive rate, n={mask.sum()}")
        
        # Check for disparate impact
        rates = list(positive_rate_by_group.values())
        if max(rates) / min(rates) > 1.25:
            print("WARNING: Potential representation bias detected")
        
        return positive_rate_by_group
    
    def train(self, X, y, drop_sensitive=True):
        """Train with optional removal of sensitive attributes"""
        X_train = X.copy()
        if drop_sensitive:
            X_train = X_train.drop(columns=[self.protected_attr], errors='ignore')
        
        self.model.fit(X_train, y)
        return self
    
    def evaluate_fairness(self, X, y, sensitive_col):
        """Post-training: Evaluate fairness metrics"""
        X_pred = X.drop(columns=[self.protected_attr], errors='ignore')
        predictions = self.model.predict(X_pred)
        
        groups = X[sensitive_col].unique()
        
        for group in groups:
            mask = X[sensitive_col] == group
            
            # Accuracy per group
            acc = accuracy_score(y[mask], predictions[mask])
            
            # Positive prediction rate (demographic parity check)
            pred_positive_rate = predictions[mask].mean()
            
            # True positive rate (equalized odds check)
            tpr = predictions[mask][y[mask] == 1].mean() if (y[mask] == 1).sum() > 0 else 0
            
            self.fairness_metrics[group] = {
                'accuracy': acc,
                'positive_rate': pred_positive_rate,
                'true_positive_rate': tpr
            }
            
            print(f"\nGroup {group}:")
            print(f"  Accuracy: {acc:.2%}")
            print(f"  Positive prediction rate: {pred_positive_rate:.2%}")
            print(f"  True positive rate: {tpr:.2%}")
        
        # Check demographic parity
        rates = [m['positive_rate'] for m in self.fairness_metrics.values()]
        disparity = max(rates) - min(rates)
        print(f"\nDemographic parity gap: {disparity:.2%}")
        
        if disparity > 0.1:
            print("WARNING: Significant demographic disparity detected")
        
        return self.fairness_metrics
    
    def explain_prediction(self, x):
        """Provide explanation for individual prediction"""
        features = x.drop(self.protected_attr, errors='ignore')
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        # Simple feature importance explanation
        coefficients = dict(zip(features.index, self.model.coef_[0]))
        top_factors = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        return {
            'prediction': prediction,
            'confidence': max(probability),
            'top_factors': top_factors,
            'explanation': f"Decision based primarily on: {[f[0] for f in top_factors]}"
        }

# Usage example
import pandas as pd

# Simulated data
np.random.seed(42)
n = 1000
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n),
    'age': np.random.randint(18, 65, n),
    'gender': np.random.choice(['M', 'F'], n),  # Protected attribute
    'approved': np.random.choice([0, 1], n)
})

pipeline = FairMLPipeline(protected_attribute='gender')

# Check data bias
print("=== Data Bias Check ===")
pipeline.check_data_bias(data, data['approved'], 'gender')

# Train (dropping sensitive attribute)
X = data.drop('approved', axis=1)
y = data['approved']
pipeline.train(X, y, drop_sensitive=True)

# Evaluate fairness
print("\n=== Fairness Evaluation ===")
pipeline.evaluate_fairness(X, y, 'gender')

# Explain individual prediction
print("\n=== Individual Explanation ===")
explanation = pipeline.explain_prediction(X.iloc[0])
print(explanation)
```

### 6. Fairness Design Patterns
- **Fairness-aware Training**: Add fairness constraint to loss
- **Post-processing**: Adjust thresholds per group
- **Adversarial Debiasing**: Train model to be unpredictive of protected class
- **Counterfactual Fairness**: Would prediction change if protected attribute changed?

### 7. Interview Tips
- Fairness often involves trade-offs with accuracy
- Different fairness metrics can conflict (impossible to satisfy all)
- Mention regulations: GDPR, CCPA, Fair Lending laws
- Human oversight is essential, not just technical solutions

---

## Question 14

**How would you approach 'Model Serving' in an environment with strict data regulations?**

**Answer:**

### 1. Definition
Model Serving under data regulations requires architectures that comply with privacy laws (GDPR, HIPAA, CCPA) while delivering predictions, including data minimization, encryption, audit trails, and user rights.

### 2. Key Regulations

| Regulation | Key Requirements |
|------------|-----------------|
| GDPR | Right to explanation, data deletion, consent |
| HIPAA | Health data protection, audit trails |
| CCPA | User data access, opt-out rights |
| SOC2 | Security controls, data handling |

### 3. Compliance Design Patterns

| Pattern | Implementation |
|---------|---------------|
| Data Minimization | Only collect/process necessary features |
| Encryption | Encrypt data at rest and in transit |
| Anonymization | Remove PII before model inference |
| Audit Logging | Log all data access and predictions |
| Right to Explanation | Provide prediction reasoning |
| Data Deletion | Delete user data on request (RTBF) |
| Consent Management | Track user consent for data usage |

### 4. Scenario Application
**Healthcare Prediction Service (HIPAA Compliant)**:

**Architecture**:
1. Data never leaves secure enclave
2. Federated learning: model trains on-premise, only gradients shared
3. Differential privacy: add noise to prevent data leakage
4. Encrypted inference: process encrypted data
5. Audit log every prediction with user, time, data accessed

### 5. Python Code Example

```python
import hashlib
import json
from datetime import datetime
from cryptography.fernet import Fernet
import uuid

class ComplianceModelServer:
    """Model serving with regulatory compliance built-in"""
    
    def __init__(self, model, encryption_key=None):
        self.model = model
        self.key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.audit_log = []
        self.user_consent = {}
        self.user_data = {}
    
    # Pattern 1: Consent Management
    def register_consent(self, user_id, purposes):
        """Track user consent for data usage"""
        self.user_consent[user_id] = {
            'purposes': purposes,  # ['inference', 'training', 'analytics']
            'timestamp': datetime.now().isoformat(),
            'version': 'v1.0'
        }
        self._log_audit(user_id, 'CONSENT_REGISTERED', {'purposes': purposes})
    
    def check_consent(self, user_id, purpose):
        """Verify consent before processing"""
        consent = self.user_consent.get(user_id, {})
        return purpose in consent.get('purposes', [])
    
    # Pattern 2: Data Minimization
    def minimize_features(self, features, required_fields):
        """Only use necessary features"""
        return {k: v for k, v in features.items() if k in required_fields}
    
    # Pattern 3: PII Anonymization
    def anonymize(self, features):
        """Remove/hash PII before inference"""
        anonymized = features.copy()
        pii_fields = ['name', 'email', 'ssn', 'address', 'phone']
        
        for field in pii_fields:
            if field in anonymized:
                # Hash instead of remove (preserves some utility)
                anonymized[field] = hashlib.sha256(
                    str(anonymized[field]).encode()
                ).hexdigest()[:16]
        
        return anonymized
    
    # Pattern 4: Encrypted Inference
    def encrypt_request(self, data):
        """Encrypt data for secure transmission"""
        json_data = json.dumps(data).encode()
        return self.cipher.encrypt(json_data)
    
    def decrypt_request(self, encrypted_data):
        """Decrypt data for processing"""
        decrypted = self.cipher.decrypt(encrypted_data)
        return json.loads(decrypted)
    
    # Pattern 5: Audit Logging
    def _log_audit(self, user_id, action, details):
        """Log all data access for compliance"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': str(uuid.uuid4()),
            'user_id': hashlib.sha256(str(user_id).encode()).hexdigest()[:16],
            'action': action,
            'details': details
        }
        self.audit_log.append(entry)
        print(f"[AUDIT] {action}: {entry['request_id']}")
    
    # Pattern 6: Right to Explanation
    def explain_prediction(self, prediction, features):
        """Provide human-readable explanation (GDPR Art. 22)"""
        return {
            'prediction': prediction,
            'explanation': f"Based on {len(features)} factors",
            'top_factors': list(features.keys())[:3],
            'human_readable': "This prediction was made considering your profile data"
        }
    
    # Pattern 7: Right to Be Forgotten
    def delete_user_data(self, user_id):
        """Delete all user data on request (GDPR Art. 17)"""
        # Delete from storage
        if user_id in self.user_data:
            del self.user_data[user_id]
        if user_id in self.user_consent:
            del self.user_consent[user_id]
        
        self._log_audit(user_id, 'DATA_DELETED', {'reason': 'User request'})
        return {'status': 'deleted', 'user_id': user_id}
    
    # Main serving endpoint
    def predict(self, user_id, features, purpose='inference'):
        """Compliant prediction endpoint"""
        
        # Step 1: Check consent
        if not self.check_consent(user_id, purpose):
            return {'error': 'Consent not provided for this purpose'}
        
        # Step 2: Minimize features
        required = ['age', 'income', 'score']  # Only what model needs
        minimized = self.minimize_features(features, required)
        
        # Step 3: Anonymize PII
        anonymized = self.anonymize(minimized)
        
        # Step 4: Log access
        self._log_audit(user_id, 'PREDICTION_REQUEST', {'features_used': list(minimized.keys())})
        
        # Step 5: Make prediction
        prediction = self.model.predict(anonymized)
        
        # Step 6: Provide explanation
        result = self.explain_prediction(prediction, minimized)
        
        return result

# Usage
class MockModel:
    def predict(self, features):
        return 'approved'

server = ComplianceModelServer(MockModel())

# Register consent
server.register_consent('user_123', ['inference', 'analytics'])

# Make prediction
features = {
    'name': 'John Doe',  # PII - will be anonymized
    'age': 30,
    'income': 75000,
    'score': 720,
    'email': 'john@example.com'  # PII
}

result = server.predict('user_123', features)
print(f"\nPrediction result: {result}")

# Right to be forgotten
deletion = server.delete_user_data('user_123')
print(f"\nDeletion result: {deletion}")

# View audit log
print(f"\nAudit log entries: {len(server.audit_log)}")
```

### 6. Key Architectural Decisions
- **On-premise deployment**: Data never leaves customer's infrastructure
- **Federated learning**: Train without centralizing data
- **Differential privacy**: Mathematical privacy guarantees
- **Secure enclaves**: Hardware-level data protection (Intel SGX)

### 7. Interview Tips
- Always mention specific regulations by name
- Discuss trade-offs: privacy vs model performance
- Mention data residency (data must stay in certain geography)
- Legal team involvement is essential for compliance

---

## Question 15

**Discuss 'Dynamic Training' approaches in a scenario where data distributions change rapidly.**

**Answer:**

### 1. Definition
Dynamic Training continuously updates models as new data arrives, adapting to rapid distribution changes (concept drift, trending topics, market shifts) rather than relying on periodic batch retraining.

### 2. Core Concepts
- **Online Learning**: Update model with each new sample
- **Mini-batch Updates**: Accumulate small batches, update frequently
- **Sliding Window**: Train only on recent data (forget old patterns)
- **Exponential Decay**: Weight recent data more heavily
- **Trigger-based Retraining**: Retrain when drift detected

### 3. Dynamic Training Strategies

| Strategy | Update Frequency | Use Case |
|----------|-----------------|----------|
| Full Online | Every sample | High-frequency trading |
| Mini-batch | Every N samples | Real-time recommendations |
| Scheduled | Hourly/Daily | News, trending content |
| Triggered | On drift detection | Stable with occasional shifts |

### 4. Scenario Application
**Trending Topic Detection for Social Media**:

**Challenge**: Topics trend and die within hours. Weekly retrained model is always outdated.

**Dynamic Training Solution**:
1. **Streaming Data**: Consume tweets in real-time via Kafka
2. **Mini-batch Updates**: Accumulate 1000 tweets, update model
3. **Sliding Window**: Keep only last 24 hours of data
4. **Ensemble**: Blend hourly model with daily model
5. **Drift Detection**: Alert if topic distribution shifts dramatically

### 5. Python Code Example

```python
import numpy as np
from collections import deque
from sklearn.linear_model import SGDClassifier
from datetime import datetime, timedelta

class DynamicTrainer:
    """Handles rapidly changing data distributions"""
    
    def __init__(self, base_model, window_hours=24, batch_size=100):
        self.model = base_model
        self.window_hours = window_hours
        self.batch_size = batch_size
        self.data_buffer = deque()
        self.drift_detector = DriftDetector()
        self.update_count = 0
    
    def add_sample(self, features, label, timestamp=None):
        """Add new sample to buffer"""
        timestamp = timestamp or datetime.now()
        self.data_buffer.append({
            'features': features,
            'label': label,
            'timestamp': timestamp
        })
        
        # Remove old samples (sliding window)
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        while self.data_buffer and self.data_buffer[0]['timestamp'] < cutoff:
            self.data_buffer.popleft()
        
        # Check if batch update needed
        if len(self.data_buffer) >= self.batch_size:
            self._update_model()
    
    def _update_model(self):
        """Perform mini-batch update"""
        recent = list(self.data_buffer)[-self.batch_size:]
        
        X = np.array([s['features'] for s in recent])
        y = np.array([s['label'] for s in recent])
        
        # Online update (partial_fit)
        self.model.partial_fit(X, y, classes=np.unique(y))
        self.update_count += 1
        
        # Check for drift
        if self.drift_detector.detect(y):
            print(f"[ALERT] Drift detected at update {self.update_count}")
        
        print(f"[UPDATE] Model updated with {len(recent)} samples. Total updates: {self.update_count}")
    
    def predict(self, features):
        """Make prediction with current model"""
        return self.model.predict([features])[0]

class DriftDetector:
    """Simple drift detection based on label distribution"""
    
    def __init__(self, window_size=100, threshold=0.2):
        self.recent_labels = deque(maxlen=window_size)
        self.baseline_dist = None
        self.threshold = threshold
    
    def detect(self, new_labels):
        """Check if distribution has shifted"""
        self.recent_labels.extend(new_labels)
        
        if len(self.recent_labels) < 50:
            return False
        
        current_dist = np.bincount(list(self.recent_labels), minlength=2) / len(self.recent_labels)
        
        if self.baseline_dist is None:
            self.baseline_dist = current_dist
            return False
        
        # Check distribution shift
        shift = np.abs(current_dist - self.baseline_dist).max()
        
        if shift > self.threshold:
            self.baseline_dist = current_dist  # Update baseline
            return True
        
        return False

class ExponentialDecayTrainer:
    """Weight recent samples more heavily"""
    
    def __init__(self, decay_rate=0.995):
        self.decay_rate = decay_rate
        self.samples = []
    
    def add_sample(self, features, label, timestamp=None):
        self.samples.append({
            'features': features,
            'label': label,
            'weight': 1.0
        })
        
        # Decay old sample weights
        for sample in self.samples:
            sample['weight'] *= self.decay_rate
        
        # Remove samples with negligible weight
        self.samples = [s for s in self.samples if s['weight'] > 0.01]
    
    def get_weighted_data(self):
        """Return features, labels, and sample weights"""
        X = np.array([s['features'] for s in self.samples])
        y = np.array([s['label'] for s in self.samples])
        weights = np.array([s['weight'] for s in self.samples])
        return X, y, weights

# Usage Example: Trending topic classifier
model = SGDClassifier(loss='log_loss', warm_start=True)
trainer = DynamicTrainer(model, window_hours=6, batch_size=50)

# Simulate streaming data
np.random.seed(42)
for i in range(200):
    # Features: word counts, sentiment, etc.
    features = np.random.rand(10)
    
    # Label distribution shifts over time
    if i < 100:
        label = np.random.choice([0, 1], p=[0.7, 0.3])
    else:
        label = np.random.choice([0, 1], p=[0.3, 0.7])  # Distribution shift
    
    trainer.add_sample(features, label)

print(f"\nFinal buffer size: {len(trainer.data_buffer)}")
print(f"Total model updates: {trainer.update_count}")
```

**Output:**
```
[UPDATE] Model updated with 50 samples. Total updates: 1
[UPDATE] Model updated with 50 samples. Total updates: 2
[ALERT] Drift detected at update 3
[UPDATE] Model updated with 50 samples. Total updates: 3
[UPDATE] Model updated with 50 samples. Total updates: 4
```

### 6. Challenges and Solutions
- **Catastrophic Forgetting**: Use replay buffer or regularization
- **Noisy Updates**: Larger batch sizes for stability
- **Concept Recovery**: Quick adaptation when old patterns return
- **Model Stability**: Ensemble stable + dynamic models

### 7. Interview Tips
- Dynamic training trades stability for freshness
- Mention specific algorithms: SGD, online random forest, streaming k-means
- Discuss validation: hard to evaluate without held-out future data
- Production: often combine with scheduled full retraining

---

