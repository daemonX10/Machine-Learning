# 🗺️ DSA Roadmap — Zero to Advanced

> **23 Topics · 1,172 Questions · 353 Coding Challenges**
> Study these in order. Each stage builds on the previous one.
> Estimated timeline: 3–4 months (1–2 topics per week)

---

## How to Use This Roadmap

- **Stages** go from foundational → intermediate → advanced
- **Within each stage**, topics are grouped — study them together as they reinforce each other
- ✅ Check off topics as you complete them
- 🔢 Numbers in parentheses = total questions available for that topic
- 💡 "Why now" explains the dependency and reasoning

---

## Stage 1 — Foundations (Start Here)

> 🎯 **Goal:** Understand what data structures are, how to measure efficiency, and master the two most fundamental structures.
> ⏱️ **Time:** Week 1–2

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 1 | [Data Structures](Data%20Structures.md) | 100 | Overview of ALL data structures — gives you the big picture before diving deep |
| 2 | [Big-O Notation](Big-O%20Notation.md) | 30 | You MUST understand time/space complexity before studying any algorithm |
| 3 | [Arrays](Arrays.md) | 60 | The most basic data structure — everything else builds on arrays |

**📌 Checkpoint:** You should be able to explain what arrays, linked lists, stacks, queues, trees, and graphs are conceptually, and analyze the time complexity of simple operations.

---

## Stage 2 — Linear Data Structures

> 🎯 **Goal:** Master all linear (sequential) data structures and their operations.
> ⏱️ **Time:** Week 3–5

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 4 | [Strings](Strings.md) | 50 | Strings are essentially character arrays — natural next step |
| 5 | [Linked Lists](Linked%20Lists.md) | 55 | Second fundamental structure; pointer-based thinking needed for trees/graphs later |
| 6 | [Stacks](Stacks.md) | 46 | Built on arrays/linked lists; needed for recursion, backtracking, expression parsing |
| 7 | [Queues](Queues.md) | 55 | Partner of stacks; needed for BFS, level-order traversal later |
| 8 | [Hash Tables](Hash%20Tables.md) | 39 | The most-used structure in interviews; key-value lookups appear everywhere |

**📌 Checkpoint:** You can implement each structure from scratch, know when to use which, and solve problems like "valid parentheses" (stack), "LRU cache" (hash + linked list).

---

## Stage 3 — Recursion & Basic Algorithms

> 🎯 **Goal:** Master recursive thinking and fundamental algorithmic techniques.
> ⏱️ **Time:** Week 6–8

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 9 | [Recursion](Recursion.md) | 53 | Foundation for EVERYTHING ahead — trees, graphs, DP, backtracking all need this |
| 10 | [Fibonacci Sequence](Fibonacci%20Sequence.md) | 35 | Classic recursion example; introduces memoization (bridge to DP) |
| 11 | [Searching Algorithms](Searching%20Algorithms.md) | 59 | Binary search is king — used in arrays, trees, and optimization problems |
| 12 | [Sorting Algorithms](Sorting%20Algorithms.md) | 60 | Merge sort, quicksort teach divide & conquer; sorting is a prerequisite for many problems |

**📌 Checkpoint:** You can write recursive solutions, convert them to iterative, explain merge sort & binary search, and identify when a problem needs recursion.

---

## Stage 4 — Trees & Hierarchical Structures

> 🎯 **Goal:** Master tree-based data structures — the most heavily tested topic in interviews.
> ⏱️ **Time:** Week 9–11

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 13 | [Tree Data Structure](Tree%20Data%20Structure.md) | 100 | General trees — traversals (BFS/DFS), properties, types |
| 14 | [Binary Tree](Binary%20Tree.md) | 53 | The most common tree variant; BST operations, balancing, rotations |
| 15 | [Heaps And Maps](Heaps%20And%20Maps.md) | 44 | Priority queues (heaps) built on arrays; needed for greedy, graph algorithms |
| 16 | [Trie Data Structure](Trie%20Data%20Structure.md) | 28 | Specialized tree for strings; autocomplete, spell check, prefix matching |

**📌 Checkpoint:** You can traverse any tree (pre/in/post/level-order), balance a BST, implement a heap, and solve "lowest common ancestor" type problems.

---

## Stage 5 — Graphs

> 🎯 **Goal:** Conquer graph problems — requires solid foundation from all previous stages.
> ⏱️ **Time:** Week 12–13

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 17 | [Graph Theory](Graph%20Theory.md) | 50 | Requires queues (BFS), stacks/recursion (DFS), heaps (Dijkstra), hash tables (adjacency lists) |

**📌 Checkpoint:** You can implement BFS, DFS, detect cycles, find shortest paths (Dijkstra, Bellman-Ford), topological sort, and identify connected components.

---

## Stage 6 — Algorithm Design Paradigms

> 🎯 **Goal:** Master the core problem-solving strategies that appear in medium/hard interview questions.
> ⏱️ **Time:** Week 14–17

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 18 | [Divide & Conquer](Divide%20%26%20Conquer.md) | 54 | Formalizes what you learned in merge sort/binary search; master theorem |
| 19 | [Greedy Algorithms](Greedy%20Algorithms.md) | 41 | Simple paradigm — make locally optimal choices; needs heaps, sorting |
| 20 | [Backtracking](Backtracking.md) | 35 | Systematic brute force with pruning; builds on recursion heavily |
| 21 | [Dynamic Programming](Dynamic%20Programming.md) | 35 | The hardest topic — save this for when recursion is second nature; memoization → tabulation |

**📌 Checkpoint:** Given a problem, you can identify whether it needs greedy, D&C, backtracking, or DP, and implement the solution.

---

## Stage 7 — Advanced & Specialized Topics

> 🎯 **Goal:** Round out your knowledge with niche but important topics.
> ⏱️ **Time:** Week 18–20

| # | Topic | Questions | Why Now |
|---|-------|-----------|---------|
| 22 | [Bit Manipulation](Bit%20Manipulation.md) | 40 | Low-level optimization; used in competitive programming and system-level interviews |
| 23 | [Blockchain](Blockchain.md) | 50 | Specialized application of data structures (hash chains, Merkle trees) — study only if relevant to target role |

**📌 Checkpoint:** You can solve XOR tricks, bit counting problems, and explain how blockchain data structures work.

---

## 📊 Summary — Study Order at a Glance

```
STAGE 1 (Foundation)     → Data Structures → Big-O → Arrays
     ↓
STAGE 2 (Linear)         → Strings → Linked Lists → Stacks → Queues → Hash Tables
     ↓
STAGE 3 (Recursion/Algo) → Recursion → Fibonacci → Searching → Sorting
     ↓
STAGE 4 (Trees)          → Trees → Binary Trees → Heaps & Maps → Tries
     ↓
STAGE 5 (Graphs)         → Graph Theory
     ↓
STAGE 6 (Paradigms)      → Divide & Conquer → Greedy → Backtracking → Dynamic Programming
     ↓
STAGE 7 (Advanced)       → Bit Manipulation → Blockchain
```

---

## 🎯 Grouping by Interview Priority

### 🔴 Must-Know (Top 10 — covers 80% of interviews)
1. Arrays
2. Strings
3. Hash Tables
4. Linked Lists
5. Stacks & Queues
6. Binary Tree
7. Graph Theory
8. Sorting Algorithms
9. Dynamic Programming
10. Recursion

### 🟡 Important (Next 5 — frequently asked)
11. Searching Algorithms
12. Heaps And Maps
13. Backtracking
14. Big-O Notation
15. Divide & Conquer

### 🟢 Good to Know (Next 5 — occasionally asked)
16. Greedy Algorithms
17. Trie Data Structure
18. Bit Manipulation
19. Tree Data Structure (general)
20. Data Structures (general)

### ⚪ Niche (Last 3 — role-specific)
21. Fibonacci Sequence
22. Blockchain
23. (Review weak areas from above)

---

## 🔄 Prerequisite Map

```
Big-O Notation ──────────────────────────► (needed everywhere)
        │
Arrays ──┬── Strings
         ├── Hash Tables
         ├── Stacks ──────┬── Backtracking
         ├── Queues ──────┤
         │                └── Graph Theory (BFS)
         └── Sorting ─────┬── Divide & Conquer
                          └── Greedy Algorithms
                          
Linked Lists ──── Stacks/Queues (alt implementations)

Recursion ──┬── Fibonacci ──── Dynamic Programming
            ├── Tree Traversals
            ├── Graph DFS
            ├── Backtracking
            └── Divide & Conquer

Trees ──┬── Binary Tree ──── Heaps
        └── Trie

Graph Theory ──── (requires: BFS, DFS, Heaps, Hash Tables)

Dynamic Programming ──── (requires: Recursion, Memoization)
```
