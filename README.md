# Generative Agents

A modular implementation of the Generative Agents architecture introduced in the paper *Generative Agents: Interactive Simulacra of Human Behavior* by Joon Sung Park et al. 

## Overview

Generative Agents are autonomous agents capable of maintaining long-term memories, reflecting on experiences, planning future actions, and producing contextually grounded behavior over time.

Rather than responding only to the immediate prompt, agents maintain an evolving internal state composed of:

- Experiences and observations
- Retrieved memories
- Reflections and insights
- Plans and goals

This repository aims to provide a clean, extensible, and production-oriented implementation of those concepts.

The architecture is built around three core capabilities:

1. **Memory**
   - Storing observations and experiences
   - Maintaining an evolving memory stream
   - Preserving historical context

2. **Retrieval**
   - Selecting relevant memories when reasoning
   - Balancing relevance, recency, and importance
   - Supplying context for downstream decisions

3. **Reasoning**
   - Generating reflections from past experiences
   - Creating and revising plans
   - Producing coherent actions over time

These mechanisms closely follow the memory, retrieval, reflection, and planning framework described in the original Generative Agents paper. 

---

## Architecture

The high-level agent lifecycle follows the pattern shown in the original paper:

```text
Perceive
   ↓
Memory Stream
   ↓
Memory Retrieval
   ↓
Reasoning / Reflection
   ↓
Planning
   ↓
Action
   ↓
Memory Update```