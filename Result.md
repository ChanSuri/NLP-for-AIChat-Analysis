# genAI-assistant-analysis

## Different Models Comparison


| Model          | Number of Clusters | Clustering Keywords | Silhouette Score (SC) | num in clusters |
|----------------|--------------------|------------------------|-----------------------|-----------------------|
| **SentenceBERT** | 5                  | Cluster 0: ['sound', 'test', 'help', 'body'] <br> Cluster 1: ['time', 'led', 'game', 'make', 'digital', 'right', 'remote', 'circuit', 'monitor', 'board'] <br> Cluster 2: ['project', 'arduino', 'display', 'programming', 'signal', 'led', 'online', 'simple'] <br> Cluster 3: ['light', 'using', 'arduino', 'control', 'project', 'robot', 'led', 'arduino uno', 'blink'] <br> Cluster 4: ['sensor', 'use', 'code', 'know', 'arduino', 'lcd', 'want', 'data', 'alarm'] | 0.4505 | 4: 257, 0: 237, 1: 222, 3: 148, 2: 136|
| **RoBERTa**        | 3                  | Cluster 0: ['help', 'code', 'sound', 'test', 'arduino', 'work', 'program', 'best'] <br> Cluster 1: ['led', 'light', 'make', 'using', 'robot', 'time', 'board', 'temperature', 'turn', 'sensor'] <br> Cluster 2: ['arduino', 'project', 'control', 'display', 'motor', 'arduino uno', 'want'] | 0.7282 | 2: 694, 1: 163, 0: 143 |
| **DistilBERT**     | 2                  | Cluster 0: ['arduino', 'help', 'project', 'code', 'sensor', 'test', 'sound', 'work'] <br> Cluster 1: ['led', 'project', 'light', 'make', 'using', 'control', 'simple', 'robot', 'time', 'use'] | 0.6881 | 1: 854, 0: 146 |
| **ALBERT**         | 2                  | Cluster 0: ['arduino', 'help', 'project', 'code', 'sensor', 'test', 'sound', 'work'] <br> Cluster 1: ['led', 'project', 'light', 'make', 'using', 'control', 'simple', 'robot', 'time', 'use'] | 0.5191 | 1: 674, 0: 326 |
| **T5**             | 2                  | Cluster 0: ['arduino', 'help', 'project', 'code', 'sensor', 'test', 'sound', 'work'] <br> Cluster 1: ['led', 'project', 'light', 'make', 'using', 'control', 'simple', 'robot', 'time', 'use'] | 0.4813 | 1: 504, 0: 496 |
| **DeBERTa**        | N/A                | No clustering results available | - | - |
| **BART**           | 3                  | Cluster 0: ['help', 'code', 'sound', 'test', 'arduino', 'work', 'program', 'best'] <br> Cluster 1: ['led', 'light', 'make', 'using', 'robot', 'time', 'board', 'temperature', 'turn', 'sensor'] <br> Cluster 2: ['arduino', 'project', 'control', 'display', 'motor', 'arduino uno', 'want'] | 0.5100 | 1: 523, 2: 314, 0: 163 |
| **ELECTRA**        | 4                  | Cluster 0: ['help', 'sound', 'work', 'best', 'smart'] <br> Cluster 1: ['led', 'time', 'game', 'light', 'turn', 'temperature', 'button', 'service', 'monitor'] <br> Cluster 2: ['project', 'arduino', 'display', 'know', 'programming', 'clock', 'data', 'projeto'] <br> Cluster 3: ['arduino', 'using', 'control', 'project', 'light', 'use', 'sensor', 'simple', 'led', 'robot'] | 0.5100 | 3: 306, 1: 276, 2: 230, 0: 188 | 

## Models performance:
RoBERTa > DistilBERT > ALBERT > BART > ELECTRA > T5 > SentenceBERT 