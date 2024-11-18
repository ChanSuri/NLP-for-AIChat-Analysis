# Project Overview

This project aims to enhance user interaction with an AI chat assistant by utilizing Natural Language Processing (NLP) techniques. The workflow includes finding the best data sources, processing data, and categorizing user inquiries.

## Workflow

1. **Data Source Identification**
   - Identify and select optimal data sources for processing.

2. **Data Processing**
   - Transform data from sources into a structured schema.

3. **User Intent Analysis**
   - Use **BERT for Sequence Classification** to categorize user intents:
     - Generate code for topics.
     - Explain code.
     - Knowledge transfer (e.g., data transfer/sensors).
     - General inquiries.
     - Collect feedback.
   - This model processes user inputs in real-time, routing different intents to appropriate response modules, enhancing response accuracy.

4. **Feedback Loop**
   - Regularly categorize and analyze user feedback to identify system weaknesses and areas for improvement, helping engineers optimize model performance and interaction workflows.

5. **Keyword Extraction and Entity Recognition**
   - Utilize NLP technologies like **SpaCy** or **NLTK** for:
     - Extracting keywords and entities from user requests (e.g., "read temperature sensor").
     - Guiding the system to relevant predefined templates or code snippets.
   - For deep semantic understanding, prefer **BERT**; for large-scale text processing, use **SpaCy**; for educational purposes, choose **NLTK**.
   - Implement Named Entity Recognition (NER) using SpaCy or NLTK to identify specific entities (e.g., device names, operation requests).

6. **Topic Clustering and User Segmentation**
   - Employ **KMeans** or **HDBSCAN** for unsupervised clustering of user requests, segmenting users based on needs (e.g., "entry-level" vs. "senior developers"). This helps in identifying main user requirements and optimizing module functionalities accordingly.

7. **Real-time Feedback and Learning Mechanism**
   - Implement continuous learning to refine the model based on user feedback. Record and address unsatisfactory responses to enhance future performance.

8. **Engineer Optimization Suggestions**
   - Present analytical results to engineers through reports or dashboards, enabling informed decision-making for:
     - Adding specific code generation templates.
     - Optimizing knowledge base content.
     - Adjusting response mechanisms to improve user experience continuously.


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

> **Data Source**: [Google BigQuery Public Data](https://cloud.google.com/bigquery/public-data) Public dataset for testing

> **Data Source**: [Arduino Data] Not collect from real data in GenAI Assistant.
