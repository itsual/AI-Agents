# 🧠 One Stop Repo For Anything AI (AI ToolKit For Developers)

This repository hosts a handon gen AI and Agentic Ai hands-on application codes, list of python librraies to choose, learning contents, 

Keywords: AI, Gen AI, LLM, Agentic AI, AI Agents

**Under construction**


## 📢 Contribution

Pull requests and feature suggestions are welcome! 🙌  

---

## **Repo Structure**
```mermaid
flowchart TD
    %% LLAMA-Vision Module
    subgraph "LLAMA-Vision Module"
        lvDemo["LLAMA_Vision_Demo.ipynb"]:::frontend
        lvReadme["README.md"]:::frontend
    end

    %% Local-Multi-Agent-RAG Module
    subgraph "Local-Multi-Agent-RAG Module"
        mmBackend["app.py"]:::backend
        subgraph "Agents"
            mmPlanner["Planner Agent"]:::agent
            mmResponder["Responder Agent"]:::agent
            mmRetrieval["Retrieval Agent"]:::agent
        end
        subgraph "Configuration"
            mmSettings["settings.yaml"]:::config
            mmIndex["index.json"]:::config
        end
        subgraph "Utilities"
            mmChunking["chunking.py"]:::utility
            mmEmbedding["embedding.py"]:::utility
            mmPDF["pdf_utils.py"]:::utility
            mmLocalEmb["local_embeddings_llm.py"]:::utility
        end
    end

    %% Local-Single-Agent-RAG Module
    subgraph "Local-Single-Agent-RAG Module"
        msBackend["app.py"]:::backend
        subgraph "Single-Agent"
            msSingle["index.py"]:::agent
            msInit["__init__.py"]:::agent
        end
        msVisual["graph.html"]:::frontend
        subgraph "Configuration"
            msSettings["settings.yaml"]:::config
            msIndex["index.json"]:::config
        end
        subgraph "Utilities"
            msChunking["chunking.py"]:::utility
            msEmbedding["embedding.py"]:::utility
            msPDF["pdf_utils.py"]:::utility
            msLocalLLM["local_llm.py"]:::utility
        end
    end

    %% Shared Python Libraries
    subgraph "Shared Python Libraries"
        pyLibs["Python Libraries"]:::library
        pyLibReadme["README.md"]:::library
    end

    %% Data Flow Connections for Local-Multi-Agent-RAG
    mmBackend -->|"calls"| mmPlanner
    mmBackend -->|"calls"| mmResponder
    mmBackend -->|"calls"| mmRetrieval
    mmPlanner -->|"uses"| mmSettings
    mmResponder -->|"uses"| mmIndex
    mmRetrieval -->|"uses"| mmPDF
    mmPlanner ---|"utilizes"| mmChunking
    mmResponder ---|"utilizes"| mmEmbedding
    mmRetrieval ---|"utilizes"| mmLocalEmb

    %% Data Flow Connections for Local-Single-Agent-RAG
    msBackend -->|"calls"| msSingle
    msSingle -->|"initializes"| msInit
    msSingle -->|"provides data to"| msVisual
    msSingle ---|"reads"| msSettings
    msSingle ---|"reads"| msIndex
    msSingle ---|"utilizes"| msChunking
    msSingle ---|"utilizes"| msEmbedding
    msSingle ---|"utilizes"| msPDF
    msSingle ---|"utilizes"| msLocalLLM

    %% Connections to Shared Python Libraries
    mmBackend ---|"extends"| pyLibs
    msBackend ---|"extends"| pyLibs
    lvDemo ---|"extends"| pyLibs

    %% Class Styles
    classDef frontend fill:#AED6F1,stroke:#1B4F72,stroke-width:2px;
    classDef backend fill:#A9DFBF,stroke:#145A32,stroke-width:2px;
    classDef agent fill:#FAD7A0,stroke:#B9770E,stroke-width:2px;
    classDef config fill:#D2B4DE,stroke:#6C3483,stroke-width:2px;
    classDef utility fill:#F5CBA7,stroke:#873600,stroke-width:2px;
    classDef library fill:#FDEBD0,stroke:#D68910,stroke-width:2px;

    %% Click Events for LLAMA-Vision Module
    click lvDemo "https://github.com/itsual/ai-agents/blob/main/LLAMA-Vision/LLAMA_Vision_Demo.ipynb"
    click lvReadme "https://github.com/itsual/ai-agents/blob/main/LLAMA-Vision/README.md"

    %% Click Events for Local-Multi-Agent-RAG Module
    click mmBackend "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/app.py"
    click mmPlanner "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/autogen_agents/planner_agent.py"
    click mmResponder "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/autogen_agents/responder_agent.py"
    click mmRetrieval "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/autogen_agents/retrieval_agent.py"
    click mmSettings "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/settings.yaml"
    click mmIndex "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/index.json"
    click mmChunking "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/utils/chunking.py"
    click mmEmbedding "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/utils/embedding.py"
    click mmPDF "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/utils/pdf_utils.py"
    click mmLocalEmb "https://github.com/itsual/ai-agents/blob/main/Local-Multi-Agent-RAG/utils/local_embeddings_llm.py"

    %% Click Events for Local-Single-Agent-RAG Module
    click msBackend "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/app.py"
    click msVisual "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/graph.html"
    click msSingle "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/graphrag/index.py"
    click msInit "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/graphrag/__init__.py"
    click msSettings "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/settings.yaml"
    click msIndex "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/index.json"
    click msChunking "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/utils/chunking.py"
    click msEmbedding "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/utils/embedding.py"
    click msPDF "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/utils/pdf_utils.py"
    click msLocalLLM "https://github.com/itsual/ai-agents/blob/main/Local-Single-Agent-RAG/utils/local_llm.py"

    %% Click Events for Shared Python Libraries
    click pyLibs "https://github.com/itsual/ai-agents/tree/main/Python Libraries"
    click pyLibReadme "https://github.com/itsual/ai-agents/blob/main/Python Libraries/README.md"
  ```
---
