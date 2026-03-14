# Contributors:
* Felipe M. Panugan III - NeoRedcraft
* Annjela Kristy Nicolas - kris-tyv
* Ron Denzel Manaog - RonDenzel
# Design and Implementation of a Retrieval-Augmented Qwen3-0.6B Chatbot for Student Mental Health Support 
A domain-specific mental health chatbot using a Retrieval-Augmented Generation (RAG) framework powered by the Qwen3-0.6B language model. The system combines document retrieval and generative language modeling to produce context-aware and grounded responses to mental health-related queries. The knowledge base is derived from the document Promoting Student Mental Health: A Guide for UC Faculty and Staff, while the Mental Health Counseling Conversations dataset from Kaggle is used for data analysis and evaluation. The document is processed through text segmentation, embedding generation using the all-MiniLM-L6-v2 model, and indexing within a ChromaDB vector database. During interaction, user queries are converted into embeddings to retrieve relevant document segments, which are then used as context for response generation. Evaluation using the RAGAS framework demonstrates strong performance in answer relevancy and faithfulness, highlighting the effectiveness of RAG-based systems for mental health information support.
## 📊 Dataset:
- [UCOP - Promoting Student Mental Health](https://www.ucop.edu/student-mental-health-resources/_files/pdf/PSMH-guide.pdf)
- [Kaggle - Mental Health Counseling Conversations](https://www.kaggle.com/datasets/melissamonfared/mental-health-counseling-conversations-k)
## 📄 File Description:
## 🌐 Web Deployment

🔗 **Live Demo:**  
