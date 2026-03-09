# README

This is a Domain-Specific Chatbot project for SDSU's AI Club completed in a group of three over the Fall 2024 semester. 
The team consisted of:
- Carlos Lopez, Group Leader
- Vaidehi Pawar
- Bibianca Baarh

The chatbot was created by using Retrieval Augmented Generation (RAG) techniques powered by the LangChain framework. It works
by extracting parsed text from an uploaded pdf document and splitting the text into chunks. Then, using an embedding model,
those text chunks are vectorized, or mapped to numerical vectors in 3D space, and stored in a vector database via Chroma.
A large language model (LLM) is then prompted to answer questions using only the information in the vector database as the 
available context via a RAG pipeline or chain. 

The purpose of our chatbot was to use a Medical Encyclopedia document to empower our chatbot answer patient questions on 
their diagnosis, or have our chatbot give a possible diagnosis based on a given set of symptoms, or to simply have a 
nursing student use the chatbot to study for their medical exam.

<p align="center">
<img width="950" src="https://github.com/user-attachments/assets/da00b390-2a37-45b3-8e18-c8be5339220f" />
</p>

This repository includes two implementations: **Using_Huggingface_model_lama_2_7B_Chat_GGML.ipynb** by Vaidehi Pawar,
and **AIClub_RAG_Ollama_LangChain_Chroma.ipynb** by Carlos Lopez. This repository also includes the semester-long
project schedule that includes the weekly goals and notes for the project. In addition, the final presentation slides are included.

Contact me here for any questions: clopez2109@sdsu.edu
