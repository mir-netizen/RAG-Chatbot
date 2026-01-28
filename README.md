A Retrieval Augmented Generation Chatbot which allows the user to upload their PDF File , and then the RAG pipeline does text extraction followed by text normalization to tokenization i.e 
converts to token ID's. 

The chunking size is 300 allowed with an overlap of 50 , followed by chunking we implement an OPENAI's embedding model which will convert the generated chunks to vectors and 
store the vectors in the vectorDB.

The vectorDB used here is ChromaDB which is a free , open-source DB which can be used for storage of vectors and the LLM used here OPENAI's gpt-4o-mini.

After retrieval of the chunks , the RAG Pipeline will augment a prompt towards the LLM and the LLM seeing the prompt given will generate an answer towards the user.
