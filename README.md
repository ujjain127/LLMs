# LLMs, GenAI, and RAGs - Basic Notes

## Large Language Models (LLMs)

Large Language Models are AI systems trained on vast amounts of text data to understand and generate human-like text. They use neural network architectures, primarily transformers, to process and generate language.

### Key Characteristics of LLMs:
- **Scale**: Trained on billions to trillions of parameters
- **Self-supervised learning**: Learn patterns from unlabeled text data
- **Zero/few-shot capabilities**: Can perform tasks with minimal examples
- **Contextual understanding**: Comprehend meaning based on surrounding text
- **Transfer learning**: Apply knowledge across different domains

### Popular LLM Examples:
- GPT series (OpenAI)
- LLaMA (Meta)
- Claude (Anthropic)
- Gemini (Google)
- Mistral (Mistral AI)

### LLM Architectures:
- **Transformer**: Core architecture using attention mechanisms to process sequences
  - **Encoder-only** (BERT): Good for understanding text, classification tasks
  - **Decoder-only** (GPT): Specialized in text generation
  - **Encoder-decoder** (T5, BART): Effective for translation, summarization
- **Mixture of Experts (MoE)**: Routes inputs to specialized sub-networks
- **Recurrent Neural Networks (RNNs)**: Process sequences through recurrent connections
- **Sparse Attention**: Optimizes computation by focusing on important tokens

## Generative AI (GenAI)

Generative AI refers to artificial intelligence systems that can create new content, including text, images, audio, code, and more.

### Key Aspects of GenAI:
- **Content creation**: Generate human-like content across modalities
- **Creative applications**: Art, music, writing, design, code generation
- **Multimodal capabilities**: Work across text, images, audio, video
- **Foundation models**: Pre-trained on broad data, adaptable to many tasks
- **Prompt engineering**: Craft inputs to guide desired outputs

### GenAI Applications:
- Content creation and summarization
- Code generation and completion
- Image and video generation
- Virtual assistants and chatbots
- Language translation and transcription

### GenAI Architectures:
- **Diffusion Models**: Generate images through iterative denoising (DALL-E, Stable Diffusion)
- **Variational Autoencoders (VAEs)**: Encode data into latent space and decode back
- **Generative Adversarial Networks (GANs)**: Generator and discriminator networks in competition
- **Autoregressive Models**: Generate outputs one element at a time (GPT for text)
- **Flow-based Models**: Transform simple distributions into complex ones via invertible functions
- **Hybrid Architectures**: Combine multiple approaches for different modalities

## Retrieval-Augmented Generation (RAG)

RAG is an approach that enhances LLMs by incorporating external knowledge retrieval to ground responses in specific information sources.

### RAG Components:
- **Retriever**: Searches and fetches relevant information from knowledge sources
- **Generator**: Language model that produces responses using retrieved information
- **Knowledge base**: Collection of documents, databases, or other information sources

### RAG Benefits:
- **Factual accuracy**: Reduces hallucinations by grounding in external data
- **Customization**: Adapts models to specific domains without fine-tuning
- **Up-to-date information**: Can access current information beyond training data
- **Source attribution**: Enables citing sources for generated content
- **Efficiency**: Reduces need for larger models by leveraging external knowledge

### RAG Implementation Steps:
1. **Document ingestion**: Process and chunk documents into a knowledge base
2. **Embedding generation**: Convert text into vector representations
3. **Vector storage**: Store embeddings in a vector database
4. **Query processing**: Transform user queries into searchable format
5. **Retrieval**: Find relevant documents using semantic search
6. **Augmentation**: Combine retrieved information with user query
7. **Generation**: Produce response using the augmented context

### RAG Architectures:
- **Classic RAG**: Original architecture with separate retrieval and generation phases
- **Hybrid RAG**: Combines dense and sparse retrieval methods
- **Recursive RAG**: Iteratively refines retrieval through multiple passes
- **Adaptive RAG**: Dynamically adjusts retrieval strategy based on query
- **Multi-vector RAG**: Uses different embedding approaches for different content types
- **Reranking RAG**: Adds a reranking step after initial retrieval to improve relevance
- **Self-querying RAG**: Generates multiple search queries from the original question
- **Agent RAG**: Integrates RAG within an agent framework for more complex reasoning