import { pipeline, cos_sim } from '@xenova/transformers';

class RAGSystem {
    constructor() {
        this.embedder = null;
        this.documents = [];
        this.embeddings = [];
    }

    async initialize() {
        // Initialize the embedding model
        this.embedder = await pipeline('feature-extraction', 'Xenova/jina-embeddings-v2-base-en', {
            quantized: false
        });
    }

    // Function to process new text content
    async processText(text) {
        try {
            // Split text into chunks (simple splitting by sentences)
            const chunks = text.match(/[^.!?]+[.!?]+/g) || [text];
            const cleanChunks = chunks.map(chunk => chunk.trim()).filter(chunk => chunk.length > 0);
            
            // Store the chunks
            this.documents.push(...cleanChunks);
            
            // Compute embeddings for each chunk
            for (const chunk of cleanChunks) {
                const output = await this.embedder(chunk, { pooling: 'mean' });
                this.embeddings.push({
                    text: chunk,
                    embedding: output.data
                });
            }
            
            return cleanChunks.length;
        } catch (error) {
            console.error('Error processing text:', error);
            throw error;
        }
    }

    // Function to find relevant context for a query
    async findRelevantContext(query, topK = 3) {
        if (this.embeddings.length === 0) {
            return [];
        }

        // Get query embedding
        const queryOutput = await this.embedder(query, { pooling: 'mean' });
        
        // Calculate similarities
        const similarities = this.embeddings.map((doc, index) => ({
            index,
            score: cos_sim(queryOutput.data, doc.embedding),
            text: doc.text
        }));

        // Sort by similarity score
        similarities.sort((a, b) => b.score - a.score);
        
        // Return top K most similar chunks
        return similarities.slice(0, topK).map(item => item.text);
    }

    // Clear all stored documents and embeddings
    clear() {
        this.documents = [];
        this.embeddings = [];
    }
}

export const ragSystem = new RAGSystem();
